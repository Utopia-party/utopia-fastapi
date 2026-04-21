from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.redis_lock import redis_lock
from models.party import Party, PartyEmbedding, PartyMember
from models.quick_match.candidate import QuickMatchCandidate, QuickMatchCandidateStatus
from models.quick_match.embedding import PartyMatchEmbedding
from models.quick_match.request import QuickMatchRequest, QuickMatchRequestStatus
from models.quick_match.result import QuickMatchResult
from models.user import User
from services.quick_match.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class QuickMatchService:
    async def create_request(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        service_id: uuid.UUID,
        preferred_conditions: dict | None,
    ):
        start_time = time.perf_counter()

        user = await db.get(User, user_id)
        if not user:
            raise Exception("USER_NOT_FOUND")

        if not user.is_active:
            raise Exception("USER_INACTIVE")

        if user.banned_until and user.banned_until > datetime.now(timezone.utc):
            raise Exception("USER_BANNED")

        existing = await db.execute(
            select(QuickMatchRequest).where(
                QuickMatchRequest.user_id == user_id,
                QuickMatchRequest.service_id == service_id,
                QuickMatchRequest.is_active.is_(True),
            )
        )
        if existing.scalar_one_or_none():
            raise Exception("ALREADY_REQUESTED")

        active_member = await db.execute(
            select(PartyMember)
            .join(Party, PartyMember.party_id == Party.id)
            .where(
                PartyMember.user_id == user_id,
                PartyMember.status == "active",
                Party.service_id == service_id,
            )
        )
        if active_member.scalar_one_or_none():
            raise Exception("ALREADY_IN_ACTIVE_PARTY")

        normalized_conditions = self._normalize_preferred_conditions(preferred_conditions)
        ai_profile = await self._build_user_ai_profile(
            db=db,
            user=user,
            service_id=service_id,
            preferred_conditions=normalized_conditions,
        )

        summary = await EmbeddingService.generate_profile_summary(ai_profile)
        embedding_vector = await EmbeddingService.generate_embedding({"text": summary})

        existing_embedding_result = await db.execute(
            select(PartyMatchEmbedding).where(
                PartyMatchEmbedding.user_id == user_id,
                PartyMatchEmbedding.service_id == service_id,
            )
        )
        embedding = existing_embedding_result.scalar_one_or_none()

        if embedding:
            embedding.embedding_vector = embedding_vector
            embedding.source_snapshot = ai_profile
            embedding.last_generated_at = datetime.now(timezone.utc)
        else:
            embedding = PartyMatchEmbedding(
                user_id=user_id,
                service_id=service_id,
                embedding_vector=embedding_vector,
                source_snapshot=ai_profile,
                last_generated_at=datetime.now(timezone.utc),
            )
            db.add(embedding)

        request = QuickMatchRequest(
            user_id=user_id,
            service_id=service_id,
            status=QuickMatchRequestStatus.REQUESTED,
            preferred_conditions=normalized_conditions,
            ai_profile_snapshot=ai_profile,
            requested_at=datetime.now(timezone.utc),
            expired_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            is_active=True,
        )

        db.add(request)
        await db.commit()
        await db.refresh(request)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "[QuickMatch] create_request done request_id=%s user_id=%s service_id=%s elapsed=%.3fs",
            request.id,
            user_id,
            service_id,
            elapsed,
        )

        return request

    async def find_candidates(
        self,
        db: AsyncSession,
        request_id: uuid.UUID,
    ):
        start_time = time.perf_counter()

        request = await db.get(QuickMatchRequest, request_id)
        if not request:
            raise Exception("REQUEST_NOT_FOUND")

        logger.info(
            "[QuickMatch] find_candidates start request_id=%s user_id=%s service_id=%s",
            request_id,
            request.user_id,
            request.service_id,
        )

        if request.status != QuickMatchRequestStatus.REQUESTED:
            raise Exception("INVALID_REQUEST_STATUS")

        user = await db.get(User, request.user_id)
        if not user:
            raise Exception("USER_NOT_FOUND")

        embedding_result = await db.execute(
            select(PartyMatchEmbedding).where(
                PartyMatchEmbedding.user_id == request.user_id,
                PartyMatchEmbedding.service_id == request.service_id,
            )
        )
        user_embedding = embedding_result.scalar_one_or_none()

        existing_candidates = await db.execute(
            select(QuickMatchCandidate).where(
                QuickMatchCandidate.request_id == request.id,
            )
        )
        for row in existing_candidates.scalars().all():
            await db.delete(row)
        await db.flush()

        party_result = await db.execute(
            select(Party)
            .options(selectinload(Party.service))
            .where(
                Party.status == "recruiting",
                Party.service_id == request.service_id,
            )
        )
        parties = party_result.scalars().all()

        if not parties:
            elapsed = time.perf_counter() - start_time
            logger.warning(
                "[QuickMatch] no recruiting party request_id=%s elapsed=%.3fs",
                request.id,
                elapsed,
            )
            await self.fail_request(
                db=db,
                request_id=request.id,
                reason="NO_RECRUITING_PARTY",
            )
            raise Exception("NO_RECRUITING_PARTY")

        existing_members_result = await db.execute(
            select(PartyMember.party_id).where(
                PartyMember.user_id == request.user_id,
            )
        )
        joined_party_ids = set(existing_members_result.scalars().all())

        preferred_conditions = self._normalize_preferred_conditions(request.preferred_conditions)
        user_trust_score = float(getattr(user, "trust_score", 0) or 0)
        user_profile = request.ai_profile_snapshot or {}

        normal_candidates_base: list[dict[str, Any]] = []
        fallback_candidates: list[dict[str, Any]] = []

        for party in parties:
            filter_reasons: dict[str, Any] = {
                "service_match": True,
                "recruiting_status": party.status == "recruiting",
            }

            party_max_members = int(getattr(party, "max_members", 0) or 0)
            party_current_members = int(getattr(party, "current_members", 0) or 0)

            remaining_seat = max((party_max_members - party_current_members), 0)
            filter_reasons["remaining_seat"] = remaining_seat
            if party_current_members >= party_max_members:
                self._reject_candidate(
                    db=db,
                    request_id=request.id,
                    party_id=party.id,
                    filter_reasons=filter_reasons,
                    reason="party_full",
                )
                continue

            min_trust_score = float(getattr(party, "min_trust_score", 0) or 0)
            filter_reasons["party_min_trust_score"] = min_trust_score
            filter_reasons["user_trust_score"] = user_trust_score
            if user_trust_score < min_trust_score:
                self._reject_candidate(
                    db=db,
                    request_id=request.id,
                    party_id=party.id,
                    filter_reasons=filter_reasons,
                    reason="trust_score_too_low",
                )
                continue

            if self._is_policy_excluded(user=user, party=party):
                self._reject_candidate(
                    db=db,
                    request_id=request.id,
                    party_id=party.id,
                    filter_reasons=filter_reasons,
                    reason="policy_excluded",
                )
                continue

            if party.id in joined_party_ids:
                self._reject_candidate(
                    db=db,
                    request_id=request.id,
                    party_id=party.id,
                    filter_reasons=filter_reasons,
                    reason="already_member",
                )
                continue

            rule_score, rule_reason = self._calculate_rule_score(
                party=party,
                user_trust_score=user_trust_score,
                preferred_conditions=preferred_conditions,
            )
            filter_reasons["rule_reason"] = rule_reason

            fallback_ok, fallback_reason = self._matches_fallback_core_conditions(
                party=party,
                preferred_conditions=preferred_conditions,
            )
            filter_reasons["fallback_core_reason"] = fallback_reason

            party_profile = self._build_party_profile(party)

            if user_embedding and user_embedding.embedding_vector:
                party_embedding = await self._get_or_create_party_embedding(
                    db=db,
                    party=party,
                    party_profile=party_profile,
                )

                if party_embedding and party_embedding.embedding_vector:
                    vector_score = self._calculate_vector_score(
                        user_embedding.embedding_vector,
                        party_embedding.embedding_vector,
                    )

                    normal_candidates_base.append(
                        {
                            "party": party,
                            "rule_score": rule_score,
                            "vector_score": vector_score,
                            "party_profile": party_profile,
                            "filter_reasons": dict(filter_reasons),
                        }
                    )
                else:
                    filter_reasons["normal_match_unavailable_reason"] = "party_embedding_not_found"
                    self._reject_candidate(
                        db=db,
                        request_id=request.id,
                        party_id=party.id,
                        filter_reasons=filter_reasons,
                        reason="party_embedding_not_found",
                    )
            else:
                filter_reasons["normal_match_unavailable_reason"] = "user_embedding_not_found"
                logger.warning(
                    "[QuickMatch] user embedding not found request_id=%s user_id=%s",
                    request.id,
                    request.user_id,
                )

            if fallback_ok:
                fallback_filter_reasons = dict(filter_reasons)
                fallback_filter_reasons["match_mode"] = "fallback"

                fallback_score = self._calculate_fallback_score(
                    party=party,
                    user_trust_score=user_trust_score,
                    preferred_conditions=preferred_conditions,
                )

                fallback_candidates.append(
                    {
                        "party": party,
                        "rule_score": rule_score,
                        "vector_score": 0.0,
                        "llm_score": 0.0,
                        "ai_score": fallback_score,
                        "filter_reasons": fallback_filter_reasons,
                    }
                )

        normal_candidates: list[dict[str, Any]] = []
        if normal_candidates_base:
            llm_results = await asyncio.gather(
                *[
                    EmbeddingService.generate_match_evaluation(
                        {
                            "user_profile": user_profile,
                            "party_profile": item["party_profile"],
                            "rule_score": item["rule_score"],
                            "vector_score": item["vector_score"],
                        }
                    )
                    for item in normal_candidates_base
                ],
                return_exceptions=True,
            )

            for item, llm_result in zip(normal_candidates_base, llm_results):
                if isinstance(llm_result, Exception):
                    llm_score = round(
                        min(
                            1.0,
                            max(
                                0.0,
                                (float(item["rule_score"]) * 0.5)
                                + (float(item["vector_score"]) * 0.5),
                            ),
                        ),
                        4,
                    )
                    llm_reason = "LLM 호출 실패로 rule/vector 기반 대체 점수 사용"
                else:
                    llm_score = float(llm_result.get("score", 0) or 0)
                    llm_reason = llm_result.get("reason")

                normal_filter_reasons = dict(item["filter_reasons"])
                normal_filter_reasons["llm_reason"] = llm_reason
                normal_filter_reasons["match_mode"] = "normal"

                ai_score = self._calculate_ai_score(
                    rule_score=float(item["rule_score"]),
                    vector_score=float(item["vector_score"]),
                    llm_score=llm_score,
                )

                normal_candidates.append(
                    {
                        "party": item["party"],
                        "rule_score": float(item["rule_score"]),
                        "vector_score": float(item["vector_score"]),
                        "llm_score": llm_score,
                        "ai_score": ai_score,
                        "filter_reasons": normal_filter_reasons,
                    }
                )

        selected_pool: list[dict[str, Any]] = []
        if normal_candidates:
            selected_pool = normal_candidates
        elif fallback_candidates:
            selected_pool = fallback_candidates

        if not selected_pool:
            rejected_result = await db.execute(
                select(QuickMatchCandidate).where(
                    QuickMatchCandidate.request_id == request.id,
                    QuickMatchCandidate.status == QuickMatchCandidateStatus.REJECTED,
                )
            )
            rejected_candidates = rejected_result.scalars().all()

            reason_counts: dict[str, int] = {}
            for candidate in rejected_candidates:
                excluded_reason = (candidate.filter_reasons or {}).get("excluded_reason", "unknown")
                reason_counts[excluded_reason] = reason_counts.get(excluded_reason, 0) + 1

            elapsed = time.perf_counter() - start_time
            logger.warning(
                "[QuickMatch] no candidate request_id=%s checked_parties=%s rejected=%s reason_counts=%s elapsed=%.3fs",
                request.id,
                len(parties),
                len(rejected_candidates),
                reason_counts,
                elapsed,
            )

            await self.fail_request(
                db=db,
                request_id=request.id,
                reason="NO_CANDIDATE",
            )
            raise Exception("NO_CANDIDATE")

        selected_pool.sort(
            key=lambda item: (
                item["ai_score"],
                item["llm_score"],
                item["vector_score"],
                item["rule_score"],
            ),
            reverse=True,
        )

        created_candidates: list[QuickMatchCandidate] = []

        for idx, item in enumerate(selected_pool, start=1):
            status = QuickMatchCandidateStatus.SELECTED if idx == 1 else QuickMatchCandidateStatus.PENDING

            candidate = QuickMatchCandidate(
                request_id=request.id,
                party_id=item["party"].id,
                rule_score=item["rule_score"],
                vector_score=item["vector_score"],
                llm_score=item["llm_score"],
                ai_score=item["ai_score"],
                rank=idx,
                status=status,
                filter_reasons=item["filter_reasons"],
            )
            db.add(candidate)
            created_candidates.append(candidate)

        await db.commit()

        for candidate in created_candidates:
            await db.refresh(candidate)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "[QuickMatch] find_candidates done request_id=%s selected_candidates=%s normal_candidates=%s fallback_candidates=%s elapsed=%.3fs",
            request.id,
            len(created_candidates),
            len(normal_candidates),
            len(fallback_candidates),
            elapsed,
        )

        return created_candidates

    async def select_party(
        self,
        db: AsyncSession,
        request_id: uuid.UUID,
    ):
        start_time = time.perf_counter()

        result = await db.execute(
            select(QuickMatchCandidate).where(QuickMatchCandidate.request_id == request_id)
        )
        candidates = result.scalars().all()

        if not candidates:
            raise Exception("NO_CANDIDATE")

        candidates.sort(
            key=lambda candidate: (
                self._get_match_mode_priority(candidate.filter_reasons),
                float(candidate.ai_score),
                float(candidate.llm_score),
                float(candidate.vector_score),
                float(candidate.rule_score),
            ),
            reverse=True,
        )
        candidate = candidates[0]

        request = await db.get(QuickMatchRequest, request_id)
        if not request:
            raise Exception("REQUEST_NOT_FOUND")

        party = await db.get(Party, candidate.party_id)
        if not party:
            raise Exception("PARTY_NOT_FOUND")

        party_current_members = int(getattr(party, "current_members", 0) or 0)
        party_max_members = int(getattr(party, "max_members", 0) or 0)

        if party.status != "recruiting":
            raise Exception("PARTY_STATUS_CHANGED")

        if party_current_members >= party_max_members:
            raise Exception("PARTY_FULL")

        request.status = QuickMatchRequestStatus.MATCHED
        request.matched_party_id = candidate.party_id
        request.matched_at = datetime.now(timezone.utc)
        request.is_active = False

        candidate.status = QuickMatchCandidateStatus.SELECTED

        existing_result = await db.execute(
            select(QuickMatchResult).where(QuickMatchResult.request_id == request.id)
        )
        result_row = existing_result.scalar_one_or_none()

        if result_row is None:
            result_row = QuickMatchResult(request_id=request.id)
            db.add(result_row)

        result_row.selected_party_id = candidate.party_id
        result_row.selected_candidate_id = candidate.id
        result_row.request_snapshot = {
            "user_id": str(request.user_id),
            "service_id": str(request.service_id),
            "preferred_conditions": request.preferred_conditions,
            "ai_profile_snapshot": request.ai_profile_snapshot,
        }
        result_row.candidate_snapshot = {
            "party_id": str(candidate.party_id),
            "rank": candidate.rank,
            "status": candidate.status.value if hasattr(candidate.status, "value") else str(candidate.status),
            "filter_reasons": candidate.filter_reasons,
        }
        result_row.final_scores = {
            "rule_score": float(candidate.rule_score),
            "vector_score": float(candidate.vector_score),
            "llm_score": float(candidate.llm_score),
            "final_score": float(candidate.ai_score),
        }
        result_row.decision_reason = self._build_decision_reason(candidate)

        await db.commit()
        await db.refresh(result_row)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "[QuickMatch] select_party done request_id=%s party_id=%s candidate_id=%s final_score=%.4f elapsed=%.3fs",
            request.id,
            candidate.party_id,
            candidate.id,
            float(candidate.ai_score),
            elapsed,
        )

        return result_row

    async def join_party(
        self,
        db: AsyncSession,
        request_id: uuid.UUID,
    ):
        start_time = time.perf_counter()

        request = await db.get(QuickMatchRequest, request_id)
        if not request:
            raise Exception("REQUEST_NOT_FOUND")

        if request.status not in [
            QuickMatchRequestStatus.MATCHED,
            QuickMatchRequestStatus.REMATCHING,
        ]:
            raise Exception("REQUEST_NOT_MATCHED")

        if not request.matched_party_id:
            raise Exception("MATCHED_PARTY_NOT_FOUND")

        party = await db.get(Party, request.matched_party_id)
        if not party:
            raise Exception("PARTY_NOT_FOUND")

        lock_key = f"quick_match_lock:{party.id}"

        async with redis_lock(lock_key=lock_key, lock_value=str(request.id), expire_seconds=30):
            await db.refresh(party)

            party_current_members = int(getattr(party, "current_members", 0) or 0)
            party_max_members = int(getattr(party, "max_members", 0) or 0)

            if party.status != "recruiting":
                logger.warning(
                    "[QuickMatch] join failed status changed request_id=%s party_id=%s",
                    request.id,
                    party.id,
                )
                await self.fail_request(db, request.id, "PARTY_STATUS_CHANGED")
                return await self.retry_match(db, request.id)

            if party_current_members >= party_max_members:
                logger.warning(
                    "[QuickMatch] join failed full party request_id=%s party_id=%s",
                    request.id,
                    party.id,
                )
                await self.fail_request(db, request.id, "PARTY_FULL")
                return await self.retry_match(db, request.id)

            existing_member_result = await db.execute(
                select(PartyMember).where(
                    PartyMember.party_id == party.id,
                    PartyMember.user_id == request.user_id,
                )
            )
            existing_member = existing_member_result.scalar_one_or_none()
            if existing_member:
                logger.warning(
                    "[QuickMatch] join failed already joined request_id=%s party_id=%s user_id=%s",
                    request.id,
                    party.id,
                    request.user_id,
                )
                await self.fail_request(db, request.id, "ALREADY_JOINED")
                return await self.retry_match(db, request.id)

            now = datetime.utcnow()

            matched_at = request.matched_at
            if matched_at is None:
                matched_at = now
            elif matched_at.tzinfo is not None:
                matched_at = matched_at.replace(tzinfo=None)

            new_member = PartyMember(
                party_id=party.id,
                user_id=request.user_id,
                role="member",
                status="active",
                joined_at=now,
                join_type="quick_match",
                match_request_id=request.id,
                matched_at=matched_at,
                approved_at=now,
                leader_review_status="approved",
            )
            db.add(new_member)

            party.current_members = party_current_members + 1
            request.status = QuickMatchRequestStatus.MATCHED
            request.is_active = False

            await db.commit()
            await db.refresh(new_member)

            elapsed = time.perf_counter() - start_time
            logger.info(
                "[QuickMatch] join_party done request_id=%s party_id=%s user_id=%s current_members=%s elapsed=%.3fs",
                request.id,
                party.id,
                request.user_id,
                party.current_members,
                elapsed,
            )

            return {
                "party_member_id": new_member.id,
                "party_id": party.id,
                "user_id": request.user_id,
                "status": new_member.status,
                "join_type": new_member.join_type,
                "current_members": party.current_members,
            }

    async def fail_request(
        self,
        db: AsyncSession,
        request_id: uuid.UUID,
        reason: str,
    ):
        request = await db.get(QuickMatchRequest, request_id)
        if not request:
            raise Exception("REQUEST_NOT_FOUND")

        request.status = QuickMatchRequestStatus.FAILED
        request.fail_reason = reason
        request.retry_count += 1
        request.is_active = False

        await db.commit()
        await db.refresh(request)

        logger.warning(
            "[QuickMatch] fail_request request_id=%s reason=%s retry_count=%s",
            request.id,
            reason,
            request.retry_count,
        )

        return request

    async def retry_match(
        self,
        db: AsyncSession,
        request_id: uuid.UUID,
    ):
        request = await db.get(QuickMatchRequest, request_id)
        if not request:
            raise Exception("REQUEST_NOT_FOUND")

        if request.retry_count >= 3:
            request.status = QuickMatchRequestStatus.EXPIRED
            request.is_active = False
            request.fail_reason = "MAX_RETRY_EXCEEDED"

            await db.commit()
            await db.refresh(request)

            logger.warning(
                "[QuickMatch] retry expired request_id=%s reason=%s",
                request.id,
                request.fail_reason,
            )

            return request

        result = await db.execute(
            select(QuickMatchCandidate).where(
                QuickMatchCandidate.request_id == request_id,
                QuickMatchCandidate.status.in_(
                    [
                        QuickMatchCandidateStatus.PENDING,
                        QuickMatchCandidateStatus.SELECTED,
                    ]
                ),
            )
        )

        candidates = result.scalars().all()
        if not candidates:
            request.status = QuickMatchRequestStatus.FAILED
            request.is_active = False
            request.fail_reason = "NO_MORE_CANDIDATES"

            await db.commit()
            await db.refresh(request)

            logger.warning(
                "[QuickMatch] retry failed no more candidates request_id=%s",
                request.id,
            )

            return request

        candidates.sort(
            key=lambda candidate: (
                self._get_match_mode_priority(candidate.filter_reasons),
                float(candidate.ai_score),
                float(candidate.llm_score),
                float(candidate.vector_score),
                float(candidate.rule_score),
            ),
            reverse=True,
        )

        current_selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.party_id == request.matched_party_id
                and candidate.status == QuickMatchCandidateStatus.SELECTED
            ),
            None,
        )
        if current_selected:
            current_selected.status = QuickMatchCandidateStatus.FAILED

        next_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.party_id != request.matched_party_id
                and candidate.status in {
                    QuickMatchCandidateStatus.PENDING,
                    QuickMatchCandidateStatus.SELECTED,
                }
            ),
            None,
        )

        if not next_candidate:
            request.status = QuickMatchRequestStatus.FAILED
            request.is_active = False
            request.fail_reason = "NO_MORE_CANDIDATES"

            await db.commit()
            await db.refresh(request)

            logger.warning(
                "[QuickMatch] retry failed no selectable candidate request_id=%s",
                request.id,
            )

            return request

        next_candidate.status = QuickMatchCandidateStatus.SELECTED

        request.matched_party_id = next_candidate.party_id
        request.status = QuickMatchRequestStatus.REMATCHING
        request.retry_count += 1
        request.is_active = True
        request.fail_reason = None

        await db.commit()

        logger.info(
            "[QuickMatch] retry_match next candidate request_id=%s next_party_id=%s retry_count=%s",
            request.id,
            next_candidate.party_id,
            request.retry_count,
        )

        return {
            "request_id": request.id,
            "next_party_id": next_candidate.party_id,
            "retry_count": request.retry_count,
            "status": request.status.value,
        }

    async def _build_user_ai_profile(
        self,
        db: AsyncSession,
        user: User,
        service_id: uuid.UUID,
        preferred_conditions: dict[str, Any],
    ) -> dict[str, Any]:
        member_result = await db.execute(
            select(PartyMember, Party)
            .join(Party, PartyMember.party_id == Party.id)
            .where(PartyMember.user_id == user.id)
        )

        memberships = member_result.all()

        service_membership_count = sum(
            1
            for membership, party in memberships
            if str(party.service_id) == str(service_id)
        )

        total_membership_count = len(memberships)

        active_membership_count = sum(
            1
            for membership, _ in memberships
            if getattr(membership, "status", None) == "active"
        )

        average_payment_amount = float(
            getattr(user, "average_payment_amount", 0)
            or getattr(user, "avg_payment_amount", 0)
            or 0
        )
        settlement_success_count = int(getattr(user, "settlement_success_count", 0) or 0)
        report_count = int(getattr(user, "report_count", 0) or 0)
        leave_count = int(getattr(user, "leave_count", 0) or 0)

        return {
            "user_id": str(user.id),
            "service_id": str(service_id),
            "trust_score": float(getattr(user, "trust_score", 0) or 0),
            "preferred_conditions": preferred_conditions,
            "activity_summary": {
                "total_party_join_count": total_membership_count,
                "service_party_join_count": service_membership_count,
                "active_party_count": active_membership_count,
                "preferred_service_id": str(service_id),
            },
            "payment_summary": {
                "average_payment_amount": average_payment_amount,
                "settlement_success_count": settlement_success_count,
            },
            "risk_summary": {
                "report_count": report_count,
                "leave_count": leave_count,
                "is_currently_banned": bool(user.banned_until and user.banned_until > datetime.now(timezone.utc)),
            },
        }

    def _build_party_profile(self, party: Party) -> dict[str, Any]:
        return {
            "party_id": str(party.id),
            "service_name": getattr(getattr(party, "service", None), "name", None),
            "monthly_per_person": float(getattr(party, "monthly_per_person", 0) or 0),
            "min_trust_score": float(getattr(party, "min_trust_score", 0) or 0),
            "max_members": int(getattr(party, "max_members", 0) or 0),
            "current_members": int(getattr(party, "current_members", 0) or 0),
            "description": getattr(party, "description", "") or getattr(party, "intro", ""),
            "duration_preference": getattr(party, "duration_preference", None),
            "status": getattr(party, "status", None),
        }

    async def _get_or_create_party_embedding(
        self,
        db: AsyncSession,
        party: Party,
        party_profile: dict[str, Any],
    ):
        party_embedding_result = await db.execute(
            select(PartyEmbedding).where(PartyEmbedding.party_id == party.id)
        )
        party_embedding = party_embedding_result.scalar_one_or_none()

        if party_embedding and party_embedding.embedding_vector:
            return party_embedding

        embedding_vector = await EmbeddingService.generate_party_embedding(party_profile)
        if not embedding_vector:
            return party_embedding

        if party_embedding:
            party_embedding.embedding_vector = embedding_vector
            if hasattr(party_embedding, "source_snapshot"):
                setattr(party_embedding, "source_snapshot", party_profile)
            if hasattr(party_embedding, "last_generated_at"):
                setattr(party_embedding, "last_generated_at", datetime.now(timezone.utc))
        else:
            party_embedding = PartyEmbedding(
                party_id=party.id,
                service_id=party.service_id,
                embedding_vector=embedding_vector,
            )
            if hasattr(party_embedding, "source_snapshot"):
                setattr(party_embedding, "source_snapshot", party_profile)
            if hasattr(party_embedding, "last_generated_at"):
                setattr(party_embedding, "last_generated_at", datetime.now(timezone.utc))
            db.add(party_embedding)

        await db.flush()
        return party_embedding

    def _reject_candidate(
        self,
        db: AsyncSession,
        request_id: uuid.UUID,
        party_id: uuid.UUID,
        filter_reasons: dict[str, Any],
        reason: str,
    ) -> None:
        rejected_reasons = dict(filter_reasons)
        rejected_reasons["excluded_reason"] = reason

        logger.info(
            "[QuickMatch] candidate rejected request_id=%s party_id=%s reason=%s details=%s",
            request_id,
            party_id,
            reason,
            rejected_reasons,
        )

        candidate = QuickMatchCandidate(
            request_id=request_id,
            party_id=party_id,
            rule_score=0,
            vector_score=0,
            llm_score=0,
            ai_score=0,
            rank=None,
            status=QuickMatchCandidateStatus.REJECTED,
            filter_reasons=rejected_reasons,
        )
        db.add(candidate)

    def _normalize_preferred_conditions(self, preferred_conditions: dict[str, Any] | None) -> dict[str, Any]:
        normalized = dict(preferred_conditions or {})

        if "estimated_price" in normalized and "price_range" not in normalized:
            normalized["price_range"] = normalized.pop("estimated_price")

        duration_preference = normalized.get("duration_preference")
        if isinstance(duration_preference, str):
            normalized["duration_preference"] = duration_preference.strip().lower()

        price_range = normalized.get("price_range")
        if isinstance(price_range, str):
            normalized["price_range"] = price_range.strip()

        return normalized

    def _is_policy_excluded(self, user: User, party: Party) -> bool:
        user_report_count = int(getattr(user, "report_count", 0) or 0)
        user_blocked = bool(getattr(user, "is_blocked_for_matching", False))
        party_blocked = bool(getattr(party, "is_blocked_for_matching", False))
        party_report_limit = int(getattr(party, "max_reported_user_count", 9999) or 9999)

        if user_blocked or party_blocked:
            return True

        if user_report_count > party_report_limit:
            return True

        return False

    def _build_decision_reason(self, candidate: QuickMatchCandidate) -> str:
        filter_reasons = candidate.filter_reasons or {}
        match_mode = str(filter_reasons.get("match_mode", "normal")).lower()

        if match_mode == "fallback":
            return (
                f"일반 AI 후보 부족으로 fallback 조건 충족 파티 선정 "
                f"(final={float(candidate.ai_score):.4f}, "
                f"rule={float(candidate.rule_score):.4f}, "
                f"vector={float(candidate.vector_score):.4f}, "
                f"llm={float(candidate.llm_score):.4f})"
            )

        return (
            f"최종 점수 {float(candidate.ai_score):.4f}로 1순위 선정 "
            f"(rule={float(candidate.rule_score):.4f}, "
            f"vector={float(candidate.vector_score):.4f}, "
            f"llm={float(candidate.llm_score):.4f})"
        )

    def _calculate_rule_score(
        self,
        party: Party,
        user_trust_score: float,
        preferred_conditions: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        score = 0.0
        detail: dict[str, Any] = {}

        min_trust_score = float(getattr(party, "min_trust_score", 0) or 0)
        if min_trust_score <= 0:
            trust_fit_score = 1.0
        elif user_trust_score >= min_trust_score:
            margin = min(user_trust_score - min_trust_score, 20)
            trust_fit_score = min(1.0, 0.7 + (margin / 20) * 0.3)
        else:
            trust_fit_score = 0.0

        score += trust_fit_score * 0.35
        detail["trust_fit_score"] = round(trust_fit_score, 4)

        party_max_members = float(getattr(party, "max_members", 0) or 0)
        party_current_members = float(getattr(party, "current_members", 0) or 0)

        if party_max_members <= 0:
            capacity_score = 0.0
        else:
            remaining = max((party_max_members - party_current_members), 0)
            capacity_score = min(1.0, remaining / max(party_max_members, 1))

        score += capacity_score * 0.2
        detail["capacity_score"] = round(capacity_score, 4)

        preferred_price = preferred_conditions.get("price_range")
        monthly_price = float(getattr(party, "monthly_per_person", 0) or 0)

        price_score = self._calculate_price_score(
            monthly_price=monthly_price,
            preferred_price=preferred_price,
        )
        score += price_score * 0.3
        detail["price_score"] = round(price_score, 4)
        detail["preferred_price"] = preferred_price
        detail["monthly_price"] = monthly_price

        duration_score = self._calculate_duration_score(
            party_duration_preference=getattr(party, "duration_preference", None),
            user_duration_preference=preferred_conditions.get("duration_preference"),
        )
        score += duration_score * 0.15
        detail["duration_score"] = round(duration_score, 4)
        detail["user_duration_preference"] = preferred_conditions.get("duration_preference")
        detail["party_duration_preference"] = getattr(party, "duration_preference", None)

        return round(min(score, 1.0), 4), detail

    def _calculate_price_score(
        self,
        monthly_price: float,
        preferred_price: str | None,
    ) -> float:
        if not preferred_price:
            return 0.7

        try:
            if "-" in preferred_price:
                low_str, high_str = preferred_price.split("-", 1)
                low = float(low_str.strip())
                high = float(high_str.strip())

                if low <= monthly_price <= high:
                    return 1.0

                if monthly_price < low:
                    diff_ratio = (low - monthly_price) / max(low, 1)
                else:
                    diff_ratio = (monthly_price - high) / max(high, 1)

                if diff_ratio <= 0.1:
                    return 0.8
                if diff_ratio <= 0.2:
                    return 0.6
                if diff_ratio <= 0.3:
                    return 0.4
                return 0.2
        except Exception:
            return 0.5

        return 0.5

    def _calculate_duration_score(
        self,
        party_duration_preference: str | None,
        user_duration_preference: str | None,
    ) -> float:
        if not user_duration_preference:
            return 0.7

        normalized_user = str(user_duration_preference).strip().lower()
        normalized_party = str(party_duration_preference).strip().lower() if party_duration_preference else ""

        if not normalized_party:
            return 0.6
        if normalized_user == normalized_party:
            return 1.0
        if {normalized_user, normalized_party} == {"long_term", "flexible"}:
            return 0.8
        if {normalized_user, normalized_party} == {"short_term", "flexible"}:
            return 0.8
        return 0.3

    def _calculate_vector_score(
        self,
        user_embedding: list[float],
        party_embedding: list[float],
    ) -> float:
        if not user_embedding or not party_embedding:
            return 0.0

        dim = min(len(user_embedding), len(party_embedding))
        if dim == 0:
            return 0.0

        a = [float(x) for x in user_embedding[:dim]]
        b = [float(x) for x in party_embedding[:dim]]

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        cosine = dot / (norm_a * norm_b)
        normalized = (cosine + 1) / 2
        return round(max(0.0, min(1.0, normalized)), 4)

    def _calculate_ai_score(
        self,
        rule_score: float,
        vector_score: float,
        llm_score: float,
    ) -> float:
        final_score = (rule_score * 0.4) + (vector_score * 0.3) + (llm_score * 0.3)
        return round(min(final_score, 1.0), 4)

    def _matches_fallback_core_conditions(
        self,
        party: Party,
        preferred_conditions: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {}

        preferred_price = preferred_conditions.get("price_range")
        user_duration_preference = preferred_conditions.get("duration_preference")
        monthly_price = float(getattr(party, "monthly_per_person", 0) or 0)
        party_duration_preference = getattr(party, "duration_preference", None)

        price_core_match = self._is_price_in_requested_range(
            monthly_price=monthly_price,
            preferred_price=preferred_price,
        )
        duration_core_match = self._is_duration_core_match(
            party_duration_preference=party_duration_preference,
            user_duration_preference=user_duration_preference,
        )

        detail["price_core_match"] = price_core_match
        detail["duration_core_match"] = duration_core_match
        detail["preferred_price"] = preferred_price
        detail["monthly_price"] = monthly_price
        detail["user_duration_preference"] = user_duration_preference
        detail["party_duration_preference"] = party_duration_preference

        return price_core_match and duration_core_match, detail

    def _is_price_in_requested_range(
        self,
        monthly_price: float,
        preferred_price: str | None,
    ) -> bool:
        if not preferred_price:
            return True

        try:
            if "-" in preferred_price:
                low_str, high_str = preferred_price.split("-", 1)
                low = float(low_str.strip())
                high = float(high_str.strip())
                return low <= monthly_price <= high
        except Exception:
            return False

        return False

    def _is_duration_core_match(
        self,
        party_duration_preference: str | None,
        user_duration_preference: str | None,
    ) -> bool:
        if not user_duration_preference:
            return True

        normalized_user = str(user_duration_preference).strip().lower()
        normalized_party = str(party_duration_preference).strip().lower() if party_duration_preference else ""

        if not normalized_party:
            return False

        return normalized_user == normalized_party

    def _calculate_fallback_score(
        self,
        party: Party,
        user_trust_score: float,
        preferred_conditions: dict[str, Any],
    ) -> float:
        min_trust_score = float(getattr(party, "min_trust_score", 0) or 0)
        party_max_members = float(getattr(party, "max_members", 0) or 0)
        party_current_members = float(getattr(party, "current_members", 0) or 0)

        if min_trust_score <= 0:
            trust_fit_score = 1.0
        elif user_trust_score >= min_trust_score:
            margin = min(user_trust_score - min_trust_score, 20)
            trust_fit_score = min(1.0, 0.7 + (margin / 20) * 0.3)
        else:
            trust_fit_score = 0.0

        if party_max_members <= 0:
            capacity_score = 0.0
        else:
            remaining = max((party_max_members - party_current_members), 0)
            capacity_score = min(1.0, remaining / max(party_max_members, 1))

        fallback_ok, _ = self._matches_fallback_core_conditions(
            party=party,
            preferred_conditions=preferred_conditions,
        )
        core_score = 1.0 if fallback_ok else 0.0

        final_score = (trust_fit_score * 0.4) + (capacity_score * 0.2) + (core_score * 0.4)
        return round(min(final_score, 1.0), 4)

    def _get_match_mode_priority(self, filter_reasons: dict[str, Any] | None) -> int:
        match_mode = str((filter_reasons or {}).get("match_mode", "normal")).lower()
        return 1 if match_mode == "normal" else 0