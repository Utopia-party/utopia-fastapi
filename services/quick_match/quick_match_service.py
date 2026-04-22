from __future__ import annotations

import logging
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
from services.notifications.party_notification_service import (
    notify_quick_match_completed,
    notify_quick_match_member_joined_to_leader,
)
from services.quick_match.embedding_service import EmbeddingService
from services.quick_match.profile_service import ProfileService
from services.quick_match.scoring_service import ScoringService

logger = logging.getLogger(__name__)


class QuickMatchService:
    LLM_TOP_N = 3

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

        normalized_conditions = ProfileService.normalize_preferred_conditions(
            preferred_conditions
        )
        ai_profile = await ProfileService.build_user_ai_profile(
            db=db,
            user=user,
            service_id=service_id,
            preferred_conditions=normalized_conditions,
        )

        summary_start = time.perf_counter()
        summary = await EmbeddingService.generate_profile_summary(ai_profile)
        logger.info(
            "[QuickMatch] profile summary done user_id=%s elapsed=%.3fs",
            user_id,
            time.perf_counter() - summary_start,
        )

        user_embedding_start = time.perf_counter()
        embedding_vector = await EmbeddingService.generate_embedding({"text": summary})
        logger.info(
            "[QuickMatch] user embedding done user_id=%s dim=%s elapsed=%.3fs",
            user_id,
            len(embedding_vector or []),
            time.perf_counter() - user_embedding_start,
        )

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
            select(PartyMember.party_id)
            .join(Party, PartyMember.party_id == Party.id)
            .where(
                PartyMember.user_id == request.user_id,
                PartyMember.status == "active",
                Party.service_id == request.service_id,
            )
        )
        joined_party_ids = set(existing_members_result.scalars().all())

        preferred_conditions = ProfileService.normalize_preferred_conditions(
            request.preferred_conditions
        )
        user_trust_score = float(getattr(user, "trust_score", 0) or 0)
        user_profile = request.ai_profile_snapshot or {}

        normal_candidates_base: list[dict[str, Any]] = []

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

            rule_score, rule_reason = ScoringService.calculate_rule_score(
                party=party,
                user_trust_score=user_trust_score,
                preferred_conditions=preferred_conditions,
            )
            filter_reasons["rule_reason"] = rule_reason

            party_profile = ProfileService.build_party_profile(party)

            if user_embedding and user_embedding.embedding_vector:
                party_embedding = await self._get_or_create_party_embedding(
                    db=db,
                    party=party,
                    party_profile=party_profile,
                )

                if party_embedding and party_embedding.embedding_vector:
                    vector_score = ScoringService.calculate_vector_score(
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
                    filter_reasons["normal_match_unavailable_reason"] = (
                        "party_embedding_not_found"
                    )
                    self._reject_candidate(
                        db=db,
                        request_id=request.id,
                        party_id=party.id,
                        filter_reasons=filter_reasons,
                        reason="party_embedding_not_found",
                    )
            else:
                filter_reasons["normal_match_unavailable_reason"] = (
                    "user_embedding_not_found"
                )
                logger.warning(
                    "[QuickMatch] user embedding not found request_id=%s user_id=%s",
                    request.id,
                    request.user_id,
                )

        normal_candidates_base.sort(
            key=lambda item: (
                float(item["vector_score"]),
                float(item["rule_score"]),
            ),
            reverse=True,
        )

        llm_target_candidates = normal_candidates_base[: self.LLM_TOP_N]
        _non_llm_candidates = normal_candidates_base[self.LLM_TOP_N :]

        logger.info(
            "[QuickMatch] llm target request_id=%s total_normal=%s llm_top_n=%s actual_llm_targets=%s",
            request.id,
            len(normal_candidates_base),
            self.LLM_TOP_N,
            len(llm_target_candidates),
        )

        normal_candidates: list[dict[str, Any]] = []
        fallback_candidates: list[dict[str, Any]] = []
        selected_pool: list[dict[str, Any]] = []

        if llm_target_candidates:
            llm_start = time.perf_counter()

            batch_candidates = [
                {
                    "party_id": str(item["party"].id),
                    "party_profile": {
                        "party_id": str(item["party"].id),
                        "service_name": item["party_profile"].get("service_name"),
                        "monthly_per_person": item["party_profile"].get(
                            "monthly_per_person"
                        ),
                        "min_trust_score": item["party_profile"].get("min_trust_score"),
                        "max_members": item["party_profile"].get("max_members"),
                        "current_members": item["party_profile"].get("current_members"),
                        "duration_preference": item["party_profile"].get(
                            "duration_preference"
                        ),
                        "status": item["party_profile"].get("status"),
                    },
                    "rule_score": float(item["rule_score"]),
                    "vector_score": float(item["vector_score"]),
                }
                for item in llm_target_candidates
            ]

            slim_user_profile = {
                "user_id": user_profile.get("user_id"),
                "service_id": user_profile.get("service_id"),
                "trust_score": user_profile.get("trust_score"),
                "preferred_conditions": user_profile.get("preferred_conditions", {}),
                "activity_summary": {
                    "service_party_join_count": (
                        (user_profile.get("activity_summary") or {}).get(
                            "service_party_join_count"
                        )
                    ),
                    "active_party_count": (
                        (user_profile.get("activity_summary") or {}).get(
                            "active_party_count"
                        )
                    ),
                },
                "risk_summary": {
                    "report_count": (
                        (user_profile.get("risk_summary") or {}).get("report_count")
                    ),
                    "leave_count": (
                        (user_profile.get("risk_summary") or {}).get("leave_count")
                    ),
                    "is_currently_banned": (
                        (user_profile.get("risk_summary") or {}).get(
                            "is_currently_banned"
                        )
                    ),
                },
            }

            try:
                llm_results = await EmbeddingService.generate_batch_match_evaluation(
                    {
                        "user_profile": slim_user_profile,
                        "candidates": batch_candidates,
                    }
                )
            except Exception as e:
                logger.exception(
                    "[QuickMatch] batch llm evaluation failed request_id=%s error=%s",
                    request.id,
                    str(e),
                )
                llm_results = []

            logger.info(
                "[QuickMatch] llm batch evaluation done request_id=%s count=%s elapsed=%.3fs",
                request.id,
                len(llm_target_candidates),
                time.perf_counter() - llm_start,
            )

            llm_result_map: dict[str, dict[str, Any]] = {}
            if isinstance(llm_results, list):
                for row in llm_results:
                    if not isinstance(row, dict):
                        continue
                    party_id = str(row.get("party_id") or "")
                    if not party_id:
                        continue
                    llm_result_map[party_id] = row

            for item in llm_target_candidates:
                party_id = str(item["party"].id)
                llm_result = llm_result_map.get(party_id)

                if not llm_result:
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
                    llm_reason = "batch LLM 결과 없음으로 rule/vector 기반 대체 점수 사용"
                else:
                    llm_score = round(float(llm_result.get("score", 0) or 0), 4)
                    llm_reason = llm_result.get("reason") or "batch LLM 평가 완료"

                normal_filter_reasons = dict(item["filter_reasons"])
                normal_filter_reasons["llm_reason"] = llm_reason
                normal_filter_reasons["match_mode"] = "normal"
                normal_filter_reasons["llm_evaluated"] = True
                normal_filter_reasons["llm_batch"] = True

                ai_score = ScoringService.calculate_ai_score(
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

        if normal_candidates:
            selected_pool = normal_candidates
        else:
            logger.info(
                "[QuickMatch] normal candidate empty, start fallback request_id=%s",
                request.id,
            )

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
                    continue

                min_trust_score = float(getattr(party, "min_trust_score", 0) or 0)
                filter_reasons["party_min_trust_score"] = min_trust_score
                filter_reasons["user_trust_score"] = user_trust_score
                if user_trust_score < min_trust_score:
                    continue

                if self._is_policy_excluded(user=user, party=party):
                    continue

                if party.id in joined_party_ids:
                    continue

                rule_score, rule_reason = ScoringService.calculate_rule_score(
                    party=party,
                    user_trust_score=user_trust_score,
                    preferred_conditions=preferred_conditions,
                )
                filter_reasons["rule_reason"] = rule_reason

                fallback_ok, fallback_reason = (
                    ScoringService.matches_fallback_core_conditions(
                        party=party,
                        preferred_conditions=preferred_conditions,
                    )
                )
                filter_reasons["fallback_core_reason"] = fallback_reason

                if not fallback_ok:
                    continue

                fallback_filter_reasons = dict(filter_reasons)
                fallback_filter_reasons["match_mode"] = "fallback"

                fallback_score = ScoringService.calculate_fallback_score(
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
                excluded_reason = (candidate.filter_reasons or {}).get(
                    "excluded_reason",
                    "unknown",
                )
                reason_counts[excluded_reason] = (
                    reason_counts.get(excluded_reason, 0) + 1
                )

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
            status = (
                QuickMatchCandidateStatus.SELECTED
                if idx == 1
                else QuickMatchCandidateStatus.PENDING
            )

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
            select(QuickMatchCandidate).where(
                QuickMatchCandidate.request_id == request_id
            )
        )
        candidates = result.scalars().all()

        if not candidates:
            raise Exception("NO_CANDIDATE")

        candidates.sort(
            key=lambda candidate: (
                ScoringService.get_match_mode_priority(candidate.filter_reasons),
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
            "status": (
                candidate.status.value
                if hasattr(candidate.status, "value")
                else str(candidate.status)
            ),
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

        async with redis_lock(
            lock_key=lock_key,
            lock_value=str(request.id),
            expire_seconds=30,
        ):
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

            user = await db.get(User, request.user_id)

            await notify_quick_match_completed(
                db=db,
                party=party,
                member_user_id=request.user_id,
                match_request_id=request.id,
            )

            await notify_quick_match_member_joined_to_leader(
                db=db,
                party=party,
                member_user_id=request.user_id,
                member_nickname=user.nickname,
                match_request_id=request.id,
            )

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
                ScoringService.get_match_mode_priority(candidate.filter_reasons),
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
                and candidate.status
                in {
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
            logger.info(
                "[QuickMatch] party embedding cache hit party_id=%s",
                party.id,
            )
            return party_embedding

        embedding_start = time.perf_counter()
        embedding_vector = await EmbeddingService.generate_party_embedding(party_profile)
        logger.info(
            "[QuickMatch] party embedding generated party_id=%s dim=%s elapsed=%.3fs",
            party.id,
            len(embedding_vector or []),
            time.perf_counter() - embedding_start,
        )

        if not embedding_vector:
            return party_embedding

        if party_embedding:
            party_embedding.embedding_vector = embedding_vector
            if hasattr(party_embedding, "source_snapshot"):
                setattr(party_embedding, "source_snapshot", party_profile)
            if hasattr(party_embedding, "last_generated_at"):
                setattr(
                    party_embedding,
                    "last_generated_at",
                    datetime.now(timezone.utc),
                )
        else:
            party_embedding = PartyEmbedding(
                party_id=party.id,
                service_id=party.service_id,
                embedding_vector=embedding_vector,
            )
            if hasattr(party_embedding, "source_snapshot"):
                setattr(party_embedding, "source_snapshot", party_profile)
            if hasattr(party_embedding, "last_generated_at"):
                setattr(
                    party_embedding,
                    "last_generated_at",
                    datetime.now(timezone.utc),
                )
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

    def _is_policy_excluded(self, user: User, party: Party) -> bool:
        user_report_count = int(getattr(user, "report_count", 0) or 0)
        user_blocked = bool(getattr(user, "is_blocked_for_matching", False))
        party_blocked = bool(getattr(party, "is_blocked_for_matching", False))
        party_report_limit = int(
            getattr(party, "max_reported_user_count", 9999) or 9999
        )

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