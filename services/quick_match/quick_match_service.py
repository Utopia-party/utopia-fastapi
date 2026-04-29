from __future__ import annotations

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
from services.notifications.party_notification_service import (
    notify_quick_match_completed,
    notify_quick_match_member_joined_to_leader,
)

logger = logging.getLogger(__name__)

# 룰점수 상위 후보만 벡터 유사도 계산 대상으로 사용한다.
TOP_VECTOR_CANDIDATES = 30

DURATION_UNDER_1_MONTH = "under_1_month"
DURATION_1_3_MONTHS = "1_3_months"
DURATION_OVER_3_MONTHS = "over_3_months"
DURATION_FLEXIBLE = "flexible"

DURATION_LABELS = {
    DURATION_UNDER_1_MONTH: "1개월 이하",
    DURATION_1_3_MONTHS: "1~3개월",
    DURATION_OVER_3_MONTHS: "3개월 이상",
    DURATION_FLEXIBLE: "상관없음",
}


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

        # 사용자 임베딩은 기존 것이 있으면 재사용한다.
        existing_embedding_result = await db.execute(
            select(PartyMatchEmbedding).where(
                PartyMatchEmbedding.user_id == user_id,
                PartyMatchEmbedding.service_id == service_id,
            )
        )
        embedding = existing_embedding_result.scalar_one_or_none()

        if embedding and embedding.embedding_vector:
            embedding_vector = embedding.embedding_vector
            if hasattr(embedding, "source_snapshot"):
                embedding.source_snapshot = ai_profile
        else:
            # LLM 요약 없이 ai_profile 자체를 임베딩한다.
            embedding_text = EmbeddingService.serialize_user_profile_text(ai_profile)
            embedding_vector = await EmbeddingService.generate_embedding({"text": embedding_text})

            if embedding:
                embedding.embedding_vector = embedding_vector
                if hasattr(embedding, "source_snapshot"):
                    embedding.source_snapshot = ai_profile
                if hasattr(embedding, "last_generated_at"):
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
            "[QuickMatch] create_request done request_id=%s user_id=%s service_id=%s elapsed=%.3fs embedding_reused=%s",
            request.id,
            user_id,
            service_id,
            elapsed,
            bool(embedding and embedding.embedding_vector),
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
        # 1단계: 하드필터 + 룰점수만 먼저 계산한다.
        # 여기서는 파티 임베딩 조회/벡터 유사도 계산을 하지 않는다.
        rule_scored_candidates: list[dict[str, Any]] = []

        for party in parties:
            filter_reasons: dict[str, Any] = {
                "service_match": True,
                "recruiting_status": party.status == "recruiting",
            }

            hard_filter_ok, hard_filter_detail = self._passes_hard_filters(
                user=user,
                party=party,
                joined_party_ids=joined_party_ids,
                preferred_conditions=preferred_conditions,
                user_trust_score=user_trust_score,
            )
            filter_reasons["hard_filter"] = hard_filter_detail

            if not hard_filter_ok:
                self._reject_candidate(
                    db=db,
                    request_id=request.id,
                    party_id=party.id,
                    filter_reasons=filter_reasons,
                    reason=str(hard_filter_detail.get("excluded_reason", "hard_filter_failed")),
                )
                continue

            rule_score, rule_reason = self._calculate_rule_score(
                party=party,
                user_trust_score=user_trust_score,
                preferred_conditions=preferred_conditions,
            )
            filter_reasons["rule_reason"] = rule_reason

            rule_scored_candidates.append(
                {
                    "party": party,
                    "rule_score": float(rule_score),
                    "party_profile": self._build_party_profile(party),
                    "filter_reasons": dict(filter_reasons),
                }
            )

        if not rule_scored_candidates:
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
                "[QuickMatch] no candidate after hard/rule request_id=%s checked_parties=%s rejected=%s reason_counts=%s elapsed=%.3fs",
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

        # 2단계: 룰점수 기준으로 먼저 정렬하고, 상위 N개만 벡터 유사도 대상으로 사용한다.
        rule_scored_candidates.sort(
            key=lambda item: item["rule_score"],
            reverse=True,
        )
        vector_targets = rule_scored_candidates[:TOP_VECTOR_CANDIDATES]

        if not user_embedding or not user_embedding.embedding_vector:
            for item in vector_targets:
                filter_reasons = dict(item["filter_reasons"])
                filter_reasons["normal_match_unavailable_reason"] = "user_embedding_not_found"
                self._reject_candidate(
                    db=db,
                    request_id=request.id,
                    party_id=item["party"].id,
                    filter_reasons=filter_reasons,
                    reason="user_embedding_not_found",
                )

            await self.fail_request(
                db=db,
                request_id=request.id,
                reason="USER_EMBEDDING_NOT_FOUND",
            )
            raise Exception("USER_EMBEDDING_NOT_FOUND")

        scored_candidates_base: list[dict[str, Any]] = []

        for item in vector_targets:
            party = item["party"]
            filter_reasons = dict(item["filter_reasons"])

            # 파티 임베딩은 빠른매칭 시 생성하지 않고 조회만 한다.
            party_embedding = await self._get_party_embedding(
                db=db,
                party_id=party.id,
            )
            if not party_embedding or not party_embedding.embedding_vector:
                filter_reasons["normal_match_unavailable_reason"] = "party_embedding_not_found"
                self._reject_candidate(
                    db=db,
                    request_id=request.id,
                    party_id=party.id,
                    filter_reasons=filter_reasons,
                    reason="party_embedding_not_found",
                )
                continue

            vector_score = self._calculate_vector_score(
                user_embedding.embedding_vector,
                party_embedding.embedding_vector,
            )

            ai_score = self._calculate_ai_score(
                rule_score=float(item["rule_score"]),
                vector_score=float(vector_score),
            )

            filter_reasons["vector_target"] = True
            filter_reasons["vector_target_limit"] = TOP_VECTOR_CANDIDATES
            filter_reasons["match_mode"] = "normal"
            filter_reasons["score_basis"] = "rule_vector_only"

            scored_candidates_base.append(
                {
                    "party": party,
                    "rule_score": float(item["rule_score"]),
                    "vector_score": float(vector_score),
                    "llm_score": 0.0,
                    "ai_score": ai_score,
                    "filter_reasons": filter_reasons,
                }
            )

        if not scored_candidates_base:
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
                "[QuickMatch] no candidate after vector request_id=%s checked_parties=%s rule_candidates=%s vector_targets=%s rejected=%s reason_counts=%s elapsed=%.3fs",
                request.id,
                len(parties),
                len(rule_scored_candidates),
                len(vector_targets),
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

        # 3단계: LLM 재판단 없이 rule + vector 최종 점수만으로 정렬한다.
        scored_candidates = scored_candidates_base
        scored_candidates.sort(
            key=lambda item: (
                item["ai_score"],
                item["vector_score"],
                item["rule_score"],
            ),
            reverse=True,
        )

        created_candidates: list[QuickMatchCandidate] = []
        for idx, item in enumerate(scored_candidates, start=1):
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
            "[QuickMatch] find_candidates done request_id=%s selected_candidates=%s hard_rule_candidates=%s vector_targets=%s elapsed=%.3fs",
            request.id,
            len(created_candidates),
            len(rule_scored_candidates),
            len(vector_targets),
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

        requested_at = request.requested_at
        elapsed_from_request = None
        if requested_at is not None:
            now_utc = datetime.now(timezone.utc)
            requested_at_utc = requested_at if requested_at.tzinfo is not None else requested_at.replace(tzinfo=timezone.utc)
            elapsed_from_request = (now_utc - requested_at_utc).total_seconds()

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
            "final_score": float(candidate.ai_score),
            "score_basis": "rule_vector_only",
        }
        result_row.decision_reason = self._build_decision_reason(candidate)

        await db.commit()
        await db.refresh(result_row)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "[QuickMatch] select_party done request_id=%s party_id=%s candidate_id=%s final_score=%.4f elapsed=%.3fs elapsed_from_request=%.3fs",
            request.id,
            candidate.party_id,
            candidate.id,
            float(candidate.ai_score),
            elapsed,
            elapsed_from_request if elapsed_from_request is not None else -1.0,
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

        party_result = await db.execute(
            select(Party)
            .options(selectinload(Party.host), selectinload(Party.service))
            .where(Party.id == request.matched_party_id)
        )
        party = party_result.scalar_one_or_none()

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

            requested_at = request.requested_at
            total_match_elapsed = None
            if requested_at is not None:
                requested_at_naive = requested_at.replace(tzinfo=None) if requested_at.tzinfo is not None else requested_at
                total_match_elapsed = (now - requested_at_naive).total_seconds()

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


            # 빠른매칭 알림
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
                member_nickname=user.nickname if user else None,
                match_request_id=request.id,
            )

            elapsed = time.perf_counter() - start_time
            logger.info(
                "[QuickMatch] join_party done request_id=%s party_id=%s user_id=%s current_members=%s elapsed=%.3fs total_match_elapsed=%.3fs",
                request.id,
                party.id,
                request.user_id,
                party.current_members,
                elapsed,
                total_match_elapsed if total_match_elapsed is not None else -1.0,
            )

            try:
                from routers.chat import manager
                from datetime import timezone
                await manager.broadcast(str(party.id), {
                    "type": "party_updated",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.warning(f"[party_updated broadcast failed] {e}")

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
        service = getattr(party, "service", None)
        return {
            "party_id": str(party.id),
            "service_id": str(getattr(party, "service_id", "") or ""),
            "service_name": getattr(service, "name", None),
            "category": self._extract_party_category(party),
            "platform": self._extract_party_platform(party),
            "min_trust_score": float(getattr(party, "min_trust_score", 0) or 0),
            "max_members": int(getattr(party, "max_members", 0) or 0),
            "current_members": int(getattr(party, "current_members", 0) or 0),
            "description": getattr(party, "description", "") or getattr(party, "intro", ""),
            "duration_preference": self._normalize_duration_preference(getattr(party, "duration_preference", None)),
            "duration_label": self._format_duration_label(getattr(party, "duration_preference", None)),
            "status": getattr(party, "status", None),
        }

    async def _get_party_embedding(
        self,
        db: AsyncSession,
        party_id: uuid.UUID,
    ):
        result = await db.execute(
            select(PartyEmbedding).where(PartyEmbedding.party_id == party_id)
        )
        return result.scalar_one_or_none()

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

        if "category" in normalized and normalized.get("category") is not None:
            normalized["category"] = str(normalized.get("category")).strip().lower()

        if "platform" in normalized and normalized.get("platform") is not None:
            normalized["platform"] = str(normalized.get("platform")).strip().lower()

        # 프론트에서는 duration_preference로 전송한다.
        # 혹시 duration_range라는 이름으로 들어와도 동일하게 처리한다.
        duration_value = normalized.get("duration_preference")
        if duration_value in (None, "") and normalized.get("duration_range") not in (None, ""):
            duration_value = normalized.get("duration_range")

        normalized_duration = self._normalize_duration_preference(duration_value)
        if normalized_duration:
            normalized["duration_preference"] = normalized_duration
        else:
            normalized.pop("duration_preference", None)
        normalized.pop("duration_range", None)

        return normalized

    def _normalize_duration_preference(self, value: Any) -> str | None:
        """
        프론트의 이용 기간 값을 백엔드 표준값으로 정규화한다.

        현재 표준값:
        - under_1_month: 1개월 이하
        - 1_3_months: 1~3개월
        - over_3_months: 3개월 이상

        기존 값(short_term / long_term / flexible)도 호환 처리한다.
        """
        if value in (None, ""):
            return None

        normalized = str(value).strip().lower()
        normalized = normalized.replace(" ", "")

        aliases = {
            "any": None,
            "all": None,
            "상관없음": None,
            "무관": None,
            "none": None,
            DURATION_UNDER_1_MONTH: DURATION_UNDER_1_MONTH,
            "under1month": DURATION_UNDER_1_MONTH,
            "under_1_months": DURATION_UNDER_1_MONTH,
            "less_than_1_month": DURATION_UNDER_1_MONTH,
            "1개월이하": DURATION_UNDER_1_MONTH,
            "short_term": DURATION_UNDER_1_MONTH,
            "short": DURATION_UNDER_1_MONTH,
            DURATION_1_3_MONTHS: DURATION_1_3_MONTHS,
            "1-3_months": DURATION_1_3_MONTHS,
            "1~3개월": DURATION_1_3_MONTHS,
            "1-3개월": DURATION_1_3_MONTHS,
            "1개월~3개월": DURATION_1_3_MONTHS,
            "1개월-3개월": DURATION_1_3_MONTHS,
            DURATION_OVER_3_MONTHS: DURATION_OVER_3_MONTHS,
            "over3months": DURATION_OVER_3_MONTHS,
            "more_than_3_months": DURATION_OVER_3_MONTHS,
            "3개월이상": DURATION_OVER_3_MONTHS,
            "long_term": DURATION_OVER_3_MONTHS,
            "long": DURATION_OVER_3_MONTHS,
            DURATION_FLEXIBLE: DURATION_FLEXIBLE,
            "flex": DURATION_FLEXIBLE,
            "유연하게가능": DURATION_FLEXIBLE,
        }

        return aliases.get(normalized, normalized)

    def _duration_preference_to_range(self, value: Any) -> tuple[float, float] | None:
        normalized = self._normalize_duration_preference(value)
        if normalized in (None, DURATION_FLEXIBLE):
            return None
        if normalized == DURATION_UNDER_1_MONTH:
            return (0.0, 1.0)
        if normalized == DURATION_1_3_MONTHS:
            return (1.0, 3.0)
        if normalized == DURATION_OVER_3_MONTHS:
            return (3.0, float("inf"))
        return None

    def _format_duration_label(self, value: Any) -> str | None:
        normalized = self._normalize_duration_preference(value)
        if normalized is None:
            return None
        return DURATION_LABELS.get(normalized, str(value))

    def _duration_ranges_overlap(self, user_value: Any, party_value: Any) -> bool:
        normalized_user = self._normalize_duration_preference(user_value)
        normalized_party = self._normalize_duration_preference(party_value)

        if normalized_user is None:
            return True
        if normalized_party is None:
            return False
        if normalized_user == DURATION_FLEXIBLE or normalized_party == DURATION_FLEXIBLE:
            return True
        if normalized_user == normalized_party:
            return True

        user_range = self._duration_preference_to_range(normalized_user)
        party_range = self._duration_preference_to_range(normalized_party)
        if not user_range or not party_range:
            return False

        user_low, user_high = user_range
        party_low, party_high = party_range

        # 1개월 이하와 1~3개월처럼 경계만 닿는 경우는 별도 구간으로 보고 불일치 처리한다.
        return max(user_low, party_low) < min(user_high, party_high)

    def _passes_hard_filters(
        self,
        user: User,
        party: Party,
        joined_party_ids: set[uuid.UUID],
        preferred_conditions: dict[str, Any],
        user_trust_score: float,
    ) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {}

        requested_category = preferred_conditions.get("category")
        party_category = self._extract_party_category(party)
        detail["requested_category"] = requested_category
        detail["party_category"] = party_category
        detail["category_match"] = self._matches_optional_string_filter(requested_category, party_category)
        if not detail["category_match"]:
            detail["excluded_reason"] = "category_mismatch"
            return False, detail

        requested_platform = preferred_conditions.get("platform")
        party_platform = self._extract_party_platform(party)
        detail["requested_platform"] = requested_platform
        detail["party_platform"] = party_platform
        detail["platform_match"] = self._matches_optional_string_filter(requested_platform, party_platform)
        if not detail["platform_match"]:
            detail["excluded_reason"] = "platform_mismatch"
            return False, detail

        user_duration_preference = preferred_conditions.get("duration_preference")
        party_duration_preference = getattr(party, "duration_preference", None)
        detail["user_duration_preference"] = user_duration_preference
        detail["party_duration_preference"] = party_duration_preference
        detail["duration_match"] = self._is_duration_core_match(
            party_duration_preference=party_duration_preference,
            user_duration_preference=user_duration_preference,
        )
        if not detail["duration_match"]:
            detail["excluded_reason"] = "duration_mismatch"
            return False, detail

        party_max_members = int(getattr(party, "max_members", 0) or 0)
        party_current_members = int(getattr(party, "current_members", 0) or 0)
        detail["remaining_seat"] = max((party_max_members - party_current_members), 0)
        if party_current_members >= party_max_members:
            detail["excluded_reason"] = "party_full"
            return False, detail

        if party.id in joined_party_ids:
            detail["excluded_reason"] = "already_member"
            return False, detail

        policy_excluded, policy_detail = self._get_policy_exclusion_detail(user=user, party=party)
        detail["policy"] = policy_detail
        if policy_excluded:
            detail["excluded_reason"] = "policy_excluded"
            return False, detail

        min_trust_score = float(getattr(party, "min_trust_score", 0) or 0)
        detail["party_min_trust_score"] = min_trust_score
        detail["user_trust_score"] = user_trust_score
        detail["trust_threshold_pass"] = user_trust_score >= min_trust_score
        if user_trust_score < min_trust_score:
            detail["excluded_reason"] = "trust_score_too_low"
            return False, detail

        return True, detail

    def _get_policy_exclusion_detail(self, user: User, party: Party) -> tuple[bool, dict[str, Any]]:
        user_report_count = int(getattr(user, "report_count", 0) or 0)
        user_blocked = bool(getattr(user, "is_blocked_for_matching", False))
        party_blocked = bool(getattr(party, "is_blocked_for_matching", False))
        party_report_limit = int(getattr(party, "max_reported_user_count", 9999) or 9999)

        detail = {
            "user_blocked": user_blocked,
            "party_blocked": party_blocked,
            "user_report_count": user_report_count,
            "party_report_limit": party_report_limit,
            "report_limit_exceeded": user_report_count > party_report_limit,
        }

        excluded = user_blocked or party_blocked or detail["report_limit_exceeded"]
        return excluded, detail

    def _matches_optional_string_filter(self, requested_value: Any, actual_value: Any) -> bool:
        if requested_value in (None, "", "any", "all"):
            return True
        if actual_value in (None, ""):
            return False
        return str(requested_value).strip().lower() == str(actual_value).strip().lower()

    def _extract_party_category(self, party: Party) -> str | None:
        service = getattr(party, "service", None)
        candidates = [
            getattr(party, "category", None),
            getattr(service, "category", None),
            getattr(service, "name", None),
        ]
        for value in candidates:
            if value not in (None, ""):
                return str(value).strip().lower()
        return None

    def _extract_party_platform(self, party: Party) -> str | None:
        service = getattr(party, "service", None)
        candidates = [
            getattr(party, "platform", None),
            getattr(service, "platform", None),
            getattr(party, "platform_name", None),
        ]
        for value in candidates:
            if value not in (None, ""):
                return str(value).strip().lower()
        return None

    def _build_decision_reason(self, candidate: QuickMatchCandidate) -> str:
        filter_reasons = candidate.filter_reasons or {}
        return (
            f"최종 점수 {float(candidate.ai_score):.4f}로 1순위 선정 "
            f"(rule={float(candidate.rule_score):.4f}, "
            f"vector={float(candidate.vector_score):.4f}, "
            f"basis={filter_reasons.get('score_basis', 'rule_vector_only')}, "
            f"mode={filter_reasons.get('match_mode', 'normal')})"
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
        else:
            margin = min(max(user_trust_score - min_trust_score, 0.0), 20.0)
            trust_fit_score = min(1.0, 0.7 + (margin / 20.0) * 0.3)

        score += trust_fit_score * 0.45
        detail["trust_fit_score"] = round(trust_fit_score, 4)
        detail["trust_margin"] = round(max(user_trust_score - min_trust_score, 0.0), 4)

        party_max_members = float(getattr(party, "max_members", 0) or 0)
        party_current_members = float(getattr(party, "current_members", 0) or 0)
        if party_max_members <= 0:
            capacity_score = 0.0
        else:
            remaining = max((party_max_members - party_current_members), 0)
            capacity_score = min(1.0, remaining / max(party_max_members, 1))

        score += capacity_score * 0.30
        detail["capacity_score"] = round(capacity_score, 4)

        duration_score = self._calculate_duration_score(
            party_duration_preference=getattr(party, "duration_preference", None),
            user_duration_preference=preferred_conditions.get("duration_preference"),
        )
        score += duration_score * 0.25
        detail["duration_score"] = round(duration_score, 4)
        detail["user_duration_preference"] = preferred_conditions.get("duration_preference")
        detail["party_duration_preference"] = getattr(party, "duration_preference", None)

        return round(min(score, 1.0), 4), detail

    def _calculate_duration_score(
        self,
        party_duration_preference: str | None,
        user_duration_preference: str | None,
    ) -> float:
        normalized_user = self._normalize_duration_preference(user_duration_preference)
        normalized_party = self._normalize_duration_preference(party_duration_preference)

        if normalized_user is None:
            return 0.7
        if normalized_party is None:
            return 0.6
        if normalized_user == DURATION_FLEXIBLE or normalized_party == DURATION_FLEXIBLE:
            return 0.8
        if normalized_user == normalized_party:
            return 1.0

        user_range = self._duration_preference_to_range(normalized_user)
        party_range = self._duration_preference_to_range(normalized_party)
        if not user_range or not party_range:
            return 0.3

        user_low, user_high = user_range
        party_low, party_high = party_range

        overlap = max(0.0, min(user_high, party_high) - max(user_low, party_low))
        if overlap > 0:
            return 0.7

        # 인접 구간은 완전 불일치보다는 낮은 근접 점수를 준다.
        if user_high == party_low or party_high == user_low:
            return 0.5

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
    ) -> float:
        """
        LLM 재판단 없이 룰 점수와 임베딩 유사도만으로 최종 매칭 점수를 계산한다.
        rule_score는 조건/정책 적합도, vector_score는 사용자-파티 프로필 유사도를 의미한다.
        """
        final_score = (rule_score * 0.5) + (vector_score * 0.5)
        return round(min(final_score, 1.0), 4)

    def _is_duration_core_match(
        self,
        party_duration_preference: str | None,
        user_duration_preference: str | None,
    ) -> bool:
        return self._duration_ranges_overlap(
            user_value=user_duration_preference,
            party_value=party_duration_preference,
        )

    def _get_match_mode_priority(self, filter_reasons: dict[str, Any] | None) -> int:
        match_mode = str((filter_reasons or {}).get("match_mode", "normal")).lower()
        return 1 if match_mode == "normal" else 0
