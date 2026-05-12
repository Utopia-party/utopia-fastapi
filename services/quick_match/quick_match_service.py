from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.redis_lock import redis_lock
from models.report import Report
from models.party import Party, PartyMember
from models.quick_match.candidate import QuickMatchCandidate, QuickMatchCandidateStatus
from models.quick_match.request import QuickMatchRequest, QuickMatchRequestStatus
from models.quick_match.result import QuickMatchResult
from models.quick_match.training_events import (
    QuickMatchTrainingEvent,
    QuickMatchTrainingLabelStatus,
)

from models.user import User
from services.notifications.party_notification_service import (
    notify_quick_match_completed,
    notify_quick_match_member_joined_to_leader,
)
from services.quick_match.training_event_service import (
    mark_training_event_excluded,
    mark_unselected_candidates_excluded,
)
from services.quick_match.training_stats_service import load_training_stats

logger = logging.getLogger(__name__)

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
        request_profile = await self._build_request_profile_snapshot(
            db=db,
            user=user,
            service_id=service_id,
            preferred_conditions=normalized_conditions,
        )

        request = QuickMatchRequest(
            user_id=user_id,
            service_id=service_id,
            status=QuickMatchRequestStatus.REQUESTED,
            preferred_conditions=normalized_conditions,
            request_profile_snapshot=request_profile,
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

        if request.status != QuickMatchRequestStatus.REQUESTED:
            raise Exception("INVALID_REQUEST_STATUS")

        user = await db.get(User, request.user_id)
        if not user:
            raise Exception("USER_NOT_FOUND")

        existing_candidates = await db.execute(
            select(QuickMatchCandidate).where(
                QuickMatchCandidate.request_id == request.id,
            )
        )
        for row in existing_candidates.scalars().all():
            await db.delete(row)
        await db.flush()

        user_trust_score = float(getattr(user, "trust_score", 0) or 0)
        remaining_seat_expr = (
            func.coalesce(Party.max_members, 0)
            - func.coalesce(Party.current_members, 0)
        )

        party_result = await db.execute(
            select(Party)
            .options(selectinload(Party.service), selectinload(Party.host))
            .where(
                Party.status == "recruiting",
                Party.service_id == request.service_id,
                func.coalesce(Party.current_members, 0) < func.coalesce(Party.max_members, 0),
                func.coalesce(Party.min_trust_score, 0) <= user_trust_score,
            )
            .order_by(
                remaining_seat_expr.desc(),
                Party.created_at.desc(),
            )
            .limit(100)
        )
        parties = party_result.scalars().all()


        if not parties:
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

        probability_stats = await self._build_probability_stats(db)

        scored_candidates: list[dict[str, Any]] = []

        for party in parties:
            filter_reasons: dict[str, Any] = {
                "service_match": True,
                "recruiting_status": party.status == "recruiting",
            }

            hard_filter_ok, hard_filter_detail = await self._passes_hard_filters(
                db=db,
                user=user,
                party=party,
                joined_party_ids=joined_party_ids,
                preferred_conditions=preferred_conditions,
                user_trust_score=user_trust_score,
            )
            filter_reasons["hard_filter"] = hard_filter_detail

            # 하드필터 통과하지 못했을 경우
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
            probability_score, probability_reason = self._calculate_probability_score(
                user=user,
                party=party,
                preferred_conditions=preferred_conditions,
                probability_stats=probability_stats,
            )
            final_score = self._calculate_final_score(
                rule_score=rule_score,
                probability_score=probability_score,
            )

            filter_reasons["rule_reason"] = rule_reason
            filter_reasons["probability_reason"] = probability_reason
            filter_reasons["score_basis"] = "rule_probability"

            scored_candidates.append(
                {
                    "party": party,
                    "rule_score": float(rule_score),
                    "probability_score": float(probability_score),
                    "final_score": float(final_score),
                    "filter_reasons": filter_reasons,
                }
            )

        if not scored_candidates:
            await self.fail_request(
                db=db,
                request_id=request.id,
                reason="NO_CANDIDATE",
            )
            raise Exception("NO_CANDIDATE")

        scored_candidates.sort(
            key=lambda item: (
                item["final_score"],
                item["probability_score"],
                item["rule_score"],
            ),
            reverse=True,
        )

        created_candidates: list[QuickMatchCandidate] = []
        for idx, item in enumerate(scored_candidates, start=1):
            status = (
                QuickMatchCandidateStatus.SELECTED
                if idx == 1
                else QuickMatchCandidateStatus.PENDING
            )

            candidate = QuickMatchCandidate(
                request_id=request.id,
                party_id=item["party"].id,
                rule_score=item["rule_score"],
                probability_score=item["probability_score"],
                final_score=item["final_score"],
                rank=idx,
                status=status,
                filter_reasons=item["filter_reasons"],
            )
            db.add(candidate)
            await db.flush()
            created_candidates.append(candidate)

            training_event = QuickMatchTrainingEvent(
                request_id=request.id,
                candidate_id=candidate.id,
                user_id=request.user_id,
                service_id=request.service_id,
                party_id=item["party"].id,
                is_selected=status == QuickMatchCandidateStatus.SELECTED,
                is_joined=False,
                label_status=QuickMatchTrainingLabelStatus.PENDING.value,
                features_snapshot={
                    "rule_score": item["rule_score"],
                    "probability_score": item["probability_score"],
                    "final_score": item["final_score"],
                    "rank": idx,
                    "candidate_status": status.value,
                    "filter_reasons": item["filter_reasons"],
                    "party": {
                        "id": str(item["party"].id),
                        "service_id": str(item["party"].service_id),
                        "max_members": item["party"].max_members,
                        "current_members": item["party"].current_members,
                        "min_trust_score": item["party"].min_trust_score,
                        "start_date": item["party"].start_date.isoformat()
                        if item["party"].start_date
                        else None,
                        "end_date": item["party"].end_date.isoformat()
                        if item["party"].end_date
                        else None,
                    },
                    "user": {
                        "id": str(request.user_id),
                        "trust_score": user_trust_score,
                    },
                    "preferred_conditions": preferred_conditions,
                },
            )
            db.add(training_event)

        # 빠른매칭 학습데이터 - 보류/제외
        await mark_unselected_candidates_excluded(
            db,
            request_id=request.id,
            reason="not_selected",
        )

        await db.commit()

        for candidate in created_candidates:
            await db.refresh(candidate)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "[QuickMatch] find_candidates done request_id=%s selected_candidates=%s elapsed=%.3fs",
            request.id,
            len(created_candidates),
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
                float(candidate.final_score),
                float(candidate.probability_score),
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
            select(QuickMatchResult).where(
                QuickMatchResult.request_id == request.id
            )
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
            "request_profile_snapshot": request.request_profile_snapshot,
        }
        result_row.candidate_snapshot = {
            "party_id": str(candidate.party_id),
            "rank": candidate.rank,
            "status": candidate.status.value,
            "filter_reasons": candidate.filter_reasons,
        }
        result_row.final_scores = {
            "rule_score": float(candidate.rule_score),
            "probability_score": float(candidate.probability_score),
            "final_score": float(candidate.final_score),
            "score_basis": "rule_probability",
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
            float(candidate.final_score),
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

        if request.status != QuickMatchRequestStatus.MATCHED:
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

        async with redis_lock(
            lock_key=lock_key,
            lock_value=str(request.id),
            expire_seconds=30,
        ):
            await db.refresh(party)

            party_current_members = int(getattr(party, "current_members", 0) or 0)
            party_max_members = int(getattr(party, "max_members", 0) or 0)

            if party.status != "recruiting":
                await self.fail_request(db, request.id, "PARTY_STATUS_CHANGED")
                return await self.retry_match(db, request.id)

            if party_current_members >= party_max_members:
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

            training_result = await db.execute(
                select(QuickMatchTrainingEvent).where(
                    QuickMatchTrainingEvent.request_id == request.id,
                    QuickMatchTrainingEvent.party_id == party.id,
                )
            )
            training_event = training_result.scalar_one_or_none()

            if training_event:
                training_event.is_joined = True
                training_event.joined_at = now
                training_event.result_snapshot = {
                    "party_member_id": str(new_member.id) if new_member.id else None,
                    "party_id": str(party.id),
                    "user_id": str(request.user_id),
                    "join_type": "quick_match",
                    "joined_at": now.isoformat(),
                }

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
                member_nickname=user.nickname if user else None,
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

            try:
                from routers.chat import manager

                await manager.broadcast(
                    str(party.id),
                    {
                        "type": "party_updated",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception as e:
                logger.warning("[party_updated broadcast failed] %s", e)

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

        # 빠른매칭 학습데이터 - 보류/제외
        if reason in {
            "NO_CANDIDATE",
            "NO_RECRUITING_PARTY",
            "PARTY_STATUS_CHANGED",
            "PARTY_FULL",
            "ALREADY_JOINED",
        }:
            await mark_training_event_excluded(
                db,
                request_id=request.id,
                party_id=request.matched_party_id,
                reason=reason.lower(),
                result_snapshot={
                    "request": {
                        "status": QuickMatchRequestStatus.FAILED.value,
                        "fail_reason": reason,
                    }
                },
            )

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
            return request

        candidates.sort(
            key=lambda candidate: (
                float(candidate.final_score),
                float(candidate.probability_score),
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
            return request

        next_candidate.status = QuickMatchCandidateStatus.SELECTED
        request.matched_party_id = next_candidate.party_id
        request.matched_at = datetime.now(timezone.utc)
        request.status = QuickMatchRequestStatus.MATCHED
        request.is_active = True
        request.fail_reason = None

        await db.commit()

        return {
            "request_id": request.id,
            "next_party_id": next_candidate.party_id,
            "retry_count": request.retry_count,
            "status": request.status.value,
        }

    async def _build_request_profile_snapshot(
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

        # 특정 서비스의 파티에 참여한 수
        service_membership_count = sum(
            1
            for _, party in memberships
            if str(party.service_id) == str(service_id)
        )

        # 총 참여한 파티 수
        total_membership_count = len(memberships)

        # 활성화 중인 파티 수
        active_membership_count = sum(
            1
            for membership, _ in memberships
            if getattr(membership, "status", None) == "active"
        )

        return {
            "user_id": str(user.id),
            "service_id": str(service_id),
            "trust_score": float(getattr(user, "trust_score", 0) or 0),
            "preferred_conditions": preferred_conditions,
            "activity_summary": {
                "total_party_join_count": total_membership_count,
                "service_party_join_count": service_membership_count,
                "active_party_count": active_membership_count,
            },
            "risk_summary": {
                
                "is_currently_banned": bool(
                    user.banned_until
                    and user.banned_until > datetime.now(timezone.utc)
                ),
            },
        }


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

        candidate = QuickMatchCandidate(
            request_id=request_id,
            party_id=party_id,
            rule_score=0,
            probability_score=0,
            final_score=0,
            rank=None,
            status=QuickMatchCandidateStatus.REJECTED,
            filter_reasons=rejected_reasons,
        )
        db.add(candidate)

    def _normalize_preferred_conditions(
        self,
        preferred_conditions: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = dict(preferred_conditions or {})

        duration_value = normalized.get("duration_preference")
        normalized_duration = self._normalize_duration_preference(duration_value)

        if normalized_duration:
            normalized["duration_preference"] = normalized_duration
        else:
            normalized.pop("duration_preference", None)

        return normalized


    def _normalize_duration_preference(self, value: Any) -> str | None:
        if value in (None, ""):
            return None

        normalized = str(value).strip().lower().replace(" ", "")

        aliases = {
            DURATION_UNDER_1_MONTH: DURATION_UNDER_1_MONTH,
            "under1month": DURATION_UNDER_1_MONTH,
            "under_1_months": DURATION_UNDER_1_MONTH,
            "1개월이하": DURATION_UNDER_1_MONTH,
            DURATION_1_3_MONTHS: DURATION_1_3_MONTHS,
            "1-3_months": DURATION_1_3_MONTHS,
            "1~3개월": DURATION_1_3_MONTHS,
            "1-3개월": DURATION_1_3_MONTHS,
            "1개월~3개월": DURATION_1_3_MONTHS,
            "1개월-3개월": DURATION_1_3_MONTHS,
            DURATION_OVER_3_MONTHS: DURATION_OVER_3_MONTHS,
            "over3months": DURATION_OVER_3_MONTHS,
            "3개월이상": DURATION_OVER_3_MONTHS,
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

    def _duration_ranges_overlap(self, user_value: Any, party_value: Any) -> bool:
        normalized_user = self._normalize_duration_preference(user_value)
        normalized_party = self._normalize_duration_preference(party_value)

        if normalized_user is None:
            return True
        if normalized_party is None:
            return True
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
        return max(user_low, party_low) < min(user_high, party_high)

    def _get_party_duration_preference(self, party: Party) -> str | None:
        party_start_date = getattr(party, "start_date", None)
        party_end_date = getattr(party, "end_date", None)

        if not party_start_date or not party_end_date:
            return None

        duration_days = (party_end_date - party_start_date).days + 1
        if duration_days <= 0:
            return None
        if duration_days <= 31:
            return DURATION_UNDER_1_MONTH
        if duration_days <= 93:
            return DURATION_1_3_MONTHS
        return DURATION_OVER_3_MONTHS

    async def _passes_hard_filters(
        self,
        db: AsyncSession,
        user: User,
        party: Party,
        joined_party_ids: set[uuid.UUID],
        preferred_conditions: dict[str, Any],
        user_trust_score: float,
    ) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {}

        user_duration_preference = preferred_conditions.get("duration_preference")
        party_start_date = getattr(party, "start_date", None)
        party_end_date = getattr(party, "end_date", None)
        party_duration_bucket = self._get_party_duration_preference(party)

        detail["user_duration_preference"] = user_duration_preference
        detail["party_start_date"] = party_start_date.isoformat() if party_start_date else None
        detail["party_end_date"] = party_end_date.isoformat() if party_end_date else None
        detail["party_duration_bucket"] = party_duration_bucket
        detail["duration_match"] = self._duration_ranges_overlap(
            user_duration_preference,
            party_duration_bucket,
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

        policy_excluded, policy_detail = await self._get_policy_exclusion_detail(
            db=db,
            user=user,
            party=party,
        )
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


    async def _get_policy_exclusion_detail(
        self,
        db: AsyncSession,
        user: User,
        party: Party,
    ) -> tuple[bool, dict[str, Any]]:
        now = datetime.now(timezone.utc)

        user_inactive = not bool(getattr(user, "is_active", True))
        user_banned = bool(
            user.banned_until and user.banned_until > now
        )

        approved_user_report_count = await db.scalar(
            select(func.count(Report.id)).where(
                func.lower(Report.target_type) == "user",
                Report.target_id == user.id,
                Report.status == "APPROVED",
            )
        ) or 0

        approved_party_report_count = await db.scalar(
            select(func.count(Report.id)).where(
                func.lower(Report.target_type) == "party",
                Report.target_id == party.id,
                Report.status == "APPROVED",
            )
        ) or 0

        user_report_limit = 3
        party_report_limit = 3

        detail = {
            "user_inactive": user_inactive,
            "user_banned": user_banned,
            "approved_user_report_count": int(approved_user_report_count),
            "approved_party_report_count": int(approved_party_report_count),
            "user_report_limit": user_report_limit,
            "party_report_limit": party_report_limit,
            "user_report_limit_exceeded": approved_user_report_count >= user_report_limit,
            "party_report_limit_exceeded": approved_party_report_count >= party_report_limit,
        }

        excluded = (
            user_inactive
            or user_banned
            or detail["user_report_limit_exceeded"]
            or detail["party_report_limit_exceeded"]
        )

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

    def _calculate_rule_score(
        self,
        party: Party,
        user_trust_score: float,
        preferred_conditions: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        score = 0.0
        detail: dict[str, Any] = {}

        baseline_trust_score = 36.5
        min_trust_score = float(getattr(party, "min_trust_score", 0) or 0)

        if min_trust_score > 0:
            if user_trust_score < min_trust_score:
                trust_fit_score = 0.0
            else:
                margin = min(user_trust_score - min_trust_score, 20.0)
                trust_fit_score = min(1.0, 0.7 + (margin / 20.0) * 0.3)
        else:
            if user_trust_score < baseline_trust_score:
                trust_fit_score = max(0.0, user_trust_score / baseline_trust_score) * 0.7
            else:
                margin = min(user_trust_score - baseline_trust_score, 20.0)
                trust_fit_score = min(1.0, 0.7 + (margin / 20.0) * 0.3)

        score += trust_fit_score * 0.45
        detail["trust_fit_score"] = round(trust_fit_score, 4)
        detail["baseline_trust_score"] = baseline_trust_score
        detail["party_min_trust_score"] = min_trust_score

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
            party_duration_preference=self._get_party_duration_preference(party),
            user_duration_preference=preferred_conditions.get("duration_preference"),
        )

        score += duration_score * 0.25
        detail["duration_score"] = round(duration_score, 4)

        return round(min(score, 1.0), 4), detail


    async def _build_probability_stats(self, db: AsyncSession) -> dict[str, Any]:
        return await load_training_stats(db)

    def _success_rate(
        self,
        bucket: dict[str, int] | None,
        *,
        prior_rate: float,
        prior_weight: int = 20,
    ) -> tuple[float, int, int]:
        if not bucket:
            return round(prior_rate, 4), 0, 0

        success = int(bucket.get("success", 0) or 0)
        total = int(bucket.get("total", 0) or 0)
        if total <= 0:
            return round(prior_rate, 4), 0, 0

        rate = (success + (prior_rate * prior_weight)) / (total + prior_weight)
        return round(min(max(rate, 0.0), 1.0), 4), success, total

    def _trust_score_to_bucket(self, value: Any) -> str | None:
        if value in (None, ""):
            return None

        try:
            score = float(value)
        except (TypeError, ValueError):
            return None

        if score < 30:
            return "under_30"
        if score < 40:
            return "30_40"
        if score < 50:
            return "40_50"
        if score < 60:
            return "50_60"
        return "over_60"

    def _capacity_to_bucket(
        self,
        *,
        max_members: Any,
        current_members: Any,
    ) -> str | None:
        try:
            max_value = float(max_members or 0)
            current_value = float(current_members or 0)
        except (TypeError, ValueError):
            return None

        if max_value <= 0:
            return "unknown"

        ratio = max(max_value - current_value, 0) / max_value
        if ratio <= 0:
            return "full"
        if ratio <= 0.25:
            return "low"
        if ratio <= 0.50:
            return "medium"
        return "high"

    def _duration_match_key(self, user_value: Any, party_value: Any) -> str:
        normalized_user = self._normalize_duration_preference(user_value)
        normalized_party = self._normalize_duration_preference(party_value)

        if normalized_user is None:
            return "no_preference"
        if normalized_party is None:
            return "party_unknown"
        if normalized_user == DURATION_FLEXIBLE or normalized_party == DURATION_FLEXIBLE:
            return "flexible"
        if normalized_user == normalized_party:
            return "exact"

        user_range = self._duration_preference_to_range(normalized_user)
        party_range = self._duration_preference_to_range(normalized_party)
        if not user_range or not party_range:
            return "unknown"

        user_low, user_high = user_range
        party_low, party_high = party_range
        overlap = max(0.0, min(user_high, party_high) - max(user_low, party_low))
        if overlap > 0:
            return "overlap"
        if user_high == party_low or party_high == user_low:
            return "boundary"
        return "mismatch"

    def _extract_training_duration_match_key(
        self,
        features_snapshot: dict[str, Any],
    ) -> str | None:
        request_snapshot = features_snapshot.get("request") or {}
        preferred_conditions = (
            request_snapshot.get("preferred_conditions")
            or features_snapshot.get("preferred_conditions")
            or {}
        )
        party_snapshot = features_snapshot.get("party") or {}
        party_duration = party_snapshot.get("duration_bucket")

        if party_duration is None:
            filter_reasons = features_snapshot.get("filter_reasons") or {}
            hard_filter = filter_reasons.get("hard_filter") or {}
            party_duration = hard_filter.get("party_duration_bucket")

        return self._duration_match_key(
            preferred_conditions.get("duration_preference"),
            party_duration,
        )

    def _calculate_probability_score(
        self,
        user: User,
        party: Party,
        preferred_conditions: dict[str, Any],
        probability_stats: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        detail: dict[str, Any] = {}

        global_bucket = probability_stats.get("global") or {}
        global_rate, global_success, global_total = self._success_rate(
            global_bucket,
            prior_rate=0.5,
            prior_weight=0,
        )
        if global_total <= 0:
            global_rate = 0.5

        service_key = str(party.service_id)
        trust_key = self._trust_score_to_bucket(getattr(user, "trust_score", None))
        duration_key = self._duration_match_key(
            preferred_conditions.get("duration_preference"),
            self._get_party_duration_preference(party),
        )
        capacity_key = self._capacity_to_bucket(
            max_members=getattr(party, "max_members", None),
            current_members=getattr(party, "current_members", None),
        )

        service_rate, service_success, service_total = self._success_rate(
            (probability_stats.get("service") or {}).get(service_key),
            prior_rate=global_rate,
        )
        trust_rate, trust_success, trust_total = self._success_rate(
            (probability_stats.get("trust_bucket") or {}).get(trust_key),
            prior_rate=global_rate,
        )
        duration_rate, duration_success, duration_total = self._success_rate(
            (probability_stats.get("duration_match") or {}).get(duration_key),
            prior_rate=global_rate,
        )
        capacity_rate, capacity_success, capacity_total = self._success_rate(
            (probability_stats.get("capacity_bucket") or {}).get(capacity_key),
            prior_rate=global_rate,
        )

        probability_score = (
            service_rate * 0.35
            + trust_rate * 0.35
            + duration_rate * 0.20
            + capacity_rate * 0.10
        )

        detail["score_basis"] = "training_success_rate"
        detail["global_success_rate"] = global_rate
        detail["global_sample_count"] = global_total
        detail["service"] = {
            "key": service_key,
            "success_rate": service_rate,
            "success_count": service_success,
            "sample_count": service_total,
            "weight": 0.35,
        }
        detail["trust_bucket"] = {
            "key": trust_key,
            "success_rate": trust_rate,
            "success_count": trust_success,
            "sample_count": trust_total,
            "weight": 0.35,
        }
        detail["duration_match"] = {
            "key": duration_key,
            "success_rate": duration_rate,
            "success_count": duration_success,
            "sample_count": duration_total,
            "weight": 0.20,
        }
        detail["capacity_bucket"] = {
            "key": capacity_key,
            "success_rate": capacity_rate,
            "success_count": capacity_success,
            "sample_count": capacity_total,
            "weight": 0.10,
        }

        return round(min(max(probability_score, 0.0), 1.0), 4), detail

    def _calculate_final_score(
        self,
        rule_score: float,
        probability_score: float,
    ) -> float:
        final_score = (rule_score * 0.55) + (probability_score * 0.45)
        return round(min(final_score, 1.0), 4)

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
        if user_high == party_low or party_high == user_low:
            return 0.5
        return 0.3

    def _build_decision_reason(self, candidate: QuickMatchCandidate) -> str:
        filter_reasons = candidate.filter_reasons or {}
        return (
            f"최종 점수 {float(candidate.final_score):.4f}로 1순위 선정 "
            f"(rule={float(candidate.rule_score):.4f}, "
            f"probability={float(candidate.probability_score):.4f}, "
            f"basis={filter_reasons.get('score_basis', 'rule_probability')})"
        )
