from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.party import Party, PartyMember
from models.user import User


class ProfileService:
    @staticmethod
    def normalize_preferred_conditions(
        preferred_conditions: dict[str, Any] | None,
    ) -> dict[str, Any]:
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

    @staticmethod
    async def build_user_ai_profile(
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
                "is_currently_banned": bool(
                    user.banned_until and user.banned_until > datetime.now(timezone.utc)
                ),
            },
        }

    @staticmethod
    def build_party_profile(party: Party) -> dict[str, Any]:
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