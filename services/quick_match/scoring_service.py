from __future__ import annotations

import math
from typing import Any


class ScoringService:
    @staticmethod
    def calculate_rule_score(
        party,
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

        price_score = ScoringService.calculate_price_score(
            monthly_price=monthly_price,
            preferred_price=preferred_price,
        )
        score += price_score * 0.3
        detail["price_score"] = round(price_score, 4)
        detail["preferred_price"] = preferred_price
        detail["monthly_price"] = monthly_price

        duration_score = ScoringService.calculate_duration_score(
            party_duration_preference=getattr(party, "duration_preference", None),
            user_duration_preference=preferred_conditions.get("duration_preference"),
        )
        score += duration_score * 0.15
        detail["duration_score"] = round(duration_score, 4)
        detail["user_duration_preference"] = preferred_conditions.get("duration_preference")
        detail["party_duration_preference"] = getattr(party, "duration_preference", None)

        return round(min(score, 1.0), 4), detail

    @staticmethod
    def calculate_price_score(
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

    @staticmethod
    def calculate_duration_score(
        party_duration_preference: str | None,
        user_duration_preference: str | None,
    ) -> float:
        if not user_duration_preference:
            return 0.7

        normalized_user = str(user_duration_preference).strip().lower()
        normalized_party = (
            str(party_duration_preference).strip().lower()
            if party_duration_preference
            else ""
        )

        if not normalized_party:
            return 0.6
        if normalized_user == normalized_party:
            return 1.0
        if {normalized_user, normalized_party} == {"long_term", "flexible"}:
            return 0.8
        if {normalized_user, normalized_party} == {"short_term", "flexible"}:
            return 0.8
        return 0.3

    @staticmethod
    def calculate_vector_score(
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

    @staticmethod
    def calculate_ai_score(
        rule_score: float,
        vector_score: float,
        llm_score: float,
    ) -> float:
        final_score = (rule_score * 0.4) + (vector_score * 0.3) + (llm_score * 0.3)
        return round(min(final_score, 1.0), 4)

    @staticmethod
    def matches_fallback_core_conditions(
        party,
        preferred_conditions: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {}

        preferred_price = preferred_conditions.get("price_range")
        user_duration_preference = preferred_conditions.get("duration_preference")
        monthly_price = float(getattr(party, "monthly_per_person", 0) or 0)
        party_duration_preference = getattr(party, "duration_preference", None)

        price_core_match = ScoringService.is_price_in_requested_range(
            monthly_price=monthly_price,
            preferred_price=preferred_price,
        )
        duration_core_match = ScoringService.is_duration_core_match(
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

    @staticmethod
    def is_price_in_requested_range(
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

    @staticmethod
    def is_duration_core_match(
        party_duration_preference: str | None,
        user_duration_preference: str | None,
    ) -> bool:
        if not user_duration_preference:
            return True

        normalized_user = str(user_duration_preference).strip().lower()
        normalized_party = (
            str(party_duration_preference).strip().lower()
            if party_duration_preference
            else ""
        )

        if not normalized_party:
            return False

        return normalized_user == normalized_party

    @staticmethod
    def calculate_fallback_score(
        party,
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

        fallback_ok, _ = ScoringService.matches_fallback_core_conditions(
            party=party,
            preferred_conditions=preferred_conditions,
        )
        core_score = 1.0 if fallback_ok else 0.0

        final_score = (trust_fit_score * 0.4) + (capacity_score * 0.2) + (core_score * 0.4)
        return round(min(final_score, 1.0), 4)

    @staticmethod
    def get_match_mode_priority(filter_reasons: dict[str, Any] | None) -> int:
        match_mode = str((filter_reasons or {}).get("match_mode", "normal")).lower()
        return 1 if match_mode == "normal" else 0