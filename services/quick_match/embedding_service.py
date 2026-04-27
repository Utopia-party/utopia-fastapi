from __future__ import annotations

import httpx

from core.config import settings


class EmbeddingService:
    @staticmethod
    async def generate_embedding(payload: dict) -> list[float]:
        """
        Ollama 임베딩 모델로 임베딩 생성
        """
        text = str(payload.get("text", "") or "").strip()
        if not text:
            return []

        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(
                f"{settings.OLLAMA_URL}/api/embeddings",
                json={
                    "model": settings.OLLAMA_EMBED_MODEL,
                    "prompt": text,
                },
            )
            res.raise_for_status()
            data = res.json()
            return data.get("embedding", [])

    @staticmethod
    def format_duration_preference(value: object) -> str | None:
        if value in (None, ""):
            return None

        normalized = str(value).strip().lower().replace(" ", "")
        labels = {
            "under_1_month": "1개월 이하",
            "under_1_months": "1개월 이하",
            "under1month": "1개월 이하",
            "less_than_1_month": "1개월 이하",
            "short_term": "1개월 이하",
            "short": "1개월 이하",
            "1개월이하": "1개월 이하",
            "1_3_months": "1~3개월",
            "1-3_months": "1~3개월",
            "1~3개월": "1~3개월",
            "1-3개월": "1~3개월",
            "1개월~3개월": "1~3개월",
            "1개월-3개월": "1~3개월",
            "over_3_months": "3개월 이상",
            "over3months": "3개월 이상",
            "more_than_3_months": "3개월 이상",
            "long_term": "3개월 이상",
            "long": "3개월 이상",
            "3개월이상": "3개월 이상",
            "flexible": "상관없음",
            "any": "상관없음",
            "all": "상관없음",
            "상관없음": "상관없음",
        }
        return labels.get(normalized, str(value))

    @staticmethod
    def serialize_user_profile_text(user_profile: dict) -> str:
        """
        LLM 요약 없이 사용자 프로필 자체를 임베딩용 텍스트로 직렬화한다.
        """
        preferred_conditions = user_profile.get("preferred_conditions", {}) or {}
        activity_summary = user_profile.get("activity_summary", {}) or {}
        payment_summary = user_profile.get("payment_summary", {}) or {}
        risk_summary = user_profile.get("risk_summary", {}) or {}

        return f"""
희망 카테고리: {preferred_conditions.get('category')}
희망 플랫폼: {preferred_conditions.get('platform')}
희망 이용 기간: {EmbeddingService.format_duration_preference(preferred_conditions.get('duration_preference'))}
신뢰도 점수: {user_profile.get('trust_score')}
전체 파티 참여 횟수: {activity_summary.get('total_party_join_count')}
서비스별 파티 참여 횟수: {activity_summary.get('service_party_join_count')}
현재 활성 파티 수: {activity_summary.get('active_party_count')}
정산 성공 횟수: {payment_summary.get('settlement_success_count')}
신고 횟수: {risk_summary.get('report_count')}
이탈 횟수: {risk_summary.get('leave_count')}
현재 밴 여부: {risk_summary.get('is_currently_banned')}
""".strip()

    @staticmethod
    def serialize_party_text(party_data: dict) -> str:
        """
        파티 프로필을 임베딩용 텍스트로 직렬화한다.
        """
        return f"""
서비스: {party_data.get('service_name')}
카테고리: {party_data.get('category')}
플랫폼: {party_data.get('platform')}
최소 신뢰도: {party_data.get('min_trust_score')}
최대 인원: {party_data.get('max_members')}
현재 인원: {party_data.get('current_members')}
설명: {party_data.get('description')}
이용 기간: {EmbeddingService.format_duration_preference(party_data.get('duration_preference'))}
상태: {party_data.get('status')}
""".strip()

    @staticmethod
    async def generate_party_embedding(party_data: dict) -> list[float]:
        """
        파티 정보를 텍스트로 직렬화해서 파티 임베딩 생성
        """
        text = EmbeddingService.serialize_party_text(party_data)
        return await EmbeddingService.generate_embedding({"text": text})
