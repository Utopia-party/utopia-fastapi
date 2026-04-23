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
희망 가격대: {preferred_conditions.get('price_range')}
이용 기간 성향: {preferred_conditions.get('duration_preference')}
신뢰도 점수: {user_profile.get('trust_score')}
전체 파티 참여 횟수: {activity_summary.get('total_party_join_count')}
서비스별 파티 참여 횟수: {activity_summary.get('service_party_join_count')}
현재 활성 파티 수: {activity_summary.get('active_party_count')}
평균 결제 금액: {payment_summary.get('average_payment_amount')}
정산 성공 횟수: {payment_summary.get('settlement_success_count')}
신고 횟수: {risk_summary.get('report_count')}
이탈 횟수: {risk_summary.get('leave_count')}
현재 밴 여부: {risk_summary.get('is_currently_banned')}
""".strip()

    @staticmethod
    async def generate_match_evaluation(payload: dict) -> dict:
        """
        상위 후보에 대해서만 사용자-파티 조합을 LLM으로 한 번 더 평가한다.
        """
        user_profile = payload.get("user_profile", {})
        party_profile = payload.get("party_profile", {})
        rule_score = float(payload.get("rule_score", 0) or 0)
        vector_score = float(payload.get("vector_score", 0) or 0)

        prompt = f"""
아래 사용자와 파티가 얼마나 잘 맞는지 0~1 사이 점수와 짧은 사유를 판단해라.
응답 형식은 반드시 다음 두 줄만 사용:
score: <0~1 숫자>
reason: <한 줄 설명>

[user_profile]
{user_profile}

[party_profile]
{party_profile}

[rule_score]
{rule_score}
[vector_score]
{vector_score}
"""

        score = round(min(1.0, max(0.0, (rule_score * 0.5) + (vector_score * 0.5))), 4)
        reason = "룰 적합도와 임베딩 유사도를 함께 반영한 LLM 재판단"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(
                    f"{settings.OLLAMA_URL}/api/generate",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                res.raise_for_status()
                data = res.json()
                text = str(data.get("response", "") or "")

                parsed_score = score
                parsed_reason = reason
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped.lower().startswith("score:"):
                        raw_value = stripped.split(":", 1)[1].strip()
                        parsed_score = float(raw_value)
                    elif stripped.lower().startswith("reason:"):
                        parsed_reason = stripped.split(":", 1)[1].strip() or reason

                score = round(min(1.0, max(0.0, parsed_score)), 4)
                reason = parsed_reason
        except Exception:
            # LLM 호출 실패 시 전체 빠른매칭이 멈추지 않도록 rule/vector 기반 대체 점수 사용
            pass

        return {
            "score": score,
            "reason": reason,
        }

    @staticmethod
    def serialize_party_text(party_data: dict) -> str:
        """
        파티 프로필을 임베딩용 텍스트로 직렬화한다.
        """
        return f"""
서비스: {party_data.get('service_name')}
카테고리: {party_data.get('category')}
플랫폼: {party_data.get('platform')}
가격대: {party_data.get('monthly_per_person')}
최소 신뢰도: {party_data.get('min_trust_score')}
최대 인원: {party_data.get('max_members')}
현재 인원: {party_data.get('current_members')}
설명: {party_data.get('description')}
이용 기간 성향: {party_data.get('duration_preference')}
상태: {party_data.get('status')}
""".strip()

    @staticmethod
    async def generate_party_embedding(party_data: dict) -> list[float]:
        """
        파티 정보를 텍스트로 직렬화해서 파티 임베딩 생성
        """
        text = EmbeddingService.serialize_party_text(party_data)
        return await EmbeddingService.generate_embedding({"text": text})
