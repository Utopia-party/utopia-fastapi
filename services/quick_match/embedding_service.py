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
    async def generate_profile_summary(payload: dict) -> str:
        """
        Ollama LLM으로 사용자 요약 생성
        """
        prompt = f"""
사용자 정보를 기반으로 매칭에 사용할 핵심 특징을 짧고 구조적으로 요약해라.
- 가격 성향
- 서비스 이용 성향
- 장기 유지 가능성
- 신뢰도/운영 안정성
- 주의할 점

사용자 정보:
{payload}
"""

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
            return str(data.get("response", "") or "").strip()

    @staticmethod
    async def generate_match_evaluation(payload: dict) -> dict:
        """
        사용자-파티 조합을 LLM으로 한 번 더 평가한다.
        현재는 파싱 안정성을 위해 숫자 점수와 사유를 단순 포맷으로 반환한다.
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
            # LLM 호출 실패 시 전체 빠른매칭이 멈추지 않도록 룰/벡터 기반 대체 점수 사용
            pass

        return {
            "score": score,
            "reason": reason,
        }

    @staticmethod
    def serialize_party_text(party_data: dict) -> str:
        return f"""
서비스: {party_data.get('service_name')}
가격대: {party_data.get('monthly_per_person')}
최소 신뢰도: {party_data.get('min_trust_score')}
최대 인원: {party_data.get('max_members')}
현재 인원: {party_data.get('current_members')}
설명: {party_data.get('description')}
장기 선호 여부: {party_data.get('duration_preference')}
""".strip()

    @staticmethod
    async def generate_party_embedding(party_data: dict) -> list[float]:
        """
        파티 정보를 텍스트로 직렬화해서 파티 임베딩 생성
        """
        text = EmbeddingService.serialize_party_text(party_data)
        return await EmbeddingService.generate_embedding({"text": text})