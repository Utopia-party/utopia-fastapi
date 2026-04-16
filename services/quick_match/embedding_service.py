import httpx

from core.config import settings


class EmbeddingService:
    @staticmethod
    async def generate_embedding(payload: dict) -> list[float]:
        """
        Ollama 임베딩 모델로 임베딩 생성
        """
        text = payload.get("text", "")

        async with httpx.AsyncClient(timeout=10.0) as client:
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
사용자 정보를 기반으로 매칭에 사용할 핵심 특징을 요약해라:
{payload}
"""

        async with httpx.AsyncClient(timeout=20.0) as client:
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
            return data.get("response", "")

    @staticmethod
    async def generate_party_embedding(party_data: dict) -> list[float]:
        """
        파티 정보를 텍스트로 직렬화해서 파티 임베딩 생성
        """
        text = f"""
서비스: {party_data.get("service")}
가격: {party_data.get("price")}
최소 신뢰도: {party_data.get("min_trust")}
설명: {party_data.get("description")}
"""
        return await EmbeddingService.generate_embedding({"text": text})