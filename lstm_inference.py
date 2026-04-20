"""
LSTM 마우스 궤적 봇 탐지 — 추론 모듈

FastAPI captcha_service.py에서 호출하는 진입점:
    from lstm_inference import predict_bot_probability
    score = await predict_bot_probability(mouse_moves)

- 서버 시작 시 best_model.pt를 한 번만 로드 (싱글턴)
- CPU 추론 (~10-50ms), GPU 불필요
- 모델 파일 없으면 graceful fallback (0.5 반환)
"""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("lstm_inference")

# ── 설정 ──────────────────────────────────────────────────
MAX_SEQ_LEN = 200       # train_lstm.py와 동일
FEATURE_DIM = 5         # dx, dy, dt, speed, angle
FALLBACK_SCORE = 0.5    # 모델 없을 때 중립 점수 (사람도 봇도 아닌 애매한 값)
MIN_MOVES = 4           # 최소 마우스 포인트 수

# 체크포인트 경로: fastapi/ 기준으로 상위의 checkpoints/ 폴더
_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "best_model.pt"

# ── 싱글턴 모델 ──────────────────────────────────────────
_model = None
_model_loaded = False


def _get_model():
    """모델을 한 번만 로드하는 싱글턴. 실패 시 None 반환."""
    global _model, _model_loaded

    if _model_loaded:
        return _model

    _model_loaded = True  # 실패해도 재시도하지 않음

    if not _CHECKPOINT_PATH.exists():
        logger.warning(
            f"[LSTM] 체크포인트 없음: {_CHECKPOINT_PATH} → fallback 모드"
        )
        return None

    try:
        # lstm_model.py는 fastapi/ 상위(ganpipeline/)에 있으므로 sys.path 추가
        import sys
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from lstm_model import MouseLSTM

        model = MouseLSTM(
            input_size=FEATURE_DIM,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
        )

        # best_model.pt 포맷 감지: state_dict만 있는 경우 vs full checkpoint
        checkpoint = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # state_dict만 저장된 경우 (OrderedDict)
            model.load_state_dict(checkpoint)

        model.eval()
        _model = model
        logger.info(
            f"[LSTM] 모델 로드 완료: {_CHECKPOINT_PATH} "
            f"(파라미터 {sum(p.numel() for p in model.parameters()):,}개)"
        )
        return _model

    except Exception as e:
        logger.error(f"[LSTM] 모델 로드 실패: {e}")
        return None


# ── 피처 추출 (train_lstm.py::extract_features와 동일) ────
def _extract_features(mouse_moves: list[dict[str, Any]]) -> np.ndarray:
    """
    raw [{x, y, t}, ...] → (seq_len-1, 5) 차분 피처

    피처: [dx, dy, dt, speed, angle]
    - dx, dy: 좌표 차분 (정규화: /1920, /1080)
    - dt: 시간 차분 (ms → s, /1000)
    - speed: √(dx²+dy²) / dt
    - angle: atan2(dy, dx) / π  (정규화: -1~1)
    """
    if len(mouse_moves) < 2:
        return np.zeros((1, FEATURE_DIM), dtype=np.float32)

    features = []
    for i in range(1, len(mouse_moves)):
        prev = mouse_moves[i - 1]
        curr = mouse_moves[i]

        dx = (curr["x"] - prev["x"]) / 1920.0
        dy = (curr["y"] - prev["y"]) / 1080.0
        dt = (curr["t"] - prev["t"]) / 1000.0

        dist = math.sqrt(dx ** 2 + dy ** 2)
        speed = dist / max(dt, 1e-6)
        angle = math.atan2(dy, dx) / math.pi

        features.append([dx, dy, dt, speed, angle])

    return np.array(features, dtype=np.float32)


# ── 추론 함수 (서비스에서 호출) ───────────────────────────
async def predict_bot_probability(
    mouse_moves: list[dict[str, Any]],
) -> tuple[float, bool]:
    """
    마우스 궤적으로 봇 확률을 예측.

    Args:
        mouse_moves: SDK에서 수집한 [{x, y, t}, ...] 리스트

    Returns:
        (bot_probability, is_available)
        - bot_probability: 0.0(사람) ~ 1.0(봇). 모델 없으면 FALLBACK_SCORE.
        - is_available: 모델이 정상 로드되어 실제 추론했는지 여부.
          False면 fallback 값이므로 final_score에 반영하지 않아야 함.
    """
    model = _get_model()

    # 모델 없음 → fallback
    if model is None:
        return FALLBACK_SCORE, False

    # 포인트 부족 → fallback
    if len(mouse_moves) < MIN_MOVES:
        logger.debug(f"[LSTM] 포인트 부족 ({len(mouse_moves)}개) → fallback")
        return FALLBACK_SCORE, False

    try:
        # 1. 피처 추출
        features = _extract_features(mouse_moves)
        seq_len = len(features)

        # 2. 트렁케이션 + 패딩
        if seq_len > MAX_SEQ_LEN:
            features = features[:MAX_SEQ_LEN]
            seq_len = MAX_SEQ_LEN

        padded = np.zeros((MAX_SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        padded[:seq_len] = features

        # 3. 텐서 변환 (batch=1)
        x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)       # (1, 200, 5)
        lengths = torch.tensor([seq_len], dtype=torch.long)               # (1,)

        # 4. 추론 (CPU, no grad)
        with torch.no_grad():
            bot_prob = model(x, lengths).item()  # float 0.0~1.0

        logger.debug(
            f"[LSTM] 추론 완료: bot_prob={bot_prob:.4f}, "
            f"seq_len={seq_len}, moves={len(mouse_moves)}"
        )

        return bot_prob, True

    except Exception as e:
        logger.error(f"[LSTM] 추론 오류: {e}")
        return FALLBACK_SCORE, False


# ── 모델 상태 확인 (헬스체크용) ───────────────────────────
def get_model_status() -> dict:
    """관리자 API에서 모델 상태를 확인할 때 사용."""
    model = _get_model()
    return {
        "loaded": model is not None,
        "checkpoint_path": str(_CHECKPOINT_PATH),
        "checkpoint_exists": _CHECKPOINT_PATH.exists(),
        "max_seq_len": MAX_SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "fallback_score": FALLBACK_SCORE,
    }
