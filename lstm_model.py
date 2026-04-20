"""
LSTM 마우스 궤적 분류 모델

입력: (batch, seq_len, 3)  ← dx, dy, dt (차분값, 정규화)
출력: bot 확률 0.0~1.0

아키텍처:
  입력 정규화 → Bidirectional LSTM(hidden=64, layers=2)
  → 마지막 hidden concat → Linear(256→64) → ReLU → Dropout
  → Linear(64→1) → Sigmoid
"""

import torch
import torch.nn as nn


class MouseLSTM(nn.Module):
    """마우스 궤적 이진 분류 LSTM"""

    def __init__(
        self,
        input_size: int = 5,     # dx, dy, dt, speed, angle
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 입력 정규화
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 분류기
        fc_input = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(fc_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) — 패딩된 시퀀스
            lengths: (batch,) — 각 시퀀스의 실제 길이 (선택)
        Returns:
            (batch,) — bot 확률 0~1
        """
        batch_size = x.size(0)

        # 입력 정규화
        x = self.input_norm(x)

        # pack if lengths provided (패딩 무시)
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            output, (hidden, _) = self.lstm(packed)
        else:
            output, (hidden, _) = self.lstm(x)

        # 마지막 레이어의 forward + backward hidden 결합
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch, hidden_size)
            h_forward = hidden[-2]   # 마지막 레이어 forward
            h_backward = hidden[-1]  # 마지막 레이어 backward
            h_combined = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_combined = hidden[-1]

        # 분류
        logit = self.classifier(h_combined).squeeze(-1)  # (batch,)
        return torch.sigmoid(logit)


def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 모델 구조 확인
    model = MouseLSTM()
    print(model)
    print(f"\n학습 가능 파라미터: {count_parameters(model):,}개")

    # 더미 입력 테스트
    dummy = torch.randn(4, 100, 5)  # batch=4, seq_len=100, features=5
    lengths = torch.tensor([100, 80, 50, 30])
    out = model(dummy, lengths)
    print(f"출력 shape: {out.shape}")  # (4,)
    print(f"출력 값: {out.detach().numpy()}")
