from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_FEATURE_DIM = 272
DEFAULT_HIDDEN_SIZE = 150
DEFAULT_NUM_LAYERS = 2
DEFAULT_NUM_CLASSES = 3
DEFAULT_CONV_OUT = 150
DEFAULT_KERNEL_SIZE = 10

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODELS_DIR = REPO_ROOT / "src" / "BeatNet" / "models"
MODEL_ID_TO_WEIGHTS = {
    1: SOURCE_MODELS_DIR / "model_1_weights.pt",
    2: SOURCE_MODELS_DIR / "model_2_weights.pt",
    3: SOURCE_MODELS_DIR / "model_3_weights.pt",
}


class StatelessBeatNetCRNN(nn.Module):
    """Stateless BeatNet CRNN compatible with the original weight files."""

    def __init__(
        self,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.conv_out = DEFAULT_CONV_OUT
        self.kernel_size = DEFAULT_KERNEL_SIZE

        pooled_dim = 2 * ((self.feature_dim - self.kernel_size + 1) // 2)

        self.conv1 = nn.Conv1d(1, 2, self.kernel_size)
        self.linear0 = nn.Linear(pooled_dim, self.conv_out)
        self.lstm = nn.LSTM(
            input_size=self.conv_out,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the stateless CRNN on feature sequences.

        Args:
            data: Feature tensor shaped `(batch, time, 272)`.

        Returns:
            Logits shaped `(batch, 3, time)`.
        """
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D input shaped (batch, time, features); got {tuple(data.shape)}"
            )

        batch_size, sequence_length, _ = data.shape
        x = data.reshape(-1, self.feature_dim).unsqueeze(1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = x.flatten(start_dim=1)
        x = self.linear0(x)
        x = x.reshape(batch_size, sequence_length, self.conv_out)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x.transpose(1, 2)


def resolve_weights_path(model_id: int | None = None, weights_path: str | Path | None = None) -> Path:
    if weights_path is not None:
        path = Path(weights_path).expanduser().resolve()
    elif model_id in MODEL_ID_TO_WEIGHTS:
        path = MODEL_ID_TO_WEIGHTS[model_id]
    else:
        raise ValueError("Provide either a valid model_id in {1, 2, 3} or an explicit weights_path.")

    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")
    return path


def build_pretrained_model(
    model_id: int | None = None,
    weights_path: str | Path | None = None,
) -> StatelessBeatNetCRNN:
    model = StatelessBeatNetCRNN()
    state_dict = torch.load(resolve_weights_path(model_id=model_id, weights_path=weights_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
