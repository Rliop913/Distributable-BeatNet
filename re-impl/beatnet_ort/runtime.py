from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .model import DEFAULT_FEATURE_DIM


def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def logits_to_activations(logits: np.ndarray) -> np.ndarray:
    """Convert BeatNet logits to the 2-channel activation layout used by inference."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 2:
        probs = _softmax(logits, axis=0)
        return probs[:2, :].T
    if logits.ndim == 3:
        probs = _softmax(logits, axis=1)
        return np.transpose(probs[:, :2, :], (0, 2, 1))
    raise ValueError(f"Expected logits with 2 or 3 dimensions; got shape {logits.shape}")


class BeatNetOrtRuntime:
    """Thin ONNX Runtime wrapper for the BeatNet CRNN."""

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=self.providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _normalize_input(self, feats: np.ndarray) -> tuple[np.ndarray, bool]:
        array = np.asarray(feats, dtype=np.float32)
        squeeze_batch = False

        if array.ndim == 2:
            array = array[None, ...]
            squeeze_batch = True
        elif array.ndim != 3:
            raise ValueError(
                f"Expected features shaped (time, {DEFAULT_FEATURE_DIM}) or (batch, time, {DEFAULT_FEATURE_DIM}); "
                f"got {array.shape}"
            )

        if array.shape[-1] != DEFAULT_FEATURE_DIM:
            raise ValueError(
                f"Expected feature dimension {DEFAULT_FEATURE_DIM}; got {array.shape[-1]}"
            )

        return np.ascontiguousarray(array), squeeze_batch

    def infer_logits(self, feats: np.ndarray) -> np.ndarray:
        array, squeeze_batch = self._normalize_input(feats)
        logits = self.session.run([self.output_name], {self.input_name: array})[0]
        if squeeze_batch:
            return logits[0]
        return logits

    def infer_activations(self, feats: np.ndarray) -> np.ndarray:
        return logits_to_activations(self.infer_logits(feats))
