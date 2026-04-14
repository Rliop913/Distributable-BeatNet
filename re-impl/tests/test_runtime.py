from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from beatnet_ort.export_onnx import export_model
from beatnet_ort.model import DEFAULT_FEATURE_DIM, build_pretrained_model
from beatnet_ort.runtime import BeatNetOrtRuntime


@pytest.fixture(scope="session")
def exported_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("onnx_models")
    return export_model(model_id=1, output_path=output_dir / "model_1.onnx")


@pytest.fixture(scope="session")
def pytorch_model() -> torch.nn.Module:
    return build_pretrained_model(model_id=1)


def test_runtime_matches_pytorch_on_batched_input(
    exported_model_path: Path,
    pytorch_model: torch.nn.Module,
) -> None:
    runtime = BeatNetOrtRuntime(exported_model_path)
    feats = np.random.randn(2, 50, DEFAULT_FEATURE_DIM).astype(np.float32)

    with torch.inference_mode():
        expected = pytorch_model(torch.from_numpy(feats)).cpu().numpy()
    actual = runtime.infer_logits(feats)

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("sequence_length", [1, 50, 200, 513])
def test_runtime_supports_dynamic_sequence_lengths(
    exported_model_path: Path,
    sequence_length: int,
) -> None:
    runtime = BeatNetOrtRuntime(exported_model_path)
    feats = np.random.randn(sequence_length, DEFAULT_FEATURE_DIM).astype(np.float32)

    logits = runtime.infer_logits(feats)
    activations = runtime.infer_activations(feats)

    assert logits.shape == (3, sequence_length)
    assert activations.shape == (sequence_length, 2)
    assert np.all(activations >= 0.0)
    assert np.all(activations <= 1.0)


def test_runtime_is_deterministic(exported_model_path: Path) -> None:
    runtime = BeatNetOrtRuntime(exported_model_path)
    feats = np.random.randn(200, DEFAULT_FEATURE_DIM).astype(np.float32)

    logits_a = runtime.infer_logits(feats)
    logits_b = runtime.infer_logits(feats)

    np.testing.assert_array_equal(logits_a, logits_b)
