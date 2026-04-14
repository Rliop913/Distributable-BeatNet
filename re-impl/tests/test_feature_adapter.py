from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("librosa")
pytest.importorskip("madmom")

from beatnet_ort.export_onnx import export_model
from beatnet_ort.feature_adapter import AudioFeatureAdapter
from beatnet_ort.runtime import BeatNetOrtRuntime


def test_audio_feature_adapter_runs_end_to_end(repo_root: Path, tmp_path: Path) -> None:
    audio_path = repo_root / "src" / "BeatNet" / "test_data" / "808kick120bpm.mp3"
    model_path = export_model(model_id=1, output_path=tmp_path / "model_1.onnx")

    adapter = AudioFeatureAdapter()
    runtime = BeatNetOrtRuntime(model_path)

    feats = adapter.audio_to_features(audio_path)
    activations = runtime.infer_activations(feats)

    assert feats.ndim == 2
    assert feats.shape[1] == 272
    assert activations.shape == (feats.shape[0], 2)
