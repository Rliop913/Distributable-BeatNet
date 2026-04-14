from __future__ import annotations

from pathlib import Path

import onnx
import onnxruntime as ort
import pytest

from beatnet_ort.export_onnx import export_model


@pytest.mark.parametrize("model_id", [1, 2, 3])
def test_export_smoke(tmp_path: Path, model_id: int) -> None:
    output_path = tmp_path / f"model_{model_id}.onnx"
    exported = export_model(model_id=model_id, output_path=output_path)

    assert exported == output_path.resolve()
    assert exported.exists()

    model = onnx.load(exported)
    onnx.checker.check_model(model)

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "feats"
    assert session.get_outputs()[0].name == "logits"
