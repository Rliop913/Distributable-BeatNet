# BeatNet ORT Re-Implementation

This directory contains a standalone ONNX Runtime path for the BeatNet CRNN.

## Scope

- Re-implements the BeatNet CRNN in a stateless form suitable for ONNX export.
- Reuses the original PyTorch weight files from `src/BeatNet/models`.
- Keeps audio feature extraction and PF/DBN post-processing outside the ONNX graph.
- Targets `online` and `offline` style batch inference only.

## Layout

- `beatnet_ort/model.py`: stateless CRNN implementation compatible with the original weights
- `beatnet_ort/export_onnx.py`: ONNX exporter
- `beatnet_ort/runtime.py`: ONNX Runtime wrapper
- `beatnet_ort/feature_adapter.py`: optional audio-to-feature adapter using the original BeatNet preprocessing
- `models/`: generated ONNX models
- `tests/`: parity and export tests

## Export

From the repository root:

```bash
PYTHONPATH=re-impl ./.venv/bin/python -m beatnet_ort.export_onnx --all
```

This writes:

- `re-impl/models/model_1.onnx`
- `re-impl/models/model_2.onnx`
- `re-impl/models/model_3.onnx`

## Runtime Usage

```python
import numpy as np

from beatnet_ort.runtime import BeatNetOrtRuntime

runtime = BeatNetOrtRuntime("re-impl/models/model_1.onnx")
features = np.random.randn(200, 272).astype(np.float32)

logits = runtime.infer_logits(features)        # (3, T)
activations = runtime.infer_activations(features)  # (T, 2)
```

## Optional Audio Features

```python
from beatnet_ort.feature_adapter import AudioFeatureAdapter
from beatnet_ort.runtime import BeatNetOrtRuntime

adapter = AudioFeatureAdapter()
runtime = BeatNetOrtRuntime("re-impl/models/model_1.onnx")

features = adapter.audio_to_features("src/BeatNet/test_data/808kick120bpm.mp3")
activations = runtime.infer_activations(features)
```

## Test

```bash
PYTHONPATH=re-impl:src ./.venv/bin/python -m pytest re-impl/tests -q
```

## Non-Goals

- No streaming or realtime stateful LSTM path
- No ONNX export of `LOG_SPECT`
- No ONNX export of particle filtering or DBN decoding
