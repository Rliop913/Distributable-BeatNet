from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from .model import DEFAULT_FEATURE_DIM, REPO_ROOT, build_pretrained_model

REIMPL_MODELS_DIR = REPO_ROOT / "re-impl" / "models"


def default_output_path(model_id: int) -> Path:
    return REIMPL_MODELS_DIR / f"model_{model_id}.onnx"


def export_model(
    model_id: int,
    output_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    opset: int = 17,
    dummy_frames: int = 50,
) -> Path:
    model = build_pretrained_model(model_id=model_id, weights_path=weights_path)
    target = Path(output_path) if output_path is not None else default_output_path(model_id)
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros(1, dummy_frames, DEFAULT_FEATURE_DIM, dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            target,
            export_params=True,
            do_constant_folding=True,
            input_names=["feats"],
            output_names=["logits"],
            dynamic_axes={
                "feats": {0: "batch", 1: "time"},
                "logits": {0: "batch", 2: "time"},
            },
            opset_version=opset,
            dynamo=False,
        )

    onnx_model = onnx.load(target)
    onnx.checker.check_model(onnx_model)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BeatNet CRNN weights to ONNX.")
    parser.add_argument(
        "--model-id",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        default=[1],
        help="Model ID(s) to export.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all bundled BeatNet models.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        help="Optional path to a custom PyTorch weight file. Only valid with a single model ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Only valid with a single model ID.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dummy-frames",
        type=int,
        default=50,
        help="Sequence length used for the dummy export input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_ids = [1, 2, 3] if args.all else args.model_id

    if args.output is not None and len(model_ids) != 1:
        raise SystemExit("--output can only be used when exporting a single model.")
    if args.weights_path is not None and len(model_ids) != 1:
        raise SystemExit("--weights-path can only be used when exporting a single model.")

    for model_id in model_ids:
        output = export_model(
            model_id=model_id,
            output_path=args.output,
            weights_path=args.weights_path,
            opset=args.opset,
            dummy_frames=args.dummy_frames,
        )
        print(output)


if __name__ == "__main__":
    main()
