import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.pipeline import ARTIFACT_DIR, TinyCNN, load_split

# Fail export when numerical agreement or CPU latency misses the project limits.
MAX_ABS_ERROR = 1e-4
MAX_P95_MS = 200.0


def main() -> None:
    # Restore the trained classifier and export one-image inference.
    model = TinyCNN()
    model.load_state_dict(torch.load(ARTIFACT_DIR / "model.pt", weights_only=True))
    model.eval()
    images, _, _ = load_split(Path("data"), "test")
    sample = torch.from_numpy(images[:1])
    program = torch.onnx.export(
        model,
        (sample,),
        input_names=["image"],
        output_names=["logit"],
        dynamo=True,
    )
    onnx_path = ARTIFACT_DIR / "model.onnx"
    program.save(onnx_path)
    onnx.checker.check_model(onnx.load(onnx_path))

    # Compare PyTorch and ONNX Runtime on the same tensor.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_output = model(sample).numpy()
    ort_output = session.run(None, {"image": sample.numpy()})[0]
    max_abs_error = float(np.abs(torch_output - ort_output).max())

    # Warm the runtime before collecting one hundred latency measurements.
    for _ in range(10):
        session.run(None, {"image": sample.numpy()})
    timings = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {"image": sample.numpy()})
        timings.append((time.perf_counter() - start) * 1000.0)
    p95_ms = sorted(timings)[math.ceil(0.95 * len(timings)) - 1]

    result = {
        "max_abs_error": max_abs_error,
        "p95_ms": p95_ms,
        "max_allowed_error": MAX_ABS_ERROR,
        "max_allowed_p95_ms": MAX_P95_MS,
    }
    (ARTIFACT_DIR / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if max_abs_error > MAX_ABS_ERROR or p95_ms > MAX_P95_MS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()