import math
import time
from pathlib import Path

import httpx

# Send generated test parts to the local inspection service.
API_URL = "http://127.0.0.1:8000"
MAX_P95_MS = 200.0


def main() -> None:
    # Measure each request while printing the returned decision.
    paths = sorted(Path("data/test").glob("*/*/*.png"))
    timings: list[float] = []
    with httpx.Client(timeout=30.0) as client:
        client.get(f"{API_URL}/health").raise_for_status()
        for index, path in enumerate(paths):
            part_id = f"LINE-{index:04d}"
            category = path.parts[-3]
            start = time.perf_counter()
            with path.open("rb") as image:
                response = client.post(
                    f"{API_URL}/inspect",
                    params={"part_id": part_id, "category": category},
                    files={"file": (path.name, image, "image/png")},
                )
            timings.append((time.perf_counter() - start) * 1000.0)
            response.raise_for_status()
            result = response.json()
            print(part_id, result["decision"], f"score={result['score']:.4f}")

        # Read the durable reject count after every image has passed the line.
        count_response = client.get(f"{API_URL}/events/count")
        count_response.raise_for_status()
        logged_rejects = int(count_response.json()["count"])

    p95_ms = sorted(timings)[math.ceil(0.95 * len(timings)) - 1]
    print({"p95_ms": p95_ms, "logged_rejects": logged_rejects})
    if p95_ms > MAX_P95_MS or logged_rejects < 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()