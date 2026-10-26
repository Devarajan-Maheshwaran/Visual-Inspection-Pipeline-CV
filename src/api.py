import json
import math
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psycopg
from fastapi import FastAPI, UploadFile
from PIL import Image

# Load the calibrated threshold and ONNX graph once during service startup.
ARTIFACT_DIR = Path("artifacts")
DATABASE_URL = os.getenv("DATABASE_URL", "")
report = json.loads((ARTIFACT_DIR / "report.json").read_text(encoding="utf-8"))
threshold = float(report["cnn"]["threshold"])
session = ort.InferenceSession(str(ARTIFACT_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
app = FastAPI(title="Visual Inspection Line")


def prepare_image(content: bytes) -> np.ndarray:
    # Match uploaded images to the model's grayscale input shape.
    with Image.open(BytesIO(content)) as image:
        array = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
    return array[None, None, :, :]


def create_event_table(connection: psycopg.Connection) -> None:
    # Keep the event schema available before inserts or counts.
    connection.execute("""
        CREATE TABLE IF NOT EXISTS defect_events (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            inspected_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
            part_id text NOT NULL,
            category text NOT NULL,
            image_name text NOT NULL,
            score double precision NOT NULL,
            threshold double precision NOT NULL,
            rejected boolean NOT NULL
        )
    """)


def log_reject(part_id: str, category: str, image_name: str, score: float) -> None:
    # Persist only rejected parts when a database is configured.
    if not DATABASE_URL:
        return
    with psycopg.connect(DATABASE_URL) as connection:
        create_event_table(connection)
        connection.execute(
            "INSERT INTO defect_events (part_id, category, image_name, score, threshold, rejected) VALUES (%s, %s, %s, %s, %s, %s)",
            (part_id, category, image_name, score, threshold, True),
        )


@app.get("/health")
def health() -> dict[str, str]:
    # Give orchestrators a lightweight readiness response.
    return {"status": "ok"}


@app.post("/inspect")
async def inspect(file: UploadFile, part_id: str = "unknown", category: str = "unknown") -> dict[str, object]:
    # Convert one upload into a calibrated ACCEPT or REJECT decision.
    tensor = prepare_image(await file.read())
    logit = float(session.run(None, {"image": tensor})[0].reshape(-1)[0])
    score = 1.0 / (1.0 + math.exp(-logit))
    rejected = score > threshold
    if rejected:
        log_reject(part_id, category, file.filename or "uploaded-image", score)
    return {
        "part_id": part_id,
        "category": category,
        "score": score,
        "threshold": threshold,
        "decision": "REJECT" if rejected else "ACCEPT",
    }


@app.get("/events/count")
def event_count() -> dict[str, object]:
    # Report how many rejected-part events reached PostgreSQL.
    if not DATABASE_URL:
        return {"count": 0, "database": "disabled"}
    with psycopg.connect(DATABASE_URL) as connection:
        create_event_table(connection)
        row = connection.execute("SELECT count(*) FROM defect_events").fetchone()
    return {"count": int(row[0]), "database": "postgresql"}