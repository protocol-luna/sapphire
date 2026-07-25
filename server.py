"""
sapphire — LLM gateway / middleware.

Classifies each user message as FUTILE (trivial) or INTERESSANT (serious)
using embedding centroid similarity (fastembed + BAAI/bge-small-en-v1.5),
then routes to the appropriate Krystal backend.

Usage:
  export KRYSTAL_GENERIC_URL=http://127.0.0.1:3124
  export KRYSTAL_SEMANTIC_URL=http://127.0.0.1:3125
  python server.py          # listens on 127.0.0.1:3123
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastembed import TextEmbedding
from pydantic import BaseModel, Field
import httpx
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="[sapphire] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sapphire")

PORT = int(os.environ.get("SAPPHIRE_PORT", "3123"))
KRYSTAL_GENERIC_URL = os.environ.get(
    "KRYSTAL_GENERIC_URL", "http://127.0.0.1:3124"
)
KRYSTAL_SEMANTIC_URL = os.environ.get(
    "KRYSTAL_SEMANTIC_URL", "http://127.0.0.1:3125"
)
CONFIDENCE_THRESHOLD = float(os.environ.get("SAPPHIRE_CONFIDENCE_THRESHOLD", "0.0"))
EXAMPLES_PATH = os.environ.get(
    "SAPPHIRE_EXAMPLES",
    str(Path(__file__).parent / "examples.yml"),
)

embedder: TextEmbedding | None = None
futile_centroid: np.ndarray | None = None
interessant_centroid: np.ndarray | None = None
http_client: httpx.AsyncClient | None = None


# ---------------------------------------------------------------------------
# Classification (embedding centroid similarity)
# ---------------------------------------------------------------------------

def load_examples(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = yaml.safe_load(f)

    def expand(items: list) -> list[str]:
        out = []
        for item in items:
            if isinstance(item, dict):
                for _ in range(item.get("weight", 1)):
                    out.append(item["text"])
            else:
                out.append(str(item))
        return out

    return expand(data.get("futile", [])), expand(data.get("interessant", []))


def compute_centroids(
    embedder: TextEmbedding,
    futile: list[str],
    interessant: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    f_emb = np.array(list(embedder.passage_embed(futile)))
    i_emb = np.array(list(embedder.passage_embed(interessant)))
    return f_emb.mean(axis=0), i_emb.mean(axis=0)


def classify(text: str) -> tuple[str, float, float, float]:
    if embedder is None or futile_centroid is None or interessant_centroid is None:
        return "FUTILE", 0.0, 0.0, 0.0
    emb = next(embedder.query_embed(text))
    norm = np.linalg.norm(emb)
    if norm == 0:
        return "FUTILE", 0.0, 0.0, 0.0
    sim_f = float(np.dot(emb, futile_centroid) / (norm * np.linalg.norm(futile_centroid)))
    sim_i = float(np.dot(emb, interessant_centroid) / (norm * np.linalg.norm(interessant_centroid)))
    diff = sim_i - sim_f
    label = "INTERESSANT" if diff > 0 else "FUTILE"
    return label, abs(diff), sim_f, sim_i


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global embedder, futile_centroid, interessant_centroid, http_client

    log.info("loading embedding model BAAI/bge-small-en-v1.5...")
    embedder = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        max_length=128,
    )

    log.info("loading examples from %s", EXAMPLES_PATH)
    futile_examples, interessant_examples = load_examples(EXAMPLES_PATH)
    log.info("  %d futile, %d interessant examples", len(futile_examples), len(interessant_examples))

    log.info("computing centroids...")
    futile_centroid, interessant_centroid = compute_centroids(
        embedder, futile_examples, interessant_examples,
    )

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=5.0),
    )

    log.info(
        "ready — FUTILE -> %s | INTERESSANT -> %s",
        KRYSTAL_GENERIC_URL,
        KRYSTAL_SEMANTIC_URL,
    )
    yield

    if http_client:
        await http_client.aclose()
    embedder = None


app = FastAPI(title="sapphire", lifespan=lifespan)


class ChatCompletionRequest(BaseModel):
    messages: list[dict] = Field(..., min_length=1)
    model: str = ""
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    id_slot: int | None = None
    cache_prompt: bool | None = None


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    if embedder is None:
        raise HTTPException(503, "model not loaded")

    user_msgs = [m for m in body.messages if m.get("role") == "user"]
    if not user_msgs:
        raise HTTPException(400, "no user message found")

    last_user_text = user_msgs[-1].get("content", "")
    label, conf, sim_f, sim_i = classify(last_user_text)

    backend = KRYSTAL_GENERIC_URL if label == "FUTILE" else KRYSTAL_SEMANTIC_URL
    log.info(
        "%s (Δ=%.3f, f=%.3f i=%.3f) | %s | %s",
        label, conf, sim_f, sim_i, backend, last_user_text[:80],
    )

    payload = body.model_dump(exclude={"stream"}, exclude_none=True)
    if "id_slot" not in payload:
        payload["id_slot"] = 0
    if "cache_prompt" not in payload:
        payload["cache_prompt"] = True

    if body.stream:
        return await _proxy_stream(backend, payload)
    return await _proxy_single(backend, payload)


async def _proxy_single(backend: str, payload: dict) -> dict:
    resp = await http_client.post(
        f"{backend}/v1/chat/completions",
        json=payload,
    )
    if not resp.is_success:
        raise HTTPException(resp.status_code, f"krystal error: {resp.text[:200]}")
    return resp.json()


async def _proxy_stream(backend: str, payload: dict) -> StreamingResponse:
    req = http_client.build_request(
        "POST",
        f"{backend}/v1/chat/completions",
        json={**payload, "stream": True},
    )
    resp = await http_client.send(req, stream=True)
    if not resp.is_success:
        raise HTTPException(resp.status_code, f"krystal error: {await resp.aread()}")

    async def event_stream():
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                yield line + "\n\n"
            elif line.strip():
                yield f"data: {line}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


class ClassifyQuery(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)


class ClassifyResult(BaseModel):
    label: str
    confidence: float
    sim_futile: float
    sim_interessant: float


@app.post("/classify", response_model=ClassifyResult)
async def classify_endpoint(query: ClassifyQuery):
    label, conf, sim_f, sim_i = classify(query.text[:2048])
    return ClassifyResult(
        label=label, confidence=round(conf, 4),
        sim_futile=round(sim_f, 4), sim_interessant=round(sim_i, 4),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "method": "embedding-centroid",
        "model": "BAAI/bge-small-en-v1.5",
        "krystal_generic": KRYSTAL_GENERIC_URL,
        "krystal_semantic": KRYSTAL_SEMANTIC_URL,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
