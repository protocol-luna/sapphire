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
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastembed import TextEmbedding
from pydantic import BaseModel, Field
import httpx
import uvicorn

from classifier import load_examples, compute_centroids, classify, get_default_examples_path
from proxy import proxy_single, proxy_stream

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
EXAMPLES_PATH = os.environ.get(
    "SAPPHIRE_EXAMPLES",
    get_default_examples_path(),
)

embedder: TextEmbedding | None = None
futile_centroid: np.ndarray | None = None
interessant_centroid: np.ndarray | None = None
http_client: httpx.AsyncClient | None = None


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
    label, conf, sim_f, sim_i = classify(
        last_user_text, embedder, futile_centroid, interessant_centroid,
    )

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
        return await proxy_stream(http_client, backend, payload)
    return await proxy_single(http_client, backend, payload)


class ClassifyQuery(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)


class ClassifyResult(BaseModel):
    label: str
    confidence: float
    sim_futile: float
    sim_interessant: float


@app.post("/classify", response_model=ClassifyResult)
async def classify_endpoint(query: ClassifyQuery):
    label, conf, sim_f, sim_i = classify(
        query.text[:2048], embedder, futile_centroid, interessant_centroid,
    )
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
