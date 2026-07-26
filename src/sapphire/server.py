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

from sapphire.classifier import load_examples, compute_centroids, classify, get_default_examples_path
from sapphire.emotion import (
    load_emotion_examples,
    compute_emotion_centroids,
    score_axes,
    get_default_emotion_examples_path,
    EmotionState,
)
from sapphire.proxy import proxy_single, proxy_stream

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
EMOTION_EXAMPLES_PATH = os.environ.get(
    "SAPPHIRE_EMOTION_EXAMPLES",
    get_default_emotion_examples_path(),
)
EMOTION_DECAY = float(os.environ.get("SAPPHIRE_EMOTION_DECAY", "0.85"))

embedder: TextEmbedding | None = None
futile_centroid: np.ndarray | None = None
interessant_centroid: np.ndarray | None = None
emotion_centroids: dict[str, np.ndarray] | None = None
emotion_state = EmotionState(decay=EMOTION_DECAY)
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global embedder, futile_centroid, interessant_centroid, emotion_centroids, http_client

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

    log.info("loading emotion examples from %s", EMOTION_EXAMPLES_PATH)
    emotion_examples = load_emotion_examples(EMOTION_EXAMPLES_PATH)
    for pole, texts in emotion_examples.items():
        log.info("  %s: %d examples", pole, len(texts))
    emotion_centroids = compute_emotion_centroids(embedder, emotion_examples)

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

    # Un seul appel embedder, réutilisé pour le routing FUTILE/INTERESSANT
    # et pour le scoring émotionnel — zéro latence supplémentaire.
    emb = next(embedder.query_embed(last_user_text)) if last_user_text.strip() else None

    label, conf, sim_f, sim_i = classify(
        last_user_text, embedder, futile_centroid, interessant_centroid,
        precomputed_emb=emb,
    )

    valence, arousal = score_axes(emb, emotion_centroids)

    # clé de conversation : id_slot sert déjà à isoler les contextes llama.cpp,
    # on le réutilise ici. À défaut, retombe sur "default" (un seul état global).
    conv_key = str(body.id_slot) if body.id_slot is not None else "default"
    state = emotion_state.update(conv_key, valence, arousal)

    backend = KRYSTAL_GENERIC_URL if label == "FUTILE" else KRYSTAL_SEMANTIC_URL
    log.info(
        "%s (Δ=%.3f, f=%.3f i=%.3f) | valence=%.3f arousal=%.3f (state v=%.3f a=%.3f) | %s | %s",
        label, conf, sim_f, sim_i, valence, arousal,
        state["valence"], state["arousal"], backend, last_user_text[:80],
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


@app.get("/emotion/{conv_key}")
async def get_emotion(conv_key: str):
    """État émotionnel courant (valence/arousal) d'une conversation.

    À appeler depuis Jade après chaque réponse générée, pour ajuster délai,
    burst mode, longueur de réponse, typo rate, etc. — voir la state machine
    comportementale existante (topic fatigue, sleep cycles).
    """
    return emotion_state.get(conv_key)


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
