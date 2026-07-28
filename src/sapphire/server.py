"""
sapphire -- LLM gateway / middleware.

Classifies each user message as FUTILE (trivial) or INTERESSANT (serious)
using embedding centroid similarity (fastembed + BAAI/bge-small-en-v1.5),
then routes to the appropriate Krystal backend.

Usage:
  export KRYSTAL_GENERIC_URL=http://127.0.0.1:3124
  export KRYSTAL_SEMANTIC_URL=http://127.0.0.1:3125
  python server.py          # listens on 127.0.0.1:3123
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastembed import TextEmbedding
from pydantic import BaseModel, Field

from sapphire.classifier import (
    centroid_path,
    classify,
    compute_centroids,
    get_default_examples_path,
    load_centroids,
    load_examples,
    save_centroids,
)
from sapphire.degenerate import is_degenerate_output
from sapphire.emotion import (
    EmotionState,
    compute_emotion_centroids,
    emotion_centroid_path,
    get_default_emotion_examples_path,
    load_emotion_centroids,
    load_emotion_examples,
    save_emotion_centroids,
    score_axes,
)
from sapphire.few_shot import (
    format_few_shot_examples,
    inject_few_shot_into_conversation,
    load_few_shot_examples,
)
from sapphire.proxy import call_backend_with_retry, proxy_single, proxy_stream
from sapphire.sessions import SessionStore

logging.basicConfig(
    level=logging.INFO,
    format="[sapphire] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sapphire")

_config_path = Path(__file__).resolve().parent.parent.parent / "config.yml"
with open(_config_path) as _f:
    _cfg = yaml.safe_load(_f)

PORT = int(_cfg.get("port", 3123))
KRYSTAL_GENERIC_URL = str(_cfg.get("krystal_generic_url", "http://127.0.0.1:3124"))
KRYSTAL_SEMANTIC_URL = str(_cfg.get("krystal_semantic_url", "http://127.0.0.1:3124"))
EXAMPLES_PATH = str(_cfg.get("examples_path", get_default_examples_path()))
EMOTION_EXAMPLES_PATH = str(
    _cfg.get("emotion_examples_path", get_default_emotion_examples_path())
)
EMOTION_DECAY = float(_cfg.get("emotion_decay", 0.85))
EMOTION_DEADZONE = float(_cfg.get("emotion_deadzone", 0.005))

BOT_NAME = str(_cfg.get("bot_name", "Luna"))

SYSTEM_PROMPT = str(
    _cfg.get(
        "system_prompt",
        f"Your name is {BOT_NAME}. You are a playful 21-year-old girl. "
        "Keep responses short and casual like a normal Discord user. "
        "Never repeat or echo the user's message back to them. "
        "Never say 'I understand' or 'I see'. Just answer naturally.",
    )
)
FEW_SHOT_ENABLED = bool(_cfg.get("few_shot_enabled", True))
FEW_SHOT_EXAMPLES_PATH = str(
    _cfg.get(
        "few_shot_examples_path",
        str(Path(__file__).resolve().parent.parent.parent / "few_shot_examples.yml"),
    )
)
LLM_N_SLOTS = int(_cfg.get("llm_n_slots", 1))
LLM_SESSION_TTL = float(_cfg.get("llm_session_ttl", 600))
LLM_MAX_HISTORY = int(_cfg.get("llm_max_history", 20))
LLM_MAX_RETRIES = int(_cfg.get("llm_max_retries", 2))

MIROSTAT_ENABLED = bool(_cfg.get("mirostat_enabled", True))
MIROSTAT_MODE = int(_cfg.get("mirostat_mode", 2))
MIROSTAT_LR = float(_cfg.get("mirostat_lr", 0.1))
MIROSTAT_ENT = float(_cfg.get("mirostat_ent", 5.0))

few_shot_examples: list[dict[str, str]] = []
session_store: SessionStore | None = None

embedder: TextEmbedding | None = None
futile_centroid: np.ndarray | None = None
interessant_centroid: np.ndarray | None = None
emotion_centroids: dict[str, np.ndarray] | None = None
emotion_state = EmotionState(decay=EMOTION_DECAY, deadzone=EMOTION_DEADZONE)
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global \
        embedder, \
        futile_centroid, \
        interessant_centroid, \
        emotion_centroids, \
        http_client
    global few_shot_examples, session_store

    log.info("loading embedding model BAAI/bge-small-en-v1.5...")
    embedder = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        max_length=128,
    )

    # Try loading prebuilt centroids first
    futile_centroid, interessant_centroid = load_centroids()
    emotion_centroids = load_emotion_centroids()

    if (
        futile_centroid is not None
        and interessant_centroid is not None
        and emotion_centroids is not None
    ):
        log.info(
            "loaded prebuilt centroids from %s/", centroid_path().rsplit("/", 1)[0]
        )
    else:
        log.info("prebuilt centroids not found, computing from examples...")
        log.info("loading examples from %s", EXAMPLES_PATH)
        futile_examples, interessant_examples = load_examples(EXAMPLES_PATH)
        log.info(
            "  %d futile, %d interessant examples",
            len(futile_examples),
            len(interessant_examples),
        )

        log.info("computing classification centroids...")
        futile_centroid, interessant_centroid = compute_centroids(
            embedder,
            futile_examples,
            interessant_examples,
        )
        save_centroids(futile_centroid, interessant_centroid)
        log.info("  saved to %s", centroid_path())

        log.info("loading emotion examples from %s", EMOTION_EXAMPLES_PATH)
        emotion_examples = load_emotion_examples(EMOTION_EXAMPLES_PATH)
        for pole, texts in emotion_examples.items():
            log.info("  %s: %d examples", pole, len(texts))
        emotion_centroids = compute_emotion_centroids(embedder, emotion_examples)
        save_emotion_centroids(emotion_centroids)
        log.info("  saved emotion centroids to %s", emotion_centroid_path())

    log.info("loading few-shot examples from %s", FEW_SHOT_EXAMPLES_PATH)
    few_shot_examples = load_few_shot_examples(FEW_SHOT_EXAMPLES_PATH)
    log.info(
        "  %d few-shot examples (enabled=%s)", len(few_shot_examples), FEW_SHOT_ENABLED
    )

    session_store = SessionStore(
        system_prompt=SYSTEM_PROMPT,
        ttl_seconds=LLM_SESSION_TTL,
        n_slots=LLM_N_SLOTS,
        max_history=LLM_MAX_HISTORY,
    )

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=5.0),
    )

    log.info(
        "ready -- FUTILE -> %s | INTERESSANT -> %s",
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
    # et pour le scoring émotionnel -- zéro latence supplémentaire.
    emb = next(embedder.query_embed(last_user_text)) if last_user_text.strip() else None

    label, conf, sim_f, sim_i = classify(
        last_user_text,
        embedder,
        futile_centroid,
        interessant_centroid,
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
        label,
        conf,
        sim_f,
        sim_i,
        valence,
        arousal,
        state["valence"],
        state["arousal"],
        backend,
        last_user_text[:80],
    )

    payload = body.model_dump(exclude={"stream"}, exclude_none=True)
    if "id_slot" not in payload:
        payload["id_slot"] = 0
    if "cache_prompt" not in payload:
        payload["cache_prompt"] = True

    if body.stream:
        return await proxy_stream(http_client, backend, payload)
    return await proxy_single(http_client, backend, payload)


class RespondRequest(BaseModel):
    username: str = ""
    text: str = Field(..., min_length=1)
    session_id: str = "default"
    stream: bool = False
    debug: bool = False


class RespondResult(BaseModel):
    text: str
    label: str
    backend: str
    valence: float
    arousal: float
    debug_prompt_tokens: int | None = None
    debug_completion_tokens: int | None = None
    debug_time_ms: float | None = None
    debug_tokens_per_second: float | None = None
    debug_emotion_state_valence: float | None = None
    debug_emotion_state_arousal: float | None = None
    debug_classification_confidence: float | None = None


import re

_USER_LEAK_RE = re.compile(r"\nUser\s*:")


def truncate_user_leak(text: str) -> str:
    m = _USER_LEAK_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def _sampling_params(valence: float = 0, arousal: float = 0) -> dict:
    temp = max(0.4, min(1.0, 0.7 + arousal * 0.3))
    penalty = max(1.1, min(1.6, 1.3 - valence * 0.15))
    if MIROSTAT_ENABLED:
        ent = max(3.0, min(8.0, MIROSTAT_ENT + arousal * 2.0))
        return {
            "mirostat": MIROSTAT_MODE,
            "mirostat_lr": MIROSTAT_LR,
            "mirostat_ent": ent,
            "repeat_penalty": round(penalty, 2),
            "repeat_last_n": 128,
        }
    return {
        "temperature": round(temp, 2),
        "top_k": 60,
        "top_p": 0.9,
        "min_p": 0.05,
        "repeat_penalty": round(penalty, 2),
        "repeat_last_n": 128,
    }


@app.post("/v1/respond")
async def respond(body: RespondRequest, request: Request):
    if embedder is None or session_store is None:
        raise HTTPException(503, "sapphire not ready")

    emb = next(embedder.query_embed(body.text)) if body.text.strip() else None

    label, conf, _sim_f, _sim_i = classify(
        body.text,
        embedder,
        futile_centroid,
        interessant_centroid,
        precomputed_emb=emb,
    )
    valence, arousal = score_axes(emb, emotion_centroids)
    emotion_state.update(body.session_id, valence, arousal)

    backend = KRYSTAL_GENERIC_URL if label == "FUTILE" else KRYSTAL_SEMANTIC_URL

    session = session_store.append_user_message(
        body.session_id, body.username, body.text
    )

    final_messages = session.messages
    if FEW_SHOT_ENABLED and few_shot_examples:
        fs_messages = format_few_shot_examples(few_shot_examples, body.username or None)
        final_messages = inject_few_shot_into_conversation(
            session.messages, fs_messages
        )

    slot = session_store.slot_for(body.session_id)
    params = _sampling_params(valence, arousal)

    if not body.stream:
        t0 = time.monotonic()
        text, usage = await call_backend_with_retry(
            http_client,
            backend,
            final_messages,
            slot,
            params,
            LLM_MAX_RETRIES,
            log=log,
        )
        elapsed = time.monotonic() - t0

        text = truncate_user_leak(text)
        session_store.append_assistant_message(body.session_id, text)
        removed = session_store.cleanup_stale()
        if removed:
            log.info("cleaned up %d stale session(s)", removed)

        emo_state = emotion_state.get(body.session_id)

        log.info(
            "%s (Δ=%.3f) | valence=%.3f arousal=%.3f | %s | %s",
            label,
            conf,
            valence,
            arousal,
            backend,
            body.text[:80],
        )

        if body.debug:
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tps = round(ct / elapsed, 1) if ct and elapsed > 0 else None
            return RespondResult(
                text=text,
                label=label,
                backend=backend,
                valence=valence,
                arousal=arousal,
                debug_prompt_tokens=pt,
                debug_completion_tokens=ct,
                debug_time_ms=round(elapsed * 1000),
                debug_tokens_per_second=tps,
                debug_emotion_state_valence=round(emo_state.get("valence", 0), 4),
                debug_emotion_state_arousal=round(emo_state.get("arousal", 0), 4),
                debug_classification_confidence=round(conf, 4),
            )

        return RespondResult(
            text=text,
            label=label,
            backend=backend,
            valence=valence,
            arousal=arousal,
        )

    # --- Streaming mode ---
    body_payload = {
        "messages": final_messages,
        "id_slot": slot,
        "cache_prompt": True,
        "max_tokens": 2000,
        "stream": True,
        **params,
    }

    req = http_client.build_request(
        "POST",
        f"{backend}/v1/chat/completions",
        json=body_payload,
    )
    resp = await http_client.send(req, stream=True)
    if not resp.is_success:
        raise HTTPException(resp.status_code, f"krystal error: {await resp.aread()}")

    full_text: list[str] = []
    t0 = time.monotonic()

    async def event_stream():
        nonlocal t0
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    delta = (
                        data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    )
                    if delta:
                        full_text.append(delta)
                        yield f"data: {delta}\n\n"
                except json.JSONDecodeError:
                    yield f"data: {payload}\n\n"

        text = "".join(full_text)
        text = truncate_user_leak(text)
        if is_degenerate_output(text):
            log.warning("degenerate output discarded from stream: %r", text[:80])
            yield "data: [DONE]\n\n"
            return

        session_store.append_assistant_message(body.session_id, text)
        removed = session_store.cleanup_stale()
        if removed:
            log.info("cleaned up %d stale session(s)", removed)

        elapsed = time.monotonic() - t0
        emo_state = emotion_state.get(body.session_id)

        log.info(
            "%s (Δ=%.3f) | valence=%.3f arousal=%.3f | %s | %s",
            label,
            conf,
            valence,
            arousal,
            backend,
            body.text[:80],
        )

        meta = (
            {
                "text": text,
                "label": label,
                "backend": backend,
                "valence": valence,
                "arousal": arousal,
                "prompt_tokens": 0,
                "completion_tokens": len(text.split()),
                "time_ms": round(elapsed * 1000),
                "tokens_per_second": round(len(text.split()) / elapsed, 1)
                if elapsed > 0
                else 0,
                "emotion_state_valence": round(emo_state.get("valence", 0), 4),
                "emotion_state_arousal": round(emo_state.get("arousal", 0), 4),
                "classification_confidence": round(conf, 4),
            }
            if body.debug
            else {
                "text": text,
                "label": label,
                "backend": backend,
                "valence": valence,
                "arousal": arousal,
            }
        )

        yield f"data: {json.dumps(meta)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/v1/reset")
async def reset_session(session_id: str | None = None):
    if session_store:
        session_store.reset(session_id)
    return {"status": "ok", "session_id": session_id or "all"}


class ClassifyQuery(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)


class ClassifyResult(BaseModel):
    label: str
    confidence: float
    sim_futile: float
    sim_interessant: float
    valence: float = 0.0
    arousal: float = 0.0


@app.post("/classify", response_model=ClassifyResult)
async def classify_endpoint(query: ClassifyQuery):
    label, conf, sim_f, sim_i = classify(
        query.text[:2048],
        embedder,
        futile_centroid,
        interessant_centroid,
    )
    emb = next(embedder.query_embed(query.text[:2048]))
    valence, arousal = score_axes(emb, emotion_centroids)
    return ClassifyResult(
        label=label,
        confidence=round(conf, 4),
        sim_futile=round(sim_f, 4),
        sim_interessant=round(sim_i, 4),
        valence=round(valence, 4),
        arousal=round(arousal, 4),
    )


@app.get("/emotion/{conv_key}")
async def get_emotion(conv_key: str):
    """État émotionnel courant (valence/arousal) d'une conversation.

    À appeler depuis Jade après chaque réponse générée, pour ajuster délai,
    burst mode, longueur de réponse, typo rate, etc. -- voir la state machine
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
        "few_shot_enabled": FEW_SHOT_ENABLED,
        "few_shot_examples": len(few_shot_examples),
        "active_sessions": len(session_store._sessions) if session_store else 0,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
