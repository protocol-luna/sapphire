"""
sapphire — LLM gateway / middleware.

Classifies each user message as GENERIC (trivial) or SEMANTIC (serious),
then routes to the appropriate Krystal backend.

Usage:
  export KRYSTAL_GENERIC_URL=http://127.0.0.1:3124
  export KRYSTAL_SEMANTIC_URL=http://127.0.0.1:3125
  python server.py          # listens on 127.0.0.1:3123
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="[sapphire] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sapphire")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
PORT = int(os.environ.get("SAPPHIRE_PORT", "3123"))
KRYSTAL_GENERIC_URL = os.environ.get(
    "KRYSTAL_GENERIC_URL", "http://127.0.0.1:3124"
)
KRYSTAL_SEMANTIC_URL = os.environ.get(
    "KRYSTAL_SEMANTIC_URL", "http://127.0.0.1:3125"
)

tokenizer: Tokenizer | None = None
session: ort.InferenceSession | None = None
http_client: httpx.AsyncClient | None = None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(text: str) -> tuple[str, float]:
    if session is None or tokenizer is None:
        return "GENERIC", 1.0
    encoding = tokenizer.encode(text[:1024])
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    logits = session.run(
        None,
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )[0]
    scores = 1.0 / (1.0 + np.exp(-logits))
    probs = scores[0] / scores[0].sum()
    if probs[1] > probs[0]:
        return "SEMANTIC", round(float(probs[1]), 4)
    return "GENERIC", round(float(probs[0]), 4)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global tokenizer, session, http_client
    log.info("loading tokenizer...")
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=64)
    tokenizer.enable_truncation(max_length=64)
    log.info("loading ONNX model...")
    session = ort.InferenceSession(
        os.path.join(MODEL_DIR, "model_quantized.onnx"),
        providers=["CPUExecutionProvider"],
    )
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    log.info(
        "ready — GENERIC -> %s | SEMANTIC -> %s",
        KRYSTAL_GENERIC_URL,
        KRYSTAL_SEMANTIC_URL,
    )
    yield
    if http_client:
        await http_client.aclose()
    tokenizer = None
    session = None


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
    if session is None:
        raise HTTPException(503, "model not loaded")

    # Extract last user message for classification
    user_msgs = [m for m in body.messages if m.get("role") == "user"]
    if not user_msgs:
        raise HTTPException(400, "no user message found")

    last_user_text = user_msgs[-1].get("content", "")
    label, score = classify(last_user_text)

    # Pick backend URL
    backend = KRYSTAL_GENERIC_URL if label == "GENERIC" else KRYSTAL_SEMANTIC_URL
    log.info(
        "%s (%.1f%%) | %s | user: %s",
        label,
        score * 100,
        backend,
        last_user_text[:60],
    )

    # Build forward payload (passthrough + extra params from headers)
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
        raise HTTPException(
            resp.status_code, f"krystal error: {resp.text[:200]}"
        )
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
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(__import__("time").time())
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                yield line + "\n\n"
            elif line.strip():
                yield f"data: {line}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Sapphire-Label": label if "label" in dir() else "",
        },
    )


@app.post("/classify")
async def classify_endpoint(data: dict):
    text = data.get("text", "")
    if not text:
        return {"label": "GENERIC", "score": 1.0}
    label, score = classify(text[:1024])
    return {"label": label, "score": score}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": session is not None,
        "krystal_generic": KRYSTAL_GENERIC_URL,
        "krystal_semantic": KRYSTAL_SEMANTIC_URL,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
