"""
Sapphire proxy -- routes requests to Krystal backends.

Supports both streaming (SSE) and non-streaming responses.
"""

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import httpx


async def proxy_single(
    client: httpx.AsyncClient,
    backend: str,
    payload: dict,
) -> dict:
    """Send a non-streaming request to a Krystal instance."""
    resp = await client.post(
        f"{backend}/v1/chat/completions",
        json=payload,
    )
    if not resp.is_success:
        raise HTTPException(resp.status_code, f"krystal error: {resp.text[:200]}")
    return resp.json()


async def proxy_stream(
    client: httpx.AsyncClient,
    backend: str,
    payload: dict,
) -> StreamingResponse:
    """Send a streaming request and return an SSE response."""
    req = client.build_request(
        "POST",
        f"{backend}/v1/chat/completions",
        json={**payload, "stream": True},
    )
    resp = await client.send(req, stream=True)
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


async def call_backend_once(
    client: httpx.AsyncClient,
    backend: str,
    messages: list[dict],
    slot: int,
    sampling_params: dict,
    max_tokens: int = 2000,
) -> tuple[str, dict]:
    body = {
        "messages": messages,
        "id_slot": slot,
        "cache_prompt": True,
        "max_tokens": max_tokens,
        **sampling_params,
    }
    resp = await client.post(f"{backend}/v1/chat/completions", json=body)
    if not resp.is_success:
        raise HTTPException(resp.status_code, f"krystal error: {resp.text[:200]}")
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return text, usage


async def call_backend_with_retry(
    client: httpx.AsyncClient,
    backend: str,
    messages: list[dict],
    slot: int,
    sampling_params: dict,
    max_retries: int,
    max_tokens: int = 2000,
    log=None,
) -> tuple[str, dict]:
    last_response = ""
    last_usage: dict = {}
    for attempt in range(max_retries + 1):
        last_response, last_usage = await call_backend_once(
            client, backend, messages, slot, sampling_params, max_tokens,
        )
        from sapphire.degenerate import is_degenerate_output
        if not is_degenerate_output(last_response):
            return last_response, last_usage
        if log:
            log.warning(
                "degenerate output detected (attempt %d/%d): %r",
                attempt + 1, max_retries + 1, last_response,
            )
    return last_response, last_usage
