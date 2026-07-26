"""
Sapphire proxy — routes requests to Krystal backends.

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
