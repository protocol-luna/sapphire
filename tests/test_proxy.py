import sys
from unittest.mock import MagicMock

sys.modules["fastembed"] = MagicMock()

import pytest
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from sapphire.proxy import proxy_single, proxy_stream, call_backend_once, call_backend_with_retry


class _MockResponse:
    def __init__(self, is_success=True, json_data=None, status_code=200, text="ok", lines=None):
        self.is_success = is_success
        self._json = json_data or {}
        self.status_code = status_code
        self.text = text
        self._lines = lines or []

    def json(self):
        return self._json

    async def aread(self):
        return self.text.encode()

    def aiter_lines(self):
        class _Aiter:
            def __init__(self, lines):
                self._lines = iter(lines)
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self._lines)
                except StopIteration:
                    raise StopAsyncIteration
        return _Aiter(self._lines)


class _MockClient:
    def __init__(self):
        self.post_calls = 0
        self.post_responses = []
        self.post_response = _MockResponse()
        self.build_request = MagicMock()

    async def post(self, url, **kwargs):
        self.post_calls += 1
        if self.post_responses:
            return self.post_responses.pop(0)
        return self.post_response

    async def send(self, req, **kwargs):
        return self.post_response


class TestProxySingle:
    @pytest.mark.asyncio
    async def test_successful_request(self):
        client = _MockClient()
        client.post_response = _MockResponse(json_data={"choices": [{"text": "hi"}]})
        result = await proxy_single(client, "http://localhost:3124", {"prompt": "hi"})
        assert result == {"choices": [{"text": "hi"}]}
        assert client.post_calls == 1

    @pytest.mark.asyncio
    async def test_error_response_raises(self):
        client = _MockClient()
        client.post_response = _MockResponse(is_success=False, status_code=500, text="internal error")
        with pytest.raises(HTTPException):
            await proxy_single(client, "http://localhost:3124", {"prompt": "hi"})


class TestProxyStream:
    @pytest.mark.asyncio
    async def test_returns_streaming_response(self):
        client = _MockClient()
        client.post_response = _MockResponse(lines=["data: hello", "data: world", "data: [DONE]"])
        result = await proxy_stream(client, "http://localhost:3124", {"prompt": "hi"})
        assert isinstance(result, StreamingResponse)

    @pytest.mark.asyncio
    async def test_error_response_raises(self):
        client = _MockClient()
        client.post_response = _MockResponse(is_success=False, status_code=502, text="bad gateway")
        with pytest.raises(HTTPException):
            await proxy_stream(client, "http://localhost:3124", {"prompt": "hi"})


class TestCallBackendOnce:
    @pytest.mark.asyncio
    async def test_returns_text_and_usage(self):
        client = _MockClient()
        client.post_response = _MockResponse(json_data={
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
        text, usage = await call_backend_once(client, "http://localhost:3124", [], 0, {"temperature": 0.7})
        assert text == "hello world"
        assert usage == {"prompt_tokens": 10, "completion_tokens": 5}

    @pytest.mark.asyncio
    async def test_error_raises_http_exception(self):
        client = _MockClient()
        client.post_response = _MockResponse(is_success=False, status_code=502, text="bad gateway")
        with pytest.raises(HTTPException):
            await call_backend_once(client, "http://localhost:3124", [], 0, {})


class TestCallBackendWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        client = _MockClient()
        client.post_response = _MockResponse(json_data={
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {},
        })
        text, usage = await call_backend_with_retry(client, "http://localhost:3124", [], 0, {}, max_retries=2)
        assert text == "hello world"
        assert client.post_calls == 1

    @pytest.mark.asyncio
    async def test_retries_on_degenerate_output(self):
        ok_resp = _MockResponse(json_data={
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {},
        })
        empty_resp = _MockResponse(json_data={
            "choices": [{"message": {"content": ""}}],
            "usage": {},
        })
        client = _MockClient()
        client.post_responses = [empty_resp, ok_resp]
        text, usage = await call_backend_with_retry(client, "http://localhost:3124", [], 0, {}, max_retries=2)
        assert text == "hello world"
        assert client.post_calls == 2

    @pytest.mark.asyncio
    async def test_returns_last_result_on_max_retries(self):
        empty_resp = _MockResponse(json_data={
            "choices": [{"message": {"content": ""}}],
            "usage": {},
        })
        client = _MockClient()
        client.post_response = empty_resp
        text, usage = await call_backend_with_retry(client, "http://localhost:3124", [], 0, {}, max_retries=2)
        assert text == ""
        assert client.post_calls == 3

    @pytest.mark.asyncio
    async def test_logs_warning_on_degenerate(self):
        empty_resp = _MockResponse(json_data={
            "choices": [{"message": {"content": ""}}],
            "usage": {},
        })
        ok_resp = _MockResponse(json_data={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        })
        client = _MockClient()
        client.post_responses = [empty_resp, ok_resp]
        log = MagicMock()
        await call_backend_with_retry(client, "http://localhost:3124", [], 0, {}, max_retries=2, log=log)
        assert log.warning.called
