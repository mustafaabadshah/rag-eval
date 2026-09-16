import json
from typing import Any

import anyio
import pytest

pytest.importorskip("fastapi")
from examples.fastapi_server import app  # noqa: E402


class ASGIResponse:
    """Wrapper around raw ASGI response."""

    def __init__(self, status_code: int, headers: list[tuple[bytes, bytes]], body: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self.content = body

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


class SimpleTestClient:
    """Lightweight ASGI test client that requires no external HTTP client dependencies."""

    def __init__(self, asgi_app: Any) -> None:
        self.app = asgi_app

    def request(self, method: str, path: str, json_data: Any = None) -> ASGIResponse:
        body = json.dumps(json_data).encode("utf-8") if json_data is not None else b""
        headers = [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("utf-8")),
        ]
        status_code = 500
        res_headers: list[tuple[bytes, bytes]] = []
        res_body = bytearray()

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            nonlocal status_code, res_headers, res_body
            if message["type"] == "http.response.start":
                status_code = message["status"]
                res_headers = message.get("headers", [])
            elif message["type"] == "http.response.body":
                res_body.extend(message.get("body", b""))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": headers,
        }

        async def _run() -> None:
            await self.app(scope, receive, send)

        anyio.run(_run)
        return ASGIResponse(status_code, res_headers, bytes(res_body))

    def get(self, path: str) -> ASGIResponse:
        return self.request("GET", path)

    def post(self, path: str, json: Any = None) -> ASGIResponse:
        return self.request("POST", path, json_data=json)


try:
    from fastapi.testclient import TestClient

    client: Any = TestClient(app)
except (ImportError, RuntimeError):
    client = SimpleTestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_evaluate_endpoint_valid() -> None:
    payload = {
        "samples": [
            {
                "question": "What is the capital of France?",
                "answer": "Paris is the capital of France.",
                "context": ["Paris is the capital of France."],
                "golden_documents": ["Paris is the capital of France."],
                "retrieved": ["Paris is the capital of France."],
                "id": "s1",
            },
            {
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "context": ["Python is an interpreted programming language."],
                "id": "s2",
            },
        ],
        "k": 5,
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_samples"] == 2
    assert "mean_faithfulness" in data
    assert len(data["per_sample"]) == 2
    assert data["per_sample"][0]["sample_id"] == "s1"


def test_evaluate_endpoint_invalid_sample_returns_422() -> None:
    payload = {
        "samples": [
            {
                "question": "Where is Rome?",
                # Missing "answer" and "context"
            }
        ]
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 422
