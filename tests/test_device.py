"""Offline tests for ML execution-device handling.

No GPU, models, keys, or network needed: CUDA availability is stubbed
and real model loading is never triggered.
"""

from __future__ import annotations

import pytest

import retrieval.models as models
from retrieval.models import device_info, resolve_device


def test_resolve_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device() == "cuda"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_falls_back_to_cpu_loudly(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        assert resolve_device() == "cpu"
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        assert resolve_device("cuda") == "cpu"


def test_resolve_device_explicit_cpu_never_warns(monkeypatch, recwarn):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("cpu") == "cpu"
    assert [w for w in recwarn.list if "falling back" in str(w.message)] == []


def test_resolve_device_env_override(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setenv("BIS_DEVICE", "cpu")
    assert resolve_device() == "cpu"


def test_resolve_device_never_names_a_gpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device() in ("cuda", "cpu")


def test_device_info_safe_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: (_ for _ in ()).throw(AssertionError("must not query device")))
    info = device_info()
    assert info["cuda_available"] is False
    assert info["device_count"] == 0
    assert info["gpu_name"] is None
    assert info["resolved_device"] == "cpu"
    assert "torch_version" in info and info["torch_version"]


def test_device_info_reports_gpu_name_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda i: "Test GPU")
    info = device_info()
    assert info["cuda_available"] is True
    assert info["gpu_name"] == "Test GPU"
    assert info["resolved_device"] == "cuda"


def test_loaded_retriever_peek_does_not_load_models():
    import sys

    from retrieval.retrieve import loaded_retriever

    module = sys.modules["retrieval.retrieve"]
    assert module._default_retriever is None
    assert loaded_retriever() is None


def test_device_endpoint_without_loaded_models():
    httpx = pytest.importorskip("httpx", reason="fastapi.testclient requires httpx")
    from fastapi.testclient import TestClient

    from app.main import build_app

    application = build_app(chunk_index=None, provider=None)
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/device")
    assert response.status_code == 200
    body = response.json()
    assert body["retriever_loaded"] is False
    assert body["encoder_device"] is None
    assert body["reranker_device"] is None
    assert body["resolved_device"] in ("cuda", "cpu")
    assert "cuda_available" in body and "gpu_name" in body
    # No secrets anywhere in the payload.
    lowered = response.text.lower()
    assert "api_key" not in lowered and "gsk_" not in lowered
