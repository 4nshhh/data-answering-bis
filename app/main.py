"""Phase 8: FastAPI web service exposing the RAG pipeline.

Endpoint ``POST /api/v1/query`` (AGENTS.md section 12) executes
``app.generator.pipeline.run_query`` and renders ``QueryResult`` as
JSON. Heavy singletons (chunk index, LLM provider) load once in the
lifespan handler; retrieval models load lazily on first query via the
frozen ``retrieval`` package.

Run (from the repository root)::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.generator import answer, warmup
from app.generator.context_builder import ChunkIndex, load_chunk_index
from app.generator.llm_client import LLMProvider, build_provider
from app.generator.pipeline import QueryResult
from app.generator.refusal import DEFAULT_THRESHOLD
from app.generator.telemetry import Telemetry

__all__ = ["app", "QueryRequest", "CitationModel", "RetrievalMetaModel", "TelemetryModel", "QueryResponse",
           "DeviceResponse", "UTF8JSONResponse"]


class UTF8JSONResponse(JSONResponse):
    """JSON with an explicit charset so clients decode UTF-8 correctly.

    Starlette's default ``Content-Type: application/json`` carries no
    charset, and non-UTF-8-defaulting clients (notably Windows/PowerShell
    tooling) then fall back to a single-byte codepage — turning correct
    UTF-8 bytes for characters like U+202F/U+3010 into ``â¯``/``ã…``
    mojibake on display. The bytes were always right; this declares them.
    """

    media_type = "application/json; charset=utf-8"


class QueryRequest(BaseModel):
    """Request payload (AGENTS.md section 12)."""

    query: str = Field(..., min_length=1, description="Natural-language BIS question")
    top_k: int = Field(3, ge=1, le=5, description="Context blocks assembled")
    expand_neighbors: bool = Field(False, description="Stitch chunk_index +/- 1 context")
    confidence_threshold: float = Field(DEFAULT_THRESHOLD, description="Refusal threshold tau")


class CitationModel(BaseModel):
    standard_no: str
    year: Optional[str] = None
    clause: str
    page: int
    chunk_id: str
    verified: bool = True


class RetrievalMetaModel(BaseModel):
    filtered_standard: Optional[str] = None
    is_mask_restricted: bool = False
    candidates_retrieved: int = 0
    top_reranker_score: Optional[float] = None
    execution_time_ms: float = 0.0


class TelemetryModel(BaseModel):
    """Per-query LLM usage telemetry (observability only, no secrets)."""

    llm_generation_attempts: int = 0
    llm_api_calls: int = 0
    groq_api_calls: int = 0
    llm_provider: str = ""
    llm_model: str = ""
    mode: str = "ask"
    correction_retry: bool = False
    widen_retry: bool = False
    retrieval_expansion: bool = False
    latency_ms: float = 0.0
    stages: dict[str, float] = Field(default_factory=dict)
    prompt_chars: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0


class QueryResponse(BaseModel):
    """Response payload (AGENTS.md section 12 + refusal extras)."""

    query: str
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    retrieval_meta: RetrievalMetaModel = Field(default_factory=RetrievalMetaModel)
    refused: bool = False
    refusal_reason: Optional[str] = None
    telemetry: TelemetryModel = Field(default_factory=TelemetryModel)


class DeviceResponse(BaseModel):
    """Runtime device diagnostics (no keys or secrets, safe to expose)."""

    cuda_available: bool
    torch_version: str
    cuda_build: Optional[str] = None
    device_count: int = 0
    gpu_name: Optional[str] = None
    resolved_device: str = "cpu"
    retriever_loaded: bool = False
    encoder_device: Optional[str] = None
    reranker_device: Optional[str] = None


def _result_to_response(result: QueryResult) -> QueryResponse:
    meta = result.retrieval_meta
    tele = result.telemetry
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        citations=[
            CitationModel(
                standard_no=c.standard_no,
                year=c.year,
                clause=c.clause,
                page=c.page,
                chunk_id=c.chunk_id,
                verified=c.verified,
            )
            for c in result.citations
        ],
        retrieval_meta=RetrievalMetaModel(
            filtered_standard=meta.filtered_standard if meta else None,
            is_mask_restricted=meta.is_mask_restricted if meta else False,
            candidates_retrieved=meta.candidates_retrieved if meta else 0,
            top_reranker_score=meta.top_reranker_score if meta else None,
            execution_time_ms=meta.execution_time_ms if meta else 0.0,
        ),
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        telemetry=TelemetryModel(
            llm_generation_attempts=tele.llm_generation_attempts if tele else 0,
            llm_api_calls=tele.llm_api_calls if tele else 0,
            groq_api_calls=tele.groq_api_calls if tele else 0,
            llm_provider=tele.llm_provider if tele else "",
            llm_model=tele.llm_model if tele else "",
            mode=tele.mode if tele else "ask",
            correction_retry=tele.correction_retry if tele else False,
            widen_retry=tele.widen_retry if tele else False,
            retrieval_expansion=tele.retrieval_expansion if tele else False,
            latency_ms=tele.latency_ms if tele else 0.0,
            stages=dict(tele.stages) if tele else {},
            prompt_chars=tele.prompt_chars if tele else 0,
            prompt_tokens_total=tele.prompt_tokens_total if tele else 0,
            completion_tokens_total=tele.completion_tokens_total if tele else 0,
        ),
    )


def build_app(
    chunk_index: ChunkIndex | None = None,
    provider: LLMProvider | None = None,
    chunks_dir: str | Path = Path(__file__).resolve().parent.parent / "data" / "chunks",
) -> FastAPI:
    """Assemble the application (singletons injectable for tests)."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.chunk_index = chunk_index if chunk_index is not None else load_chunk_index(chunks_dir)
        # Provider from LLM_PROVIDER (Groq default); same singleton serves
        # every request in this process.
        application.state.provider = provider if provider is not None else build_provider()
        # Optional one-time retriever warmup (BIS_WARMUP=1) delegating to
        # the core library's warmup(): loads BGE-M3 + CrossEncoder and
        # runs one dummy retrieval at startup so the first real query
        # pays ~0.5s instead of ~40s model load. No LLM call, no quota,
        # best-effort (lazy loading remains the fallback). Opt-in so
        # test lifespans stay fast. Direct-library backends that never
        # start this server call warmup() themselves instead.
        import os as _os

        if _os.environ.get("BIS_WARMUP", "0") == "1":
            try:
                warmup()
            except Exception as exc:  # noqa: BLE001 - lazy path still works
                import logging as _logging

                _logging.getLogger(__name__).warning("retriever warmup failed: %s", exc)
        yield

    application = FastAPI(title="BIS Standards Assistant (Answering/RAG)", lifespan=lifespan)

    @application.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/v1/device", response_model=DeviceResponse,
                      response_class=UTF8JSONResponse)
    def device() -> DeviceResponse:
        """Report ML execution-device diagnostics for the live process.

        ``retriever_loaded`` is False until the first query lazily
        constructs the shared ``Retriever``; device fields are populated
        from the actually loaded models once present.
        """
        from retrieval import device_info as _device_info
        from retrieval import loaded_retriever as _loaded_retriever

        info = _device_info()
        shared = _loaded_retriever()
        return DeviceResponse(
            cuda_available=info["cuda_available"],
            torch_version=info["torch_version"],
            cuda_build=info["cuda_build"],
            device_count=info["device_count"],
            gpu_name=info["gpu_name"],
            resolved_device=info["resolved_device"],
            retriever_loaded=shared is not None,
            encoder_device=str(getattr(shared._model, "device", None))
            if shared is not None else None,
            reranker_device=str(getattr(shared._reranker, "device", None))
            if shared is not None else None,
        )

    @application.post("/api/v1/query", response_model=QueryResponse,
                       response_class=UTF8JSONResponse)
    def post_query(payload: QueryRequest, request: Request) -> QueryResponse:
        from retrieval import retrieve as retrieve_fn

        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="query must be a non-blank string")
        try:
            # Thin adapter: the canonical pipeline lives in
            # app.generator.answer(); HTTP only validates + serializes.
            # The versioned query endpoint always answers in ask mode.
            telemetry = Telemetry()
            result = answer(
                payload.query,
                payload.top_k,
                expand_neighbors=payload.expand_neighbors,
                threshold=payload.confidence_threshold,
                retrieve_fn=retrieve_fn,
                chunk_index=request.app.state.chunk_index,
                provider=request.app.state.provider,
                telemetry=telemetry,
                mode="ask",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc
        return _result_to_response(result)

    return application


app: Any = build_app()
