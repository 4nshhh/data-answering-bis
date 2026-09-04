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

from app.generator.context_builder import ChunkIndex, load_chunk_index
from app.generator.llm_client import GroqProvider, LLMProvider
from app.generator.pipeline import QueryResult, run_query
from app.generator.refusal import DEFAULT_THRESHOLD

__all__ = ["app", "QueryRequest", "CitationModel", "RetrievalMetaModel", "QueryResponse",
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


class QueryResponse(BaseModel):
    """Response payload (AGENTS.md section 12 + refusal extras)."""

    query: str
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    retrieval_meta: RetrievalMetaModel = Field(default_factory=RetrievalMetaModel)
    refused: bool = False
    refusal_reason: Optional[str] = None


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
    )


def build_app(
    chunk_index: ChunkIndex | None = None,
    provider: LLMProvider | None = None,
    chunks_dir: str | Path = Path("data/chunks"),
) -> FastAPI:
    """Assemble the application (singletons injectable for tests)."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.chunk_index = chunk_index if chunk_index is not None else load_chunk_index(chunks_dir)
        application.state.provider = provider if provider is not None else GroqProvider()
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
            result = run_query(
                payload.query,
                top_k=payload.top_k,
                expand_neighbors=payload.expand_neighbors,
                threshold=payload.confidence_threshold,
                retrieve_fn=retrieve_fn,
                chunk_index=request.app.state.chunk_index,
                provider=request.app.state.provider,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc
        return _result_to_response(result)

    return application


app: Any = build_app()
