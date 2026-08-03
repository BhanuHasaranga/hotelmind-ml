from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.schemas_genai import RagIndexRequest, RagIndexResponse, RagQueryRequest, RagQueryResponse, RagStatsResponse
from genai.llm.factory import get_llm_provider
from genai.rag.chains.qa_chain import NoContextError, answer_question, stream_answer
from genai.rag.indexer import build_index, incremental_index, index_stats
from genai.rag.retriever import HybridRetriever, MetadataFilter
from genai.rag.vector_store.faiss_store import FaissVectorStore
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _safe_llm_provider():
    try:
        return get_llm_provider()
    except Exception as exc:
        logger.warning("LLM provider unavailable for RAG query: %s", exc)
        return None


def _build_retriever() -> HybridRetriever:
    store = FaissVectorStore()
    store.load()
    return HybridRetriever(store)


@router.post("/rag/index", response_model=RagIndexResponse)
def index(request: RagIndexRequest) -> RagIndexResponse:
    try:
        build_index(include_warehouse=request.include_warehouse, include_predictions=request.include_predictions)
        stats = index_stats()
    except Exception as exc:
        logger.exception("RAG indexing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RagIndexResponse(**stats)


@router.post("/rag/reindex", response_model=RagIndexResponse)
def reindex(request: RagIndexRequest) -> RagIndexResponse:
    try:
        incremental_index(include_warehouse=request.include_warehouse, include_predictions=request.include_predictions)
        stats = index_stats()
    except Exception as exc:
        logger.exception("RAG reindexing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RagIndexResponse(**stats)


@router.get("/rag/stats", response_model=RagStatsResponse)
def stats() -> RagStatsResponse:
    try:
        result = index_stats()
    except Exception as exc:
        logger.exception("Fetching RAG stats failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RagStatsResponse(**result)


@router.post("/rag/query")
def query(request: RagQueryRequest):
    metadata_filter = MetadataFilter(doc_type=request.doc_type, branch_id=request.branch_id)
    llm_provider = _safe_llm_provider()

    if request.stream:
        def event_generator():
            try:
                retriever = _build_retriever()
                for chunk in stream_answer(
                    request.query,
                    retriever,
                    llm_provider,
                    persona=request.persona,
                    metadata_filter=metadata_filter,
                ):
                    yield {"event": "token", "data": chunk}
                yield {"event": "done", "data": ""}
            except NoContextError as exc:
                yield {"event": "error", "data": str(exc)}
            except Exception as exc:
                logger.exception("RAG streaming query failed")
                yield {"event": "error", "data": str(exc)}

        return EventSourceResponse(event_generator())

    try:
        retriever = _build_retriever()
        result = answer_question(
            request.query,
            retriever,
            llm_provider,
            persona=request.persona,
            session_id=request.session_id,
            metadata_filter=metadata_filter,
        )
    except NoContextError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RagQueryResponse(
        answer=result.answer,
        citations=[c.__dict__ for c in result.citations],
        used_llm=result.used_llm,
    )
