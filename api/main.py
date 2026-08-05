from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.model_cache import warm_cache
from api.routers import churn, insights, occupancy, pricing, rag, restaurant, reviews, staffing
from genai.config.genai_settings import genai_settings
from src.mlops.monitoring.metrics_middleware import instrument_app
from src.mlops.observability.logging_config import configure_mlops_logging
from src.utils.logging import get_logger

configure_mlops_logging("api", "api.log")

logger = get_logger(__name__)


def _warm_rag_index() -> None:
    """Optionally preload the FAISS index at startup when ENABLE_RAG=true.
    Never fails startup if the index hasn't been built yet -- /rag/index
    can build it later; this just makes /rag/stats reflect an existing
    index sooner and warms the embedding model into memory."""
    if not genai_settings.ENABLE_RAG:
        logger.info("ENABLE_RAG is false; skipping RAG index warm-up")
        return
    try:
        from genai.rag.embeddings import get_embedder
        from genai.rag.vector_store.faiss_store import FaissVectorStore

        get_embedder().embed_text("warmup")

        store = FaissVectorStore()
        if store.load():
            logger.info("RAG index warmed: %d vectors", store.size)
        else:
            logger.info("No existing RAG index found; call POST /rag/index to build one")
    except Exception as exc:
        logger.warning("RAG index warm-up skipped due to error: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_cache()
    _warm_rag_index()
    yield


app = FastAPI(title="HotelMind ML Prediction API", lifespan=lifespan)
instrument_app(app)

app.include_router(occupancy.router, tags=["occupancy"])
app.include_router(pricing.router, tags=["pricing"])
app.include_router(restaurant.router, tags=["restaurant"])
app.include_router(staffing.router, tags=["staffing"])
app.include_router(churn.router, tags=["churn"])
app.include_router(reviews.router, tags=["reviews"])
app.include_router(rag.router, tags=["rag"])
app.include_router(insights.router, tags=["insights"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
