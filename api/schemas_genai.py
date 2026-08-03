from pydantic import BaseModel


# --- Reviews ---
class ReviewsAnalyzeRequest(BaseModel):
    limit: int | None = None


class ReviewsAnalyzeResponse(BaseModel):
    reviews_analyzed: int
    output_dir: str
    keywords: list[str]
    topics: list[dict]


class ReviewsSummaryResponse(BaseModel):
    summary: str
    keywords: list[str]
    csat_by_hotel: list[dict]


class ReviewsTopicsResponse(BaseModel):
    topics: list[dict]


class ReviewsComplaintsResponse(BaseModel):
    complaints: list[dict]


class ReviewsTrendsResponse(BaseModel):
    grain: str
    series: list[dict]
    trend_by_hotel: dict


# --- RAG ---
class RagIndexRequest(BaseModel):
    include_warehouse: bool = True
    include_predictions: bool = True


class RagIndexResponse(BaseModel):
    indexed: bool
    vector_count: int
    store_dir: str


class RagQueryRequest(BaseModel):
    query: str
    persona: str = "hotel_analyst"
    session_id: str | None = None
    doc_type: str | None = None
    branch_id: str | None = None
    stream: bool = False


class RagCitation(BaseModel):
    source: str
    doc_type: str | None = None
    score: float = 0.0
    extra: dict = {}


class RagQueryResponse(BaseModel):
    answer: str
    citations: list[RagCitation]
    used_llm: bool


class RagStatsResponse(BaseModel):
    indexed: bool
    vector_count: int
    store_dir: str


# --- Insights ---
class InsightItem(BaseModel):
    category: str
    title: str
    metric: str
    metric_delta: float | None
    severity: str
    branch_id: object | None = None
    supporting_data: dict
    citation: str
    priority_score: float
    recommendation: str


class InsightsResponse(BaseModel):
    insights: list[InsightItem]


class ExecutiveInsightsResponse(BaseModel):
    narrative: str
    top_findings: list[InsightItem]


class RecommendationItem(BaseModel):
    category: str
    title: str
    recommendation: str
    priority_score: float


class RecommendationsResponse(BaseModel):
    recommendations: list[RecommendationItem]


class AnomaliesResponse(BaseModel):
    anomalies: list[InsightItem]
