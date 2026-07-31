from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.model_cache import warm_cache
from api.routers import churn, occupancy, pricing, restaurant, staffing


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_cache()
    yield


app = FastAPI(title="HotelMind ML Prediction API", lifespan=lifespan)

app.include_router(occupancy.router, tags=["occupancy"])
app.include_router(pricing.router, tags=["pricing"])
app.include_router(restaurant.router, tags=["restaurant"])
app.include_router(staffing.router, tags=["staffing"])
app.include_router(churn.router, tags=["churn"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
