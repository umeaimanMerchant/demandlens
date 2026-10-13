from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .forecasting import forecast_sales


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "sales_forecasting_artifact.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained models once when the server starts.
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {ARTIFACT_PATH}")

    app.state.artifact = joblib.load(ARTIFACT_PATH)
    yield


app = FastAPI(
    title="Sales Forecasting API",
    description="Generate recursive daily sales forecasts using LightGBM.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the local Next.js frontend to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store_nbr: int = Field(ge=1, le=54)
    family: str = Field(min_length=1)
    start_date: date
    horizon: Annotated[int, Field(strict=True)]
    promotion_counts: list[Annotated[int, Field(strict=True, ge=0)]] | None = None
    assume_zero_promotions: bool = False

    @model_validator(mode="after")
    def validate_forecast_inputs(self):
        if self.horizon not in {7, 14, 30}:
            raise ValueError("horizon must be 7, 14, or 30")

        if (self.promotion_counts is None) == (
            not self.assume_zero_promotions
        ):
            raise ValueError(
                "Provide promotion_counts or set "
                "assume_zero_promotions=true, but not both."
            )

        if (
            self.promotion_counts is not None
            and len(self.promotion_counts) != self.horizon
        ):
            raise ValueError(
                "promotion_counts must contain one value per forecast day."
            )

        return self


class ForecastDay(BaseModel):
    date: date
    store_nbr: int
    family: str
    onpromotion: int
    predicted_sales: float
    promotion_assumption: str


class ForecastResponse(BaseModel):
    forecast: list[ForecastDay]


@app.get("/")
def root():
    return {"message": "Sales Forecasting API", "docs": "/docs"}


@app.get("/health")
def health(request: Request):
    artifact_loaded = hasattr(request.app.state, "artifact")

    return {
        "status": "ok" if artifact_loaded else "not_ready",
        "model_loaded": artifact_loaded,
    }


@app.post("/forecast", response_model=ForecastResponse)
def create_forecast(
    payload: ForecastRequest,
    request: Request,
):
    try:
        result = forecast_sales(
            artifact=request.app.state.artifact,
            store_nbr=payload.store_nbr,
            family=payload.family,
            start_date=payload.start_date.isoformat(),
            horizon=payload.horizon,
            promotion_counts=payload.promotion_counts,
            assume_zero_promotions=payload.assume_zero_promotions,
        )

        records = result.to_dict(orient="records")

        # Convert pandas/numpy values into ordinary Python values.
        forecast_days = [
            {
                "date": row["date"].date(),
                "store_nbr": int(row["store_nbr"]),
                "family": str(row["family"]),
                "onpromotion": int(row["onpromotion"]),
                "predicted_sales": float(row["predicted_sales"]),
                "promotion_assumption": str(row["promotion_assumption"]),
            }
            for row in records
        ]

        return {"forecast": forecast_days}

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc