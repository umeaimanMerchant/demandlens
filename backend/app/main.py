"""
This module contains FastAPI endpoints for generating sales forecasts using a pre-trained LightGBM model.
"""

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .forecasting import forecast_sales

# find model path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "sales_forecasting_artifact.joblib"

# loading the model to server
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained models once when the server starts.
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {ARTIFACT_PATH}")

    app.state.artifact = joblib.load(ARTIFACT_PATH)
    yield

# This creates the FastAPI application object
app = FastAPI(
    title="Sales Forecasting API",
    description="Generate recursive daily sales forecasts using LightGBM.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the defined Next.js frontend to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://demandlens-api.netlify.app",  ## replace with your deployed frontend URL
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Pydantic: defining the request contract. Validation is performed automatically by FastAPI when a request is received.
class ForecastRequest(BaseModel):
    """
    Request model for generating sales forecasts.
    """

    model_config = ConfigDict(extra="forbid") # Rejects unexpected request fields.
    store_nbr: int = Field(ge=1, le=54) # Restrict store_nbr to be between 1 and 54, inclusive.
    family: str = Field(min_length=1) # Restrict family to be a non-empty string.
    start_date: date # The date from which the forecast should start.
    horizon: Annotated[int, Field(strict=True)] # Restrict horizon to be an integer and validate it in the model validator.
    promotion_counts: list[Annotated[int, Field(strict=True, ge=0)]] | None = None # Restrict promotion_counts to be a list of non-negative integers and validate it in the model validator.
    assume_zero_promotions: bool = False # Default to False, indicating that promotion counts are provided unless explicitly set to True.

    # Model validator for additional input validation
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

# Pydantic: defining the response contract
class ForecastDay(BaseModel):
    date: date
    store_nbr: int
    family: str
    onpromotion: int
    predicted_sales: float
    promotion_assumption: str

# The response contains a list of daily forecasts.
class ForecastResponse(BaseModel):
    forecast: list[ForecastDay]


#sample response:{
#   "forecast": [
#     {
#       "date": "2017-08-16",
#       "store_nbr": 1,
#       "family": "GROCERY I",
#       "onpromotion": 0,
#       "predicted_sales": 250.5,
#       "promotion_assumption": "zero_promotions"
#     }
#   ]
# }


# Endpoint one: the root route
@app.get("/")
def root():
    return {"message": "Sales Forecasting API", "docs": "/docs"}


# Endpoint two: health check route
@app.get("/health")
def health(request: Request):
    artifact_loaded = hasattr(request.app.state, "artifact")

    return {
        "status": "ok" if artifact_loaded else "not_ready",
        "model_loaded": artifact_loaded,
    }

# Endpoint three: the main forecasting route
@app.post("/forecast", response_model=ForecastResponse) # This tells FastAPI that the endpoint's response should conform to the ForecastResponse schema.
def create_forecast( 
    # FastAPI recognizes that payload is a Pydantic model. 
    # It parses the incoming JSON, validates the fields, and supplies a ForecastRequest instance to your function.
    payload: ForecastRequest, 
    request: Request,
):
    try:
        # The forecast_sales function is called with the validated input data and the loaded model artifact.
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
    # If a ValueError is raised during the forecast generation, it is caught and an HTTPException with a 422 status code is raised, along with the error message.
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc