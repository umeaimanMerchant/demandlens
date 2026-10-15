
from __future__ import annotations

import numpy as np
import pandas as pd


ALLOWED_HORIZONS = {7, 14, 30}
KEY_COLUMNS = ["date", "store_nbr", "family"]


def _holiday_lookup(holidays: pd.DataFrame) -> dict:
    """Aggregate raw holiday events using the training-time definitions."""
    events = holidays.copy()
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()

    events = events.assign(
        is_national=events["locale"].eq("National"),
        is_regional=events["locale"].eq("Regional"),
        is_local=events["locale"].eq("Local"),
        is_transferred=(
            events["transferred"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        ),
    )

    daily = (
        events.groupby("date", as_index=False)
        .agg(
            holiday_event_count=("type", "size"),
            has_national_event=("is_national", "max"),
            has_regional_event=("is_regional", "max"),
            has_local_event=("is_local", "max"),
            has_transferred_event=("is_transferred", "max"),
        )
    )

    return daily.set_index("date").to_dict(orient="index")


def _lag_features(history_sales: list[float]) -> dict:
    """Calculate features from past actuals and previously generated predictions."""
    result = {}

    for lag in [1, 7, 14, 28, 364]:
        result[f"lag_{lag}"] = (
            history_sales[-lag]
            if len(history_sales) >= lag
            else np.nan
        )

    for window in [7, 28, 56]:
        values = history_sales[-window:]
        result[f"rolling_mean_{window}"] = (
            float(np.mean(values))
            if len(values) == window
            else np.nan
        )

    values_7 = history_sales[-7:]
    result["rolling_std_7"] = (
        float(np.std(values_7, ddof=1))
        if len(values_7) == 7
        else np.nan
    )

    return result


def forecast_sales(
    artifact: dict,
    store_nbr: int,
    family: str,
    start_date,
    horizon: int,
    promotion_counts: list[int] | None = None,
    assume_zero_promotions: bool = False,
) -> pd.DataFrame:
    """
    Generate recursive daily sales forecasts for one store and family.

    Supply one promotion count per forecast day, or explicitly set
    assume_zero_promotions=True.
    """
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError("horizon must be 7, 14, or 30")

    if (promotion_counts is None) == (not assume_zero_promotions):
        raise ValueError(
            "Provide promotion_counts or explicitly set "
            "assume_zero_promotions=True, but not both."
        )

    if assume_zero_promotions:
        promotion_counts = [0] * horizon

    if len(promotion_counts) != horizon:
        raise ValueError("promotion_counts must match the forecast horizon")

    if any(
        not isinstance(x, (int, np.integer))
        or isinstance(x, (bool, np.bool_))
        or x < 0
        for x in promotion_counts
    ):
        raise ValueError("Promotion counts must be nonnegative integers")


    store_nbr = int(store_nbr)
    family = str(family)
    start_date = pd.Timestamp(start_date).normalize()

    # Filter the original history first. Copy only the selected series.
    raw_history = artifact["history"]

    series = raw_history.loc[
        raw_history["store_nbr"].eq(store_nbr)
        & raw_history["family"].eq(family),
        ["date", "sales"],
    ].copy()

    series["date"] = pd.to_datetime(series["date"]).dt.normalize()
    series = series.sort_values("date")


    last_date = pd.Timestamp(artifact["last_train_date"]).normalize()
    expected_start = last_date + pd.Timedelta(days=1)

    if start_date != expected_start:
        raise ValueError(
            f"For this MVP, start_date must be {expected_start.date()}, "
            "the day after the latest observed sales."
        )

    if series.empty:
        raise ValueError("Unknown store or product family combination")

    if series["date"].max() != last_date:
        raise ValueError("Selected series does not have the latest history date")

    if series["date"].duplicated().any():
        raise ValueError("Duplicate dates found in the selected sales history")

    stores = artifact["stores"].copy()
    store_rows = stores.loc[stores["store_nbr"].astype(int) == store_nbr]

    if store_rows.empty:
        raise ValueError(f"Unknown store_nbr: {store_nbr}")

    store_info = store_rows.iloc[0]
    cluster_id = int(store_info["cluster"])

    if cluster_id not in artifact["models"]:
        raise ValueError(f"No model found for cluster {cluster_id}")

    model = artifact["models"][cluster_id]
    feature_cols = artifact["feature_cols"]
    category_levels = artifact["category_levels"]
    holiday_by_date = _holiday_lookup(artifact["holidays_events"])

    # Keep the target history separate from feature construction.
    # Each prediction is appended only after its forecast is generated.
    history_sales = series["sales"].astype(float).tolist()
    predictions = []

    for step in range(horizon):
        current_date = start_date + pd.Timedelta(days=step)

        holiday = holiday_by_date.get(
            current_date,
            {
                "holiday_event_count": 0,
                "has_national_event": False,
                "has_regional_event": False,
                "has_local_event": False,
                "has_transferred_event": False,
            },
        )

        row = {
            "store_nbr": store_nbr,
            "family": family,
            "onpromotion": int(promotion_counts[step]),
            "city": store_info["city"],
            "state": store_info["state"],
            "type": store_info["type"],
            "day_of_week": current_date.dayofweek,
            "day_of_month": current_date.day,
            "month": current_date.month,
            "year": current_date.year,
            "is_weekend": int(current_date.dayofweek >= 5),
            "holiday_event_count": int(holiday["holiday_event_count"]),
            "has_national_event": int(holiday["has_national_event"]),
            "has_regional_event": int(holiday["has_regional_event"]),
            "has_local_event": int(holiday["has_local_event"]),
            "has_transferred_event": int(holiday["has_transferred_event"]),
            **_lag_features(history_sales),
        }

        X = pd.DataFrame([row], columns=feature_cols)

        for col in artifact["categorical_cols"]:
            X[col] = pd.Categorical(
                X[col],
                categories=category_levels[col],
            )

        prediction = float(model.predict(X)[0])

        # Sales volume cannot be negative.
        prediction = max(0.0, prediction)

        predictions.append(
            {
                "date": current_date,
                "store_nbr": store_nbr,
                "family": family,
                "onpromotion": int(promotion_counts[step]),
                "predicted_sales": prediction,
                "promotion_assumption": (
                    "zero_promotions"
                    if assume_zero_promotions
                    else "user_provided"
                ),
            }
        )

        # Recursive step: future lag features can now use this prediction.
        history_sales.append(prediction)

    return pd.DataFrame(predictions)
