from __future__ import annotations

import io
import json
import math
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from xgboost import XGBRegressor

MAX_UPLOAD_BYTES = 35 * 1024 * 1024
FORECAST_MONTHS = 3
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "backend" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RSO Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUN_STORE: dict[str, dict[str, Any]] = {}


class ChatRequest(BaseModel):
    run_id: str
    message: str


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return normalized.strip("_")


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    return value


def load_dataframe(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    if suffix == ".csv":
        return pd.read_csv(buffer)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer)

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")


def detect_schema(df: pd.DataFrame) -> dict[str, Any]:
    columns = list(df.columns)
    col_set = set(columns)

    def first_match(options: list[str]) -> str | None:
        for option in options:
            if option in col_set:
                return option
        return None

    date_column = first_match(
        [
            "date",
            "sale_date",
            "transaction_date",
            "invoice_date",
            "order_date",
        ]
    )
    year_column = "year" if "year" in col_set else None
    month_column = "month" if "month" in col_set else None
    time_column = "sale_time" if "sale_time" in col_set else None

    target_column = first_match(
        [
            "retail_sales",
            "total_sale",
            "total_amount",
            "sales",
            "sales_amount",
            "quantity",
            "quantiy",
            "warehouse_sales",
        ]
    )

    item_column = first_match(
        [
            "item_description",
            "product_category",
            "category",
            "item_code",
            "product_id",
            "product",
            "item",
            "supplier",
        ]
    )

    item_group_column = first_match(
        [
            "item_type",
            "product_category",
            "category",
            "supplier",
        ]
    )

    sales_columns = [column for column in columns if any(token in column for token in ["sales", "sale", "amount", "quantity"])]

    return {
        "date_column": date_column,
        "year_column": year_column,
        "month_column": month_column,
        "time_column": time_column,
        "target_column": target_column,
        "item_column": item_column,
        "item_group_column": item_group_column,
        "sales_columns": sales_columns,
        "can_auto_detect": bool(target_column and (date_column or (year_column and month_column))),
    }


def build_datetime_series(df: pd.DataFrame, schema: dict[str, Any]) -> pd.Series:
    if schema["date_column"]:
        date_series = df[schema["date_column"]].astype(str)
        if schema["time_column"]:
            time_series = df[schema["time_column"]].astype(str)
            return pd.to_datetime(date_series + " " + time_series, errors="coerce")
        return pd.to_datetime(date_series, errors="coerce")

    if schema["year_column"] and schema["month_column"]:
        year = pd.to_numeric(df[schema["year_column"]], errors="coerce")
        month = pd.to_numeric(df[schema["month_column"]], errors="coerce")
        date_strings = year.fillna(0).astype(int).astype(str) + "-" + month.fillna(1).astype(int).astype(str) + "-01"
        return pd.to_datetime(date_strings, errors="coerce")

    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def clean_dataframe(raw_df: pd.DataFrame, schema: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = raw_df.copy()
    original_columns = list(df.columns)
    df.columns = [normalize_column_name(column) for column in df.columns]

    before_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        df = df.drop_duplicates().reset_index(drop=True)

    trimmed_columns: list[str] = []
    for column in df.columns:
        if df[column].dtype == object:
            original = df[column]
            cleaned = original.astype(str).str.strip()
            if not cleaned.equals(original.astype(str)):
                trimmed_columns.append(column)
            df[column] = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    numeric_columns = list({schema.get("target_column"), *schema.get("sales_columns", []), "price_per_unit", "cogs", "age"})
    numeric_columns = [column for column in numeric_columns if column and column in df.columns]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    detected_date = build_datetime_series(df, schema)
    df["analysis_date"] = detected_date
    invalid_dates_removed = int(df["analysis_date"].isna().sum())
    if invalid_dates_removed:
        df = df.dropna(subset=["analysis_date"]).reset_index(drop=True)

    missing_before = df.isna().sum()
    filled_numeric: dict[str, float] = {}
    filled_categorical: dict[str, str] = {}

    for column in df.columns:
        if column == "analysis_date":
            continue
        if df[column].dtype.kind in {"i", "u", "f"}:
            if df[column].isna().any():
                fill_value = float(df[column].median()) if not df[column].dropna().empty else 0.0
                df[column] = df[column].fillna(fill_value)
                filled_numeric[column] = round(fill_value, 4)
        else:
            if df[column].isna().any():
                mode = df[column].mode(dropna=True)
                fill_value = str(mode.iloc[0]) if not mode.empty else "Unknown"
                df[column] = df[column].fillna(fill_value)
                filled_categorical[column] = fill_value

    missing_after = df.isna().sum()

    cleaning_summary = {
        "original_rows": before_rows,
        "rows_after_cleaning": int(len(df)),
        "duplicates_removed": duplicate_rows,
        "invalid_dates_removed": invalid_dates_removed,
        "trimmed_text_columns": trimmed_columns,
        "missing_values_before": {column: int(value) for column, value in missing_before.items() if int(value) > 0},
        "missing_values_after": {column: int(value) for column, value in missing_after.items() if int(value) > 0},
        "filled_numeric_columns": filled_numeric,
        "filled_categorical_columns": filled_categorical,
        "cleaning_applied": bool(
            duplicate_rows
            or invalid_dates_removed
            or filled_numeric
            or filled_categorical
            or trimmed_columns
        ),
        "original_columns": original_columns,
        "normalized_columns": list(df.columns),
    }

    return df, cleaning_summary


def build_monthly_series(df: pd.DataFrame, schema: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    if not schema["target_column"] or schema["target_column"] not in df.columns:
        raise HTTPException(status_code=400, detail="Could not identify a target sales column in the dataset.")

    target_column = schema["target_column"]
    working = df.copy()
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce").fillna(0.0)
    monthly = (
        working.assign(month_bucket=working["analysis_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month_bucket", as_index=False)[target_column]
        .sum()
        .rename(columns={"month_bucket": "date", target_column: "target"})
        .sort_values("date")
    )

    full_range = pd.date_range(monthly["date"].min(), monthly["date"].max(), freq="MS")
    monthly = monthly.set_index("date").reindex(full_range, fill_value=0.0).rename_axis("date").reset_index()
    return monthly, target_column


def create_feature_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    frame = monthly.copy()
    frame["time_index"] = np.arange(len(frame))
    frame["month"] = frame["date"].dt.month
    frame["quarter"] = frame["date"].dt.quarter

    for lag in [1, 2, 3, 6]:
        frame[f"lag_{lag}"] = frame["target"].shift(lag)

    frame["rolling_mean_3"] = frame["target"].shift(1).rolling(3).mean()
    frame["rolling_mean_6"] = frame["target"].shift(1).rolling(6).mean()
    frame["rolling_std_3"] = frame["target"].shift(1).rolling(3).std().fillna(0)
    return frame.dropna().reset_index(drop=True)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if len(actual) == 0:
        return {"mae": None, "rmse": None, "mape": None, "r2": None}

    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    non_zero_actual = np.where(actual == 0, 1, actual)
    mape = float(np.mean(np.abs((actual - predicted) / non_zero_actual)) * 100)

    r2 = None
    if len(actual) > 1:
        try:
            r2 = r2_score(actual, predicted)
        except Exception:
            r2 = None

    return sanitize_for_json({"mae": mae, "rmse": rmse, "mape": mape, "r2": r2})


def recursive_forecast(model: Any, history: list[float], future_dates: pd.DatetimeIndex, feature_template: pd.DataFrame) -> list[dict[str, Any]]:
    values = history.copy()
    forecasts: list[dict[str, Any]] = []
    next_index = int(feature_template["time_index"].max()) + 1

    for future_date in future_dates:
        feature_row = {
            "time_index": next_index,
            "month": int(future_date.month),
            "quarter": int(((future_date.month - 1) // 3) + 1),
            "lag_1": values[-1],
            "lag_2": values[-2],
            "lag_3": values[-3],
            "lag_6": values[-6],
            "rolling_mean_3": float(np.mean(values[-3:])),
            "rolling_mean_6": float(np.mean(values[-6:])),
            "rolling_std_3": float(np.std(values[-3:])),
        }
        feature_frame = pd.DataFrame([feature_row])
        prediction = float(model.predict(feature_frame)[0])
        values.append(prediction)
        forecasts.append({"date": future_date.strftime("%Y-%m-%d"), "value": round(prediction, 2)})
        next_index += 1

    return forecasts


def run_tree_models(monthly: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    feature_frame = create_feature_frame(monthly)
    if len(feature_frame) < 8:
        raise HTTPException(status_code=400, detail="Not enough time history to train forecasting models reliably.")

    holdout = 3 if len(feature_frame) < 15 else min(6, max(3, len(feature_frame) // 5))
    train = feature_frame.iloc[:-holdout]
    test = feature_frame.iloc[-holdout:]

    feature_columns = [
        "time_index",
        "month",
        "quarter",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_std_3",
    ]

    x_train = train[feature_columns]
    y_train = train["target"]
    x_test = test[feature_columns]
    y_test = test["target"].to_numpy()

    future_dates = pd.date_range(monthly["date"].max() + pd.offsets.MonthBegin(1), periods=FORECAST_MONTHS, freq="MS")
    history = monthly["target"].tolist()

    models: list[tuple[str, Any]] = [
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=400,
                random_state=42,
                max_depth=8,
                min_samples_leaf=2,
            ),
        ),
        (
            "XGBoost",
            XGBRegressor(
                n_estimators=350,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            ),
        ),
    ]

    model_results: list[dict[str, Any]] = []
    forecasts: dict[str, list[dict[str, Any]]] = {}

    for name, model in models:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = compute_metrics(y_test, predictions)
        forecast = recursive_forecast(model, history, future_dates, feature_frame)
        forecasts[name] = forecast
        model_results.append(
            {
                "model": name,
                **metrics,
                "status": "trained",
            }
        )

    return model_results, forecasts


def run_lstm_model(monthly: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = monthly["target"].astype(float).to_numpy()
    if len(values) < 16:
        return (
            {"model": "LSTM", "status": "skipped", "reason": "Not enough history for stable LSTM training."},
            [],
        )

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    window = 6

    x_sequences: list[np.ndarray] = []
    y_sequences: list[float] = []
    for index in range(window, len(scaled)):
        x_sequences.append(scaled[index - window : index])
        y_sequences.append(scaled[index])

    x = np.array(x_sequences).reshape(-1, window, 1)
    y = np.array(y_sequences)

    holdout = 3 if len(x) < 18 else min(6, max(3, len(x) // 5))
    x_train, x_test = x[:-holdout], x[-holdout:]
    y_train, y_test = y[:-holdout], y[-holdout:]

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window, 1)),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=80,
        batch_size=8,
        verbose=0,
        callbacks=callbacks,
    )

    predicted_scaled = model.predict(x_test, verbose=0).flatten()
    predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    metrics = compute_metrics(actual, predicted)

    rolling_window = scaled[-window:].tolist()
    future_dates = pd.date_range(monthly["date"].max() + pd.offsets.MonthBegin(1), periods=FORECAST_MONTHS, freq="MS")
    forecast: list[dict[str, Any]] = []
    for future_date in future_dates:
        input_array = np.array(rolling_window[-window:]).reshape(1, window, 1)
        next_scaled = float(model.predict(input_array, verbose=0).flatten()[0])
        rolling_window.append(next_scaled)
        next_value = float(scaler.inverse_transform([[next_scaled]])[0][0])
        forecast.append({"date": future_date.strftime("%Y-%m-%d"), "value": round(next_value, 2)})

    return {"model": "LSTM", **metrics, "status": "trained"}, forecast


def choose_best_model(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    trained = [result for result in model_results if result.get("status") == "trained" and result.get("rmse") is not None]
    if not trained:
        return {"model": "Unavailable", "reason": "No models were trained successfully."}
    return min(trained, key=lambda result: float(result["rmse"]))


def build_recommendations(df: pd.DataFrame, schema: dict[str, Any], target_column: str) -> list[dict[str, Any]]:
    item_column = schema.get("item_column")
    group_column = schema.get("item_group_column")
    if not item_column or item_column not in df.columns:
        return []

    working = df.copy()
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce").fillna(0.0)
    working["month_bucket"] = working["analysis_date"].dt.to_period("M").dt.to_timestamp()
    latest_month = working["month_bucket"].max()
    recent_months = pd.date_range(latest_month - pd.offsets.MonthBegin(2), latest_month, freq="MS")
    prior_months = pd.date_range(latest_month - pd.offsets.MonthBegin(5), latest_month - pd.offsets.MonthBegin(3), freq="MS")

    recent = (
        working[working["month_bucket"].isin(recent_months)]
        .groupby(item_column, as_index=False)[target_column]
        .sum()
        .rename(columns={target_column: "recent_sales"})
    )
    prior = (
        working[working["month_bucket"].isin(prior_months)]
        .groupby(item_column, as_index=False)[target_column]
        .sum()
        .rename(columns={target_column: "prior_sales"})
    )

    recommendation_frame = recent.merge(prior, on=item_column, how="left").fillna({"prior_sales": 0.0})

    if "warehouse_sales" in df.columns:
        warehouse_recent = (
            working[working["month_bucket"].isin(recent_months)]
            .groupby(item_column, as_index=False)["warehouse_sales"]
            .sum()
            .rename(columns={"warehouse_sales": "recent_warehouse_sales"})
        )
        recommendation_frame = recommendation_frame.merge(warehouse_recent, on=item_column, how="left").fillna({"recent_warehouse_sales": 0.0})
    else:
        recommendation_frame["recent_warehouse_sales"] = 0.0

    if group_column and group_column in df.columns:
        group_lookup = (
            working[[item_column, group_column]]
            .drop_duplicates()
            .groupby(item_column, as_index=False)
            .first()
        )
        recommendation_frame = recommendation_frame.merge(group_lookup, on=item_column, how="left")

    def classify(row: pd.Series) -> tuple[str, str, str]:
        prior_sales = float(row["prior_sales"])
        recent_sales = float(row["recent_sales"])
        warehouse_sales = float(row["recent_warehouse_sales"])
        growth = 1.0 if prior_sales <= 0 and recent_sales > 0 else ((recent_sales - prior_sales) / prior_sales if prior_sales > 0 else 0.0)

        if recent_sales > 0 and warehouse_sales > 0 and recent_sales > warehouse_sales:
            return "high", "Increase stock immediately", "Retail demand is running ahead of warehouse support."
        if growth >= 0.25:
            return "medium", "Pre-order extra stock", "Recent demand is climbing quickly over the previous period."
        return "low", "Maintain current stock plan", "Demand looks stable relative to recent months."

    recommendations: list[dict[str, Any]] = []
    recommendation_frame = recommendation_frame.sort_values(["recent_sales"], ascending=False).head(5)
    for _, row in recommendation_frame.iterrows():
        risk, action, reason = classify(row)
        prior_sales = float(row["prior_sales"])
        recent_sales = float(row["recent_sales"])
        growth_pct = 100.0 if prior_sales <= 0 and recent_sales > 0 else (((recent_sales - prior_sales) / prior_sales) * 100 if prior_sales > 0 else 0.0)

        item_name = str(row[item_column])
        group_value = str(row[group_column]) if group_column and group_column in row and not pd.isna(row[group_column]) else "General"
        recommendations.append(
            {
                "item": item_name,
                "group": group_value,
                "recent_sales": round(recent_sales, 2),
                "prior_sales": round(prior_sales, 2),
                "growth_pct": round(growth_pct, 2),
                "warehouse_support": round(float(row["recent_warehouse_sales"]), 2),
                "risk": risk,
                "action": action,
                "reason": reason,
            }
        )

    return recommendations


def build_overview(df: pd.DataFrame, schema: dict[str, Any], target_column: str) -> dict[str, Any]:
    item_column = schema.get("item_column")
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "date_range": {
            "start": df["analysis_date"].min().strftime("%Y-%m-%d"),
            "end": df["analysis_date"].max().strftime("%Y-%m-%d"),
        },
        "target_column": target_column,
        "target_sum": round(float(df[target_column].sum()), 2),
        "target_mean": round(float(df[target_column].mean()), 2),
        "item_count": int(df[item_column].nunique()) if item_column and item_column in df.columns else None,
        "columns": list(df.columns),
    }


def preview_rows(df: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    preview = df.head(limit).copy()
    for column in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[column]):
            preview[column] = preview[column].dt.strftime("%Y-%m-%d")
    return sanitize_for_json(preview.to_dict(orient="records"))


def process_dataset(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    raw_df = load_dataframe(file_name, file_bytes)
    raw_df = raw_df.dropna(axis=1, how="all")
    if raw_df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file did not contain usable tabular data.")

    normalized_preview = raw_df.copy()
    normalized_preview.columns = [normalize_column_name(column) for column in raw_df.columns]
    schema = detect_schema(normalized_preview)
    if not schema["can_auto_detect"]:
        raise HTTPException(
            status_code=400,
            detail="The dataset structure could not be auto-detected well enough. Please use a retail sales file with date and sales columns.",
        )

    cleaned_df, cleaning_summary = clean_dataframe(raw_df, schema)
    monthly, target_column = build_monthly_series(cleaned_df, schema)
    overview = build_overview(cleaned_df, schema, target_column)

    model_results, tree_forecasts = run_tree_models(monthly)
    lstm_result, lstm_forecast = run_lstm_model(monthly)
    model_results.append(lstm_result)
    if lstm_forecast:
        tree_forecasts["LSTM"] = lstm_forecast

    best_model = choose_best_model(model_results)
    recommendations = build_recommendations(cleaned_df, schema, target_column)

    forecast_summary = tree_forecasts.get(best_model.get("model"), next(iter(tree_forecasts.values()), []))

    run_id = uuid.uuid4().hex
    result = sanitize_for_json(
        {
            "run_id": run_id,
            "file": {
                "name": file_name,
                "size_bytes": len(file_bytes),
            },
            "schema": schema,
            "cleaning_summary": cleaning_summary,
            "overview": overview,
            "monthly_series": monthly.assign(date=monthly["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
            "preview_rows": preview_rows(cleaned_df),
            "model_results": model_results,
            "best_model": best_model,
            "forecasts": tree_forecasts,
            "forecast_summary": forecast_summary,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    )

    RUN_STORE[run_id] = result
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def answer_chat_question(run: dict[str, Any], message: str) -> str:
    question = message.lower().strip()
    cleaning = run["cleaning_summary"]
    overview = run["overview"]
    best_model = run["best_model"]
    recommendations = run.get("recommendations", [])
    forecast = run.get("forecast_summary", [])

    if any(keyword in question for keyword in ["duplicate", "missing", "clean", "format"]):
        return (
            f"The upload pipeline {'did' if cleaning['cleaning_applied'] else 'did not need to'} apply cleaning. "
            f"It removed {cleaning['duplicates_removed']} duplicate rows and {cleaning['invalid_dates_removed']} rows with invalid dates. "
            f"Missing numeric fills were applied to {len(cleaning['filled_numeric_columns'])} columns and categorical fills to {len(cleaning['filled_categorical_columns'])} columns."
        )

    if any(keyword in question for keyword in ["best model", "accuracy", "mae", "rmse", "xgboost", "random forest", "lstm", "model"]):
        model_lines = []
        for result in run["model_results"]:
            if result.get("status") == "trained":
                model_lines.append(
                    f"{result['model']}: MAE {result['mae']}, RMSE {result['rmse']}, R2 {result['r2']}"
                )
            else:
                model_lines.append(f"{result['model']}: {result.get('reason', 'not trained')}")
        return f"The best model is {best_model.get('model', 'Unavailable')}. " + " | ".join(model_lines)

    if any(keyword in question for keyword in ["forecast", "next", "future", "predict"]):
        if not forecast:
            return "A future forecast is not available for this run yet."
        values = ", ".join(f"{point['date']}: {point['value']}" for point in forecast)
        return f"The current 3-month forecast based on {best_model.get('model', 'the best available model')} is {values}."

    if any(keyword in question for keyword in ["stock", "restock", "recommend", "inventory"]):
        if not recommendations:
            return "This dataset did not provide enough item-level detail to generate restocking recommendations."
        top = recommendations[0]
        return (
            f"The top restocking priority is {top['item']} in {top['group']}. "
            f"Recent sales were {top['recent_sales']} versus {top['prior_sales']} before, giving {top['growth_pct']}% growth. "
            f"Recommended action: {top['action']}."
        )

    if any(keyword in question for keyword in ["chart", "trend", "summary", "dataset", "column", "data"]):
        return (
            f"The dataset covers {overview['date_range']['start']} to {overview['date_range']['end']} with "
            f"{overview['row_count']} cleaned rows across {overview['column_count']} columns. "
            f"The detected target column is {overview['target_column']} with total observed value {overview['target_sum']}."
        )

    if recommendations:
        top = recommendations[0]
        return (
            f"This run used {overview['row_count']} rows and found {best_model.get('model', 'the best available model')} as the strongest model. "
            f"The top recommendation is {top['item']} with action '{top['action']}'. Ask me about cleaning, forecasts, models, or recommendations for more detail."
        )

    return (
        f"This run analyzed {overview['row_count']} rows and selected {best_model.get('model', 'the best available model')} as the strongest model. "
        "Ask me about cleaning, forecasts, models, charts, or recommendations."
    )


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum supported size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    return process_dataset(file.filename, contents)


@app.post("/api/chat")
def chat_with_run(payload: ChatRequest) -> dict[str, Any]:
    run = RUN_STORE.get(payload.run_id)
    if not run:
        run_path = RUNS_DIR / f"{payload.run_id}.json"
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Analysis run not found.")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        RUN_STORE[payload.run_id] = run

    answer = answer_chat_question(run, payload.message)
    return {"answer": answer, "run_id": payload.run_id}
