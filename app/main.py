# app/main.py

"""Main module for the FastAPI application."""

import os
import logging
from typing import Any, Dict
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from retail_bank_risk.feature_engineering_utils import (
    bin_age_into_groups,
    bin_numeric_into_quantiles,
    calculate_ratio,
    clean_column_names,
    encode_categorical_features,
)

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tuned_tuned_xgboost_checkpoint.pkl")

with open(MODEL_PATH, "rb") as f:
    checkpoint = pickle.load(f)

model = checkpoint["model"]
selected_features = checkpoint["selected_features"]

binary_features = [
    "reg_city_not_work_city",
    "name_contract_type",
    "code_gender",
    "flag_own_car",
    "flag_own_realty",
    "is_anomaly",
]

categorical_features = [
    "name_type_suite",
    "name_income_type",
    "name_family_status",
    "name_housing_type",
    "weekday_appr_process_start",
]

ordinal_features = [
    "region_rating_client",
    "region_rating_client_w_city",
    "name_education_type",
    "age_group",
    "income_group",
    "credit_amount_group",
]

ordinal_orders = {
    "region_rating_client": ["3", "2", "1"],
    "region_rating_client_w_city": ["3", "2", "1"],
    "name_education_type": [
        "Lower secondary",
        "Secondary / secondary special",
        "Incomplete higher",
        "Higher education",
        "Academic degree",
    ],
    "age_group": ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    "income_group": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "credit_amount_group": ["Q1", "Q2", "Q3", "Q4", "Q5"],
}


class PredictionInput(BaseModel):
    """Schema for input data for prediction."""

    amt_income_total: float = Field(..., description="Total income amount")
    amt_credit: float = Field(..., description="Credit amount")
    amt_annuity: float = Field(..., description="Annuity amount")
    amt_goods_price: float = Field(..., description="Goods price")
    name_income_type: str = Field(..., description="Income type")
    name_housing_type: str = Field(..., description="Housing type")
    name_family_status: str = Field(..., description="Family status")
    name_education_type: str = Field(..., description="Education type")
    code_gender: str = Field(..., description="Gender code")
    flag_own_car: str = Field(..., description="Car ownership flag")
    flag_own_realty: str = Field(..., description="Realty ownership flag")
    days_birth: int = Field(..., description="Days since birth")
    reg_city_not_work_city: int = Field(..., description="Work city different from registration city")
    name_contract_type: str = Field(..., description="Contract type")
    is_anomaly: int = Field(..., description="Anomaly flag")
    name_type_suite: str = Field(..., description="Type of suite")
    weekday_appr_process_start: str = Field(..., description="Weekday of application process start")
    region_rating_client: int = Field(..., description="Client's region rating")
    region_rating_client_w_city: int = Field(..., description="Client's region rating with city")


def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocesses input data for model prediction.

    Args:
        data: A dictionary containing input features.

    Returns:
        A pandas DataFrame with preprocessed features.
    """
    df = pd.DataFrame([data])
    df["age_group"] = bin_age_into_groups(df, "days_birth")
    df["income_group"] = bin_numeric_into_quantiles(df, "amt_income_total")
    df["credit_amount_group"] = bin_numeric_into_quantiles(df, "amt_credit")
    df["debt_to_income_ratio"] = calculate_ratio(df, "amt_credit", "amt_income_total")
    df["credit_to_goods_ratio"] = calculate_ratio(df, "amt_credit", "amt_goods_price")
    df["annuity_to_income_ratio"] = calculate_ratio(df, "amt_annuity", "amt_income_total")
    df = clean_column_names(df)
    df_encoded, _ = encode_categorical_features(
        df,
        df,
        binary_features,
        categorical_features,
        ordinal_features=ordinal_features,
        ordinal_orders=ordinal_orders,
    )
    for feature in selected_features:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    return df_encoded[selected_features]



@app.get("/")
async def read_root():
    return {"message": "Welcome to the retail bank risk evaluation API!"}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Endpoint for making a prediction.

    Args:
        input_data: An instance of PredictionInput containing input features.

    Returns:
        A JSON response with the probability of default.
    """
    logger.info("Received prediction request")
    try:
        df = preprocess_input(input_data.dict())
        prediction = model.predict_proba(df)[0, 1]
        logger.info(f"Prediction made: {prediction}")
        return {"probability_of_default": float(prediction)}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        A JSON response indicating the service is healthy.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
