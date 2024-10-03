# app/main.py

import os
import logging
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
import pickle
from datetime import datetime

# Import your utility functions
from retail_bank_risk.feature_engineering_utils import (
    bin_age_into_groups,
    bin_numeric_into_quantiles,
    calculate_ratio,
    clean_column_names,
    encode_categorical_features,
)
from retail_bank_risk.data_preprocessing_utils import flag_anomalies

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the base directory and model path
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH: str = os.path.join(BASE_DIR, "models", "tuned_tuned_xgboost_checkpoint.pkl")

# Set up templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))

# Load the trained model
with open(MODEL_PATH, "rb") as f:
    checkpoint: Dict[str, Any] = pickle.load(f)

model: Any = checkpoint["model"]
selected_features: List[str] = checkpoint["selected_features"]

# Define feature categories based on DataFrame
binary_features: List[str] = [
    "code_gender",
    "flag_own_car",
    "flag_own_realty",
]

categorical_features: List[str] = [
    "name_income_type",
    "name_housing_type",
    "name_family_status",
]

ordinal_features: List[str] = [
    "name_education_type",
]

# Define the order for ordinal features
ordinal_orders: Dict[str, List[Any]] = {
    "name_education_type": [
        "Lower secondary",
        "Secondary / secondary special",
        "Incomplete higher",
        "Higher education",
        "Academic degree",
    ],
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

    @field_validator('days_birth')
    @classmethod
    def validate_days_birth(cls, v):
        """Ensure that days_birth is negative to represent age."""
        if v >= 0:
            raise ValueError('days_birth must be negative to represent age in days')
        return v

    class Config:
        """Configuration for the PredictionInput model."""
        arbitrary_types_allowed = True

def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocesses input data for model prediction."""
    logger.info("Starting preprocessing of input data")
    df: pd.DataFrame = pd.DataFrame([data])

    # Convert days_birth to positive age in days
    df["days_birth"] = df["days_birth"].abs()

    # Create binned features
    df["age_group"] = bin_age_into_groups(df, "days_birth")
    df["income_group"] = bin_numeric_into_quantiles(df, "amt_income_total")
    df["credit_amount_group"] = bin_numeric_into_quantiles(df, "amt_credit")

    # Calculate ratios
    df["debt_to_income_ratio"] = calculate_ratio(df, "amt_credit", "amt_income_total")
    df["credit_to_goods_ratio"] = calculate_ratio(df, "amt_credit", "amt_goods_price")
    df["annuity_to_income_ratio"] = calculate_ratio(df, "amt_annuity", "amt_income_total")

    # Flag anomalies
    numerical_features = ["amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price", "days_birth"]
    df["is_anomaly"] = flag_anomalies(df, numerical_features).astype(int)

    # Handle weekday
    current_datetime = datetime.utcnow()
    weekday = current_datetime.strftime("%A").lower()
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        df[f"weekday_appr_process_start_{day}"] = 1 if day == weekday else 0

    # Clean column names
    df = clean_column_names(df)

    # Encode categorical features
    df_encoded, _ = encode_categorical_features(
        df_train=df,
        df_test=df,  # Using the same df for both train and test as we're only encoding one sample
        binary_features=binary_features,
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        ordinal_orders=ordinal_orders,
    )

    # Ensure all required features are present
    for feature in selected_features:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0

    # Reorder columns to match the model's expected input
    df_encoded = df_encoded[selected_features]

    # Convert categorical columns to 'category' dtype
    categorical_cols = ['name_education_type', 'code_gender', 'flag_own_car', 'flag_own_realty']
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype('category')

    logger.info("Preprocessed dataframe shape: %s", df_encoded.shape)
    return df_encoded

@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    """Serve the loan application form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict(input_data: PredictionInput) -> Dict[str, float]:
    """Endpoint for making a prediction."""
    logger.info("Received prediction request")
    try:
        # Convert input_data to dict and preprocess
        processed_data = preprocess_input(input_data.dict())

        # Make prediction
        prediction: float = float(model.predict_proba(processed_data)[0, 1])
        logger.info("Prediction made: %f", prediction)
        return {"probability_of_default": prediction}
    except ValueError as ve:
        logger.error("ValueError during prediction: %s", str(ve))
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
