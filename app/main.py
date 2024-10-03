# main.py
import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError, validator
import pickle
from starlette.staticfiles import StaticFiles

from retail_bank_risk.feature_engineering_utils import (
    engineer_features
)

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR: str = os.path.join(BASE_DIR, "models")
MODEL_PATH: str = os.path.join(MODEL_DIR, "tuned_tuned_xgboost_checkpoint.pkl")

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    logger.warning(f"Created missing models directory at {MODEL_DIR}")

TEMPLATES_DIR = os.path.join(BASE_DIR, "app", "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

STATIC_DIR = os.path.join(BASE_DIR, "app", "static")
if not os.path.isdir(STATIC_DIR):
    os.makedirs(STATIC_DIR)
    logger.warning(f"Created missing static directory at {STATIC_DIR}")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if not os.path.isfile(MODEL_PATH):
    logger.error(f"Model file not found at {MODEL_PATH}")
    model = None
    selected_features = []
else:
    with open(MODEL_PATH, "rb") as f:
        checkpoint: Dict[str, Any] = pickle.load(f)

    model: Any = checkpoint.get("model")
    selected_features: List[str] = checkpoint.get("selected_features", [])

    if not model or not selected_features:
        logger.error("Model or selected features not found in the checkpoint.")
        model = None
        selected_features = []
    else:
        logger.info(f"Model loaded successfully. Selected features: {selected_features}")

class IncomeType(str, Enum):
    working = "Working"
    state_servant = "State servant"
    commercial_associate = "Commercial associate"
    pensioner = "Pensioner"
    unemployed = "Unemployed"
    student = "Student"
    businessman = "Businessman"
    maternity_leave = "Maternity leave"

class HousingType(str, Enum):
    house_apartment = "House / apartment"
    rented_apartment = "Rented apartment"
    with_parents = "With parents"
    municipal_apartment = "Municipal apartment"
    office_apartment = "Office apartment"
    co_op_apartment = "Co-op apartment"

class FamilyStatus(str, Enum):
    single_not_married = "Single / not married"
    married = "Married"
    civil_marriage = "Civil marriage"
    widow = "Widow"
    separated = "Separated"
    unknown = "Unknown"

class EducationType(str, Enum):
    lower_secondary = "Lower secondary"
    secondary = "Secondary / secondary special"
    incomplete_higher = "Incomplete higher"
    higher_education = "Higher education"
    academic_degree = "Academic degree"

class PredictionInput(BaseModel):
    """Schema for input data for prediction."""

    amt_income_total: float = Field(..., description="Total income amount", ge=0.0)
    amt_credit: float = Field(..., description="Credit amount", gt=0.0)
    amt_goods_price: float = Field(..., description="Goods price", ge=0.0)
    name_income_type: IncomeType = Field(..., description="Income type")
    name_housing_type: HousingType = Field(..., description="Housing type")
    name_family_status: FamilyStatus = Field(..., description="Family status")
    name_education_type: EducationType = Field(..., description="Education type")
    code_gender: str = Field(..., description="Gender code", pattern="^(M|F)$")
    flag_own_car: str = Field(..., description="Car ownership flag", pattern="^(Y|N)$")
    flag_own_realty: str = Field(..., description="Realty ownership flag", pattern="^(Y|N)$")
    days_birth: Optional[int] = Field(None, description="Days before current date of birth")
    age: Optional[int] = Field(None, description="Age in years", ge=18, le=100)
    buy_goods: bool = Field(..., description="Whether goods are being purchased")
    loan_term_years: int = Field(..., description="Loan term in years", ge=1, le=30)

    @validator('amt_goods_price')
    def validate_amt_goods_price(cls, v):
        if v < 0:
            raise ValueError('amt_goods_price must be non-negative')
        return v

    @validator('days_birth')
    def validate_days_birth(cls, v, values):
        if v is not None and v > 0:
            raise ValueError('days_birth must be negative')
        if v is None and 'age' not in values:
            raise ValueError('Either days_birth or age must be provided')
        return v

    @validator('age')
    def validate_age(cls, v, values):
        if v is None and 'days_birth' not in values:
            raise ValueError('Either days_birth or age must be provided')
        return v

    class Config:
        use_enum_values = True

class AnomalyRecord(BaseModel):
    """Schema for a single anomaly record."""
    feature: str
    value: float
    lower_bound: float
    upper_bound: float

class PredictionResponse(BaseModel):
    """Schema for the prediction response."""
    probability_of_default: float
    monthly_payment: float
    risk_level: Optional[str] = None
    loan_approval: str
    anomalies_detected: bool
    anomalies_details: Optional[List[AnomalyRecord]] = []
    rejection_reason: Optional[str] = None

def preprocess_input(data: Dict[str, Any]) -> Tuple[pd.DataFrame, bool, List[Dict[str, Any]]]:
    """
    Preprocesses input data for model prediction, including anomaly detection.

    Parameters:
    - data: Dictionary containing input features.

    Returns:
    - Tuple containing the processed DataFrame, anomaly detection flag, and list of anomalies.
    """
    logger.info("Starting preprocessing of input data")
    logger.info(f"Input data: {data}")

    df = pd.DataFrame([data])

    if 'age' in df.columns and pd.notnull(df['age']).all():
        df['days_birth'] = -df['age'] * 365
        df = df.drop(columns=['age'])
        logger.info(f"Converted age to days_birth: {df['days_birth'].iloc[0]} days")
    elif 'days_birth' in df.columns:
        if pd.isnull(df['days_birth']).all():
            if 'age' in df.columns and pd.notnull(df['age']).all():
                df['days_birth'] = -df['age'] * 365
                df = df.drop(columns=['age'])
                logger.info(f"Converted age to days_birth: {df['days_birth'].iloc[0]} days")
            else:
                logger.error("Either days_birth or age must be provided and valid.")
                raise ValueError("Either days_birth or age must be provided and valid.")

    if 'amt_annuity' not in df.columns or pd.isnull(df['amt_annuity']).all():
        loan_amount = df['amt_credit'].iloc[0]
        loan_term_years = df['loan_term_years'].iloc[0]
        interest_rate = 0.05
        loan_term_months = loan_term_years * 12
        monthly_interest_rate = interest_rate / 12
        if monthly_interest_rate == 0:
            amt_annuity = loan_amount / loan_term_months
        else:
            amt_annuity = (loan_amount * monthly_interest_rate * (1 + monthly_interest_rate) ** loan_term_months) / (
                (1 + monthly_interest_rate) ** loan_term_months - 1
            )
        df['amt_annuity'] = amt_annuity
        logger.info(f"Calculated amt_annuity: {amt_annuity}")

    df = engineer_features(df)
    logger.info("Completed feature engineering.")

    numerical_features = ['amt_income_total', 'amt_credit', 'amt_goods_price', 'amt_annuity']
    anomaly_bounds = {
        'amt_income_total': {'lower': 0, 'upper': 1000000},
        'amt_credit': {'lower': 1000, 'upper': 1000000},
        'amt_goods_price': {'lower': 0, 'upper': 500000},
        'amt_annuity': {'lower': 100, 'upper': 100000},
    }

    anomalies_detected = False
    anomalies_details = []

    for feature in numerical_features:
        value = df.at[0, feature]
        lower = anomaly_bounds[feature]['lower']
        upper = anomaly_bounds[feature]['upper']
        if value < lower or value > upper:
            anomalies_detected = True
            anomaly_record = {
                'feature': feature,
                'value': value,
                'lower_bound': lower,
                'upper_bound': upper
            }
            anomalies_details.append(anomaly_record)
            logger.warning(f"Anomaly detected in feature '{feature}': {value} not in [{lower}, {upper}]")

    if not anomalies_detected:
        logger.info("No anomalies detected.")

    binary_features_mapping = {
        'code_gender': {'M': 1, 'F': 0},
        'flag_own_car': {'Y': 1, 'N': 0},
        'flag_own_realty': {'Y': 1, 'N': 0}
    }
    for feature, mapping in binary_features_mapping.items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping).fillna(0).astype(int)
            logger.info(f"Encoded binary feature '{feature}' with mapping {mapping}")

    categorical_features = ['name_income_type', 'name_housing_type', 'name_family_status', 'name_education_type']
    categories_mapping = {
        'name_income_type': ["Working", "State servant", "Commercial associate", "Pensioner", "Unemployed", "Student", "Businessman", "Maternity leave"],
        'name_housing_type': ["House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment", "Co-op apartment"],
        'name_family_status': ["Single / not married", "Married", "Civil marriage", "Widow", "Separated", "Unknown"],
        'name_education_type': ["Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education", "Academic degree"]
    }

    for feature in categorical_features:
        if feature in df.columns:
            dummies = pd.get_dummies(df[feature], prefix=feature, dummy_na=False).astype(int)
            dummies.columns = [col.lower().replace(' ', '_').replace('/', '_').replace('-', '_') for col in dummies.columns]
            expected_cols = [f"{feature}_{cat.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}" for cat in categories_mapping[feature]]
            for col in expected_cols:
                if col not in dummies.columns:
                    dummies[col] = 0
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[feature])
            logger.info(f"One-hot encoded feature '{feature}'. Added columns: {expected_cols}")

    current_weekday = datetime.now().strftime('%A').lower()
    weekday_features = [
        'weekday_appr_process_start_monday',
        'weekday_appr_process_start_tuesday',
        'weekday_appr_process_start_wednesday',
        'weekday_appr_process_start_thursday',
        'weekday_appr_process_start_friday',
        'weekday_appr_process_start_saturday',
        'weekday_appr_process_start_sunday'
    ]

    for day_feature in weekday_features:
        day = day_feature.split('_')[-1]
        df[day_feature] = 1 if day == current_weekday else 0
        logger.info(f"Set '{day_feature}' to {df[day_feature].iloc[0]}")

    df['is_anomaly'] = 1 if anomalies_detected else 0
    logger.info(f"Set 'is_anomaly' to {df['is_anomaly'].iloc[0]}")

    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0
            logger.info(f"Added missing feature: {feature}")

    df = df[selected_features]
    logger.info("Reordered DataFrame to match selected features.")

    df.replace([np.inf, -np.inf], 1e6, inplace=True)
    logger.info("Replaced infinite values with 1e6.")

    logger.info(f"Preprocessed dataframe shape: {df.shape}")
    logger.info(f"Preprocessed data columns: {df.columns.tolist()}")
    logger.info(f"Sample of preprocessed data: {df.iloc[0].to_dict()}")

    return df, anomalies_detected, anomalies_details

def calculate_monthly_payment(amt_credit: float, loan_term_years: int, interest_rate: float = 0.05) -> float:
    """
    Calculate the monthly payment for a loan.

    Parameters:
    - amt_credit: The loan amount
    - loan_term_years: The term of the loan in years
    - interest_rate: Annual interest rate (default 5%)

    Returns:
    - Monthly payment amount
    """
    loan_term_months = loan_term_years * 12
    monthly_interest_rate = interest_rate / 12
    if monthly_interest_rate == 0:
        return amt_credit / loan_term_months
    else:
        return (amt_credit * monthly_interest_rate * (1 + monthly_interest_rate) ** loan_term_months) / (
            (1 + monthly_interest_rate) ** loan_term_months - 1
        )

def adjust_default_probability(original_prob: float, debt_to_income_ratio: float) -> float:
    """
    Adjust the probability of default based on the debt-to-income ratio.

    Parameters:
    - original_prob: The original predicted probability of default
    - debt_to_income_ratio: The debt-to-income ratio

    Returns:
    - Adjusted probability of default
    """
    adjustment_factor = 0.10
    adjusted_prob = original_prob + (adjustment_factor * debt_to_income_ratio)
    return min(adjusted_prob, 1.0)

def loan_to_income_check(monthly_payment: float, amt_income_total: float, max_ratio: float = 0.4) -> bool:
    """
    Check if the monthly payment exceeds the maximum allowable ratio of income.

    Parameters:
    - monthly_payment: The calculated monthly payment
    - amt_income_total: The total income of the applicant
    - max_ratio: The maximum allowable ratio (default 0.4 for 40%)

    Returns:
    - True if within the ratio, False otherwise
    """
    if amt_income_total == 0:
        return False
    ratio = monthly_payment / amt_income_total
    logger.info(f"Monthly payment to income ratio: {ratio:.2f}")
    return ratio <= max_ratio

@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    """
    Serve the loan application form.

    Parameters:
    - request: FastAPI request object

    Returns:
    - Rendered HTML template for the loan application form
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=JSONResponse, response_model=PredictionResponse)
async def predict(request: Request) -> PredictionResponse:
    """
    Endpoint for making a loan risk prediction.

    Parameters:
    - request: FastAPI request object containing loan application data

    Returns:
    - JSON response with prediction details including probability of default, monthly payment,
      risk level, loan approval status, anomaly detection information, and rejection reason if applicable
    """
    logger.info("Received prediction request")
    try:
        raw_data = await request.json()
        logger.info(f"Raw incoming data: {raw_data}")

        try:
            input_data = PredictionInput(**raw_data)
        except ValidationError as ve:
            logger.error(f"Validation error: {ve.json()}")
            raise HTTPException(status_code=422, detail=ve.errors()) from ve

        if model is None:
            raise HTTPException(status_code=503, detail="Model is not available")

        input_dict = input_data.dict()
        processed_data, has_anomalies, anomalies_info = preprocess_input(input_dict)
        logger.info(f"Processed data: {processed_data.to_dict()}")

        prediction_proba = model.predict_proba(processed_data)
        original_prediction: float = float(prediction_proba[0, 1])
        logger.info(f"Raw prediction probabilities: {prediction_proba}")
        logger.info(f"Original prediction made: {original_prediction}")

        loan_amount = float(input_data.amt_credit)
        loan_term_years = input_data.loan_term_years
        monthly_payment = calculate_monthly_payment(loan_amount, loan_term_years)
        logger.info(f"Calculated monthly payment: {monthly_payment:.2f}")

        debt_to_income_ratio = float(processed_data['debt_to_income_ratio'].iloc[0])
        adjusted_prediction = adjust_default_probability(original_prediction, debt_to_income_ratio)
        logger.info(f"Adjusted probability of default: {adjusted_prediction:.4f}")

        max_monthly_payment = 0.4 * input_data.amt_income_total
        logger.info(f"Maximum allowable monthly payment based on income: {max_monthly_payment:.2f}")

        rejection_reason = None
        if not loan_to_income_check(monthly_payment, input_data.amt_income_total):
            risk_level = "High"
            loan_approval = "Rejected"
            rejection_reason = "Monthly payment exceeds 40% of income."
            adjusted_prediction = 1.0  # Maximum risk due to 40% rule
            logger.warning("Loan rejected due to monthly payment exceeding 40% of income.")
        else:
            if adjusted_prediction < 0.3:
                risk_level = "Low"
            elif adjusted_prediction < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"

            loan_approval = "Approved" if risk_level in ["Low", "Medium"] else "Rejected"
            if loan_approval == "Rejected" and not rejection_reason:
                rejection_reason = "Risk level classified as High."

        logger.info(f"Risk Level: {risk_level if loan_approval != 'Rejected' else 'N/A'}")
        logger.info(f"Loan Approval: {loan_approval}")
        logger.info(f"Monthly Payment: {monthly_payment:.2f}")
        if has_anomalies:
            logger.warning(f"Anomalies detected: {anomalies_info}")

        anomalies_details = [AnomalyRecord(**record) for record in anomalies_info] if has_anomalies else []

        response_data = PredictionResponse(
            probability_of_default=float(round(adjusted_prediction, 4)),
            monthly_payment=float(round(monthly_payment, 2)),
            risk_level=risk_level if loan_approval != "Rejected" else None,
            loan_approval=loan_approval,
            anomalies_detected=has_anomalies,
            anomalies_details=anomalies_details,
            rejection_reason=rejection_reason
        )

        return response_data
    except ValueError as ve:
        logger.error(f"ValueError during prediction: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
    - JSON response indicating the health status of the application and model loading status
    """
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
