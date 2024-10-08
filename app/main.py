"""
Main module for the Retail Bank Risk Evaluation application.

This module sets up the FastAPI application, defines data models,
handles feature engineering, processes loan applications, and provides
endpoints for prediction and health checks.

The application evaluates the risk of loan defaults based on user-provided
data and a pre-trained machine learning model.
"""

import logging
import os
import pickle
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError, validator
from starlette.staticfiles import StaticFiles

from retail_bank_risk.feature_engineering_utils import engineer_features

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR: str = os.path.join(BASE_DIR, "models")
MODEL_PATH: str = os.path.join(MODEL_DIR, "tuned_tuned_xgboost_checkpoint.pkl")

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    logger.warning("Created missing models directory at %s", MODEL_DIR)

TEMPLATES_DIR = os.path.join(BASE_DIR, "app", "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

STATIC_DIR = os.path.join(BASE_DIR, "app", "static")
if not os.path.isdir(STATIC_DIR):
    os.makedirs(STATIC_DIR)
    logger.warning("Created missing static directory at %s", STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if not os.path.isfile(MODEL_PATH):
    logger.error("Model file not found at %s", MODEL_PATH)
    MODEL = None
    SELECTED_FEATURES = []
else:
    with open(MODEL_PATH, "rb") as f:
        checkpoint: Dict[str, Any] = pickle.load(f)

    MODEL: Any = checkpoint.get("model")
    SELECTED_FEATURES: List[str] = checkpoint.get("selected_features", [])

    if not MODEL or not SELECTED_FEATURES:
        logger.error("Model or selected features not found in the checkpoint.")
        MODEL = None
        SELECTED_FEATURES = []
    else:
        logger.info(
            "Model loaded successfully. Selected features: %s",
            SELECTED_FEATURES,
        )


class IncomeType(str, Enum):
    """Enumeration for types of income."""

    WORKING = "Working"
    STATE_SERVANT = "State servant"
    COMMERCIAL_ASSOCIATE = "Commercial associate"
    PENSIONER = "Pensioner"
    UNEMPLOYED = "Unemployed"
    STUDENT = "Student"
    BUSINESSMAN = "Businessman"
    MATERNITY_LEAVE = "Maternity leave"


class HousingType(str, Enum):
    """Enumeration for types of housing."""

    HOUSE_APARTMENT = "House / apartment"
    RENTED_APARTMENT = "Rented apartment"
    WITH_PARENTS = "With parents"
    MUNICIPAL_APARTMENT = "Municipal apartment"
    OFFICE_APARTMENT = "Office apartment"
    CO_OP_APARTMENT = "Co-op apartment"


class FamilyStatus(str, Enum):
    """Enumeration for family status."""

    SINGLE_NOT_MARRIED = "Single / not married"
    MARRIED = "Married"
    CIVIL_MARRIAGE = "Civil marriage"
    WIDOW = "Widow"
    SEPARATED = "Separated"
    UNKNOWN = "Unknown"


class EducationType(str, Enum):
    """Enumeration for education types."""

    LOWER_SECONDARY = "Lower secondary"
    SECONDARY = "Secondary / secondary special"
    INCOMPLETE_HIGHER = "Incomplete higher"
    HIGHER_EDUCATION = "Higher education"
    ACADEMIC_DEGREE = "Academic degree"


class PredictionInput(BaseModel):
    """Schema for input data for prediction."""

    amt_income_total: float = Field(
        ..., description="Total monthly income amount", ge=0.0
    )
    amt_credit: float = Field(..., description="Credit amount", gt=0.0)
    amt_goods_price: float = Field(..., description="Goods price", ge=0.0)
    name_income_type: IncomeType = Field(..., description="Income type")
    name_housing_type: HousingType = Field(..., description="Housing type")
    name_family_status: FamilyStatus = Field(..., description="Family status")
    name_education_type: EducationType = Field(
        ..., description="Education type"
    )
    code_gender: str = Field(..., description="Gender code", pattern="^(M|F)$")
    flag_own_car: str = Field(
        ..., description="Car ownership flag", pattern="^(Y|N)$"
    )
    flag_own_realty: str = Field(
        ..., description="Realty ownership flag", pattern="^(Y|N)$"
    )
    days_birth: Optional[int] = Field(
        None, description="Days before current date of birth"
    )
    age: Optional[int] = Field(None, description="Age in years", ge=18, le=100)
    buy_goods: bool = Field(
        ..., description="Whether goods are being purchased"
    )
    loan_term_years: int = Field(
        ..., description="Loan term in years", ge=1, le=30
    )
    existing_mortgage_payment: float = Field(
        0, description="Existing monthly mortgage payment", ge=0
    )
    existing_loan_payments: float = Field(
        0, description="Total of other existing monthly loan payments", ge=0
    )

    @validator("amt_goods_price")
    @classmethod
    def validate_amt_goods_price(cls, v):
        """Ensure amt_goods_price is non-negative."""
        if v < 0:
            raise ValueError("amt_goods_price must be non-negative")
        return v

    @validator("days_birth")
    @classmethod
    def validate_days_birth(cls, v, values):
        """Ensure days_birth is negative and either days_birth or age is provided."""
        if v is not None and v > 0:
            raise ValueError("days_birth must be negative")
        if v is None and "age" not in values:
            raise ValueError("Either days_birth or age must be provided")
        return v

    @validator("age")
    @classmethod
    def validate_age(cls, v, values):
        """Ensure either age or days_birth is provided."""
        if v is None and "days_birth" not in values:
            raise ValueError("Either days_birth or age must be provided")
        return v

    class Config:
        """Configuration for the PredictionInput model."""

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
    risk_level: str
    loan_approval: str
    anomalies_detected: bool
    anomalies_details: Optional[List[AnomalyRecord]] = []


def preprocess_input(
    data: Dict[str, Any],
) -> Tuple[pd.DataFrame, bool, List[Dict[str, Any]]]:
    """
    Preprocess input data for model prediction, including anomaly detection.

    Args:
        data (Dict[str, Any]): Dictionary containing input features.

    Returns:
        Tuple[pd.DataFrame, bool, List[Dict[str, Any]]]:
            - Processed DataFrame
            - Anomaly detection flag
            - List of anomalies
    """
    logger.info("Starting preprocessing of input data")
    logger.info("Input data: %s", data)

    df = pd.DataFrame([data])

    if "age" in df.columns and pd.notnull(df["age"]).all():
        df["days_birth"] = -df["age"] * 365
        df = df.drop(columns=["age"])
        logger.info(
            "Converted age to days_birth: %s days",
            df["days_birth"].iloc[0],  # pylint: disable=unsubscriptable-object
        )
    elif "days_birth" in df.columns:
        if pd.isnull(df["days_birth"]).all():
            if "age" in df.columns and pd.notnull(df["age"]).all():
                df["days_birth"] = -df["age"] * 365
                df = df.drop(columns=["age"])
                logger.info(
                    "Converted age to days_birth: %s days",
                    df["days_birth"].iloc[ # pylint: disable=unsubscriptable-object
                        0
                    ],
                )
            else:
                logger.error(
                    "Either days_birth or age must be provided and valid."
                )
                raise ValueError(
                    "Either days_birth or age must be provided and valid."
                )

    if "amt_annuity" not in df.columns or pd.isnull(df["amt_annuity"]).all():
        loan_amount = df["amt_credit"].iloc[0]
        loan_term_years = df["loan_term_years"].iloc[0]
        interest_rate = 0.05
        loan_term_months = loan_term_years * 12
        monthly_interest_rate = interest_rate / 12
        if monthly_interest_rate == 0:
            amt_annuity = loan_amount / loan_term_months
        else:
            amt_annuity = (
                loan_amount
                * monthly_interest_rate
                * (1 + monthly_interest_rate) ** loan_term_months
            ) / ((1 + monthly_interest_rate) ** loan_term_months - 1)
        df["amt_annuity"] = amt_annuity
        logger.info("Calculated amt_annuity: %s", amt_annuity)

    df = engineer_features(df)
    logger.info("Completed feature engineering.")

    numerical_features = [
        "amt_income_total",
        "amt_credit",
        "amt_goods_price",
        "amt_annuity",
    ]
    anomaly_bounds = {
        "amt_income_total": {"lower": 0, "upper": 1000000},
        "amt_credit": {"lower": 1000, "upper": 1000000},
        "amt_goods_price": {"lower": 0, "upper": 500000},
        "amt_annuity": {"lower": 100, "upper": 100000},
    }

    anomalies_detected = False
    anomalies_details = []

    for feature in numerical_features:
        value = df.at[0, feature]
        lower = anomaly_bounds[feature]["lower"]
        upper = anomaly_bounds[feature]["upper"]
        if value < lower or value > upper:
            anomalies_detected = True
            anomaly_record = {
                "feature": feature,
                "value": value,
                "lower_bound": lower,
                "upper_bound": upper,
            }
            anomalies_details.append(anomaly_record)
            logger.warning(
                "Anomaly detected in feature '%s': %s not in [%s, %s]",
                feature,
                value,
                lower,
                upper,
            )

    if not anomalies_detected:
        logger.info("No anomalies detected.")

    binary_features_mapping = {
        "code_gender": {"M": 1, "F": 0},
        "flag_own_car": {"Y": 1, "N": 0},
        "flag_own_realty": {"Y": 1, "N": 0},
    }
    for feature, mapping in binary_features_mapping.items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping).fillna(0).astype(int)
            logger.info(
                "Encoded binary feature '%s' with mapping %s", feature, mapping
            )

    categorical_features = [
        "name_income_type",
        "name_housing_type",
        "name_family_status",
        "name_education_type",
    ]
    categories_mapping = {
        "name_income_type": [
            "Working",
            "State servant",
            "Commercial associate",
            "Pensioner",
            "Unemployed",
            "Student",
            "Businessman",
            "Maternity leave",
        ],
        "name_housing_type": [
            "House / apartment",
            "Rented apartment",
            "With parents",
            "Municipal apartment",
            "Office apartment",
            "Co-op apartment",
        ],
        "name_family_status": [
            "Single / not married",
            "Married",
            "Civil marriage",
            "Widow",
            "Separated",
            "Unknown",
        ],
        "name_education_type": [
            "Lower secondary",
            "Secondary / secondary special",
            "Incomplete higher",
            "Higher education",
            "Academic degree",
        ],
    }

    for feature in categorical_features:
        if feature in df.columns:
            dummies = pd.get_dummies(
                df[feature], prefix=feature, dummy_na=False
            ).astype(int)
            dummies.columns = [
                col.lower()
                .replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
                for col in dummies.columns
            ]
            expected_cols = [
                f"{feature}_{cat.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}"
                for cat in categories_mapping[feature]
            ]
            for col in expected_cols:
                if col not in dummies.columns:
                    dummies[col] = 0
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[feature])
            logger.info(
                "One-hot encoded feature '%s'. Added columns: %s",
                feature,
                expected_cols,
            )

    current_weekday = datetime.now().strftime("%A").lower()
    weekday_features = [
        "weekday_appr_process_start_monday",
        "weekday_appr_process_start_tuesday",
        "weekday_appr_process_start_wednesday",
        "weekday_appr_process_start_thursday",
        "weekday_appr_process_start_friday",
        "weekday_appr_process_start_saturday",
        "weekday_appr_process_start_sunday",
    ]

    for day_feature in weekday_features:
        day = day_feature.rsplit("_", maxsplit=1)[-1]
        df[day_feature] = 1 if day == current_weekday else 0
        logger.info("Set '%s' to %s", day_feature, df[day_feature].iloc[0])

    df["is_anomaly"] = 1 if anomalies_detected else 0
    logger.info("Set 'is_anomaly' to %s", df["is_anomaly"].iloc[0])

    for feature in SELECTED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0
            logger.info("Added missing feature: %s", feature)

    df = df[SELECTED_FEATURES]
    logger.info("Reordered DataFrame to match selected features.")

    df.replace([np.inf, -np.inf], 1e6, inplace=True)
    logger.info("Replaced infinite values with 1e6.")

    logger.info("Preprocessed dataframe shape: %s", df.shape)
    logger.info("Preprocessed data columns: %s", df.columns.tolist())
    logger.info("Sample of preprocessed data: %s", df.iloc[0].to_dict())

    return df, anomalies_detected, anomalies_details


def calculate_monthly_payment(
    amt_credit: float, loan_term_years: int, interest_rate: float = 0.05
) -> float:
    """
    Calculate the monthly payment for a loan.

    Args:
        amt_credit (float): The loan amount.
        loan_term_years (int): The term of the loan in years.
        interest_rate (float, optional): Annual interest rate. Defaults to 0.05.

    Returns:
        float: Monthly payment amount.
    """
    loan_term_months = loan_term_years * 12
    monthly_interest_rate = interest_rate / 12
    if monthly_interest_rate == 0:
        return amt_credit / loan_term_months
    else:
        return (
            amt_credit
            * monthly_interest_rate
            * (1 + monthly_interest_rate) ** loan_term_months
        ) / ((1 + monthly_interest_rate) ** loan_term_months - 1)


def adjust_default_probability(
    original_prob: float, debt_to_income_ratio: float
) -> float:
    """
    Adjust the probability of default based on the debt-to-income ratio.

    Args:
        original_prob (float): The original predicted probability of default.
        debt_to_income_ratio (float): The debt-to-income ratio.

    Returns:
        float: Adjusted probability of default.
    """
    adjustment_factor = (
        0.05  # Reduced from 0.10 to make the adjustment less aggressive
    )
    adjusted_prob = original_prob + (adjustment_factor * debt_to_income_ratio)
    return min(adjusted_prob, 1.0)


def determine_risk_level(probability_of_default: float) -> str:
    """
    Determine the risk level based on the probability of default.

    Args:
        probability_of_default (float): The probability of default.

    Returns:
        str: Risk level as a string.
    """
    if probability_of_default < 0.3:
        return "Low"
    elif probability_of_default < 0.6:
        return "Medium"
    else:
        return "High"


@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request) -> HTMLResponse:
    """
    Serve the loan application form.

    Args:
        request (Request): FastAPI request object.

    Returns:
        HTMLResponse: Rendered HTML template for the loan application form.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post(
    "/predict",
    response_class=JSONResponse,
    response_model=PredictionResponse,
)
async def predict(request: Request) -> PredictionResponse:
    """
    Endpoint for making a loan risk prediction.

    Args:
        request (Request): FastAPI request object containing loan application data.

    Returns:
        PredictionResponse: JSON response with prediction details including probability of default,
                            monthly payment, risk level, loan approval status, and anomaly detection
                            information.
    """
    logger.info("Received prediction request")
    try:
        raw_data = await request.json()
        logger.info("Raw incoming data: %s", raw_data)

        try:
            input_data = PredictionInput(**raw_data)
        except ValidationError as ve:
            logger.error("Validation error: %s", ve.json())
            raise HTTPException(status_code=422, detail=ve.errors()) from ve

        if MODEL is None:
            raise HTTPException(
                status_code=503, detail="Model is not available"
            )

        input_dict = input_data.dict()
        processed_data, has_anomalies, anomalies_info = preprocess_input(
            input_dict
        )
        logger.info("Processed data: %s", processed_data.to_dict())

        prediction_proba = MODEL.predict_proba(processed_data)
        original_prediction: float = float(prediction_proba[0, 1])
        logger.info("Raw prediction probabilities: %s", prediction_proba)
        logger.info("Original prediction made: %s", original_prediction)

        loan_amount = float(input_data.amt_credit)
        loan_term_years = input_data.loan_term_years
        monthly_payment = calculate_monthly_payment(
            loan_amount, loan_term_years
        )
        logger.info("Calculated monthly payment: %.2f", monthly_payment)

        debt_to_income_ratio = monthly_payment / input_data.amt_income_total
        adjusted_prediction = adjust_default_probability(
            original_prediction, debt_to_income_ratio
        )
        logger.info(
            "Adjusted probability of default: %.4f", adjusted_prediction
        )

        total_monthly_debt = (
            monthly_payment
            + input_data.existing_mortgage_payment
            + input_data.existing_loan_payments
        )
        total_debt_to_income_ratio = (
            total_monthly_debt / input_data.amt_income_total
        )
        logger.info(
            "Total monthly debt to income ratio: %.2f",
            total_debt_to_income_ratio,
        )

        if total_debt_to_income_ratio <= 0.4:
            loan_approval = "Approved"
            risk_level = determine_risk_level(adjusted_prediction)
            logger.info(
                "Loan approved. Debt-to-income ratio: %.2f",
                total_debt_to_income_ratio,
            )
        else:
            loan_approval = "Rejected"
            risk_level = "High"
            logger.info(
                "Loan rejected. Debt-to-income ratio: %.2f exceeds 40%% threshold",
                total_debt_to_income_ratio,
            )

        logger.info("Risk Level: %s", risk_level)
        logger.info("Loan Approval: %s", loan_approval)
        logger.info("Monthly Payment: %.2f", monthly_payment)

        if has_anomalies:
            logger.warning("Anomalies detected: %s", anomalies_info)

        anomalies_details = (
            [AnomalyRecord(**record) for record in anomalies_info]
            if has_anomalies
            else []
        )

        response_data = PredictionResponse(
            probability_of_default=float(round(adjusted_prediction, 4)),
            monthly_payment=float(round(monthly_payment, 2)),
            risk_level=risk_level,
            loan_approval=loan_approval,
            anomalies_detected=has_anomalies,
            anomalies_details=anomalies_details,
        )

        return response_data
    except ValidationError as ve:
        logger.error("Validation error: %s", ve.json())
        return JSONResponse(status_code=422, content={"detail": ve.errors()})
    except ValueError as ve:
        logger.error("ValueError during prediction: %s", str(ve))
        return JSONResponse(status_code=400, content={"detail": str(ve)})
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error("Error during prediction: %s", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Dict[str, str]: JSON response indicating the health status of
        the application and model loading status.
    """
    return {"status": "healthy", "model_loaded": MODEL is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
