import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


CATEGORICAL_COLUMNS = [
    "city",
    "plan",
    "acquisition_channel",
]


def clean_data(data):
    """Apply data-quality rules shared by both production models."""
    data = data.copy()

    data = data.drop(
        columns=["customer_id", "join_date"],
        errors="ignore",
    )

    for col in CATEGORICAL_COLUMNS:
        if col in data.columns:
            data[col] = data[col].astype("string").str.strip().str.lower()

    if "age" in data.columns:
        data.loc[
            (data["age"] >= 99) | (data["age"] < 18),
            "age",
        ] = np.nan

    if "monthly_income_pln" in data.columns:
        data.loc[
            (data["monthly_income_pln"] < 0)
            | (data["monthly_income_pln"] == 999999),
            "monthly_income_pln",
        ] = np.nan

    return data


def create_features(data, task="classification"):
    """Create task-specific features and remove all target columns."""
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")

    data = data.copy()

    X = data.drop(
        columns=["monthly_spend_pln", "churned"],
        errors="ignore",
    )

    if task == "classification":
        X["satisfaction_per_tenure"] = (
            X["satisfaction_score"] / (X["tenure_months"] + 1)
        )
    else:
        X["income_per_tenure"] = (
            X["monthly_income_pln"] / (X["tenure_months"] + 1)
        )

    return X


def create_preprocessor():
    """Return the shared leakage-safe preprocessing pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                make_column_selector(dtype_include="number"),
            ),
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_include="object"),
            ),
        ]
    )
