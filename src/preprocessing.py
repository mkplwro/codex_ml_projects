import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def clean_data(data):
    data = data.copy()

    # Remove columns that are not used by the model
    data = data.drop(columns=["customer_id", "join_date"], errors="ignore")

    # Clean categorical values
    categorical_cols = [
        "city",
        "plan",
        "acquisition_channel"
    ]

    for col in categorical_cols:
        data[col] = data[col].str.strip().str.lower()

    # Replace invalid ages with missing values
    data.loc[
        (data["age"] >= 99) | (data["age"] < 18),
        "age"
    ] = np.nan

    # Replace invalid income values with missing values
    data.loc[
        (data["monthly_income_pln"] < 0) |
        (data["monthly_income_pln"] == 999999),
        "monthly_income_pln"
    ] = np.nan

    return data


def create_features(data):
    data = data.copy()

    # Remove target-related columns from features
    X = data.drop(
        columns=["monthly_spend_pln", "churned"],
        errors="ignore"
    )

    # Feature engineering used in the classification notebook
    X["satisfaction_per_tenure"] = (
        X["satisfaction_score"] /
        (X["tenure_months"] + 1)
    )

    return X


def create_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                make_column_selector(dtype_include="number")
            ),
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_include="object")
            )
        ]
    )

    return preprocessor
