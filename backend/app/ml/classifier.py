import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

NUMERIC_FEATURES = ["penghasilan", "jumlah_tanggungan"]
CATEGORICAL_FEATURES = ["pekerjaan", "kondisi_rumah", "lokasi"]
TARGET_COL = "layak"

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

def prepare_data(df: pd.DataFrame, sensitive_col: str = "lokasi"):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)
    sensitive = df[sensitive_col].astype(str)

    return train_test_split(
        X, y, sensitive, test_size=0.25, random_state=42, stratify=y
    )

def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    preprocessor = build_preprocessor()
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return model