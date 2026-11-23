import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from app.ml.classifier import build_preprocessor

def train_debiased_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_train: pd.Series,
    constraint_type: str = "demographic_parity",
):
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)

    if constraint_type == "equalized_odds":
        constraint = EqualizedOdds(difference_bound=0.05)
    else:
        constraint = DemographicParity(difference_bound=0.05)

    base_estimator = LogisticRegression(max_iter=1000, random_state=42)
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        eps=0.02,
    )
    mitigator.fit(X_train_transformed, y_train, sensitive_features=sensitive_train)

    return preprocessor, mitigator

def predict_debiased(preprocessor, mitigator, X: pd.DataFrame):
    X_transformed = preprocessor.transform(X)
    return mitigator.predict(X_transformed)