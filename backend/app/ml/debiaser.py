import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from app.ml.classifier import build_preprocessor


def compute_reweighing_weights(y_train: pd.Series, sensitive_train: pd.Series) -> np.ndarray:
    """Menghitung bobot sampel (Reweighing - Pre-processing) agar proporsi antar kelompok setara."""
    df_w = pd.DataFrame({"y": y_train.values, "a": sensitive_train.values})
    weights = np.ones(len(df_w))

    for a_val in df_w["a"].unique():
        for y_val in df_w["y"].unique():
            mask = (df_w["a"] == a_val) & (df_w["y"] == y_val)
            p_a = (df_w["a"] == a_val).mean()
            p_y = (df_w["y"] == y_val).mean()
            p_ay = mask.mean()
            if p_ay > 0:
                weights[mask] = (p_a * p_y) / p_ay

    return weights


def train_debiased_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_train: pd.Series,
    mitigation_method: str = "reweighing",
    constraint_type: str = "demographic_parity",
):
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    base_estimator = LogisticRegression(max_iter=1000, random_state=42)

    if mitigation_method == "reweighing":
        sample_weights = compute_reweighing_weights(y_train, sensitive_train)
        base_estimator.fit(X_train_transformed, y_train, sample_weight=sample_weights)
        return preprocessor, base_estimator

    # Default alternatif: Exponentiated Gradient (In-processing)
    if constraint_type == "equalized_odds":
        constraint = EqualizedOdds(difference_bound=0.05)
    else:
        constraint = DemographicParity(difference_bound=0.05)

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