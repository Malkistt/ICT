import pandas as pd
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)

def evaluate_fairness(y_true, y_pred, sensitive_features) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    dp_diff = float(
        demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
    )
    eo_diff = float(
        equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
    )

    metric_frame = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    group_breakdown = {}
    for group, row in metric_frame.by_group.iterrows():
        group_breakdown[str(group)] = {
            "selection_rate": round(float(row["selection_rate"]), 4),
            "accuracy": round(float(row["accuracy"]), 4),
        }

    # Ambang batas toleransi bias standar: > 0.10 (10%) dianggap terindikasi bias
    is_biased = dp_diff > 0.10 or eo_diff > 0.10

    return {
        "accuracy": round(acc, 4),
        "demographic_parity_diff": round(dp_diff, 4),
        "equalized_odds_diff": round(eo_diff, 4),
        "is_biased": is_biased,
        "group_metrics": group_breakdown,
    }