from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_logistic():
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    )

def build_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

def build_xgboost(scale_pos_weight: float):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

