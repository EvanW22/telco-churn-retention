from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models():
    return {
        "logistic": LogisticRegression(
            max_iter=1000
        ),
        "xgboost": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "gbm_sklearn": GradientBoostingClassifier(
            random_state=42
        )
    }