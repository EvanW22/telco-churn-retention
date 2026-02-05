from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_models():
    return {
        "logistic": LogisticRegression(max_iter=1000),
        "xgboost": XGBClassifier(
            eval_metric="logloss"
        )
    }