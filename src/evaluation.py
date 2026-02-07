from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_pr_auc(pipe, X, y):
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="average_precision"
    )
    return scores.mean(), scores.std()