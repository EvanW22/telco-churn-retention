from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_preprocessor(df):
    categorical = df.select_dtypes(include="object").columns
    numeric = df.select_dtypes(exclude="object").columns.drop("Churn Value")

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", numeric_pipeline, numeric)
        ]
    )

    return preprocessor