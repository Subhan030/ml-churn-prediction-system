from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

def build_model(preprocessor, model_name="logistic"):
    if model_name == "logistic":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)