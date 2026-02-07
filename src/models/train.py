from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from src.features.build_features import build_preprocessor


def train_gradient_boosting(
    X_train,
    y_train,
    numerical_features,
    categorical_features,
    binary_features,
):
    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
        binary_features
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(
                learning_rate=0.1,
                n_estimators=100,
                max_depth=2,
                random_state=42
            ))
        ]
    )

    model.fit(X_train, y_train)
    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
