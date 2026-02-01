from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_feature_lists():
    numerical_features = ["tenure","MonthlyCharges","TotalCharges"]
    categorical_features = ["Contract","PaymentMethod","InternetService"]
    binary_features = ["Partner","Dependents","PaperlessBilling"]
    return numerical_features, categorical_features, binary_features

def build_preprocessor(numerical_features, categorical_features, binary_features):
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", "passthrough", binary_features),
        ]
    )

    return preprocessor