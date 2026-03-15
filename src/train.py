import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.data_prep import load_and_clean_data, split_features_target


def build_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include="object").columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor


def train_models(filepath="data/raw/telco_churn.csv"):
    df = load_and_clean_data(filepath)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=43,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    logreg_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    logreg_model.fit(X_train, y_train)

    y_pred_lr = logreg_model.predict(X_test)
    y_proba_lr = logreg_model.predict_proba(X_test)[:, 1]

    print("Logistic Regression Report:")
    print(classification_report(y_test, y_pred_lr))
    print("Logistic Regression Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))
    print("Logistic Regression ROC AUC:", roc_auc_score(y_test, y_proba_lr))

    rf_stronger = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=750,
            max_depth=10,
            min_samples_split=12,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    rf_stronger.fit(X_train, y_train)

    y_pred_rf = rf_stronger.predict(X_test)
    y_proba_rf = rf_stronger.predict_proba(X_test)[:, 1]

    print("\nStronger Random Forest Report:")
    print(classification_report(y_test, y_pred_rf))
    print("Stronger Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("Stronger Random Forest ROC AUC:", roc_auc_score(y_test, y_proba_rf))

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_stronger, "models/churn_model.pkl")

    print("\nSaved model to models/churn_model.pkl")

    return rf_stronger, X_test, y_test


if __name__ == "__main__":
    train_models()
