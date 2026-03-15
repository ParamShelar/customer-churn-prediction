import os
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

from src.data_prep import load_and_clean_data, split_features_target
from sklearn.model_selection import train_test_split


def generate_shap_plots(model_path="models/churn_model.pkl", data_path="data/raw/telco_churn.csv"):
    model = joblib.load(model_path)

    df = load_and_clean_data(data_path)
    X, y = split_features_target(df)

    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=43,
        stratify=y
    )

    X_test_transformed = model.named_steps["preprocessor"].transform(X_test)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    rf_classifier = model.named_steps["classifier"]

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    if isinstance(shap_values, list):
        shap_for_class1 = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_for_class1 = shap_values[:, :, 1]
    else:
        shap_for_class1 = shap_values

    os.makedirs("outputs/figures", exist_ok=True)

    shap.summary_plot(
        shap_for_class1,
        X_test_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_for_class1,
        X_test_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved SHAP plots to outputs/figures/")


if __name__ == "__main__":
    generate_shap_plots()
