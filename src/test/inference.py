import os
import sys
import yaml
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Import clean dataset pipeline
from src.pipelines.build_dataset_test import process_inference_data


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_ensemble_inference():
    print("========== Starting Ensemble Submission Generator ==========")
    config = load_config()

    # 1. Run inference data cleaner pipeline
    X_test, sub = process_inference_data()

    # 2. Filter test features using active feature selection registry
    yaml_drop_path = "models/artifacts/optimized_dropped_features.yaml"
    if os.path.exists(yaml_drop_path):
        print(f" -> Loading active feature registry from: {yaml_drop_path}")
        with open(yaml_drop_path, "r") as f:
            drop_data = yaml.safe_load(f)
        selected_features = drop_data.get("selected_features", [])

        if selected_features:
            keep_cols = [c for c in selected_features if c in X_test.columns]
            X_test = X_test[keep_cols]

    # Remove volatile interaction layers if present
    X_test.drop(columns=["P_email_ProductCD", "R_email_ProductCD"], errors="ignore", inplace=True)

    # Downcast floats to float32
    float64_cols = X_test.select_dtypes(include=["float64"]).columns.tolist()
    if float64_cols:
        X_test[float64_cols] = X_test[float64_cols].astype(np.float32)

    cat_cols = X_test.select_dtypes(include=["category", "object", "string"]).columns.tolist()
    if "ProductCD" not in cat_cols and "ProductCD" in X_test.columns:
        cat_cols.append("ProductCD")

    # 3. Load ensemble weights
    weights_path = "models/artifacts/ensemble_weights.yaml"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing blending weights file: '{weights_path}'. Run blend_opt.py first.")

    with open(weights_path, "r") as f:
        weight_data = yaml.safe_load(f)

    weights = weight_data.get("weights", {"w_lightgbm": 0.3333, "w_catboost": 0.3333, "w_xgboost": 0.3334})
    print(f"Loaded Blending Weights: {weights}")

    # =========================================================================
    # LIGHTGBM PREDICTIONS
    # =========================================================================
    lgb_model_path = "models/final_lightgbm_model.txt"
    if os.path.exists(lgb_model_path) and weights.get("w_lightgbm", 0) > 0:
        print(f"\nEvaluating LightGBM predictions from '{lgb_model_path}'...")
        X_lgb = X_test.copy()
        for col in cat_cols:
            X_lgb[col] = X_lgb[col].astype("category")

        lgb_model = lgb.Booster(model_file=lgb_model_path)
        sub["lgb_pred"] = lgb_model.predict(X_lgb)
        del X_lgb, lgb_model
        gc.collect()
    else:
        sub["lgb_pred"] = 0.0

    # =========================================================================
    # CATBOOST PREDICTIONS
    # =========================================================================
    cb_model_path = "models/final_catboost_model.cbm"
    if os.path.exists(cb_model_path) and weights.get("w_catboost", 0) > 0:
        print(f"\nEvaluating CatBoost predictions from '{cb_model_path}'...")
        X_cb = X_test.copy()
        for col in cat_cols:
            X_cb[col] = X_cb[col].astype(str).fillna("missing")

        cb_model = CatBoostClassifier()
        cb_model.load_model(cb_model_path)
        sub["catboost_pred"] = cb_model.predict_proba(X_cb)[:, 1]
        del X_cb, cb_model
        gc.collect()
    else:
        sub["catboost_pred"] = 0.0

    # =========================================================================
    # XGBOOST PREDICTIONS
    # =========================================================================
    xgb_model_path = "models/final_xgboost_model.json"
    cat_map_path = "models/artifacts/cat_map.yaml"

    if os.path.exists(xgb_model_path) and weights.get("w_xgboost", 0) > 0:
        print(f"\nEvaluating XGBoost predictions from '{xgb_model_path}'...")
        X_xgb = X_test.copy()

        # Load saved category lists from training
        if os.path.exists(cat_map_path):
            with open(cat_map_path, "r") as f:
                cat_map = yaml.safe_load(f)

            for col, train_categories in cat_map.items():
                if col in X_xgb.columns:
                    # 1. Clean missing representation strings
                    clean_series = X_xgb[col].astype(str).replace(['nan', 'None', '<NA>', 'missing'], np.nan)

                    # 2. Align against training categories (unseen categories automatically turn into NaN)
                    X_xgb[col] = pd.Categorical(clean_series, categories=train_categories)
        else:
            # Fallback if cat_map.yaml is missing: cast to standard categories
            for col in cat_cols:
                if col in X_xgb.columns:
                    X_xgb[col] = X_xgb[col].astype("category")

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_model_path)

        sub["xgb_pred"] = xgb_model.predict_proba(X_xgb)[:, 1]
        del X_xgb, xgb_model
        gc.collect()
    else:
        sub["xgb_pred"] = 0.0
    # =========================================================================
    # BLEND PREDICTIONS AND SAVE SUBMISSION
    # =========================================================================
    print("\nApplying weighted probability blend...")
    sub["isFraud"] = (
        (sub["lgb_pred"] * weights.get("w_lightgbm", 0)) +
        (sub["catboost_pred"] * weights.get("w_catboost", 0)) +
        (sub["xgb_pred"] * weights.get("w_xgboost", 0))
    )

    # Format output submission
    final_submission = sub[["TransactionID", "isFraud"]]

    output_sub_path = "outputs/submission.csv"
    final_submission.to_csv(output_sub_path, index=False)

    print(f"\nSuccessfully generated submission file: '{output_sub_path}'!")
    print(f"Shape: {final_submission.shape}")
    print("\nFirst 5 Rows:")
    print(final_submission.head())


if __name__ == "__main__":
    run_ensemble_inference()
