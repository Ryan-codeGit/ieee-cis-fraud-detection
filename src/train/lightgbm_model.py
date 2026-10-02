import os
import sys
import yaml
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train_lightgbm():
    print("========== Starting LightGBM Pipeline Training ==========")
    config = load_config()

    processed_train_path = config["paths"]["processed_train_output"]
    print(f"Loading engineered dataset from: {processed_train_path}")
    df = pd.read_parquet(processed_train_path)

    print("Sorting dataset chronologically by TransactionDT...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    X = df.drop(columns=["isFraud", "TransactionDT"], errors="ignore")
    y = df["isFraud"].astype(np.int8)
    del df
    gc.collect()

    # 1. Feature Selection Registry Ingestion
    yaml_drop_path = "models/artifacts/optimized_dropped_features.yaml"
    if os.path.exists(yaml_drop_path):
        with open(yaml_drop_path, "r") as f:
            drop_data = yaml.safe_load(f)
        selected_features = drop_data.get("selected_features", [])
        if selected_features:
            keep_cols = [c for c in selected_features if c in X.columns]
            X = X[keep_cols]

    X.drop(columns=["P_email_ProductCD", "R_email_ProductCD"], errors="ignore", inplace=True)

    float64_cols = X.select_dtypes(include=["float64"]).columns.tolist()
    if float64_cols:
        X[float64_cols] = X[float64_cols].astype(np.float32)

    cat_cols = X.select_dtypes(include=["category", "object", "string"]).columns.tolist()
    if "ProductCD" not in cat_cols and "ProductCD" in X.columns:
        cat_cols.append("ProductCD")

    categorical_dtypes_map = {}
    for col in cat_cols:
        clean_series = X[col].astype(str).replace(['nan', 'None', '<NA>'], np.nan)
        unique_categories = sorted([x for x in clean_series.dropna().unique()])
        categorical_dtypes_map[col] = pd.CategoricalDtype(categories=unique_categories, ordered=False)
        X[col] = clean_series.astype(categorical_dtypes_map[col])

    # 2. Load Tuned Hyperparameters from Config Path
    lgb_param_path = config["paths"].get("lightgbm_params", "models/artifacts/best_lightgbm_params.yaml")
    if os.path.exists(lgb_param_path):
        print(f" -> Loading parameters from: {lgb_param_path}")
        with open(lgb_param_path, "r") as f:
            lgb_params = yaml.safe_load(f)
    else:
        lgb_params = {"objective": "binary", "metric": "auc", "boosting_type": "gbdt", "verbosity": -1, "random_state": 42}

    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = pd.DataFrame({"isFraud": y, "fold": -1, "lgb_pred": np.nan})
    fold_scores = []
    best_iterations = []

    # 3. Time-Series Cross Validation & OOF Capture
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n--- LightGBM Fold {fold + 1}/5 ---")
        X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]

        for col in cat_cols:
            if col in X_tr.columns:
                X_tr[col] = X_tr[col].astype(str).replace(['nan', 'None', '<NA>'], np.nan)
                X_tr[col] = X_tr[col].astype(categorical_dtypes_map[col])
            if col in X_val.columns:
                X_val[col] = X_val[col].astype(str).replace(['nan', 'None', '<NA>'], np.nan)
                X_val[col] = X_val[col].astype(categorical_dtypes_map[col])

        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=True)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols, free_raw_data=True)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1200,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        auc_score = roc_auc_score(y_val, preds)
        fold_scores.append(auc_score)
        best_iterations.append(model.best_iteration)

        # Record OOF predictions
        oof_preds.loc[val_idx, "lgb_pred"] = preds
        oof_preds.loc[val_idx, "fold"] = fold + 1
        print(f"Fold {fold + 1} ROC-AUC: {auc_score:.5f} | Best Iter: {model.best_iteration}")

        del train_data, val_data, X_tr, X_val, model
        gc.collect()

    # Save OOF Predictions Artifact
    oof_path = "models/artifacts/oof_lightgbm.csv"
    os.makedirs(os.path.dirname(oof_path), exist_ok=True)
    oof_preds.dropna(subset=["lgb_pred"]).to_csv(oof_path, index=False)
    print(f"\nSaved LightGBM OOF Validation Predictions to: {oof_path}")

    # 4. Final Production Retraining on Full Dataset
    final_rounds = int(np.mean(best_iterations))
    print(f"\n⚡ Retraining final LightGBM model on 100% data ({final_rounds} rounds)...")
    full_data = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=True)
    final_model = lgb.train(lgb_params, full_data, num_boost_round=final_rounds)

    model_save_path = "models/final_lightgbm_model.txt"
    final_model.save_model(model_save_path)
    print(f"Production model saved to: {model_save_path}")


if __name__ == "__main__":
    train_lightgbm()
