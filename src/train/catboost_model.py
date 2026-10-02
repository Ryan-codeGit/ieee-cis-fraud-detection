import os
import sys
import yaml
import gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train_catboost():
    print("========== Starting CatBoost Pipeline Training ==========")
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

    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("missing")

    # Load Hyperparameters
    cb_param_path = config["paths"].get("cat_boost_params", "models/artifacts/best_catboost_params.yaml")
    if os.path.exists(cb_param_path):
        print(f" -> Loading parameters from: {cb_param_path}")
        with open(cb_param_path, "r") as f:
            cb_params = yaml.safe_load(f)
    else:
        cb_params = {"loss_function": "Logloss", "eval_metric": "AUC", "random_seed": 42, "verbose": False}

    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = pd.DataFrame({"isFraud": y, "fold": -1, "catboost_pred": np.nan})
    fold_scores = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n--- CatBoost Fold {fold + 1}/5 ---")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**cb_params, iterations=1000)
        model.fit(
            X_tr, y_tr,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            early_stopping_rounds=40,
            verbose=False
        )

        preds = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, preds)
        fold_scores.append(auc_score)
        best_iterations.append(model.get_best_iteration())

        oof_preds.loc[val_idx, "catboost_pred"] = preds
        oof_preds.loc[val_idx, "fold"] = fold + 1
        print(f"Fold {fold + 1} ROC-AUC: {auc_score:.5f} | Best Iter: {model.get_best_iteration()}")

    # Save OOF Artifact
    oof_path = "models/artifacts/oof_catboost.csv"
    os.makedirs(os.path.dirname(oof_path), exist_ok=True)
    oof_preds.dropna(subset=["catboost_pred"]).to_csv(oof_path, index=False)
    print(f"\nSaved CatBoost OOF Validation Predictions to: {oof_path}")

    # Retrain on Full Dataset
    final_rounds = int(np.mean(best_iterations))
    print(f"\n⚡ Retraining final CatBoost model on 100% data ({final_rounds} rounds)...")
    final_model = CatBoostClassifier(**cb_params, iterations=final_rounds)
    final_model.fit(X, y, cat_features=cat_cols, verbose=False)

    model_save_path = "models/final_catboost_model.cbm"
    final_model.save_model(model_save_path)
    print(f"Production model saved to: {model_save_path}")


if __name__ == "__main__":
    train_catboost()
