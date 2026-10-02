import os
import sys
import yaml
import optuna
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# Disable Optuna verbosity for clean console logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


def load_oof_predictions():
    print("========== Loading Out-of-Fold (OOF) Predictions ==========")
    
    lgb_path = "models/artifacts/oof_lightgbm.csv"
    cb_path = "models/artifacts/oof_catboost.csv"
    xgb_path = "models/artifacts/oof_xgboost.csv"

    if not (os.path.exists(lgb_path) and os.path.exists(cb_path) and os.path.exists(xgb_path)):
        raise FileNotFoundError(
            "Missing one or more OOF files. Please ensure train_lightgbm.py, "
            "train_catboost.py, and train_xgboost.py have been executed."
        )

    df_lgb = pd.read_csv(lgb_path)
    df_cb = pd.read_csv(cb_path)
    df_xgb = pd.read_csv(xgb_path)

    # Merge OOF data on Target and Fold
    oof = df_lgb[["isFraud", "fold", "lgb_pred"]].copy()
    oof["catboost_pred"] = df_cb["catboost_pred"]
    oof["xgb_pred"] = df_xgb["xgb_pred"]

    # Filter out non-validation rows if any exist
    oof = oof.dropna(subset=["lgb_pred", "catboost_pred", "xgb_pred"]).reset_index(drop=True)

    print(f"Loaded {len(oof)} validation rows across folds {sorted(oof['fold'].unique().tolist())}.")
    return oof


def evaluate_individual_models(oof):
    print("\n--- Baseline Individual Out-of-Fold ROC-AUC Scores ---")
    y_true = oof["isFraud"]

    # Focus on the last 2 time-series folds for out-of-time evaluation
    oot_mask = oof["fold"].isin([4, 5])
    y_true_oot = oof.loc[oot_mask, "isFraud"]

    for col, name in [("lgb_pred", "LightGBM"), ("catboost_pred", "CatBoost"), ("xgb_pred", "XGBoost")]:
        overall_auc = roc_auc_score(y_true, oof[col])
        oot_auc = roc_auc_score(y_true_oot, oof.loc[oot_mask, col])
        print(f"  * {name:<10} | Full OOF ROC-AUC: {overall_auc:.5f} | OOT (Folds 4 & 5): {oot_auc:.5f}")


def rank_average_blend(oof):
    print("\n--- 1. Evaluating Rank Averaging Blend ---")
    
    # Rank probabilities per fold to prevent scale mismatch
    oof_ranked = oof.copy()
    for col in ["lgb_pred", "catboost_pred", "xgb_pred"]:
        oof_ranked[col] = oof_ranked.groupby("fold")[col].rank(pct=True)

    oof_ranked["rank_blend"] = (
        oof_ranked["lgb_pred"] + oof_ranked["catboost_pred"] + oof_ranked["xgb_pred"]
    ) / 3.0

    oot_mask = oof_ranked["fold"].isin([4, 5])
    full_auc = roc_auc_score(oof_ranked["isFraud"], oof_ranked["rank_blend"])
    oot_auc = roc_auc_score(oof_ranked.loc[oot_mask, "isFraud"], oof_ranked.loc[oot_mask, "rank_blend"])

    print(f"  * Rank Average Blend | Full OOF ROC-AUC: {full_auc:.5f} | OOT (Folds 4 & 5): {oot_auc:.5f}")
    return rank_average_blend


def optimize_weights_optuna(oof, n_trials=100):
    print("\n--- 2. Optimizing Weighted Probability Ensemble (Optuna) ---")

    oot_mask = oof["fold"].isin([4, 5])
    y_true_oot = oof.loc[oot_mask, "isFraud"].values

    p_lgb = oof.loc[oot_mask, "lgb_pred"].values
    p_cb = oof.loc[oot_mask, "catboost_pred"].values
    p_xgb = oof.loc[oot_mask, "xgb_pred"].values

    def objective(trial):
        # Sample raw weights
        w_lgb = trial.suggest_float("w_lgb", 0.0, 1.0)
        w_cb = trial.suggest_float("w_cb", 0.0, 1.0)
        w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)

        total_w = w_lgb + w_cb + w_xgb
        if total_w == 0:
            return 0.0

        # Normalize weights to sum to 1.0
        w_lgb /= total_w
        w_cb /= total_w
        w_xgb /= total_w

        blend_pred = (w_lgb * p_lgb) + (w_cb * p_cb) + (w_xgb * p_xgb)
        return roc_auc_score(y_true_oot, blend_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_w = study.best_params
    total_w = sum(best_w.values())
    
    normalized_weights = {
        "w_lightgbm": round(best_w["w_lgb"] / total_w, 4),
        "w_catboost": round(best_w["w_cb"] / total_w, 4),
        "w_xgboost": round(best_w["w_xgb"] / total_w, 4),
    }

    print(f"Optimal OOT ROC-AUC: {study.best_value:.5f}")
    print("Optimal Blending Weights:")
    for k, v in normalized_weights.items():
        print(f"  {k:<12}: {v:.4f}")

    # Save ensemble manifest
    output_path = "models/artifacts/ensemble_weights.yaml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ensemble_manifest = {
        "best_oot_auc": float(study.best_value),
        "weights": normalized_weights
    }

    with open(output_path, "w") as f:
        yaml.dump(ensemble_manifest, f, default_flow_style=False)

    print(f"\nSaved optimal blending configuration to: '{output_path}'")


if __name__ == "__main__":
    oof_data = load_oof_predictions()
    evaluate_individual_models(oof_data)
    rank_average_blend(oof_data)
    optimize_weights_optuna(oof_data, n_trials=200)
