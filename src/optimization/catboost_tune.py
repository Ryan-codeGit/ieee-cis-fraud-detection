import os
import sys
import gc
import yaml
import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# Enable Optuna logging output
optuna.logging.set_verbosity(optuna.logging.INFO)

PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_catboost_tuning(n_trials=20):
    print("========== Starting CatBoost Hyperparameter Search ==========")
    config = load_config()

    processed_train_path = config["paths"]["processed_train_output"]

    print(f"Loading engineered feature database from: {processed_train_path}")
    df = pd.read_parquet(processed_train_path)

    print("Sorting dataset chronologically by TransactionDT...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Separate targets safely
    X = df.drop(columns=["isFraud", "TransactionDT"], errors="ignore")
    y = df["isFraud"].astype(np.int8)

    del df
    gc.collect()

    # 1. Load active feature selection registry
    yaml_drop_path = "models/artifacts/optimized_dropped_features.yaml"
    if os.path.exists(yaml_drop_path):
        print(f" -> Loading active feature selection registry from: {yaml_drop_path}")
        with open(yaml_drop_path, "r") as f:
            drop_data = yaml.safe_load(f)
        selected_features = drop_data.get("selected_features", [])

        if selected_features:
            print(f" -> Active Feature Selection Engaged: Keeping {len(selected_features)} features.")
            keep_cols = [c for c in selected_features if c in X.columns]
            X = X[keep_cols]
    else:
        print(f" -> Warning: {yaml_drop_path} not found. Proceeding with all base features.")

    # Drop volatile interaction layers
    X.drop(columns=["P_email_ProductCD", "R_email_ProductCD"], errors="ignore", inplace=True)

    # Downcast floats to optimize execution speed and RAM
    float64_cols = X.select_dtypes(include=["float64"]).columns.tolist()
    if float64_cols:
        X[float64_cols] = X[float64_cols].astype(np.float32)

    # CatBoost expects string/object representations for categorical features
    cat_cols = X.select_dtypes(include=["category", "object", "string"]).columns.tolist()
    if "ProductCD" not in cat_cols and "ProductCD" in X.columns:
        cat_cols.append("ProductCD")

    print(f"Formatting {len(cat_cols)} categorical columns as strings for CatBoost...")
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("missing")

    print(f"Dataset Shape: {X.shape} | Categorical Features: {len(cat_cols)}")

    # -------------------------------------------------------------------------
    # PRE-SPLIT CV INDEXES (Fast, Low Memory Footprint)
    # -------------------------------------------------------------------------
    print("Pre-splitting Cross-Validation folds...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_splits = []

    for train_idx, val_idx in tscv.split(X):
        cv_splits.append((
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[val_idx], y.iloc[val_idx]
        ))

    # Clean base variables to save RAM
    del X, y
    gc.collect()

    # -------------------------------------------------------------------------
    # OPTUNA OBJECTIVE FUNCTION
    # -------------------------------------------------------------------------
    def objective(trial):
        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "task_type": "CPU", 
            "thread_count": -1,
            "verbose": False,
            
            # Fast Feature Discretization & Subsampling
            "border_count": 128,
            "one_hot_max_size": 20,
            "rsm": 0.8,
            
            # Search Space
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
            "depth": trial.suggest_int("depth", 4, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }

        print(f"\n--- [Trial {trial.number + 1}/{n_trials}] Testing Params: lr={params['learning_rate']:.4f}, depth={params['depth']}, l2={params['l2_leaf_reg']:.3f} ---")

        fold_scores = []

        for fold_idx, (X_tr, y_tr, X_val, y_val) in enumerate(cv_splits, start=1):
            model = CatBoostClassifier(**params, iterations=700)
            
            # Fit directly using DataFrames & cat_features list
            model.fit(
                X_tr, y_tr,
                cat_features=cat_cols,
                eval_set=(X_val, y_val),
                early_stopping_rounds=30,
                verbose=False
            )

            preds = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, preds)
            fold_scores.append(auc_score)

            best_iter = model.get_best_iteration()
            print(f"  Fold {fold_idx}/5 ROC-AUC: {auc_score:.5f} (Best Iter: {best_iter})")

            trial.report(auc_score, fold_idx)
            if trial.should_prune():
                print(f"  --> Trial {trial.number + 1} Pruned at Fold {fold_idx}")
                raise optuna.TrialPruned()

        mean_oot_auc = np.mean(fold_scores[-2:])
        print(f"  ==> Trial {trial.number + 1} Finished | OOT Mean ROC-AUC (Folds 4 & 5): {mean_oot_auc:.5f}")
        return mean_oot_auc

    print(f"\n⚡ Starting Optuna Search Loop ({n_trials} Trials)...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    print("\n================ CatBoost Tuning Optimization Complete ================")
    print(f"Best Trial OOT ROC-AUC: {study.best_value:.5f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "task_type": "CPU",
        "thread_count": -1,
        "verbose": False,
        "border_count": 128,
        "one_hot_max_size": 20,
        **study.best_params,
    }

    output_params_path = "models/artifacts/best_catboost_params.yaml"
    os.makedirs(os.path.dirname(output_params_path), exist_ok=True)

    with open(output_params_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    print(f"\nSuccessfully saved optimal CatBoost hyperparameters to: '{output_params_path}'!")


if __name__ == "__main__":
    run_catboost_tuning(n_trials=20)
