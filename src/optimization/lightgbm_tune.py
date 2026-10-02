import os
import sys
import gc
import yaml
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# Enable Optuna logging output
optuna.logging.set_verbosity(optuna.logging.INFO)

PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_hyperparameter_tuning(n_trials=30):
    print("========== Starting LightGBM Hyperparameter Search ==========")
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

    # Secure and format categorical parameters
    cat_cols = X.select_dtypes(include=["category", "object", "string"]).columns.tolist()
    if "ProductCD" not in cat_cols and "ProductCD" in X.columns:
        cat_cols.append("ProductCD")

    # Pre-compile global categorical mappings
    print("Pre-compiling global categorical maps...")
    categorical_dtypes_map = {}
    for col in cat_cols:
        clean_series = X[col].astype(str).replace(["nan", "None", "<NA>"], np.nan)
        unique_categories = sorted([x for x in clean_series.dropna().unique()])
        categorical_dtypes_map[col] = pd.CategoricalDtype(categories=unique_categories, ordered=False)
        X[col] = clean_series.astype(categorical_dtypes_map[col])

    print(f"Dataset Shape: {X.shape} | Categorical Features: {len(cat_cols)}")

    # -------------------------------------------------------------------------
    # PRE-BUILD CV FOLDS & LIGHTGBM DATASETS
    # -------------------------------------------------------------------------
    print("Pre-building Cross-Validation Datasets in memory...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_datasets = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]

        for col in cat_cols:
            if col in X_tr.columns:
                X_tr[col] = X_tr[col].astype(str).replace(["nan", "None", "<NA>"], np.nan)
                X_tr[col] = X_tr[col].astype(categorical_dtypes_map[col])
            if col in X_val.columns:
                X_val[col] = X_val[col].astype(str).replace(["nan", "None", "<NA>"], np.nan)
                X_val[col] = X_val[col].astype(categorical_dtypes_map[col])

        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols, free_raw_data=False)

        cv_datasets.append((train_data, val_data, y_val, X_val))

    # Clear base DataFrame from system memory
    del X, y
    gc.collect()

    # Objective function for Optuna trial evaluation
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": 42,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        print(f"\n--- [Trial {trial.number + 1}/{n_trials}] Testing Params: lr={params['learning_rate']:.4f}, leaves={params['num_leaves']}, depth={params['max_depth']} ---")

        fold_scores = []

        for fold_idx, (train_data, val_data, y_val, X_val) in enumerate(cv_datasets, start=1):
            model = lgb.train(
                params,
                train_data,
                num_boost_round=800,
                valid_sets=[train_data, val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            auc_score = roc_auc_score(y_val, preds)
            fold_scores.append(auc_score)

            print(f"  Fold {fold_idx}/5 ROC-AUC: {auc_score:.5f} (Best Iter: {model.best_iteration})")

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

    print("\n================ Tuning Optimization Complete ================")
    print(f"Best Trial OOT ROC-AUC: {study.best_value:.5f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
        **study.best_params,
    }

    output_params_path = "models/artifacts/best_lightgbm_params.yaml"
    os.makedirs(os.path.dirname(output_params_path), exist_ok=True)

    with open(output_params_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    print(f"\nSuccessfully saved optimal hyperparameters to: '{output_params_path}'!")


if __name__ == "__main__":
    run_hyperparameter_tuning(n_trials=30)
