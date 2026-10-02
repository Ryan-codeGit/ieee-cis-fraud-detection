import gc
import os
import sys
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Ensure src import path is available
PROJECT_ROOT = os.getcwd()
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from train.lightgbm_train import preprocess_data


def run_feature_optimization():
    print("==================================================")
    print("=== STARTING 3-STEP FEATURE OPTIMIZATION ENGINE ===")
    print("==================================================\n")

    # 1. Load Data
    print("Loading engineered features dataset...")
    df = pd.read_parquet('data/processed/train_engineered.parquet')
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    # Preprocess
    df, feature_cols, cat_cols = preprocess_data(df)

    # Isolate Training & Validation Sets (80/20 Chronological Time Split)
    split_idx = int(len(df) * 0.80)

    X_tr = df[feature_cols].iloc[:split_idx].copy()
    y_tr = df['isFraud'].iloc[:split_idx].astype(int)

    X_val = df[feature_cols].iloc[split_idx:].copy()
    y_val = df['isFraud'].iloc[split_idx:].astype(int)

    del df
    gc.collect()

    initial_feature_count = len(feature_cols)
    current_features = list(feature_cols)

    # Train baseline model to obtain gain feature importances
    print("\n--- Training Fast Baseline Model for Initial Diagnostics ---")
    active_cats = [c for c in cat_cols if c in current_features]
    
    train_data = lgb.Dataset(
        X_tr[current_features],
        label=y_tr,
        categorical_feature=active_cats,
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_val[current_features],
        label=y_val,
        reference=train_data,
        categorical_feature=active_cats,
        free_raw_data=False,
    )

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 45,
        'max_depth': 8,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    base_val_preds = model.predict(X_val[current_features])
    base_auc = roc_auc_score(y_val, base_val_preds)
    print(f"Baseline Validation ROC-AUC: {base_auc:.5f}")

    # -------------------------------------------------------------
    # STEP 1: GAIN PRUNING (Drop Zero Gain)
    # -------------------------------------------------------------
    print("\n==================================================")
    print("=== STEP 1: GAIN PRUNING (Zero-Gain Removal) ===")
    print("==================================================")

    importance_df = pd.DataFrame({
        'feature': model.feature_name(),
        'gain': model.feature_importance(importance_type='gain'),
    })

    zero_gain_cols = importance_df[importance_df['gain'] == 0]['feature'].tolist()
    current_features = [c for c in current_features if c not in zero_gain_cols]

    print(f"Features with 0 Gain: {len(zero_gain_cols)}")
    print(
        f"Step 1 Complete -> Remaining Features: {len(current_features)}"
        f" / {initial_feature_count}"
    )

    # -------------------------------------------------------------
    # STEP 2: CORRELATION PRUNING (> 0.98 Pearson Correlation)
    # -------------------------------------------------------------
    print("\n==================================================")
    print("=== STEP 2: CORRELATION PRUNING (Threshold > 0.98) ===")
    print("==================================================")

    num_features_remaining = [
        c for c in current_features if c not in cat_cols
    ]
    print(f"Computing correlation matrix across {len(num_features_remaining)} numerical features...")

    corr_matrix = X_tr[num_features_remaining].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    collinear_drops = [
        col for col in upper_tri.columns if any(upper_tri[col] > 0.98)
    ]
    current_features = [c for c in current_features if c not in collinear_drops]

    print(f"Identified Collinear Features (>0.98): {len(collinear_drops)}")
    print(
        f"Step 2 Complete -> Remaining Features: {len(current_features)}"
        f" / {initial_feature_count}"
    )

    # -------------------------------------------------------------
    # STEP 3: PERMUTATION IMPORTANCE CHECK
    # -------------------------------------------------------------
    print("\n==================================================")
    print("=== STEP 3: PERMUTATION IMPORTANCE CHECK ===")
    print("==================================================")

    active_cats = [c for c in cat_cols if c in current_features]
    train_data_pruned = lgb.Dataset(
        X_tr[current_features],
        label=y_tr,
        categorical_feature=active_cats,
        free_raw_data=False,
    )
    val_data_pruned = lgb.Dataset(
        X_val[current_features],
        label=y_val,
        reference=train_data_pruned,
        categorical_feature=active_cats,
        free_raw_data=False,
    )

    model_pruned = lgb.train(
        params,
        train_data_pruned,
        num_boost_round=300,
        valid_sets=[train_data_pruned, val_data_pruned],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    clean_base_preds = model_pruned.predict(X_val[current_features])
    clean_base_auc = roc_auc_score(y_val, clean_base_preds)
    print(f"Pruned Baseline Validation ROC-AUC: {clean_base_auc:.5f}")

    noisy_features = []
    np.random.seed(42)

    print("Evaluating permutation impact per feature on validation set...")
    for idx, col in enumerate(current_features, start=1):
        X_val_permuted = X_val[current_features].copy()
        
        # Preserve original categorical structure during permutation
        if col in active_cats:
            orig_type = X_val_permuted[col].dtype
            shuffled_vals = np.random.permutation(X_val_permuted[col].values)
            X_val_permuted[col] = pd.Series(shuffled_vals, index=X_val_permuted.index).astype(orig_type)
        else:
            X_val_permuted[col] = np.random.permutation(X_val_permuted[col].values)

        perm_preds = model_pruned.predict(X_val_permuted)
        perm_auc = roc_auc_score(y_val, perm_preds)
        auc_drop = clean_base_auc - perm_auc

        if auc_drop <= 0:
            noisy_features.append(col)

        if idx % 50 == 0 or idx == len(current_features):
            print(
                f" Evaluated {idx}/{len(current_features)} features | Current Noisy"
                f" Count: {len(noisy_features)}"
            )

    final_selected_features = [
        c for c in current_features if c not in noisy_features
    ]

    print("\n==================================================")
    print("=== FEATURE OPTIMIZATION SUMMARY ===")
    print("==================================================")
    print(f"Initial Features:               {initial_feature_count}")
    print(f"Zero Gain Features Removed:     {len(zero_gain_cols)}")
    print(f"Collinear Features Removed:     {len(collinear_drops)}")
    print(f"Noisy Permutation Features:     {len(noisy_features)}")
    print(f"Final Selected Features:        {len(final_selected_features)}")
    print("==================================================")

    # Save optimized feature manifest
    os.makedirs('models/artifacts', exist_ok=True)
    output_yaml = 'models/artifacts/optimized_dropped_features.yaml'

    with open(output_yaml, 'w') as f:
        yaml.dump({'selected_features': final_selected_features}, f)

    print(f"\nOptimized feature list successfully saved to '{output_yaml}'!")


if __name__ == '__main__':
    run_feature_optimization()
