import os
import sys
import yaml
import gc
import pandas as pd

PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from data.identity_features import process_and_standardize_identity
from data.transaction_features import run_transaction_feature_pipeline
from data.vesta_feature_handling import VestaVColumnCompressor


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def process_inference_data():
    print("========== Starting High-Performance Hybrid Inference Pipeline ==========")
    config = load_config()

    train_trans_path = config["paths"]["train_transaction"]
    test_trans_path = config["paths"]["test_transaction"]
    test_id_path = config["paths"]["test_identity"]
    v_compressor_artifact_path = config["paths"]["v_compressor_artifact"]

    # =========================================================================
    # STEP 1: DYNAMIC LIGHTWEIGHT COLUMN LOADING (COMPLETELY EXCLUDE ISFRAUD)
    # =========================================================================
    print("Analyzing train schema to isolate lightweight tracking columns...")
    train_sample = pd.read_csv(train_trans_path, nrows=1)
    
    lightweight_cols = [
        c for c in train_sample.columns 
        if not (c.startswith("V") and c[1:].isdigit()) and c != "isFraud"
    ]

    print("Loading lightweight train tracks...")
    train_lightweight = pd.read_csv(train_trans_path, usecols=lightweight_cols)
    train_lightweight["is_test_set"] = 0 

    print("Loading lightweight test tracks...")
    test_lightweight = pd.read_csv(test_trans_path, usecols=lightweight_cols)
    test_lightweight["is_test_set"] = 1 

    # =========================================================================
    # STEP 2: STITCH TIMELINES & RUN COMPUTATION LOOP
    # =========================================================================
    print("Stitching lightweight timelines together...")
    combined_df = pd.concat([train_lightweight, test_lightweight], axis=0, ignore_index=True)
    
    del train_lightweight, test_lightweight
    gc.collect()

    print("🧹 De-fragmenting combined tracking frame...")
    combined_df = combined_df.copy()
    gc.collect()

    # Calculate rolling statistics and expanding user behavior tracks over time
    combined_df = run_transaction_feature_pipeline(combined_df)

    # =========================================================================
    # STEP 3: EXTRACT TEST SET ROWS 
    # =========================================================================
    print("Slicing engineered historical tracks back into test space...")
    inference_df = combined_df[combined_df["is_test_set"] == 1].copy()
    
    del combined_df
    gc.collect()

    inference_df.drop(columns=["is_test_set"], inplace=True, errors="ignore")

    # =========================================================================
    # STEP 4: RE-MERGE HEAVY VESTA PAYLOADS (TEST ROWS ONLY)
    # =========================================================================
    print("Loading raw test Vesta columns (Test set rows only)...")
    full_raw_test = pd.read_csv(test_trans_path)
    expected_test_rows = len(full_raw_test)

    heavy_v_cols = [c for c in full_raw_test.columns if (c.startswith("V") and c[1:].isdigit())]
    heavy_v_df = full_raw_test[["TransactionID"] + heavy_v_cols]

    print("Merging heavy V-columns back onto processed test rows...")
    inference_df = inference_df.merge(heavy_v_df, on="TransactionID", how="left")
    
    del full_raw_test, heavy_v_df
    gc.collect()

    if len(inference_df) != expected_test_rows:
        raise ValueError(f"Critical Error: Row count mismatch! Expected {expected_test_rows}, got {len(inference_df)}")

    # =========================================================================
    # STEP 5: PRODUCING IDENTITY PAYLOAD AND COMPRESSING BLOCKS
    # =========================================================================
    if os.path.exists(test_id_path):
        print("Loading local raw test identity data...")
        test_id = pd.read_csv(test_id_path)
        
        # FIX: Handle Kaggle's hyphenated column naming convention in test_identity.csv
        test_id.columns = [c.replace("-", "_") for c in test_id.columns]
        
        test_id = process_and_standardize_identity(test_id)
        inference_df = inference_df.merge(test_id, on="TransactionID", how="left")
        del test_id
        gc.collect()
    else:
        print(" -> Identity data missing on disk. Generating empty tracking matrices.")
        inference_df = process_and_standardize_identity(inference_df)

    if os.path.exists(v_compressor_artifact_path):
        print(f"Restoring pre-fitted PCA compressor parameters from: {v_compressor_artifact_path}")
        v_compressor = VestaVColumnCompressor.load(v_compressor_artifact_path)
        inference_df = v_compressor.transform(inference_df)
    else:
        raise FileNotFoundError(f"Missing PCA compressor file: {v_compressor_artifact_path}")

    # =========================================================================
    # STEP 6: FORMAT BASE TEMPLATE FOR THE INFERENCE ENGINE
    # =========================================================================
    submission_base = pd.DataFrame({
        "TransactionID": inference_df["TransactionID"]
    })

    # Drop operational keys to match training dimension signatures exactly
    drop_cols = ["uid_card", "TransactionID", "isFraud"]
    inference_df.drop(columns=drop_cols, inplace=True, errors="ignore")

    print(f"========== Hybrid Test Pipeline Complete! Shape: {inference_df.shape} ==========")
    return inference_df, submission_base


if __name__ == "__main__":
    X_test, sub_df = process_inference_data()
