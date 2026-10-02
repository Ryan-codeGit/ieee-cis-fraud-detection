import os
import sys
import yaml
import pandas as pd

PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from data.identity_features import process_and_standardize_identity
# Import the unified orchestration pipeline to handle execution order automatically
from data.transaction_features import run_transaction_feature_pipeline
# Import the stateful compressor class
from data.vesta_feature_handling import VestaVColumnCompressor


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_master_training_pipeline():
    print("========== Starting Training Feature Database Builder ==========")

    config = load_config()

    # Retrieve all operational directory targets directly from configuration maps
    train_trans_path = config["paths"]["train_transaction"]
    train_id_path = config["paths"]["train_identity"]
    output_processed_path = config["paths"]["processed_train_output"]
    v_compressor_artifact_path = config["paths"]["v_compressor_artifact"]

    # Secure directories exist on disk before running modifications
    os.makedirs(os.path.dirname(output_processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(v_compressor_artifact_path), exist_ok=True)

    print(f"Loading raw transaction data from: {train_trans_path}")
    train_df = pd.read_csv(train_trans_path)

    if os.path.exists(train_id_path):
        print(f"Loading raw identity telemetry from: {train_id_path}")
        train_id = pd.read_csv(train_id_path)
        train_df = train_df.merge(train_id, on="TransactionID", how="left")
        del train_id

    # Execute functional updates via the unified chronological feature pipeline
    train_df = run_transaction_feature_pipeline(train_df)
    train_df = process_and_standardize_identity(train_df)

    # Instantiate, fit, apply, and serialize your V-Block scaling and PCA axes
    v_compressor = VestaVColumnCompressor(variance_threshold=0.95)

    print("Fitting PCA mapping parameters on training variance structure...")
    v_compressor.fit(train_df)

    print("Transforming V-Block spatial dimensions...")
    train_df = v_compressor.transform(train_df)

    # Save the pipeline mapping state to disk
    v_compressor.save(v_compressor_artifact_path)

    # Clean high-cardinality processing anchors to avoid tree overfitting
    drop_cols = ["uid_card", "TransactionID"]
    train_df.drop(columns=drop_cols, inplace=True, errors="ignore")

    print(f"Exporting compressed feature database to: {output_processed_path}")
    train_df.to_parquet(output_processed_path)
    print("========== Dataset successfully generated for local training! ==========")


if __name__ == "__main__":
    run_master_training_pipeline()
