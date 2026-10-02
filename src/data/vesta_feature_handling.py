import gc
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


class VestaVColumnCompressor:

    def __init__(self, variance_threshold=0.95, chunk_size=50000):
        self.variance_threshold = variance_threshold
        self.chunk_size = chunk_size
        self.fitted_states = {}

    def fit(self, df):
        print("Fitting missingness-pattern mapped V-Column PCA transformers...")
        v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]

        # Low-overhead missingness detection
        null_signatures = df[v_cols].isna().to_numpy().T
        signature_hashes = [hash(tuple(row)) for row in null_signatures]

        signature_series = pd.Series(signature_hashes, index=v_cols)
        unique_pattern_groups = signature_series.groupby(signature_series).groups

        print(
            f"Detected {len(unique_pattern_groups)} independent missingness vectors."
        )

        for pattern_hash, col_indices in unique_pattern_groups.items():
            group_cols = list(col_indices)

            self.fitted_states[pattern_hash] = {
                "columns": group_cols,
                "scaler": None,
                "pca": None,
            }

            if len(group_cols) <= 1:
                continue

            print(
                f" -> Calibrating block pattern {abs(pattern_hash)} with {len(group_cols)} columns..."
            )

            # Isolate and downcast immediately
            filled_block = df[group_cols].fillna(-1).astype(np.float32)

            scaler = StandardScaler()
            scaled_block = scaler.fit_transform(filled_block)
            del filled_block
            gc.collect()

            n_samples, n_features = scaled_block.shape
            max_components = min(n_samples, n_features)

            # Limit components to a conservative pool to minimize output column size inflation
            target_comps = min(max_components, 15)
            ipca = IncrementalPCA(
                n_components=target_comps, batch_size=self.chunk_size
            )

            for i in range(0, scaled_block.shape[0], self.chunk_size):
                chunk = scaled_block[i : i + self.chunk_size]
                if len(chunk) >= ipca.n_components:
                    ipca.partial_fit(chunk)

            del scaled_block
            gc.collect()

            self.fitted_states[pattern_hash]["scaler"] = scaler
            self.fitted_states[pattern_hash]["pca"] = ipca

        return self

    def transform(self, df):
        print("Transforming V-Block spatial dimensions via destructive extraction...")

        # NO df.copy() here. We operate directly on the passed reference dataframe.
        for pattern_hash, state in self.fitted_states.items():
            group_cols = state["columns"]
            existing_cols = [c for c in group_cols if c in df.columns]

            if not existing_cols:
                continue

            if len(group_cols) <= 1:
                df[group_cols[0]] = df[group_cols[0]].fillna(-1).astype(np.float32)
                continue

            # Destructive Extraction: POP the columns entirely out of df to free up RAM
            filled_block = pd.DataFrame()
            for col in existing_cols:
                filled_block[col] = df.pop(col).fillna(-1).astype(np.float32)

            scaler = state["scaler"]
            ipca = state["pca"]

            # Transform the chunked block
            scaled_block = scaler.transform(filled_block)
            del filled_block
            gc.collect()

            pca_features = ipca.transform(scaled_block)
            del scaled_block
            gc.collect()

            # Append the dense components back directly into the dataframe
            for comp_idx in range(pca_features.shape[1]):
                feat_name = f"V_pattern_{abs(pattern_hash)}_PCA_{comp_idx+1}"
                df[feat_name] = pca_features[:, comp_idx].astype(np.float32)

            del pca_features
            gc.collect()

        return df

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved V-Column PCA Mapping parameters safely to: {filepath}")

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)
