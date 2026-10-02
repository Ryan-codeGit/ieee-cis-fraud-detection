import numpy as np
import pandas as pd


def process_and_standardize_identity(df):
    print("Standardizing high-cardinality machine and hardware attributes...")

    # Fix: Deploy explicit anchored boundary markers to isolate genuine IE footprints
    if "id_31" in df.columns:
        df["id_31"] = df["id_31"].fillna("missing").astype(str).str.lower()

        df.loc[df["id_31"].str.contains("chrome"), "id_31"] = "chrome"
        df.loc[df["id_31"].str.contains("safari"), "id_31"] = "safari"
        df.loc[df["id_31"].str.contains("firefox"), "id_31"] = "firefox"
        df.loc[
            df["id_31"].str.contains(r"\bie\b|msie|trident|edge"), "id_31"
        ] = "ie_edge"
        df.loc[df["id_31"].str.contains("opera"), "id_31"] = "opera"

        allowed_browsers = [
            "chrome",
            "safari",
            "firefox",
            "ie_edge",
            "opera",
            "missing",
        ]
        df.loc[~df["id_31"].isin(allowed_browsers), "id_31"] = "other_browser"
        df["id_31"] = df["id_31"].astype("category")

    if "id_30" in df.columns:
        df["id_30"] = df["id_30"].fillna("missing").astype(str).str.lower()

        df.loc[df["id_30"].str.contains("windows"), "id_30"] = "windows"
        df.loc[df["id_30"].str.contains("mac|ios"), "id_30"] = "apple"
        df.loc[df["id_30"].str.contains("android"), "id_30"] = "android"
        df.loc[df["id_30"].str.contains("linux"), "id_30"] = "linux"

        allowed_os = ["windows", "apple", "android", "linux", "missing"]
        df.loc[~df["id_30"].isin(allowed_os), "id_30"] = "other_os"
        df["id_30"] = df["id_30"].astype("category")

    if "id_33" in df.columns:
        df["id_33"] = df["id_33"].fillna("0x0").astype(str)
        split_res = df["id_33"].str.split("x", expand=True)

        df["screen_area"] = pd.to_numeric(
            split_res[0], errors="coerce"
        ).fillna(0) * pd.to_numeric(split_res[1], errors="coerce").fillna(0)
        df["screen_area"] = df["screen_area"].replace(0, -1)
        df.drop(columns=["id_33"], inplace=True)

    if "DeviceInfo" in df.columns:
        df["DeviceInfo"] = df["DeviceInfo"].fillna("missing")
        device_counts = df["DeviceInfo"].value_counts()
        rare_devices = device_counts[device_counts < 100].index

        df.loc[df["DeviceInfo"].isin(rare_devices), "DeviceInfo"] = (
            "other_device"
        )
        df["DeviceInfo"] = df["DeviceInfo"].astype("category")

# Systematically process ID features by rule:
    # id_01 to id_12 are continuous numericals
    # id_13 and above are categorical features
    id_cols = [c for c in df.columns if c.startswith("id_")]
    for col in id_cols:
        try:
            num = int(col.split("_")[1])
            if num <= 12:
                # Continuous numericals
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)
            else:
                # Categoricals (id_13 and higher)
                if col not in ["id_30", "id_31"]:  # Skip if processed above
                    # FIX: Cast to str FIRST to prevent PyArrow float/str conversion errors
                    df[col] = df[col].fillna("missing").astype(str).astype("category")
        except ValueError:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(-1)
            else:
                df[col] = df[col].fillna("missing").astype(str).astype("category")
    return df
