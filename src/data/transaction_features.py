import numpy as np
import pandas as pd


class SmoothedTargetEncoder:
    """Computes target encodings with additive Bayesian smoothing inside cross-

    validation folds to eliminate out-of-fold data leakage.
    """

    def __init__(self, cols, smoothing=20.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mapping = {}
        self.global_mean = 0.0

    def fit(self, df, target_col):
        self.global_mean = df[target_col].astype(float).mean()
        for col in self.cols:
            stats = df.groupby(col)[target_col].agg(["count", "mean"])
            smoothed = (
                stats["count"] * stats["mean"]
                + self.smoothing * self.global_mean
            ) / (stats["count"] + self.smoothing)
            self.mapping[col] = smoothed.to_dict()
        return self

    def transform(self, df):
        out = df.copy()
        for col in self.cols:
            if col in out.columns:
                out[f"{col}_target_enc"] = (
                    out[col].map(self.mapping[col]).fillna(self.global_mean)
                )
        return out


def process_time_and_amount(df):
    print("Engineering time coordinates and transaction magnitudes...")
    # Clean up floating-point noise from pennies
    df["TransactionAmt_cents"] = (df["TransactionAmt"] % 1).round(4)
    # Extract cyclical time signatures
    df["Transaction_hour"] = (df["TransactionDT"] // 3600) % 24
    df["Transaction_day"] = (df["TransactionDT"] // 86400) % 7
    return df


def normalize_d_features(df):
    print("Normalizing dynamic account age timelines...")
    for i in range(1, 16):
        col = f"D{i}"
        if col in df.columns:
            # Reconstruct absolute historical day tracking positions
            df[f"{col}_normalized"] = df["TransactionDT"] / 86400 - df[col]
            # Safeguard: Explicitly fill normalized values with -1 to prevent string UID poisoning
            df[f"{col}_normalized"] = df[f"{col}_normalized"].fillna(-1)
            # Retain standard filled state for raw baseline arrays
            df[col] = df[col].fillna(-1)
    return df


def engineer_advanced_behavioral_features(df):
    print("Executing sequential tracking and dynamic expanding risk engines...")

    # 1. Establish the master calendar day coordinate
    df["date_day"] = df["TransactionDT"] / 86400
    df["date_day_int"] = np.floor(df["date_day"]).astype(int)

    # 2. Build the structural baseline User ID (Card Bin + Birth Day)
    df["D1_normalized_round"] = np.round(df["D1_normalized"], 0)

    df["uid_card"] = (
        df["card1"].astype(str)
        + "_"
        + df["card2"].astype(str)
        + "_"
        + df["card3"].astype(str)
        + "_"
        + df["addr1"].astype(str)
        + "_"
        + df["D1_normalized_round"].astype(str)
    )

    # 3. Soft Identity Safeguards: EXPANDING D-column drift states
    # Using expanding().std() so a row only sees the variance of transactions up to that exact moment
    for d_num in [4, 10, 15]:
        norm_col = f"D{d_num}_normalized"
        if norm_col in df.columns:
            df[f"uid_{norm_col}_std"] = (
                df.groupby("uid_card")[norm_col]
                .transform(lambda x: x.expanding().std())
                .fillna(0)
            )
            df[f"uid_{norm_col}_mean"] = (
                df.groupby("uid_card")[norm_col]
                .transform(lambda x: x.expanding().mean())
                .fillna(-1)
            )

    # 4. Pricing Fingerprints & EXPANDING Historical Deviations
    # Only calculate the median of amounts seen *so far* for this user
    uid_expanding_medians = (
        df.groupby("uid_card")["TransactionAmt"]
        .transform(lambda x: x.expanding().median())
    )
    df["amt_dev_from_expanding_median"] = df["TransactionAmt"] / (
        uid_expanding_medians + 1e-5
    )

    df["is_round_number"] = (df["TransactionAmt_cents"] == 0.0000).astype(int)
    df["is_95_cents"] = (df["TransactionAmt_cents"] == 0.9500).astype(int)

    # Cumulative running tracking counts passed forward across time
    df["uid_cum_round_nums"] = df.groupby("uid_card")[
        "is_round_number"
    ].cumsum()
    df["uid_cum_95_cents"] = df.groupby("uid_card")["is_95_cents"].cumsum()

    print("Computing regional amount deviation indices...")
    addr1_medians = df.groupby("addr1")["TransactionAmt"].transform(
        lambda x: x.expanding().median()
    )
    addr1_stds = (
        df.groupby("addr1")["TransactionAmt"]
        .transform(lambda x: x.expanding().std())
        .fillna(1)
    )
    df["addr1_amt_dev_median"] = df["TransactionAmt"] - addr1_medians
    df["addr1_amt_dev_std"] = df["TransactionAmt"] / (addr1_stds + 1e-6)

    # 5. Spatial Memory Processing: Strict Cumulative Counts (Replacing all total counts)
    df["uid_card_cumcount"] = df.groupby("uid_card").cumcount() + 1

    df["uid_amt_key"] = (
        df["uid_card"] + "__" + df["TransactionAmt"].astype(str)
    )
    # The current row knows exactly how many times this amount has been hit in the past
    df["amt_historical_cumcount"] = df.groupby("uid_amt_key").cumcount() + 1

    # Track time interval variation to isolate chaotic low-and-slow patterns dynamically
    df["days_since_last_same_amt"] = df.groupby("uid_amt_key")[
        "date_day"
    ].diff()

    df["amt_time_delta_expanding_std"] = (
        df.groupby("uid_amt_key")["days_since_last_same_amt"]
        .transform(lambda x: x.expanding().std())
        .fillna(-1)
    )
    df["amt_time_delta_expanding_mean"] = (
        df.groupby("uid_amt_key")["days_since_last_same_amt"]
        .transform(lambda x: x.expanding().mean())
        .fillna(-1)
    )

    # 6. Compounded Multiplicative Interaction Rating (Completely Historical)
    df["uid_day_amt_key"] = (
        df["uid_card"]
        + "__"
        + df["date_day_int"].astype(str)
        + "__"
        + df["TransactionAmt"].astype(str)
    )
    # Cumulative velocity: 1st time today = 1, 4th time today = 4
    df["same_day_amt_velocity_cum"] = (
        df.groupby("uid_day_amt_key").cumcount() + 1
    )

    # Highly optimized EXPANDING nunique for C13
    # Marks a 1 every time a NEW C13 value appears for this specific uid_card
    df["is_new_c13"] = (~df.duplicated(subset=["uid_card", "C13"])).astype(int)
    df["uid_c13_nunique_expanding"] = df.groupby("uid_card")[
        "is_new_c13"
    ].cumsum()

    # Risk rating strictly based on events that have happened UP TO this millisecond
    df["signal_c13_discrepancy"] = 1.0 - (
        df["uid_c13_nunique_expanding"] / df["uid_card_cumcount"]
    )
    df["signal_c13_discrepancy"] = df["signal_c13_discrepancy"].clip(
        lower=0.0, upper=1.0
    )

    # If same-day velocity is 1, risk is 0. As it climbs, risk scales proportionally to total history.
    df["signal_velocity_risk"] = (
        df["same_day_amt_velocity_cum"] - 1
    ) / df["uid_card_cumcount"]

    df["multiplicative_bot_risk_rating"] = (
        df["signal_velocity_risk"] * df["signal_c13_discrepancy"]
    )

    # Drop intermediate keys to keep memory clear
    cleanup_keys = [
        "date_day",
        "date_day_int",
        "D1_normalized_round",
        "uid_amt_key",
        "uid_day_amt_key",
        "days_since_last_same_amt",
        "is_new_c13",
    ]
    df.drop(columns=cleanup_keys, inplace=True, errors="ignore")

    return df


def process_demographics_and_distances_pure_anomaly(df):
    print("Mapping pure categorical anomaly geography flags...")

    # Both columns are already filled with -1, avoiding dropping keys during groupby
    card3_to_addr2_mode = (
        df.groupby("card3")["addr2"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else -1)
        .to_dict()
    )

    df["expected_addr2"] = df["card3"].map(card3_to_addr2_mode)
    df["geo_mismatch_flag"] = (df["expected_addr2"] != df["addr2"]).astype(int)

    df["dist1"] = df["dist1"].fillna(-1)
    df["dist2_has_data"] = df["dist2"].notna().astype(int)
    df["dist2"] = df["dist2"].fillna(-1)

    df.drop(columns=["expected_addr2"], inplace=True)
    return df


def process_c_m_features(df):
    print("Imputing standard counting variables and transaction tracks...")
    c_cols = [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]
    for col in c_cols:
        df[col] = df[col].fillna(-1)

    # Convert M features cleanly to category format with string conversions
    m_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]
    for col in m_cols:
        df[col] = df[col].fillna("missing").astype(str).astype("category")
    return df


def process_emails_(df):
    print("Formatting communications layout vectors and variance maps...")
    
    # 1. Fill missing values as strings first
    p_email = df["P_emaildomain"].fillna("missing").astype(str)
    r_email = df["R_emaildomain"].fillna("missing").astype(str)

    # 2. Compute the mismatch vector while they are standard string Series
    df["email_mismatch_flag"] = (p_email != r_email).astype(int)

    # 3. Cast to categorical types after the comparison
    df["P_emaildomain"] = p_email.astype("category")
    df["R_emaildomain"] = r_email.astype("category")
    
    return df

def run_transaction_feature_pipeline(df):
    """Orchestrates the complete feature engineering phase in the correct
    chronological and logical order.
    """
    print("⚡ Starting Transaction Feature Engineering Pipeline...")

    # 1. Force strict chronological timeline order
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # 2. Compute counting features first to ensure counters like C13 are ready
    df = process_c_m_features(df)

    # 3. Standardize and cast categorical attributes cleanly to 'category'
    if "ProductCD" in df.columns:
        df["ProductCD"] = (
            df["ProductCD"].fillna("missing").astype(str).astype("category")
        )

    for col in ["card1", "card2", "card3", "card5"]:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    for col in ["card4", "card6"]:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str).astype("category")

    for col in ["addr1", "addr2"]:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # 4. Core sequential transformations
    df = process_time_and_amount(df)
    df = normalize_d_features(df)
    df = engineer_advanced_behavioral_features(df)
    df = process_demographics_and_distances_pure_anomaly(df)
    df = process_emails_(df)

    # Ensure high-cardinality identity categoricals are cleanly structured as category
    for col in ["card1", "card2", "card3", "addr1", "addr2"]:
        if col in df.columns:
            df[f"{col}_cat"] = (
                df[col].fillna("-1").astype(str).astype("category")
            )

    print("✨ Pipeline Complete! Data is ready for model training.")
    return df
