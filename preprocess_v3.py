import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("code-rush-26-ml-module")
OUTPUT_DIR = Path("processed_data_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading data...")
train_raw = pd.read_csv(DATA_DIR / "Coderush-26-ML-Train.csv")
test_raw  = pd.read_csv(DATA_DIR / "Coderush-26-ML-test.csv")

# ────────────────────────────────────────────────────────────────
# 1. Drop constant / leakage / ID columns (same as before)
# ────────────────────────────────────────────────────────────────
DROP_COLS = [
    "processing_flag", "survey_year", "currency_code", "poverty_line_usd",
    "is_adult_flag", "person_idx", "annual_hours_est", "interviewer_id"
]
train_raw.drop(columns=[c for c in DROP_COLS if c in train_raw.columns], inplace=True)
test_raw.drop(columns=[c for c in DROP_COLS if c in test_raw.columns], inplace=True)

# Keep bag_id for grouping, will drop later
train_bag_ids = train_raw["bag_id"].values
test_bag_ids  = test_raw["bag_id"].values

# Target mapping
label_map = {"lower": 0, "middle": 1, "upper": 2}
train_raw["label_int"] = train_raw["label"].map(label_map)

# ────────────────────────────────────────────────────────────────
# 2. Bag‑level aggregation – one row per bag
# ────────────────────────────────────────────────────────────────
def aggregate_bag(df, is_train=True):
    """Return bag-level DataFrame with all features."""
    # Start with basic bag stats
    bag = df.groupby("bag_id").agg(
        bag_size = ("bag_size", "first"),
        bag_education_mean = ("education_num", "mean"),
        bag_education_std = ("education_num", "std"),
        bag_hours_mean = ("hours_per_week", "mean"),
        bag_hours_std = ("hours_per_week", "std"),
        bag_age_mean = ("year_of_birth", lambda x: (1994 - x).mean()),
        bag_age_std = ("year_of_birth", lambda x: (1994 - x).std()),
        bag_capital_gain_max = ("capital_gain", "max"),
        bag_capital_gain_mean = ("capital_gain", "mean"),
        bag_capital_loss_max = ("capital_loss", "max"),
        bag_capital_loss_mean = ("capital_loss", "mean"),
        bag_net_capital_max = ("net_capital_asset", "max"),
        bag_net_capital_mean = ("net_capital_asset", "mean"),
        bag_duration_mean = ("survey_duration_mins", "mean"),
        bag_unique_relationships = ("relationship", "nunique"),
        bag_unique_occupations = ("occupation", "nunique"),
        bag_unique_workclass = ("workclass", "nunique"),
        bag_has_capital_activity = ("capital_activity_flag", "max"),
    ).reset_index()

    # Add range features (v1)
    edu_range = df.groupby("bag_id")["education_num"].agg(lambda x: x.max() - x.min())
    bag["bag_education_range"] = edu_range.values
    age_range = df.groupby("bag_id")["year_of_birth"].agg(lambda x: x.max() - x.min())
    bag["bag_age_range"] = age_range.values
    hours_range = df.groupby("bag_id")["hours_per_week"].agg(lambda x: x.max() - x.min())
    bag["bag_hours_range"] = hours_range.values

    # Fill NaN std with 0 (single‑member bags)
    bag["bag_education_std"] = bag["bag_education_std"].fillna(0)
    bag["bag_hours_std"] = bag["bag_hours_std"].fillna(0)

    # ─── V3 new simple features (6) ─────────────────────────────
    # a) low education ratio (<=9)
    low_ed = df.groupby("bag_id")["education_num"].apply(lambda x: (x <= 9).mean())
    bag["bag_low_ed_ratio"] = low_ed

    # b) full‑time ratio (>=35 hours)
    full_time = df.groupby("bag_id")["hours_per_week"].apply(lambda x: (x >= 35).mean())
    bag["bag_full_time_ratio"] = full_time

    # c) capital activity ratio (any gain/loss)
    cap_act = df.groupby("bag_id")["capital_activity_flag"].mean()
    bag["bag_capital_activity_ratio"] = cap_act

    # d) zero capital gain ratio
    zero_cap = df.groupby("bag_id")["capital_gain"].apply(lambda x: (x == 0).mean())
    bag["bag_zero_capital_ratio"] = zero_cap

    # e) dependency ratio (corrected)
    ft_count = df[df["hours_per_week"] >= 35].groupby("bag_id").size()
    bag_size = df.groupby("bag_id").size()
    dep_ratio = (bag_size - ft_count) / np.maximum(1, ft_count)
    bag["bag_dependency_ratio"] = bag["bag_id"].map(dep_ratio).fillna(bag_size / 1.0)  # fallback if no ft

    # f) proportion below bag education mean
    bag_means = df.groupby("bag_id")["education_num"].mean()
    df_with_mean = df.merge(bag_means.rename("bag_edu_mean"), on="bag_id")
    below_mean = df_with_mean.groupby("bag_id")["education_num"].apply(lambda x: (x < x.name).mean())
    bag["bag_education_vs_below"] = below_mean

    if is_train:
        # Add label (same for all rows in bag)
        label = df.groupby("bag_id")["label_int"].first()
        bag["label"] = bag["bag_id"].map(label)

    return bag

print("\nAggregating bags...")
train_bag = aggregate_bag(train_raw, is_train=True)
test_bag  = aggregate_bag(test_raw, is_train=False)

print(f"  Train bags: {len(train_bag)} | Test bags: {len(test_bag)}")

# ────────────────────────────────────────────────────────────────
# 3. Encode categorical features (target encoding per bag)
#    We use bag‑level mode of categorical columns, then target encode.
# ────────────────────────────────────────────────────────────────
# Compute most frequent category per bag for selected columns
CAT_COLS = ["occupation", "workclass", "education", "native_country", "race",
            "relationship", "marital_status", "sex", "interview_mode"]

def mode_per_bag(df, col):
    return df.groupby("bag_id")[col].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)

for col in CAT_COLS:
    if col in train_raw.columns:
        train_bag[f"bag_mode_{col}"] = train_bag["bag_id"].map(mode_per_bag(train_raw, col))
        test_bag[f"bag_mode_{col}"]  = test_bag["bag_id"].map(mode_per_bag(test_raw, col))

# Target encoding for these mode features (only on training bags)
for col in [f"bag_mode_{c}" for c in CAT_COLS if c != "label"]:
    if col in train_bag.columns:
        # Use label from train_bag (already present)
        global_mean = train_bag["label"].mean()
        col_means = train_bag.groupby(col)["label"].mean()
        col_counts = train_bag.groupby(col).size()
        smoothing = 10
        encoded = (col_means * col_counts + global_mean * smoothing) / (col_counts + smoothing)
        train_bag[f"{col}_target_enc"] = train_bag[col].map(encoded)
        test_bag[f"{col}_target_enc"] = test_bag[col].map(encoded).fillna(global_mean)
        # drop the raw mode column
        train_bag.drop(columns=[col], inplace=True)
        test_bag.drop(columns=[col], inplace=True)

# ────────────────────────────────────────────────────────────────
# 4. Log transform capital features (bag aggregates)
# ────────────────────────────────────────────────────────────────
CAP_AGG = ["bag_capital_gain_mean", "bag_capital_loss_mean", "bag_net_capital_mean",
           "bag_capital_gain_max", "bag_capital_loss_max", "bag_net_capital_max"]
for col in CAP_AGG:
    if col in train_bag.columns:
        train_bag[f"log_{col}"] = np.log1p(train_bag[col])
        test_bag[f"log_{col}"] = np.log1p(test_bag[col])
        # Optionally keep original? We'll keep both for now.

# ────────────────────────────────────────────────────────────────
# 5. Drop unused columns (IDs, bag_size from raw, etc.)
# ────────────────────────────────────────────────────────────────
drop_final = ["bag_id", "label"]  # keep label for now, will separate
if "bag_size_x" in train_bag.columns:
    drop_final.append("bag_size_x")
X_train = train_bag.drop(columns=[c for c in drop_final if c in train_bag.columns])
y_train = train_bag["label"]
X_test = test_bag.drop(columns=["bag_id"], errors="ignore")

# Ensure same columns
common_cols = sorted(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# Fill any remaining NaNs (should be none, but safe)
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_train.median(), inplace=True)  # use train median for test

# ────────────────────────────────────────────────────────────────
# 6. Save
# ────────────────────────────────────────────────────────────────
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False, header=["label"])
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
np.save(OUTPUT_DIR / "train_bag_ids.npy", train_bag_ids)
np.save(OUTPUT_DIR / "test_bag_ids.npy", test_bag_ids)

print(f"\nSaved to {OUTPUT_DIR}/")
print(f"  X_train shape: {X_train.shape} (bags × features)")
print(f"  y_train shape: {y_train.shape}")
print(f"  X_test shape:  {X_test.shape}")
print("\n✅ Ready for train_universal.py v3")