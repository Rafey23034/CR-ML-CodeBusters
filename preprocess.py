import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle

warnings.filterwarnings("ignore")

DATA_DIR = Path("code-rush-26-ml-module")
OUTPUT_DIR = Path("processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════
# CONFIG — based on EDA findings
# ════════════════════════════════════════════════════════

DROP_COLS = [
    "processing_flag",      # constant = 1.0
    "survey_year",          # constant = 1994
    "currency_code",        # constant = USD
    "poverty_line_usd",     # constant = 15141
    "is_adult_flag",        # near-constant (99.4% = 1)
    "bag_id",               # identifier
    "person_idx",           # position index within bag
    "annual_hours_est",     # redundant with hours_per_week
]

ORDINAL_COLS = {
    "education_tier": {"Primary": 0, "Secondary": 1, "Higher": 2}
}

TARGET_ENCODE_COLS = ["occupation", "workclass", "education", "native_country", "race"]
LABEL_ENCODE_COLS = ["relationship", "marital_status", "sex", "interview_mode"]

CAPITAL_COLS = ["capital_gain", "capital_loss", "net_capital_asset"]
OUTLIER_CLIP_PCT = 99.5


# ════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════

print("Loading data...")
train = pd.read_csv(DATA_DIR / "Coderush-26-ML-Train.csv")
test = pd.read_csv(DATA_DIR / "Coderush-26-ML-test.csv")

print(f"  Train: {train.shape} | Test: {test.shape}")
print(f"  Train bags: {train['bag_id'].nunique()} | Test bags: {test['bag_id'].nunique()}")

# ════════════════════════════════════════════════════════
# STEP 1: DROP LEAKAGE / NOISE / REDUNDANT COLUMNS
# ════════════════════════════════════════════════════════

print("\n[1] Dropping leakage/noise/redundant columns...")
# Keep bag_id for bag-level feature engineering, drop it later
cols_to_drop_step1 = [c for c in DROP_COLS if c != "bag_id" and c in train.columns]
train = train.drop(columns=cols_to_drop_step1)
test = test.drop(columns=[c for c in DROP_COLS if c != "bag_id" and c in test.columns])
print(f"  Dropped {len(cols_to_drop_step1)} columns: {cols_to_drop_step1}")
print(f"  Remaining columns: {train.shape[1]}")


# ════════════════════════════════════════════════════════
# STEP 2: BAG-LEVEL FEATURE ENGINEERING
# ════════════════════════════════════════════════════════

print("\n[2] Engineering bag-level features...")

def engineer_bag_features(df):
    bag_stats = df.groupby("bag_id").agg(
        bag_member_count=("bag_size", "first"),
        bag_education_mean=("education_num", "mean"),
        bag_education_std=("education_num", "std"),
        bag_age_mean=("year_of_birth", "mean"),
        bag_hours_mean=("hours_per_week", "mean"),
        bag_hours_std=("hours_per_week", "std"),
        bag_capital_gain_max=("capital_gain", "max"),
        bag_capital_gain_mean=("capital_gain", "mean"),
        bag_capital_loss_max=("capital_loss", "max"),
        bag_capital_loss_mean=("capital_loss", "mean"),
        bag_net_capital_max=("net_capital_asset", "max"),
        bag_net_capital_mean=("net_capital_asset", "mean"),
        bag_duration_mean=("survey_duration_mins", "mean"),
        bag_unique_relationships=("relationship", "nunique"),
        bag_unique_occupations=("occupation", "nunique"),
        bag_unique_workclass=("workclass", "nunique"),
        bag_has_capital_activity=("capital_activity_flag", "max"),
        bag_size=("bag_size", "first"),
    ).reset_index()

    # Range features computed after aggregation
    edu_range = df.groupby("bag_id")["education_num"].agg(["max", "min"])
    bag_stats["bag_education_range"] = edu_range["max"] - edu_range["min"]

    age_range = df.groupby("bag_id")["year_of_birth"].agg(["max", "min"])
    bag_stats["bag_age_range"] = age_range["max"] - age_range["min"]

    hours_range = df.groupby("bag_id")["hours_per_week"].agg(["max", "min"])
    bag_stats["bag_hours_range"] = hours_range["max"] - hours_range["min"]

    bag_stats["bag_education_std"] = bag_stats["bag_education_std"].fillna(0)
    bag_stats["bag_hours_std"] = bag_stats["bag_hours_std"].fillna(0)
    return bag_stats

train_bag = engineer_bag_features(train)
test_bag = engineer_bag_features(test)

print(f"  Bag features created: {len(train_bag.columns) - 1}")

# Merge bag features back to individual rows
train = train.merge(train_bag, on="bag_id", how="left", suffixes=("", "_bag"))
test = test.merge(test_bag, on="bag_id", how="left", suffixes=("", "_bag"))

print(f"  Train shape after bag merge: {train.shape}")
print(f"  Test shape after bag merge: {test.shape}")

# Store bag_ids for GroupKFold before dropping
train_bag_ids = train["bag_id"].values
test_bag_ids = test["bag_id"].values

# Now drop bag_id from feature sets
train = train.drop(columns=["bag_id"])
test = test.drop(columns=["bag_id"])
print(f"  Dropped bag_id after bag feature engineering")
print(f"  Train bag_ids saved: {len(train_bag_ids)} | Test bag_ids saved: {len(test_bag_ids)}")


# ════════════════════════════════════════════════════════
# STEP 3: INDIVIDUAL-LEVEL FEATURE ENGINEERING
# ════════════════════════════════════════════════════════

print("\n[3] Engineering individual-level features...")

# Age from year_of_birth (assume reference year 1994 based on survey_year)
train["age"] = 1994 - train["year_of_birth"]
test["age"] = 1994 - test["year_of_birth"]

# Age squared (non-linear effect)
train["age_squared"] = train["age"] ** 2
test["age_squared"] = test["age"] ** 2

# Net capital = gain - loss (already exists, but create clean version)
train["net_capital_clean"] = train["capital_gain"] - train["capital_loss"]
test["net_capital_clean"] = test["capital_gain"] - test["capital_loss"]

# Capital ratio
train["capital_ratio"] = train["capital_gain"] / (train["capital_loss"] + 1)
test["capital_ratio"] = test["capital_gain"] / (test["capital_loss"] + 1)

# Binary flags
train["has_capital_gain"] = (train["capital_gain"] > 0).astype(int)
test["has_capital_gain"] = (test["capital_gain"] > 0).astype(int)
train["has_capital_loss"] = (train["capital_loss"] > 0).astype(int)
test["has_capital_loss"] = (test["capital_loss"] > 0).astype(int)

# Capital per working hour
train["capital_per_hour"] = train["capital_gain"] / (train["hours_per_week"] + 1)
test["capital_per_hour"] = test["capital_gain"] / (test["hours_per_week"] + 1)

# Hours categories
train["works_full_time"] = (train["hours_per_week"] >= 35).astype(int)
test["works_full_time"] = (test["hours_per_week"] >= 35).astype(int)
train["works_overtime"] = (train["hours_per_week"] > 40).astype(int)
test["works_overtime"] = (test["hours_per_week"] > 40).astype(int)

# Age-education interaction
train["age_x_education"] = train["age"] * train["education_num"]
test["age_x_education"] = test["age"] * test["education_num"]

# Log transforms for capital features
for col in CAPITAL_COLS:
    if col in train.columns:
        train[f"log_{col}"] = np.log1p(train[col])
        test[f"log_{col}"] = np.log1p(test[col])

print(f"  Individual features created")


# ════════════════════════════════════════════════════════
# STEP 4: ENCODE CATEGORICAL FEATURES
# ════════════════════════════════════════════════════════

print("\n[4] Encoding categorical features...")

# 4a. Ordinal encoding
for col, mapping in ORDINAL_COLS.items():
    train[col] = train[col].map(mapping)
    test[col] = test[col].map(mapping)
    print(f"  Ordinal: {col} -> {mapping}")

# 4b. Label encoding for low-cardinality binary-like features
for col in LABEL_ENCODE_COLS:
    train[col] = train[col].astype("category").cat.codes
    test[col] = test[col].astype("category").cat.codes
    print(f"  Label encoded: {col}")

# 4c. Target encoding for high-cardinality features
target_map = {}
for col in TARGET_ENCODE_COLS:
    if col not in train.columns:
        continue
    label_encoded = train["label"].map({"lower": 0, "middle": 1, "upper": 2})
    global_mean = label_encoded.mean()
    col_means = label_encoded.groupby(train[col]).mean()
    col_counts = train.groupby(col).size()
    smoothing = 10
    col_smoothed = (col_means * col_counts + global_mean * smoothing) / (col_counts + smoothing)

    train[f"{col}_target_enc"] = train[col].map(col_smoothed)
    test[f"{col}_target_enc"] = test[col].map(col_smoothed).fillna(global_mean)

    target_map[col] = col_smoothed.to_dict()
    print(f"  Target encoded: {col} (smoothed, k={smoothing})")

    # Drop original high-cardinality column
    train = train.drop(columns=[col])
    test = test.drop(columns=[col])


# ════════════════════════════════════════════════════════
# STEP 5: CLIP OUTLIERS
# ════════════════════════════════════════════════════════

print("\n[5] Clipping outliers at {}th percentile...".format(OUTLIER_CLIP_PCT))

clip_cols = ["capital_gain", "capital_loss", "net_capital_asset", "net_capital_clean",
             "capital_ratio", "capital_per_hour",
             "bag_capital_gain_max", "bag_capital_gain_mean",
             "bag_capital_loss_max", "bag_capital_loss_mean",
             "bag_net_capital_max", "bag_net_capital_mean"]

clip_cols = [c for c in clip_cols if c in train.columns]

for col in clip_cols:
    upper = np.percentile(train[col], OUTLIER_CLIP_PCT)
    train[col] = train[col].clip(upper=upper)
    test[col] = test[col].clip(upper=upper)
    print(f"  Clipped {col} at {upper:,.0f}")


# ════════════════════════════════════════════════════════
# STEP 6: FINALIZE — SEPARATE FEATURES AND TARGET
# ════════════════════════════════════════════════════════

print("\n[6] Finalizing feature sets...")

label_to_int = {"lower": 0, "middle": 1, "upper": 2}
int_to_label = {0: "lower", 1: "middle", 2: "upper"}

# Features for individual-level prediction
X_train = train.drop(columns=["label"])
y_train = train["label"].map(label_to_int)
X_test = test.copy()

# Ensure train and test have same columns
common_cols = sorted(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"  Feature columns: {len(common_cols)}")
print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  Train bag_ids: {len(train_bag_ids)} | Test bag_ids: {len(test_bag_ids)}")

# Fill any remaining NaNs with median
for col in X_train.columns:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

remaining_nulls_train = X_train.isnull().sum().sum()
remaining_nulls_test = X_test.isnull().sum().sum()
print(f"  Remaining nulls — train: {remaining_nulls_train}, test: {remaining_nulls_test}")


# ════════════════════════════════════════════════════════
# STEP 7: SAVE PROCESSED DATA
# ════════════════════════════════════════════════════════

print("\n[7] Saving processed data...")

# Save as CSV for easy inspection
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False, header=["label"])
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

# Save bag_ids for GroupKFold
np.save(OUTPUT_DIR / "train_bag_ids.npy", train_bag_ids)
np.save(OUTPUT_DIR / "test_bag_ids.npy", test_bag_ids)

# Save preprocessing artifacts for inference
artifacts = {
    "label_to_int": label_to_int,
    "int_to_label": int_to_label,
    "ordinal_mappings": {k: v for k, v in ORDINAL_COLS.items()},
    "target_encodings": target_map,
    "drop_cols": DROP_COLS,
    "feature_cols": common_cols,
    "clip_values": {col: float(np.percentile(train[col], OUTLIER_CLIP_PCT)) for col in clip_cols},
}
with open(OUTPUT_DIR / "preprocessing_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print(f"\n  Saved to {OUTPUT_DIR}/:")
print(f"    X_train.csv      — {X_train.shape[0]} rows x {X_train.shape[1]} features")
print(f"    y_train.csv      — {y_train.shape[0]} labels")
print(f"    X_test.csv       — {X_test.shape[0]} rows x {X_test.shape[1]} features")
print(f"    train_bag_ids.npy — for GroupKFold")
print(f"    test_bag_ids.npy  — for GroupKFold")
print(f"    preprocessing_artifacts.pkl — encoding maps and config")

# ════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print(f"""
ORIGINAL TRAIN:  16,776 rows x 29 columns
PROCESSED TRAIN: {X_train.shape[0]} rows x {X_train.shape[1]} features

DROPPED ({len(DROP_COLS)} columns):
  {', '.join(DROP_COLS)}

ENGINEERED FEATURES:
  Bag-level (21): member_count, education_mean/std/range,
    age_mean/range, hours_mean/std/range, capital_gain/loss max/mean,
    net_capital max/mean, duration_mean, unique_relationships/
    occupations/workclass, has_capital_activity, adult_ratio, bag_size

  Individual (12): age, age_squared, net_capital_clean, capital_ratio,
    has_capital_gain/loss, capital_per_hour, works_full_time/overtime,
    age_x_education, log_capital_gain/loss/asset

ENCODING:
  Ordinal: education_tier (Primary=0, Secondary=1, Higher=2)
  Label: relationship, marital_status, sex, interview_mode
  Target: occupation, workclass, education, native_country, race

OUTLIER HANDLING:
  Clipped at {OUTLIER_CLIP_PCT}th percentile for capital-related features

NEXT STEP:
  Train with GroupKFold(bag_id) to prevent data leakage
  Use Macro F1 for evaluation
""")
