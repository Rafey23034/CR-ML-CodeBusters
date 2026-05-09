import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle

warnings.filterwarnings("ignore")

DATA_DIR = Path("code-rush-26-ml-module")
OUTPUT_DIR = Path("processed_data_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════
# CONFIG — based on EDA findings + class gap analysis
# ════════════════════════════════════════════════════════

DROP_COLS = [
    "processing_flag",
    "survey_year",
    "currency_code",
    "poverty_line_usd",
    "is_adult_flag",
    "bag_id",
    "person_idx",
    "annual_hours_est",
]

ORDINAL_COLS = {
    "education_tier": {"Primary": 0, "Secondary": 1, "Higher": 2}
}

TARGET_ENCODE_COLS = ["occupation", "workclass", "education", "native_country", "race"]
LABEL_ENCODE_COLS = ["relationship", "marital_status", "sex", "interview_mode"]

CAPITAL_COLS = ["capital_gain", "capital_loss", "net_capital_asset"]
OUTLIER_CLIP_PCT = 99.5

# Class weight tuning — aggressive lower emphasis
# Auto-balanced would be: {0: 1.06, 1: 0.84, 2: 1.05}
CLASS_WEIGHTS = {0: 1.5, 1: 0.7, 2: 0.8}


# ════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════

print("Loading data...")
train = pd.read_csv(DATA_DIR / "Coderush-26-ML-Train.csv")
test = pd.read_csv(DATA_DIR / "Coderush-26-ML-test.csv")

print(f"  Train: {train.shape} | Test: {test.shape}")


# ════════════════════════════════════════════════════════
# STEP 1: DROP LEAKAGE / NOISE / REDUNDANT COLUMNS
# ════════════════════════════════════════════════════════

print("\n[1] Dropping leakage/noise/redundant columns...")
cols_to_drop_step1 = [c for c in DROP_COLS if c != "bag_id" and c in train.columns]
train = train.drop(columns=cols_to_drop_step1)
test = test.drop(columns=[c for c in DROP_COLS if c != "bag_id" and c in test.columns])
print(f"  Dropped {len(cols_to_drop_step1)} columns: {cols_to_drop_step1}")


# ════════════════════════════════════════════════════════
# STEP 2: BAG-LEVEL FEATURE ENGINEERING (ENHANCED)
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

    # Range features
    edu_range = df.groupby("bag_id")["education_num"].agg(["max", "min"])
    bag_stats["bag_education_range"] = edu_range["max"] - edu_range["min"]

    age_range = df.groupby("bag_id")["year_of_birth"].agg(["max", "min"])
    bag_stats["bag_age_range"] = age_range["max"] - age_range["min"]

    hours_range = df.groupby("bag_id")["hours_per_week"].agg(["max", "min"])
    bag_stats["bag_hours_range"] = hours_range["max"] - hours_range["min"]

    # NEW: Capital std features (capture capital inequality within bag)
    bag_stats = bag_stats.merge(
        df.groupby("bag_id")["capital_gain"].std().rename("bag_capital_gain_std").reset_index(),
        on="bag_id"
    )
    bag_stats = bag_stats.merge(
        df.groupby("bag_id")["capital_loss"].std().rename("bag_capital_loss_std").reset_index(),
        on="bag_id"
    )

    # NEW: Education concentration ratios
    high_ed = df.groupby("bag_id")["education_num"].apply(lambda x: (x >= 13).mean()).rename("bag_high_ed_ratio")
    low_ed = df.groupby("bag_id")["education_num"].apply(lambda x: (x <= 9).mean()).rename("bag_low_ed_ratio")
    bag_stats = bag_stats.merge(high_ed.reset_index(), on="bag_id")
    bag_stats = bag_stats.merge(low_ed.reset_index(), on="bag_id")

    # NEW: Work pattern concentration
    ft_ratio = df.groupby("bag_id")["hours_per_week"].apply(lambda x: (x >= 35).mean()).rename("bag_full_time_ratio")
    unemployed_ratio = df.groupby("bag_id")["hours_per_week"].apply(lambda x: (x == 0).mean()).rename("bag_unemployed_ratio")
    bag_stats = bag_stats.merge(ft_ratio.reset_index(), on="bag_id")
    bag_stats = bag_stats.merge(unemployed_ratio.reset_index(), on="bag_id")

    # NEW: Age structure ratios
    young_ratio = df.groupby("bag_id")["year_of_birth"].apply(lambda x: ((1994 - x) < 30).mean()).rename("bag_young_ratio")
    senior_ratio = df.groupby("bag_id")["year_of_birth"].apply(lambda x: ((1994 - x) > 55).mean()).rename("bag_senior_ratio")
    bag_stats = bag_stats.merge(young_ratio.reset_index(), on="bag_id")
    bag_stats = bag_stats.merge(senior_ratio.reset_index(), on="bag_id")

    # NEW: Capital activity ratio (proportion of bag with capital activity)
    cap_act = df.groupby("bag_id")["capital_activity_flag"].mean().rename("bag_capital_activity_ratio")
    bag_stats = bag_stats.merge(cap_act.reset_index(), on="bag_id")

    # Fill NaN std for single-member bags (shouldn't exist but safe)
    bag_stats["bag_education_std"] = bag_stats["bag_education_std"].fillna(0)
    bag_stats["bag_hours_std"] = bag_stats["bag_hours_std"].fillna(0)
    bag_stats["bag_capital_gain_std"] = bag_stats["bag_capital_gain_std"].fillna(0)
    bag_stats["bag_capital_loss_std"] = bag_stats["bag_capital_loss_std"].fillna(0)

    return bag_stats

train_bag = engineer_bag_features(train)
test_bag = engineer_bag_features(test)

print(f"  Bag features created: {len(train_bag.columns) - 1}")

train = train.merge(train_bag, on="bag_id", how="left", suffixes=("", "_bag"))
test = test.merge(test_bag, on="bag_id", how="left", suffixes=("", "_bag"))

print(f"  Train shape after bag merge: {train.shape}")
print(f"  Test shape after bag merge: {test.shape}")

train_bag_ids = train["bag_id"].values
test_bag_ids = test["bag_id"].values

train = train.drop(columns=["bag_id"])
test = test.drop(columns=["bag_id"])
print(f"  Dropped bag_id after bag feature engineering")


# ════════════════════════════════════════════════════════
# STEP 3: INDIVIDUAL-LEVEL FEATURE ENGINEERING (ENHANCED)
# ════════════════════════════════════════════════════════

print("\n[3] Engineering individual-level features...")

# --- ORIGINAL features (from v1) ---
train["age"] = 1994 - train["year_of_birth"]
test["age"] = 1994 - test["year_of_birth"]
train["age_squared"] = train["age"] ** 2
test["age_squared"] = test["age"] ** 2
train["net_capital_clean"] = train["capital_gain"] - train["capital_loss"]
test["net_capital_clean"] = test["capital_gain"] - test["capital_loss"]
train["capital_ratio"] = train["capital_gain"] / (train["capital_loss"] + 1)
test["capital_ratio"] = test["capital_gain"] / (test["capital_loss"] + 1)
train["has_capital_gain"] = (train["capital_gain"] > 0).astype(int)
test["has_capital_gain"] = (test["capital_gain"] > 0).astype(int)
train["has_capital_loss"] = (train["capital_loss"] > 0).astype(int)
test["has_capital_loss"] = (test["capital_loss"] > 0).astype(int)
train["capital_per_hour"] = train["capital_gain"] / (train["hours_per_week"] + 1)
test["capital_per_hour"] = test["capital_gain"] / (test["hours_per_week"] + 1)
train["works_full_time"] = (train["hours_per_week"] >= 35).astype(int)
test["works_full_time"] = (test["hours_per_week"] >= 35).astype(int)
train["works_overtime"] = (train["hours_per_week"] > 40).astype(int)
test["works_overtime"] = (test["hours_per_week"] > 40).astype(int)
train["age_x_education"] = train["age"] * train["education_num"]
test["age_x_education"] = test["age"] * test["education_num"]

for col in CAPITAL_COLS:
    if col in train.columns:
        train[f"log_{col}"] = np.log1p(train[col])
        test[f"log_{col}"] = np.log1p(test[col])

# --- NEW: Lower/Middle boundary features ---

# Education vs bag context: person's deviation from household average
train["education_vs_bag"] = train["education_num"] - train["bag_education_mean"]
test["education_vs_bag"] = test["education_num"] - test["bag_education_mean"]

# Capital vs bag context
train["capital_vs_bag"] = train["capital_gain"] - train["bag_capital_gain_mean"]
test["capital_vs_bag"] = test["capital_gain"] - test["bag_capital_gain_mean"]

# Poverty/struggle indicators (target lower class signal)
train["low_ed_low_hours"] = ((train["education_num"] < 9) & (train["hours_per_week"] < 30)).astype(int)
test["low_ed_low_hours"] = ((test["education_num"] < 9) & (test["hours_per_week"] < 30)).astype(int)

train["no_capital_low_ed"] = ((train["education_num"] < 9) & (train["has_capital_gain"] == 0)).astype(int)
test["no_capital_low_ed"] = ((test["education_num"] < 9) & (test["has_capital_gain"] == 0)).astype(int)

train["hours_per_ed"] = train["hours_per_week"] / (train["education_num"] + 1)
test["hours_per_ed"] = test["hours_per_week"] / (test["education_num"] + 1)

train["young_low_ed"] = ((train["age"] < 30) & (train["education_num"] < 9)).astype(int)
test["young_low_ed"] = ((test["age"] < 30) & (test["education_num"] < 9)).astype(int)

# --- NEW: Capital structure features ---

train["capital_dominance"] = train["bag_capital_gain_max"] / (train["bag_capital_gain_mean"] + 1)
test["capital_dominance"] = test["bag_capital_gain_max"] / (test["bag_capital_gain_mean"] + 1)

train["capital_cv"] = train["bag_capital_gain_std"] / (train["bag_capital_gain_mean"] + 1)
test["capital_cv"] = test["bag_capital_gain_std"] / (test["bag_capital_gain_mean"] + 1)

train["negative_net_capital"] = (train["net_capital_clean"] < 0).astype(int)
test["negative_net_capital"] = (test["net_capital_clean"] < 0).astype(int)

train["capital_both"] = ((train["has_capital_gain"] == 1) & (train["has_capital_loss"] == 1)).astype(int)
test["capital_both"] = ((test["has_capital_gain"] == 1) & (test["has_capital_loss"] == 1)).astype(int)

# Is this person the primary earner in the household?
train["is_primary_earner"] = (train["capital_gain"] >= train["bag_capital_gain_max"]).astype(int)
test["is_primary_earner"] = (test["capital_gain"] >= test["bag_capital_gain_max"]).astype(int)

# Education relative to bag mean minus one std
train["education_vs_bag_max"] = train["education_num"] - train["bag_education_mean"] - train["bag_education_std"]
test["education_vs_bag_max"] = test["education_num"] - test["bag_education_mean"] - test["bag_education_std"]

# --- NEW: Household dependency ratio ---
train["dependency_ratio"] = train["bag_member_count"] / (train["bag_full_time_ratio"] * train["bag_member_count"] + 1)
test["dependency_ratio"] = test["bag_member_count"] / (test["bag_full_time_ratio"] * test["bag_member_count"] + 1)

# --- NEW: Capital efficiency features ---
train["capital_per_capita"] = train["bag_capital_gain_mean"] / train["bag_member_count"]
test["capital_per_capita"] = test["bag_capital_gain_mean"] / test["bag_member_count"]

train["hours_inequality"] = train["bag_hours_std"] / (train["bag_hours_mean"] + 1)
test["hours_inequality"] = test["bag_hours_std"] / (test["bag_hours_mean"] + 1)

print(f"  Individual features added. Current shape: train={train.shape}, test={test.shape}")


# ════════════════════════════════════════════════════════
# STEP 4: ENCODE CATEGORICAL FEATURES
# ════════════════════════════════════════════════════════

print("\n[4] Encoding categorical features...")

# 4a. Ordinal encoding
for col, mapping in ORDINAL_COLS.items():
    train[col] = train[col].map(mapping)
    test[col] = test[col].map(mapping)

# 4b. Label encoding
for col in LABEL_ENCODE_COLS:
    train[col] = train[col].astype("category").cat.codes
    test[col] = test[col].astype("category").cat.codes

# 4c. Target encoding with smoothing
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
    train = train.drop(columns=[col])
    test = test.drop(columns=[col])

    print(f"  Target encoded: {col}")

# Education-occupation/workclass interactions (after target encoding)
print("  Creating education interaction features...")
train["education_x_workclass"] = train["education_num"] * train["workclass_target_enc"]
test["education_x_workclass"] = test["education_num"] * test["workclass_target_enc"]
train["education_x_occupation"] = train["education_num"] * train["occupation_target_enc"]
test["education_x_occupation"] = test["education_num"] * test["occupation_target_enc"]


# ════════════════════════════════════════════════════════
# STEP 5: CLIP OUTLIERS
# ════════════════════════════════════════════════════════

print("\n[5] Clipping outliers at {}th percentile...".format(OUTLIER_CLIP_PCT))

clip_cols = ["capital_gain", "capital_loss", "net_capital_asset", "net_capital_clean",
             "capital_ratio", "capital_per_hour",
             "bag_capital_gain_max", "bag_capital_gain_mean",
             "bag_capital_loss_max", "bag_capital_loss_mean",
             "bag_net_capital_max", "bag_net_capital_mean",
             "capital_vs_bag", "capital_dominance"]

clip_cols = [c for c in clip_cols if c in train.columns]

for col in clip_cols:
    upper = np.percentile(train[col], OUTLIER_CLIP_PCT)
    train[col] = train[col].clip(upper=upper)
    test[col] = test[col].clip(upper=upper)


# ════════════════════════════════════════════════════════
# STEP 6: FINALIZE
# ════════════════════════════════════════════════════════

print("\n[6] Finalizing feature sets...")

label_to_int = {"lower": 0, "middle": 1, "upper": 2}
int_to_label = {0: "lower", 1: "middle", 2: "upper"}

X_train = train.drop(columns=["label"])
y_train = train["label"].map(label_to_int)
X_test = test.copy()

common_cols = sorted(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"  Feature columns: {len(common_cols)}")

# Fill remaining NaNs with median
for col in X_train.columns:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)


# ════════════════════════════════════════════════════════
# STEP 7: COMPUTE SAMPLE WEIGHTS
# ════════════════════════════════════════════════════════

print("\n[7] Computing sample-level weights...")

# Base weights from class distribution
sample_weights = np.ones(len(y_train))

for cls, weight in CLASS_WEIGHTS.items():
    mask = y_train == cls
    sample_weights[mask] = weight

# Bonus weights for rare, highly informative lower-class samples
# Lower class + has capital gain = very discriminative
mask = (y_train == 0) & (X_train["has_capital_gain"] == 1)
sample_weights[mask] *= 2.0

# Lower class + high education = rare and informative
mask = (y_train == 0) & (X_train["education_num"] >= 13)
sample_weights[mask] *= 1.8

# Lower class + works full time + has capital = signal-rich
mask = (y_train == 0) & (X_train["works_full_time"] == 1) & (X_train["has_capital_gain"] == 1)
sample_weights[mask] *= 1.5

print(f"  Weight stats: min={sample_weights.min():.2f}, mean={sample_weights.mean():.2f}, max={sample_weights.max():.2f}")
print(f"  Weighted lower class effective count: {sample_weights[y_train == 0].sum():.0f} (original: {(y_train == 0).sum()})")
print(f"  Weighted middle class effective count: {sample_weights[y_train == 1].sum():.0f} (original: {(y_train == 1).sum()})")
print(f"  Weighted upper class effective count: {sample_weights[y_train == 2].sum():.0f} (original: {(y_train == 2).sum()})")


# ════════════════════════════════════════════════════════
# STEP 8: SAVE
# ════════════════════════════════════════════════════════

print("\n[8] Saving processed data...")

X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False, header=["label"])
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
np.save(OUTPUT_DIR / "sample_weights.npy", sample_weights)
np.save(OUTPUT_DIR / "train_bag_ids.npy", train_bag_ids)
np.save(OUTPUT_DIR / "test_bag_ids.npy", test_bag_ids)

artifacts = {
    "label_to_int": label_to_int,
    "int_to_label": int_to_label,
    "ordinal_mappings": ORDINAL_COLS,
    "target_encodings": target_map,
    "drop_cols": DROP_COLS,
    "feature_cols": common_cols,
    "clip_values": {col: float(np.percentile(train[col], OUTLIER_CLIP_PCT)) for col in clip_cols},
    "class_weights": CLASS_WEIGHTS,
    "feature_changes": {
        "new_bag_features": [
            "bag_capital_gain_std", "bag_capital_loss_std",
            "bag_high_ed_ratio", "bag_low_ed_ratio",
            "bag_full_time_ratio", "bag_unemployed_ratio",
            "bag_young_ratio", "bag_senior_ratio",
            "bag_capital_activity_ratio",
        ],
        "new_individual_features": [
            "education_vs_bag", "capital_vs_bag",
            "low_ed_low_hours", "no_capital_low_ed",
            "hours_per_ed", "young_low_ed",
            "education_x_workclass", "education_x_occupation",
            "capital_dominance", "capital_cv",
            "negative_net_capital", "capital_both",
            "education_vs_bag_max", "dependency_ratio",
            "capital_per_capita", "hours_inequality",
            "is_primary_earner",
        ],
    },
}
with open(OUTPUT_DIR / "preprocessing_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print(f"\n  Saved to {OUTPUT_DIR}/:")
print(f"    X_train.csv            — {X_train.shape[0]} rows x {X_train.shape[1]} features")
print(f"    y_train.csv            — {y_train.shape[0]} labels")
print(f"    X_test.csv             — {X_test.shape[0]} rows x {X_test.shape[1]} features")
print(f"    sample_weights.npy     — per-sample weights for training")
print(f"    train_bag_ids.npy      — for GroupKFold")
print(f"    test_bag_ids.npy       — for GroupKFold")
print(f"    preprocessing_artifacts.pkl — encoding maps, config, feature changes")

# ════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY (v2 — Class Gap Focused)")
print("=" * 60)
print(f"""
ORIGINAL:        16,776 rows x 29 columns
PROCESSED (v2):  {X_train.shape[0]} rows x {X_train.shape[1]} features

NEW BAG FEATURES (9):
  {', '.join(artifacts['feature_changes']['new_bag_features'])}

NEW INDIVIDUAL FEATURES ({len(artifacts['feature_changes']['new_individual_features'])}):
  {', '.join(artifacts['feature_changes']['new_individual_features'])}

CLASS WEIGHTS: {CLASS_WEIGHTS}
SAMPLE WEIGHT RANGE: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]

NEXT STEP:
  Run train_v2.py with:
    1. Class weight tuning (already applied)
    2. Sample-level weighting (already computed)
    3. Two-stage classification framework (in train_v2.py)
""")
