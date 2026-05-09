import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import pickle
import warnings
import sys

warnings.filterwarnings("ignore")

DATA_DIR = Path("processed_data_v3")

# ════════════════════════════════════════════════════════
# LOAD BASE DATA
# ════════════════════════════════════════════════════════

print("Loading base data...")
raw_train = pd.read_csv(Path("code-rush-26-ml-module") / "Coderush-26-ML-Train.csv")
raw_test = pd.read_csv(Path("code-rush-26-ml-module") / "Coderush-26-ML-test.csv")

X_train_base = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")["label"].values
X_test_base = pd.read_csv(DATA_DIR / "X_test.csv")
train_bag_ids = np.load(DATA_DIR / "train_bag_ids.npy")
test_bag_ids = np.load(DATA_DIR / "test_bag_ids.npy")

with open(DATA_DIR / "preprocessing_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

class_weights = artifacts["class_weights"]
int_to_label = artifacts["int_to_label"]
v3_features = artifacts["v3_new_features"]
feature_cols = artifacts["feature_cols"]

print(f"  Base features: {X_train_base.shape[1]}")
print(f"  V3 new features: {v3_features}")
print(f"  Class weights: {class_weights}")


# ════════════════════════════════════════════════════════
# PER-FOLD BAG FEATURE RECOMPUTATION FUNCTIONS
# These mirror the functions in preprocess_v3.py but operate
# on fold-specific data to prevent leakage.
# ════════════════════════════════════════════════════════

def compute_v3_bag_features_fold(df):
    """
    Recompute the 6 v3 bag features from raw individual-level data.
    Called inside each CV fold with only training-fold data.
    """
    bag_ratios = df.groupby("bag_id").agg(
        bag_low_ed_ratio=("education_num", lambda x: (x <= 9).mean()),
        bag_full_time_ratio=("hours_per_week", lambda x: (x >= 35).mean()),
        bag_capital_activity_ratio=("capital_activity_flag", "mean"),
        bag_zero_capital_ratio=("capital_gain", lambda x: (x == 0).mean()),
        bag_education_mean=("education_num", "mean"),
    ).reset_index()

    ft_count = df[df["hours_per_week"] >= 35].groupby("bag_id").size().rename("ft_count")
    bag_size = df.groupby("bag_id").size().rename("bag_size_raw")
    dep_df = pd.DataFrame({"ft_count": ft_count, "bag_size_raw": bag_size}).fillna(0)
    dep_df["bag_dependency_ratio"] = (dep_df["bag_size_raw"] - dep_df["ft_count"]) / np.maximum(1, dep_df["ft_count"])
    bag_ratios = bag_ratios.merge(dep_df[["bag_dependency_ratio"]], left_on="bag_id", right_index=True, how="left")

    df_with_mean = df.merge(bag_ratios[["bag_id", "bag_education_mean"]], on="bag_id", how="left")
    df_with_mean["below_mean"] = (df_with_mean["education_num"] < df_with_mean["bag_education_mean"]).astype(int)
    below_ratio = df_with_mean.groupby("bag_id")["below_mean"].mean().rename("bag_education_vs_below")
    bag_ratios = bag_ratios.merge(below_ratio.reset_index(), on="bag_id")

    return bag_ratios


def apply_v3_features_per_fold(raw_fold_df, base_fold_df):
    """
    Recompute v3 bag features from raw_fold_df (training-only data),
    then merge them into base_fold_df.
    Returns base_fold_df with fresh v3 features.
    """
    v3_fold = compute_v3_bag_features_fold(raw_fold_df)

    # Drop old v3 features from base
    v3_cols_to_drop = [c for c in v3_features if c in base_fold_df.columns]
    result = base_fold_df.drop(columns=v3_cols_to_drop, errors="ignore")

    # Merge fresh v3 features
    result = result.merge(v3_fold.drop(columns=["bag_education_mean"]), on="bag_id", how="left")

    # Fill any missing (bags in validation not in training — shouldn't happen in GroupKFold)
    for col in v3_features:
        if col in result.columns:
            result[col] = result[col].fillna(result[col].median() if result[col].notna().any() else 0)

    return result


# ════════════════════════════════════════════════════════
# PREPARE RAW DATA (same preprocessing as preprocess_v3.py, minus v3 features)
# We need raw train data with bag_id to recompute per-fold
# ════════════════════════════════════════════════════════

print("\nPreparing raw data for per-fold recomputation...")

# Drop columns that were dropped in preprocessing
drop_cols = [c for c in artifacts["drop_cols"] if c != "bag_id" and c in raw_train.columns]
raw_train_clean = raw_train.drop(columns=drop_cols)
raw_test_clean = raw_test.drop(columns=[c for c in artifacts["drop_cols"] if c != "bag_id" and c in raw_test.columns])


# ════════════════════════════════════════════════════════
# GROUPKFOLD WITH PER-FOLD V3 FEATURE RECOMPUTATION
# ════════════════════════════════════════════════════════

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_train_base, y_train, groups=train_bag_ids))

params = {
    "objective": "multiclass",
    "num_class": 3,
    "verbose": -1,
    "class_weight": class_weights,
    "random_state": 42,
    "num_leaves": 127,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 50,
    "reg_lambda": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

print(f"\nModel params: {params}")
print(f"Using same v1 hyperparameters — no re-tuning yet.")
print(f"\nRunning GroupKFold (5 folds) with per-fold v3 feature recomputation...")

oof_preds = np.zeros((len(y_train), 3))
fold_macros = []
fold_per_class = []

for fold_idx, (tr_idx, val_idx) in enumerate(splits):
    print(f"\n--- Fold {fold_idx + 1}/5 ---")

    # Get base features for this fold
    X_tr_base = X_train_base.iloc[tr_idx].copy()
    X_val_base = X_train_base.iloc[val_idx].copy()
    y_tr = y_train[tr_idx]
    y_val = y_train[val_idx]

    # Get raw data for this fold (needed to recompute v3 features)
    raw_tr = raw_train_clean.iloc[tr_idx]
    raw_val = raw_train_clean.iloc[val_idx]

    # Recompute v3 bag features from training data only
    X_tr = apply_v3_features_per_fold(raw_tr, X_tr_base)

    # For validation, use training-fold bag stats (apply the same mapping)
    # This simulates what happens with truly unseen test bags
    X_val = apply_v3_features_per_fold(raw_tr, X_val_base)

    # Ensure same columns
    common_cols = sorted(set(X_tr.columns) & set(X_val.columns))
    X_tr = X_tr[common_cols]
    X_val = X_val[common_cols]

    # Train
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr)

    # Predict
    oof_preds[val_idx] = model.predict_proba(X_val)

    # Evaluate
    val_preds = np.argmax(oof_preds[val_idx], axis=1)
    macro = f1_score(y_val, val_preds, average="macro")
    per_class = f1_score(y_val, val_preds, average=None, labels=[0, 1, 2])
    fold_macros.append(macro)
    fold_per_class.append(per_class)

    print(f"  Fold {fold_idx + 1}: macro={macro:.4f} | lo={per_class[0]:.3f} mid={per_class[1]:.3f} up={per_class[2]:.3f}")

# Overall OOF evaluation
oof_labels = np.argmax(oof_preds, axis=1)
overall_macro = f1_score(y_train, oof_labels, average="macro")
overall_per_class = f1_score(y_train, oof_labels, average=None, labels=[0, 1, 2])

print("\n" + "=" * 60)
print("V3 RESULTS (with per-fold v3 feature recomputation)")
print("=" * 60)
print(f"\n  Per-fold macro F1: {[f'{m:.4f}' for m in fold_macros]}")
print(f"  Mean macro F1: {np.mean(fold_macros):.4f} ± {np.std(fold_macros):.4f}")
print(f"  Overall macro F1: {overall_macro:.4f}")
print(f"  Per-class: lower={overall_per_class[0]:.3f} middle={overall_per_class[1]:.3f} upper={overall_per_class[2]:.3f}")

# Fold variance check
fold_var = np.std(fold_macros)
if fold_var > 0.02:
    print(f"\n  ⚠️ Fold variance {fold_var:.4f} > 0.02 — possible overfitting!")
else:
    print(f"\n  ✓ Fold variance {fold_var:.4f} ≤ 0.02 — stable across folds.")

# Compare to v1 baseline
v1_macro = 0.710
v1_lower = 0.567
gain = overall_macro - v1_macro
gain_lower = overall_per_class[0] - v1_lower

print(f"\n  Comparison to v1 baseline:")
print(f"    v1 macro F1: {v1_macro:.3f} → v3 macro F1: {overall_macro:.3f} (gain: {gain:+.3f})")
print(f"    v1 lower F1: {v1_lower:.3f} → v3 lower F1: {overall_per_class[0]:.3f} (gain: {gain_lower:+.3f})")

# Confusion matrix
print(f"\n  Confusion matrix:")
print(confusion_matrix(y_train, oof_labels))

# Feature importance
print(f"\n  Top 15 feature importances:")
# Train on full data for importance analysis
model_full = lgb.LGBMClassifier(**params)
model_full.fit(X_train_base[feature_cols], y_train)
importance = model_full.feature_importances_
feat_importance = pd.DataFrame({"feature": feature_cols, "importance": importance})
feat_importance = feat_importance.sort_values("importance", ascending=False)
for _, row in feat_importance.head(15).iterrows():
    marker = " [V3]" if row["feature"] in v3_features else ""
    print(f"    {row['feature']:<35} {row['importance']:<8}{marker}")

# Check v3 feature performance
v3_importances = feat_importance[feat_importance["feature"].isin(v3_features)]
print(f"\n  V3 features importance:")
for _, row in v3_importances.iterrows():
    pct = row["importance"] / feat_importance["importance"].max() * 100
    print(f"    {row['feature']:<35} importance={row['importance']:<8} ({pct:.1f}% of max)")

# Save results
results = {
    "overall_macro": overall_macro,
    "overall_per_class": overall_per_class.tolist(),
    "fold_macros": fold_macros,
    "fold_per_class": fold_per_class,
    "feature_importance": feat_importance.to_dict("records"),
    "params": params,
    "v3_features": v3_features,
}
with open(DATA_DIR / "v3_results.pkl", "wb") as f:
    pickle.dump(results, f)

oof_df = pd.DataFrame({
    "y_true": y_train,
    "y_pred": oof_labels,
    "bag_id": train_bag_ids,
})
oof_df.to_csv(DATA_DIR / "v3_oof_predictions.csv", index=False)

print(f"\n  Saved: {DATA_DIR}/v3_results.pkl")
print(f"  Saved: {DATA_DIR}/v3_oof_predictions.csv")
