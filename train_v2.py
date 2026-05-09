import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.special import softmax
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path("processed_data_v2")

# ════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")["label"].values
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
train_bag_ids = np.load(DATA_DIR / "train_bag_ids.npy")
test_bag_ids = np.load(DATA_DIR / "test_bag_ids.npy")
sample_weights = np.load(DATA_DIR / "sample_weights.npy")

with open(DATA_DIR / "preprocessing_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

int_to_label = artifacts["int_to_label"]
class_weights = artifacts["class_weights"]

print(f"X_train: {X_train.shape} | y_train: {y_train.shape} | X_test: {X_test.shape}")
print(f"Classes: {np.unique(y_train)} — {int_to_label}")
print(f"Class weights: {class_weights}")
print(f"Sample weights range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
print(f"Features: {X_train.shape[1]}")

# ════════════════════════════════════════════════════════
# GROUPKFOLD SETUP
# ════════════════════════════════════════════════════════

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_train, y_train, groups=train_bag_ids))

def evaluate_fold(y_true, y_pred, fold_idx=None):
    macro = f1_score(y_true, y_pred, average="macro")
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    prefix = f"Fold {fold_idx} |" if fold_idx is not None else "Overall |"
    print(f"  {prefix} macro={macro:.4f} | lo={per_class[0]:.3f} mid={per_class[1]:.3f} up={per_class[2]:.3f}")
    return macro, per_class


# ════════════════════════════════════════════════════════
# STRATEGY 1: CLASS WEIGHT TUNING
# ════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STRATEGY 1: CLASS WEIGHT TUNING")
print("=" * 60)

weight_configs = [
    {0: 1.5, 1: 0.7, 2: 0.8},   # current default
    {0: 1.8, 1: 0.6, 2: 0.7},   # more aggressive
    {0: 2.0, 1: 0.5, 2: 0.6},   # very aggressive
    {0: 1.3, 1: 0.8, 2: 0.9},   # conservative
]

best_w_macro = 0
best_w_config = None
best_w_oof_preds = None

for cw in weight_configs:
    print(f"\n  Testing weights: {cw}")
    fold_macros = []
    oof_preds = np.zeros((len(y_train), 3))

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        sw_tr = sample_weights[tr_idx]

        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, verbose=-1,
            class_weight=cw, random_state=42,
            num_leaves=127, learning_rate=0.05, n_estimators=300,
            min_child_samples=50, reg_lambda=0.1,
            subsample=0.8, colsample_bytree=0.8,
        )
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
        oof_preds[val_idx] = model.predict_proba(X_val)

    oof_labels = np.argmax(oof_preds, axis=1)
    macro, _ = evaluate_fold(y_train, oof_labels, "OOF")
    fold_macros.append(macro)

    if macro > best_w_macro:
        best_w_macro = macro
        best_w_config = cw
        best_w_oof_preds = oof_preds

print(f"\n  Best class weights: {best_w_config} (macro={best_w_macro:.4f})")


# ════════════════════════════════════════════════════════
# STRATEGY 2: SAMPLE-LEVEL WEIGHTING
# ════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STRATEGY 2: SAMPLE-LEVEL WEIGHTING")
print("=" * 60)

# Compare: no sample weights vs current sample weights vs amplified
weight_variants = {
    "uniform": np.ones(len(y_train)),
    "class_only": np.array([class_weights[y] for y in y_train]),
    "class+signal": sample_weights,  # already computed in preprocess_v2
    "class+signal_amplified": np.where(sample_weights > 2.0, sample_weights * 1.5, sample_weights),
}

best_s_macro = 0
best_s_variant = None
best_s_oof_preds = None

for name, sw_variant in weight_variants.items():
    print(f"\n  Testing: {name}")
    oof_preds = np.zeros((len(y_train), 3))

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, verbose=-1,
            class_weight=best_w_config, random_state=42,
            num_leaves=127, learning_rate=0.05, n_estimators=300,
            min_child_samples=50, reg_lambda=0.1,
            subsample=0.8, colsample_bytree=0.8,
        )
        model.fit(X_tr, y_tr, sample_weight=sw_variant[tr_idx])
        oof_preds[val_idx] = model.predict_proba(X_val)

    oof_labels = np.argmax(oof_preds, axis=1)
    macro, _ = evaluate_fold(y_train, oof_labels, "OOF")

    if macro > best_s_macro:
        best_s_macro = macro
        best_s_variant = name
        best_s_oof_preds = oof_preds

print(f"\n  Best sample weight variant: {best_s_variant} (macro={best_s_macro:.4f})")


# ════════════════════════════════════════════════════════
# STRATEGY 3: TWO-STAGE CLASSIFICATION
# ════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STRATEGY 3: TWO-STAGE CLASSIFICATION")
print("=" * 60)

# Stage 1: lower (0) vs not-lower (1+2)
# Stage 2: middle (1) vs upper (2) among non-lower

best_2s_macro = 0
best_2s_oof_labels = None
best_2s_configs = {}

# Tune stage 1 class weights
stage1_weight_options = [
    {0: 2.0, 1: 1.0},
    {0: 2.5, 1: 1.0},
    {0: 3.0, 1: 1.0},
    {0: 3.5, 1: 1.0},
]

# Sample weight variants for two-stage
sw_variants_2s = {
    "class_only": np.array([class_weights[y] for y in y_train]),
    "class+signal": sample_weights,
}

for s1_w in stage1_weight_options:
    for sw_name, sw_variant in sw_variants_2s.items():
        print(f"\n  Stage1 weights={s1_w}, sample_weights={sw_name}")
        oof_labels = np.full(len(y_train), -1, dtype=int)
        fold_macros = []

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            sw_tr = sw_variant[tr_idx]

            # Stage 1: lower vs not-lower
            y_tr_s1 = (y_tr == 0).astype(int)
            y_val_s1 = (y_val == 0).astype(int)

            model_s1 = lgb.LGBMClassifier(
                objective="binary", verbose=-1,
                class_weight=s1_w, random_state=42,
                num_leaves=63, learning_rate=0.05, n_estimators=300,
                min_child_samples=50, reg_lambda=0.1,
            )
            model_s1.fit(X_tr, y_tr_s1, sample_weight=sw_tr)
            prob_s1 = model_s1.predict_proba(X_val)[:, 1]

            # Predict lower if probability > threshold (tune threshold)
            for threshold in [0.3, 0.4, 0.5]:
                pred_lower = (prob_s1 >= threshold).astype(int)

                # Stage 2: middle vs upper (only for non-lower predictions)
                non_lower_mask = pred_lower == 0
                y_tr_non_lower = y_tr[non_lower_mask]
                X_tr_non_lower = X_tr.iloc[non_lower_mask]
                sw_tr_non_lower = sw_tr[non_lower_mask]

                if len(np.unique(y_tr_non_lower)) < 2:
                    continue

                y_tr_s2 = (y_tr_non_lower == 2).astype(int)

                model_s2 = lgb.LGBMClassifier(
                    objective="binary", verbose=-1,
                    class_weight={0: 1.0, 1: 1.0}, random_state=42,
                    num_leaves=63, learning_rate=0.05, n_estimators=300,
                    min_child_samples=50, reg_lambda=0.1,
                )
                model_s2.fit(X_tr_non_lower, y_tr_s2, sample_weight=sw_tr_non_lower)

                if non_lower_mask.sum() > 0:
                    X_val_non_lower = X_val[non_lower_mask]
                    prob_s2 = model_s2.predict_proba(X_val_non_lower)[:, 1]
                    pred_s2 = (prob_s2 >= 0.5).astype(int)
                    oof_labels_local = np.where(
                        pred_lower == 0,
                        np.where(pred_s2 == 1, 2, 1),
                        0
                    )
                else:
                    oof_labels_local = pred_lower

                if fold_idx == 0:  # only evaluate once (on first fold) for threshold tuning
                    macro, per_class = evaluate_fold(y_val, oof_labels_local, "val")
                    if macro > best_2s_macro:
                        best_2s_macro = macro
                        best_2s_configs = {"stage1_weights": s1_w, "sw_variant": sw_name, "threshold": threshold}

# Full OOF evaluation with best config
print(f"\n  Best two-stage config: {best_2s_configs}")
print(f"  Best two-stage OOF macro: {best_2s_macro:.4f}")

# Run full OOF with best config for two-stage
if best_2s_configs:
    oof_2s_labels = np.full(len(y_train), -1, dtype=int)
    s1_w = best_2s_configs["stage1_weights"]
    sw_name = best_2s_configs["sw_variant"]
    threshold = best_2s_configs["threshold"]
    sw_variant = sw_variants_2s[sw_name]

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        sw_tr = sw_variant[tr_idx]

        y_tr_s1 = (y_tr == 0).astype(int)
        model_s1 = lgb.LGBMClassifier(
            objective="binary", verbose=-1, class_weight=s1_w, random_state=42,
            num_leaves=63, learning_rate=0.05, n_estimators=300,
            min_child_samples=50, reg_lambda=0.1,
        )
        model_s1.fit(X_tr, y_tr_s1, sample_weight=sw_tr)
        prob_s1 = model_s1.predict_proba(X_val)[:, 1]
        pred_lower = (prob_s1 >= threshold).astype(int)

        non_lower_mask = pred_lower == 0
        y_tr_non_lower = y_tr[non_lower_mask]
        X_tr_non_lower = X_tr.iloc[non_lower_mask]
        sw_tr_non_lower = sw_tr[non_lower_mask]

        if len(np.unique(y_tr_non_lower)) >= 2:
            y_tr_s2 = (y_tr_non_lower == 2).astype(int)
            model_s2 = lgb.LGBMClassifier(
                objective="binary", verbose=-1, class_weight={0: 1.0, 1: 1.0},
                random_state=42, num_leaves=63, learning_rate=0.05, n_estimators=300,
                min_child_samples=50, reg_lambda=0.1,
            )
            model_s2.fit(X_tr_non_lower, y_tr_s2, sample_weight=sw_tr_non_lower)

            if non_lower_mask.sum() > 0:
                X_val_non_lower = X_val[non_lower_mask]
                prob_s2 = model_s2.predict_proba(X_val_non_lower)[:, 1]
                pred_s2 = (prob_s2 >= 0.5).astype(int)
                oof_2s_labels[val_idx] = np.where(
                    pred_lower == 0,
                    np.where(pred_s2 == 1, 2, 1),
                    0
                )
            else:
                oof_2s_labels[val_idx] = pred_lower
        else:
            oof_2s_labels[val_idx] = pred_lower

    valid_mask = oof_2s_labels >= 0
    macro_2s, per_class_2s = evaluate_fold(y_train[valid_mask], oof_2s_labels[valid_mask], "OOF 2S")


# ════════════════════════════════════════════════════════
# COMPARISON & THRESHOLD TUNING
# ════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STRATEGY COMPARISON")
print("=" * 60)

# Strategy 1: best class weights (already have oof preds)
oof_s1_labels = np.argmax(best_w_oof_preds, axis=1)
macro_s1, pc_s1 = evaluate_fold(y_train, oof_s1_labels, "S1")

# Strategy 2: best sample weights (already have oof preds)
oof_s2_labels = np.argmax(best_s_oof_preds, axis=1)
macro_s2, pc_s2 = evaluate_fold(y_train, oof_s2_labels, "S2")

# Strategy 3: two-stage
valid_mask = oof_2s_labels >= 0
macro_s3, pc_s3 = evaluate_fold(y_train[valid_mask], oof_2s_labels[valid_mask], "S3")

print("\n" + "-" * 60)
print("  Strategy        | Macro F1 | Lower | Middle | Upper")
print("-" * 60)
print(f"  S1: Class Weights | {macro_s1:.4f}   | {pc_s1[0]:.3f} | {pc_s1[1]:.3f}   | {pc_s1[2]:.3f}")
print(f"  S2: Sample Weights| {macro_s2:.4f}   | {pc_s2[0]:.3f} | {pc_s2[1]:.3f}   | {pc_s2[2]:.3f}")
print(f"  S3: Two-Stage     | {macro_s3:.4f}   | {pc_s3[0]:.3f} | {pc_s3[1]:.3f}   | {pc_s3[2]:.3f}")

# Find best single strategy
strategies = {
    "S1_class_weights": (macro_s1, oof_s1_labels, None),
    "S2_sample_weights": (macro_s2, oof_s2_labels, None),
    "S3_two_stage": (macro_s3, oof_2s_labels[valid_mask], valid_mask),
}

best_name = max(strategies, key=lambda k: strategies[k][0])
best_macro = strategies[best_name][0]
best_labels = strategies[best_name][1]

print(f"\n  Best strategy: {best_name} (macro={best_macro:.4f})")

# Threshold tuning on best strategy's OOF predictions
print("\n" + "-" * 60)
print("THRESHOLD TUNING (best strategy)")
print("-" * 60)

if best_name == "S3_two_stage":
    print("  Two-stage model uses binary thresholds; skipping probability threshold tuning.")
else:
    best_probs = None
    if best_name == "S1_class_weights":
        best_probs = best_w_oof_preds
    elif best_name == "S2_sample_weights":
        best_probs = best_s_oof_preds

    if best_probs is not None:
        best_thresh_macro = 0
        best_thresh = (None, None, None)

        for t0 in np.arange(0.15, 0.40, 0.05):
            for t1 in np.arange(0.20, 0.45, 0.05):
                t2 = 1.0 - t0 - t1
                if t2 <= 0.05 or t2 >= 0.70:
                    continue

                probs_norm = best_probs.copy()
                probs_norm[:, 0] /= (t0 + 1e-10)
                probs_norm[:, 1] /= (t1 + 1e-10)
                probs_norm[:, 2] /= (t2 + 1e-10)
                preds = np.argmax(probs_norm, axis=1)

                m = f1_score(y_train, preds, average="macro")
                if m > best_thresh_macro:
                    best_thresh_macro = m
                    best_thresh = (t0, t1, t2)

        t0, t1, t2 = best_thresh
        probs_adj = best_probs.copy()
        probs_adj[:, 0] /= (t0 + 1e-10)
        probs_adj[:, 1] /= (t1 + 1e-10)
        probs_adj[:, 2] /= (t2 + 1e-10)
        tuned_preds = np.argmax(probs_adj, axis=1)

        tuned_macro, tuned_pc = evaluate_fold(y_train, tuned_preds, "tuned")
        print(f"  Default: {best_macro:.4f}")
        print(f"  Tuned thresholds: t0={t0:.2f}, t1={t1:.2f}, t2={t2:.2f}")
        print(f"  Tuned macro: {tuned_macro:.4f}")


# ════════════════════════════════════════════════════════
# SAVE OOF PREDICTIONS FOR ANALYSIS
# ════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SAVING OOF PREDICTIONS")
print("=" * 60)

oof_df = pd.DataFrame({
    "y_true": y_train,
    "pred_s1": oof_s1_labels,
    "pred_s2": oof_s2_labels,
    "pred_s3": np.where(valid_mask, oof_2s_labels, -1),
    "bag_id": train_bag_ids,
})
oof_df.to_csv(DATA_DIR / "oof_predictions.csv", index=False)

print("  Saved: processed_data_v2/oof_predictions.csv")
print("\n  Confusion matrix (best strategy):")
print(confusion_matrix(y_train, best_labels))
