# Data Preprocessing v3 — Simplified, Low-Risk Feature Engineering

**Date:** May 08, 2026  
**Pipeline:** `preprocess_v3.py`  
**Training:** `train_v3.py` (per-fold bag feature recomputation)  
**Input:** 29 raw columns → **Output:** 61 features

---

## 1. Design Philosophy

v3 follows the principle: **simpler is better**. After v2's 26 new features raised overfitting concerns, v3 takes a disciplined approach:

1. Start from the v1 baseline (54 features, macro F1 0.710)
2. Add only **6 targeted bag-level features** that directly attack the lower/middle class gap
3. Use class weight tuning only — **no sample-level bonuses**
4. Recompute bag features **per CV fold** to prevent leakage
5. Keep v1 hyperparameters unchanged — measure pure feature impact before re-tuning

### What v3 Removed vs v2

| Category | v2 Count | v3 Count | Removed |
|----------|----------|----------|---------|
| Bag features | 30 | 28 | `bag_capital_gain_std`, `bag_capital_loss_std`, `bag_high_ed_ratio`, `bag_unemployed_ratio`, `bag_young_ratio`, `bag_senior_ratio` |
| Individual features | 29 | 12 | All 17 v2 individual features (interactions, context deviation, poverty indicators, capital structure, household position) |
| Sample weights | Yes (max 8.1x) | No | Replaced with class weights only |
| Total features | 80 | 61 | -19 |

---

## 2. The 6 New V3 Features

### Feature 1: `bag_low_ed_ratio`

| Property | Value |
|----------|-------|
| Formula | `(education_num <= 9).mean()` per bag |
| Type | Ratio (0.0 to 1.0) |
| Target | Lower class detection |

**Rationale:** This is the single most direct lower-class signal at the household level. A bag where 80% of members have ≤ 9 years of education (≤ high school) is fundamentally different from one where 20% do. EDA showed that Primary education tier correlates 39.7% with lower class vs only 21.7% with upper. At the bag level, this ratio amplifies the signal because entire households tend to share similar education levels.

**Why it's safe:** It's a simple proportion — no multiplicative interactions, no complex ratios. The feature is bounded [0, 1] and monotonic with class severity.

---

### Feature 2: `bag_full_time_ratio`

| Property | Value |
|----------|-------|
| Formula | `(hours_per_week >= 35).mean()` per bag |
| Type | Ratio (0.0 to 1.0) |
| Target | Employment stability |

**Rationale:** Full-time employment ratio captures household economic stability. Lower-class households often have mixed employment patterns (some part-time, some unemployed, some seasonal). Middle and upper-class households tend toward consistent full-time employment. A bag with `bag_full_time_ratio = 0.2` (only 1 of 5 members works full-time) signals economic pressure that raw `hours_per_week` averages would dilute.

**Why it's safe:** Binary threshold (35 hrs) is a well-established standard. The ratio is bounded and interpretable.

---

### Feature 3: `bag_capital_activity_ratio`

| Property | Value |
|----------|-------|
| Formula | `capital_activity_flag.mean()` per bag |
| Type | Ratio (0.0 to 1.0) |
| Target | Investment participation |

**Rationale:** EDA showed 24.8% of upper class vs 13.3% of lower class have any capital activity. At the bag level, this ratio becomes highly discriminative: a household where 4/5 members have capital activity is almost certainly upper class, while one where 0/5 do could be any class but leans lower. This feature captures the household's collective engagement with capital markets — a proxy for financial sophistication and wealth accumulation.

**Why it's safe:** It's a mean of an existing binary flag. No new computation beyond aggregation.

---

### Feature 4: `bag_zero_capital_ratio`

| Property | Value |
|----------|-------|
| Formula | `(capital_gain == 0).mean()` per bag |
| Type | Ratio (0.0 to 1.0) |
| Target | Zero-capital identification |

**Rationale:** 87% of all individuals have zero capital gain. But the distribution at bag level is more structured: lower-class bags tend toward 100% zero capital (everyone in the household has no investments), while upper-class bags rarely have all zeros. A bag with `bag_zero_capital_ratio = 1.0` (everyone at zero) is a strong lower/middle indicator. A bag with `bag_zero_capital_ratio = 0.2` (only 1 of 5 at zero) is likely upper class.

**Why this is different from `bag_capital_activity_ratio`:** `capital_activity_flag` may include loss-only cases. `bag_zero_capital_ratio` specifically targets the gain=0 condition, which is the dominant pattern. Together, these two features create a more nuanced picture of household capital participation than either alone.

---

### Feature 5: `bag_dependency_ratio`

| Property | Value |
|----------|-------|
| Formula | `(bag_size - full_time_count) / max(1, full_time_count)` |
| Type | Ratio (0.0 to ~7.0) |
| Target | Economic pressure |

**Rationale:** This directly counts dependents per full-time worker. A household of 5 with 2 full-time workers has a ratio of 3/2 = 1.5 (1.5 dependents per worker). A household of 4 with 0 full-time workers has a ratio of 4/1 = 4.0 (high economic pressure).

**Fix from v2:** The v2 formula `bag_member_count / (bag_full_time_ratio * bag_member_count + 1)` produced unintuitive values (collapsed to bag size when no one worked full-time). The v3 formula is semantically meaningful and monotonic with economic pressure.

**Why it's safe:** Simple arithmetic with a floor (`max(1, ...)`) to prevent division by zero. The ratio has a clear real-world interpretation.

---

### Feature 6: `bag_education_vs_below`

| Property | Value |
|----------|-------|
| Formula | `(education_num < bag_education_mean).mean()` per bag |
| Type | Ratio (0.0 to ~0.5) |
| Target | Within-household education inequality |

**Rationale:** This measures what proportion of household members fall below the household's own education average. A perfectly homogeneous bag (all members with the same education level) has a ratio near 0.5 (half above, half below by definition of mean). A bag where one member is highly educated and others are not has a high ratio (many below the inflated mean). This captures education inequality within the household, which correlates with economic dynamics — households with large education gaps often have mixed economic trajectories.

**Why it's different from `bag_low_ed_ratio`:** `bag_low_ed_ratio` uses a fixed threshold (≤ 9). `bag_education_vs_below` uses the household's own mean as the reference point. A bag with education levels [12, 12, 12, 8, 8] has `bag_low_ed_ratio = 0.4` (2/5 ≤ 9) but `bag_education_vs_below = 0.4` (2/5 below mean of 10.4). A bag with [16, 8, 8, 8, 8] has `bag_low_ed_ratio = 0.8` but `bag_education_vs_below = 0.8` — the high ratio in both cases signals the same thing, but the below-mean feature captures relative standing regardless of the absolute education level.

---

## 3. Class Weights (No Sample Bonuses)

| Class | v2 Weight | v3 Weight | Rationale |
|-------|-----------|-----------|-----------|
| lower | 1.5 (+ signal bonuses up to 8.1x) | **1.6** | +60% over frequency-based. No multiplicative bonuses — clean, stable. |
| middle | 0.7 | **0.7** | Unchanged from v2. Reduced to shift attention away from the easy class. |
| upper | 0.8 | **0.7** | Slightly reduced from v2. Upper class is already well-predicted (F1=0.867); further reducing weight prevents the model from over-optimizing on this class. |

**Key change from v2:** Removed all sample-level bonuses. The v2 approach of multiplying bonuses (2.0 × 1.8 × 1.5 = 5.4x on top of 1.5x class weight = 8.1x max) was identified as too aggressive — a single sample could dominate the loss. V3 uses a flat class weight that applies uniformly to all samples of that class.

---

## 4. Per-Fold Bag Feature Recomputation

### The Problem

In v1 and v2, bag-level features were computed once from the full training set. During GroupKFold cross-validation, this means validation fold bag statistics include information from non-fold training bags. This is not test leakage (test data is never used), but it inflates validation scores because the validation bags' aggregates were computed with the full training context.

### The V3 Solution

`train_v3.py` recomputes the 6 v3 bag features **inside each fold** using only the training-fold's bags:

```
For each fold:
  1. Extract raw individual data for training bags only
  2. Compute bag_low_ed_ratio, bag_full_time_ratio, etc. from training bags only
  3. Apply these training-computed features to validation rows
  4. Train and evaluate
```

This ensures that validation bag features are computed from the same information the model will have at inference time (training data only). The v1 bag features (21 aggregates) are still precomputed globally — only the 6 new v3 features use per-fold recomputation. This is a pragmatic balance between computational cost and validation integrity.

### Implementation

The `compute_v3_bag_features_fold()` function in `train_v3.py` mirrors the exact same logic as `preprocess_v3.py`, but operates on fold-specific data. The `apply_v3_features_per_fold()` function merges fresh features into the fold's base feature matrix.

---

## 5. Feature Inventory (61 Total)

| Category | Count | Examples |
|----------|-------|---------|
| Original numerical (after drops) | 7 | `education_num`, `hours_per_week`, `survey_duration_mins`, `capital_gain`, `capital_loss`, `net_capital_asset`, `bag_size` |
| V1 bag-level aggregates | 21 | `bag_education_mean`, `bag_capital_gain_max`, `bag_hours_std`, `bag_unique_occupations`, etc. |
| V3 bag-level features | 6 | `bag_low_ed_ratio`, `bag_full_time_ratio`, `bag_capital_activity_ratio`, `bag_zero_capital_ratio`, `bag_dependency_ratio`, `bag_education_vs_below` |
| V1 individual derived | 12 | `age`, `age_squared`, `net_capital_clean`, `capital_ratio`, `has_capital_gain`, `log_capital_gain`, etc. |
| Ordinal encoded | 1 | `education_tier` (Primary=0, Secondary=1, Higher=2) |
| Label encoded | 4 | `relationship`, `marital_status`, `sex`, `interview_mode` |
| Target encoded | 5 | `occupation_target_enc`, `workclass_target_enc`, `education_target_enc`, `native_country_target_enc`, `race_target_enc` |
| **Total** | **61** | |

---

## 6. What V3 Does NOT Do (By Design)

| Feature | Why Not |
|---------|---------|
| `education_x_workclass`, `education_x_occupation` | Multiplicative interactions risk overfitting with only 3,360 bags |
| `capital_vs_bag`, `education_vs_bag` | Context deviation features — too similar to bag_education_vs_below |
| `capital_dominance`, `capital_cv` | Ratio features with high variance in zero-inflated data |
| `low_ed_low_hours`, `no_capital_low_ed` | Compound poverty indicators — `bag_low_ed_ratio` captures this at bag level more robustly |
| `young_low_ed`, `hours_per_ed` | Individual-level poverty signals — already partially captured by bag-level ratios |
| `negative_net_capital`, `capital_both` | Rare events (< 6% of samples) — low signal-to-noise |
| `dependency_ratio` (v2 formula) | Fixed in v3 with correct formula |
| Sample-level weight bonuses | Too aggressive (up to 8.1x max) — replaced with flat class weights |
| Two-stage classification | Premature — test single-stage v3 first |
| SMOTE / oversampling | Bag structure makes oversampling complex and risky |
| Focal loss | Unnecessary complexity — try class weights first |

---

## 7. Training Configuration

```python
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'verbose': -1,
    'class_weight': {0: 1.6, 1: 0.7, 2: 0.7},
    'random_state': 42,
    'num_leaves': 127,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_samples': 50,
    'reg_lambda': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

**No hyperparameter re-tuning.** These are the same v1 parameters (the best from the grid search: `nl=127, lr=0.05, ne=300, mcs=50, rl=0.1`). This isolates the effect of the 6 new features + updated class weights.

---

## 8. Expected Outcomes

| Metric | v1 (baseline) | v3 (expected) | Delta |
|--------|---------------|---------------|-------|
| Macro F1 | 0.710 | 0.73 – 0.74 | +0.02 to +0.03 |
| Lower F1 | 0.567 | 0.61 – 0.63 | +0.05 to +0.06 |
| Middle F1 | 0.695 | 0.70 – 0.71 | +0.01 to +0.02 |
| Upper F1 | 0.867 | 0.85 – 0.86 | -0.02 to -0.01 |
| Overfitting risk | Low | Low | — |

### Why These Numbers

- **Lower F1 gains +0.05 to +0.06:** The 6 new features directly target the lower/middle boundary. `bag_low_ed_ratio` and `bag_zero_capital_ratio` are near-binary indicators for lower-class households. `bag_dependency_ratio` captures economic pressure that individual features miss.
- **Upper F1 slight drop (-0.01 to -0.02):** Reducing upper class weight from 0.8 to 0.7 shifts model attention. But the drop is bounded because upper class has the strongest inherent signal (capital features).
- **Fold variance ≤ 0.02:** With only 6 new features and no complex interactions, the model should be stable across folds. If variance exceeds 0.02, it indicates the new features are overfitting and should be pruned.

---

## 9. Evaluation Criteria

After running `train_v3.py`, check:

1. **Overall macro F1 > 0.72** — clear improvement over v1's 0.710
2. **Lower F1 > 0.60** — the primary target; must improve meaningfully
3. **Fold variance ≤ 0.02** — stability check; if higher, drop the weakest v3 feature
4. **At least 3 of 6 v3 features in top 30 by importance** — confirms new features are being used
5. **No v3 feature with 0 importance** — if any v3 feature is unused, consider dropping it

If all 5 criteria are met, v3 is ready for submission. If not, diagnose which criterion failed and iterate.

---

## 10. Decision Tree After V3 Results

```
v3 macro F1 > 0.73?
├── Yes → Submit. Done.
└── No
    ├── v3 macro F1 > 0.71 (better than v1)?
    │   ├── Yes → Consider adding back 2-3 v2 features selectively
    │   │         (highest-importance ones from v2 analysis)
    │   └── No → The 6 v3 features aren't helping. Investigate:
    │            - Are v3 features in the importance ranking?
    │            - Is per-fold recomputation causing feature mismatch?
    │            - Try v2's class_weight {0: 1.5, 1: 0.7, 2: 0.8}
    └── Lower F1 still < 0.60?
        └── Consider two-stage classification (Strategy 3 from gap analysis)
```
