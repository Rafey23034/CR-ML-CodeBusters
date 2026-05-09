# Class Performance Gap Analysis & Improvement Plan

**Date:** May 08, 2026
**Model:** Ensemble LightGBM + RandomForest (equal weights)
**Current Macro F1:** 0.7100 (after threshold tuning)

---

## 1. The Problem

| Class | F1 Score | Support | Gap from Mean |
|-------|----------|---------|---------------|
| lower | 0.567 | ~4,730 (28%) | -0.13 |
| middle | 0.695 | ~6,696 (40%) | 0.00 |
| upper | 0.867 | ~5,350 (32%) | +0.17 |

**Gap between best and worst: 0.30** — this is severe. The model predicts "upper" class with high accuracy but struggles badly with the "lower" class. Since Macro F1 treats all classes equally, the low "lower" F1 drags down the overall score significantly.

### Root Causes

1. **Minority class signal is diffuse** — "lower" class has the weakest capital signals (mean capital_gain=979 vs 2,854 for upper). When capital is zero (87% of rows), the model has few strong signals to distinguish lower from middle.
2. **Feature overlap between lower and middle** — Education, work hours, and occupation distributions overlap heavily between these two classes. The boundary is fuzzy.
3. **Balanced weights are insufficient** — `class_weight='balanced'` compensates for class frequency but not for signal strength. If the lower class is inherently harder to separate, equal weighting won't fix it.
4. **Bag-level features favor upper class** — Household aggregates (mean capital, max education) amplify the signal for upper-class bags but provide less differentiation for lower-class bags where all members tend to have similar low values.

---

## 2. Solutions for Class Performance Gap

### Priority 1: Class Weight Tuning (Immediate, High Impact)

Replace `class_weight='balanced'` with manually tuned weights that over-penalize misclassifying the lower class:

```python
# Current (auto-balanced):
# lower: 1.06, middle: 0.84, upper: 1.05

# Proposed (aggressive lower emphasis):
class_weight = {0: 1.5, 1: 0.7, 2: 0.8}  # lower weighted 2x upper

# Or even more aggressive:
class_weight = {0: 1.8, 1: 0.6, 2: 0.7}
```

**How to find optimal weights:**
```python
from sklearn.metrics import f1_score
import numpy as np

weight_grid = []
for w0 in np.arange(1.2, 2.5, 0.1):
    w1 = 2.0 - w0 * 0.4  # keep total weight ~constant
    w2 = 2.0 - w0 * 0.3
    weight_grid.append({0: w0, 1: w1, 2: w2})

# Grid search over weights using GroupKFold CV
```

**Expected impact:** +0.02 to +0.04 macro F1, with lower F1 improving to ~0.60-0.62.

---

### Priority 2: Sample-Level Weighting with Signal-Based Weights

Go beyond class frequency. Weight samples by how informative they are:

```python
# Weight lower-class samples with capital activity higher
# (they are rare and highly discriminative)
sample_weights = np.ones(len(y_train))

# Lower class + has capital gain = very informative
mask = (y_train == 0) & (X_train['has_capital_gain'] == 1)
sample_weights[mask] = 3.0

# Lower class + high education = rare signal
mask = (y_train == 0) & (X_train['education_num'] >= 13)
sample_weights[mask] = 2.5
```

---

### Priority 3: Two-Stage Classification

Instead of 3-class classification, use a hierarchical approach:

```
Stage 1: Binary classifier — "lower" vs "not-lower" (middle+upper)
  - Optimized for recall on lower class
  - Use higher threshold for "lower" prediction

Stage 2: Binary classifier — "middle" vs "upper" (only on "not-lower" samples)
  - Standard binary classification
```

**Why this works:** The lower vs non-lower boundary is the hardest part. A dedicated binary model can focus on the subtle signals that distinguish lower from middle.

```python
# Stage 1
y_train_stage1 = (y_train == 0).astype(int)  # 0=not-lower, 1=lower
model_stage1 = RandomForestClassifier(class_weight={0: 0.6, 1: 1.8}, ...)

# Stage 2 (only on non-lower samples)
mask_non_lower = y_train != 0
y_train_stage2 = (y_train[mask_non_lower] == 2).astype(int)
model_stage2 = RandomForestClassifier(...)
```

**Expected impact:** +0.03 to +0.05 macro F1.

---

### Priority 4: SMOTE with Group Awareness

Standard SMOTE would leak information across bag members. Use a bag-aware variant:

1. Apply SMOTE **at the bag level** — aggregate bag features first, then oversample minority bags
2. Or use ADASYN which focuses on harder-to-learn minority samples
3. Alternative: **Bag-level undersampling** of majority class bags

```python
from imblearn.over_sampling import SMOTE

# Only use bag-level features for SMOTE (not individual features)
bag_features = X_train[[col for col in X_train.columns if col.startswith('bag_')]]
smote = SMOTE(k_neighbors=3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(bag_features, y_train)
```

**Caution:** SMOTE on tabular data with mixed types (target-encoded, clipped, log-transformed) can create unrealistic samples. Use conservatively (k_neighbors=3, not 5).

---

### Priority 5: Focal Loss or Custom Objective

For LightGBM, use a custom objective that focuses on hard-to-classify samples:

```python
def focal_loss_objective(y_pred, train_data):
    y_true = train_data.get_label()
    gamma = 2.0  # focusing parameter

    # Convert to one-hot
    n_classes = 3
    y_oh = np.zeros((len(y_true), n_classes))
    y_oh[np.arange(len(y_true)), y_true.astype(int)] = 1

    probs = softmax(y_pred.reshape(-1, n_classes), axis=1)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)

    # Focal loss gradient
    pt = np.sum(y_oh * probs, axis=1)
    focal_weight = (1 - pt) ** gamma

    grad = focal_weight[:, None] * (probs - y_oh)
    hess = focal_weight[:, None] * probs * (1 - probs)

    return grad.flatten(), hess.flatten()
```

**Expected impact:** +0.01 to +0.03 macro F1.

---

## 3. Feature Engineering Recommendations

### 3.1 Features Targeting the Lower/Middle Boundary

The biggest confusion is between lower and middle class. Create features that specifically differentiate them:

#### Education-Occupation Interactions
```python
# Is person over-educated for their occupation?
# Lower class: high education + low-skill occupation = signal
df['education_x_workclass'] = df['education_num'] * df['workclass_target_enc']
df['education_x_occupation'] = df['education_num'] * df['occupation_target_enc']

# Education premium: how much education exceeds bag average
df['education_vs_bag'] = df['education_num'] - df['bag_education_mean']

# Same for capital
df['capital_vs_bag'] = df['capital_gain'] - df['bag_capital_gain_mean']
```

#### Poverty/Struggle Indicators
```python
# Low education + low hours = strong lower class signal
df['low_ed_low_hours'] = ((df['education_num'] < 9) & (df['hours_per_week'] < 30)).astype(int)

# Low education + no capital activity
df['no_capital_low_ed'] = ((df['education_num'] < 9) & (df['has_capital_gain'] == 0)).astype(int)

# Works many hours but low education (manual labor proxy)
df['hours_per_ed'] = df['hours_per_week'] / (df['education_num'] + 1)

# Young + low education = very likely lower class
df['young_low_ed'] = ((df['age'] < 30) & (df['education_num'] < 9)).astype(int)
```

#### Capital Structure Features
```python
# Capital concentration: does one person dominate household capital?
# (Already have bag_capital_gain_max and bag_capital_gain_mean)
df['capital_dominance'] = df['bag_capital_gain_max'] / (df['bag_capital_gain_mean'] + 1)

# Capital diversity: std/mean ratio
df['capital_cv'] = df['bag_capital_gain_std'] / (df['bag_capital_gain_mean'] + 1)
# Need to add bag_capital_gain_std to engineer_bag_features

# Net capital negative flag
df['negative_net_capital'] = (df['net_capital_clean'] < 0).astype(int)

# Capital stability: both gain and loss present
df['capital_both'] = ((df['has_capital_gain'] == 1) & (df['has_capital_loss'] == 1)).astype(int)
```

### 3.2 Bag-Level Features to Add

```python
# Add these to engineer_bag_features():

# Capital features
bag_capital_gain_std = ("capital_gain", "std")
bag_capital_loss_std = ("capital_loss", "std")
bag_net_capital_std = ("net_capital_asset", "std")

# Education concentration
bag_high_ed_ratio = lambda x: (x >= 13).mean()  # proportion with bachelor's+
bag_low_ed_ratio = lambda x: (x <= 9).mean()    # proportion with <= high school

# Work pattern concentration
bag_full_time_ratio = lambda x: (x >= 35).mean()
bag_unemployed_ratio = lambda x: (x == 0).mean()

# Age structure
bag_young_ratio = lambda x: ((1994 - x) < 30).mean()
bag_senior_ratio = lambda x: ((1994 - x) > 55).mean()

# Capital activity ratio
bag_capital_activity_ratio = ("capital_activity_flag", "mean")
```

### 3.3 Cross-Features (Individual x Bag Context)

```python
# How does this person compare to their household?
df['education_percentile_in_bag'] = df.groupby('bag_id')['education_num'].rank(pct=True)
df['capital_percentile_in_bag'] = df.groupby('bag_id')['capital_gain'].rank(pct=True)
df['age_percentile_in_bag'] = df.groupby('bag_id')['age'].rank(pct=True)

# Is this person the primary earner?
df['is_primary_earner'] = (df['capital_gain'] == df['bag_capital_gain_max']).astype(int)

# Is this person the most educated?
df['is_most_educated'] = (df['education_num'] == df.groupby('bag_id')['education_num'].transform('max')).astype(int)

# Household dependency ratio
df['dependency_ratio'] = df['bag_member_count'] / (df['bag_full_time_ratio'] * df['bag_member_count'] + 1)
```

### 3.4 Feature Selection After Engineering

After adding ~20-30 new features, apply feature selection:

```python
from sklearn.feature_selection import mutual_info_classif, SelectKBest

# Select top 60 features by mutual information
selector = SelectKBest(mutual_info_classif, k=60)
X_selected = selector.fit_transform(X_train, y_train)

# Or use LightGBM feature importance
model = lgb.LGBMClassifier(...)
model.fit(X_train, y_train)
importance = model.feature_importances_
top_features = np.argsort(importance)[-60:]
```

---

## 4. Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
- [ ] Tune class weights manually (grid search over weight combinations)
- [ ] Add 5-10 targeted features (education_vs_bag, capital_vs_bag, low_ed_low_hours, hours_per_ed, negative_net_capital)
- [ ] Retrain with GroupKFold

### Phase 2: Structural Changes (2-4 hours)
- [ ] Implement two-stage classification (lower vs rest, then middle vs upper)
- [ ] Add all bag-level features from Section 3.2
- [ ] Add cross-features from Section 3.3
- [ ] Apply feature selection (mutual information or importance-based)

### Phase 3: Advanced (4-8 hours)
- [ ] Try focal loss custom objective for LightGBM
- [ ] Experiment with SMOTE on bag-level features
- [ ] Hyperparameter tuning on the new feature set
- [ ] Ensemble with a third model (e.g., CatBoost or XGBoost)

---

## 5. Expected Improvements

| Change | Expected Macro F1 | Lower F1 | Middle F1 | Upper F1 |
|--------|-------------------|----------|-----------|----------|
| Current baseline | 0.7100 | 0.567 | 0.695 | 0.867 |
| + Class weight tuning | 0.72-0.73 | 0.59-0.61 | 0.70-0.71 | 0.85-0.86 |
| + Targeted features | 0.73-0.75 | 0.61-0.64 | 0.71-0.73 | 0.86-0.87 |
| + Two-stage model | 0.75-0.77 | 0.64-0.67 | 0.73-0.75 | 0.86-0.88 |
| Combined | 0.77-0.79 | 0.67-0.70 | 0.74-0.76 | 0.87-0.88 |

**Goal:** Reach macro F1 of 0.75+ with lower class F1 above 0.65.

---

## 6. Diagnostic Checks

Before implementing changes, run these diagnostics:

```python
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
# Check: are lower samples being predicted as middle? (most likely)

# Analyze misclassified lower samples
mask_lower = (y_true == 0)
mask_wrong = (y_pred != y_true)
wrong_lower = X_train[mask_lower & mask_wrong]
print(wrong_lower[['education_num', 'capital_gain', 'hours_per_week']].describe())
# Compare with correctly classified lower samples to find the gap

# Feature importance by class
# Train separate binary models and compare feature importance
```
