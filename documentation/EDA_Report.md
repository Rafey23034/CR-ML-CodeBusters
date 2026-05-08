# Exploratory Data Analysis Report

Dataset: Coderush-26 ML Module

Date: May 08, 2026

Files: Coderush-26-ML-Train.csv (16,776 rows, 29 columns),
Coderush-26-ML-test.csv (1,981 rows, 28 columns)

# 1. Dataset Overview

The dataset consists of individual-level survey records grouped into
household bags. The task is a multi-class classification problem to
predict economic class (label) with three categories: lower, middle, and
upper.

| **Property**         | **Train**           | **Test**      |
| -------------------- | ------------------- | ------------- |
| Rows                 | 16,776              | 1,981         |
| Columns              | 29 (includes label) | 28 (no label) |
| Numerical features   | 15                  | 15            |
| Categorical features | 11                  | 11            |
| Unique bags          | 3,360               | ~397 (est.)   |

# 2. Target Variable: Class Balance

The target variable 'label' has three classes. The distribution is
moderately imbalanced, which has direct implications for model
evaluation and training strategy.

| **Class** | **Count** | **Percentage** |
| --------- | --------- | -------------- |
| lower     | 4,730     | 28.2%          |
| middle    | 6,696     | 39.9%          |
| upper     | 5,350     | 31.9%          |
| Total     | 16,776    | 100%           |

### Key Observations:

- The minority class is 'lower' at 28.2%, and the majority is 'middle' at 39.9%.
- Imbalance ratio (majority/minority): 1.4x — moderate but not extreme.
- Macro F1 is the evaluation metric. This punishes models that ignore the minority class, so accuracy alone is misleading. A naive model predicting all 'middle' would score ~33% Macro F1.
- Class weights or oversampling strategies should be applied during training.

# 3. Bag Structure Analysis

Each person belongs to a bag (household/group). Understanding bag
composition is critical for deciding the prediction unit (row-level vs
bag-level) and cross-validation strategy.

| **Property**                     | **Value**             |
| -------------------------------- | --------------------- |
| Total bags                       | 3,360                 |
| Bag sizes                        | 3, 4, 5, 6, 7 members |
| Most common size                 | 4 members (857 bags)  |
| Mean bag size                    | 5.33 members          |
| Bags with mixed labels           | 0 (0%)                |
| bag_size vs actual members match | 100% consistent       |

### Key Observations:

- All members within a bag share the same label. There are zero mixed-label bags. This means predicting at row-level or bag-level yields the same target.
- Bags represent households of 3 to 7 members. The prediction should ideally be aggregated at bag level (majority vote or mean probability) to ensure consistency.
- Cross-validation must preserve bag membership — do not split members of the same bag across train/validation folds. Use GroupKFold with bag_id as the group key.
- Bag-level features can be engineered: member count, relationship diversity, age range within bag, education spread, etc.

# 4. Missing Value Analysis

No null values were found in either the training or test dataset. All 29
columns in train and 28 columns in test are fully populated. This
simplifies preprocessing — no imputation is needed.

| **Property**               | **Value** |
| -------------------------- | --------- |
| Columns with nulls (train) | 0         |
| Columns with nulls (test)  | 0         |
| Total missing values       | 0         |

# 5. Leakage and Noise Feature Analysis

Several features were identified as constant, near-constant, or
potential leakage vectors. These should be dropped to reduce noise and
prevent spurious correlations.

| **Feature**      | **Reason to Drop** | **Details**                                |
| ---------------- | ------------------ | ------------------------------------------ |
| processing_flag  | Constant           | All values = 1.0 (100%)                    |
| survey_year      | Constant           | All values = 1994 (100%)                   |
| currency_code    | Constant           | All values = 'USD' (100%)                  |
| poverty_line_usd | Constant           | All values = 15141 (100%)                  |
| is_adult_flag    | Near-constant      | 99.4% = 1, only 107 rows = 0               |
| bag_id           | Identifier         | 3,360 unique values — no predictive signal |
| person_idx       | Identifier         | 0-7 within each bag — position index only  |

### Recommended Action:

- Drop all 7 features listed above before model training.
- interviewer_id has 500 unique values and no interviewer sees only one class, but its high cardinality makes it a noise risk. Drop unless target encoding proves useful.

# 6. Capital Features — Key Discriminative Signals

Capital gain, capital loss, and derived net capital asset are the most
economically meaningful numerical features. They show a clear gradient
across classes but are heavily zero-inflated.

| **Feature**           | **lower (mean)** | **middle (mean)** | **upper (mean)** | **Zero %** |
| --------------------- | ---------------- | ----------------- | ---------------- | ---------- |
| capital_gain          | 979.5            | 2,160.5           | 2,854.4          | 87.0%      |
| capital_loss          | 93.3             | 122.1             | 158.4            | 93.5%      |
| net_capital_asset     | 886.2            | 2,038.4           | 2,696.0          | 80.5%      |
| capital_activity_flag | 13.3%            | 19.7%             | 24.8%            | 80.5%      |

### Key Observations:

- Clear monotonic trend: upper > middle > lower across all capital metrics. This is the strongest economic signal in the dataset.
- capital_gain: 87% zeros, max = 99,999. Highly right-skewed with extreme outliers. Log transformation or binning will be needed.
- capital_loss: 93.5% zeros, max = 3,900. Less discriminative than gain but still useful.
- net_capital_asset = capital_gain - capital_loss. It captures the net effect and may be more stable than raw gain/loss.
- capital_activity_flag indicates whether a person had any capital activity. 24.8% of upper class have capital activity vs only 13.3% of lower class.
- Feature engineering opportunity: create capital_gain_minus_loss ratio, capital_gain per hour worked, and binary has_capital_gain/has_capital_loss flags.

# 7. Feature Correlation with Target

Correlation coefficients (Pearson) between numerical features and the
encoded target (lower=0, middle=1, upper=2). All correlations are
modest, indicating no single feature dominates — a good sign for ensemble
models.

| **Feature**           | **Correlation (r)** | **Interpretation**                                    |
| --------------------- | ------------------- | ----------------------------------------------------- |
| education_num         | +0.144              | Higher education correlates with higher class         |
| hours_per_week        | +0.114              | More hours worked correlates with higher class        |
| annual_hours_est      | +0.114              | Same as above (derived from hours_per_week)           |
| capital_activity_flag | +0.113              | Capital activity correlates with higher class         |
| year_of_birth         | -0.114              | Younger people slightly more likely to be upper class |
| capital_gain          | +0.070              | Weak positive correlation                             |
| net_capital_asset     | +0.068              | Weak positive correlation                             |
| capital_loss          | +0.052              | Weakest capital correlation                           |
| bag_size              | -0.016              | Negligible                                            |
| survey_duration_mins  | +0.007              | Negligible                                            |

### Key Observations:

- education_num has the strongest correlation at r=+0.144. Education is the single best numerical predictor of economic class.
- hours_per_week and annual_hours_est are perfectly correlated (r would be ~1.0 between them). Drop one to avoid redundancy.
- year_of_birth has a negative correlation — younger individuals trend toward higher classes, possibly reflecting career stage.
- No feature has a correlation above 0.15, confirming that the signal is distributed across many features rather than concentrated in one.

# 8. Categorical Features by Class

## 8.1 Education Tier

Education tier shows a clear gradient across classes:

| **Education Tier** | **lower %** | **middle %** | **upper %** |
| ------------------ | ----------- | ------------ | ----------- |
| Primary            | 39.7%       | 38.5%        | 21.7%       |
| Secondary          | 30.4%       | 39.8%        | 29.8%       |
| Higher             | 21.1%       | 40.6%        | 38.3%       |

- Primary education: 39.7% are lower class, only 21.7% upper.
- Higher education: 38.3% are upper class, only 21.1% lower.
- Secondary is the most balanced but still peaks at middle class (39.8%).
- Education tier is likely one of the most predictive categorical features.

## 8.2 Other Categorical Features

Key patterns observed across other categorical features:

- occupation: High-skill occupations (Exec-managerial, Prof-specialty) skew upper. Service and labor occupations skew lower.
- workclass: Federal-gov and Self-emp skew upper. Private is spread across all classes.
- marital_status: Married-civ-spouse skews middle/upper. Never-married skews lower.
- sex: Distribution is relatively balanced across classes — weak predictor.
- race: White dominates the dataset; other races are minorities. Class distribution varies slightly by race but not dramatically.
- native_country: United-States dominates (>90%). Other countries are sparse.
- interview_mode: 'in-person' dominates. Minimal variation across classes.

# 9. Key Takeaways

## 9.1 What Works in Our Favor

- No missing values — clean dataset, no imputation needed.
- Zero mixed-label bags — row-level and bag-level predictions are consistent.
- Clear economic gradient in capital features and education — signal is present.
- Moderate imbalance (1.4x) — manageable with class weights.

## 9.2 Challenges

- Modest correlations — no single strong predictor. Requires good feature engineering and model capacity.
- Zero-inflated capital features — 80-93% zeros. Needs special handling (log transform, binning, or separate binary features).
- High-cardinality categoricals — occupation, interviewer_id, native_country need careful encoding.
- Bag-level aggregation adds complexity — must ensure CV strategy respects bag structure.

## 9.3 Recommended Preprocessing Pipeline

- Drop: processing_flag, survey_year, currency_code, poverty_line_usd, is_adult_flag, bag_id, person_idx
- Drop one of: hours_per_week or annual_hours_est (redundant)
- Encode: education_tier as ordinal, occupation/workclass as target encoding or frequency encoding
- Transform: log(capital_gain + 1), log(capital_loss + 1), clip outliers at 99th percentile
- Engineer: net_capital = gain - loss, capital_ratio = gain / (loss + 1), has_capital_activity flag
- Aggregate: bag-level statistics (mean education_num, max capital_gain, mode occupation)

## 9.4 Recommended Modeling Strategy

- Baseline: Logistic Regression with class_weight='balanced' to establish a floor.
- Primary model: LightGBM or XGBoost with class_weight or scale_pos_weight per class.
- Validation: GroupKFold with bag_id groups, 5 folds. Ensure no bag leakage across splits.
- Evaluation: Macro F1 as primary metric. Also track per-class F1 to monitor minority class performance.
- Ensemble: Blend LightGBM + XGBoost + potentially a neural network for marginal gains.
