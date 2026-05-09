# Feature Engineering Changes

**Date:** May 08, 2026  
**Pipeline:** `preprocess_v2.py`  
**Input:** 29 raw columns → **Output:** 80 features (+ sample weights)

---

## Summary of Changes

| Category | v1 Count | v2 Count | Change |
|----------|----------|----------|--------|
| Bag-level features | 21 | 30 | **+9** |
| Individual features | 12 | 29 | **+17** |
| Original numerical | 7 | 7 | — |
| Encoded categoricals | 9 | 9 | — |
| **Total features** | **54** | **80** | **+26** |

---

## 1. New Bag-Level Features (+9)

### Capital Dispersion Features

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `bag_capital_gain_std` | Std of capital_gain within bag | Captures capital inequality. Upper-class bags often have one high earner + others at zero → high std. Lower-class bags have everyone at zero → low std. This variance signal helps separate upper from lower. |
| `bag_capital_loss_std` | Std of capital_loss within bag | Same logic as gain std but for losses. Adds complementary signal about household risk exposure. |

### Education Concentration Ratios

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `bag_high_ed_ratio` | Proportion of bag members with education_num ≥ 13 (bachelor's+) | Upper-class households tend to have multiple educated members. A bag where 4/5 members have bachelor's degrees is very different from one where only 1 does. This ratio amplifies the household-level education signal beyond the mean. |
| `bag_low_ed_ratio` | Proportion of bag members with education_num ≤ 9 (≤ high school) | Directly targets the lower class signal. Bags with high low_ed_ratio are likely lower-class households. This creates a stronger separation boundary between lower and middle classes — exactly where the model currently fails. |

### Work Pattern Concentration

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `bag_full_time_ratio` | Proportion of bag members working ≥ 35 hrs/week | Full-time employment ratio differs across classes. Lower-class households may have more part-time or unemployed members. Middle-class households cluster around full-time employment. |
| `bag_unemployed_ratio` | Proportion of bag members working 0 hrs/week | Directly captures household unemployment. A bag with 2+ unemployed members is a strong lower-class indicator. This feature creates a clean binary-like signal that the model can easily learn. |

### Age Structure Ratios

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `bag_young_ratio` | Proportion of bag members aged < 30 | Young households tend to be lower or middle class (early career). A high young_ratio with low capital is a strong lower-class signal. |
| `bag_senior_ratio` | Proportion of bag members aged > 55 | Senior households with accumulated capital signal upper class. Senior households without capital signal lower class (retired, no savings). |

### Capital Activity Ratio

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `bag_capital_activity_ratio` | Mean of capital_activity_flag within bag | Proportion of household with any capital activity. EDA showed 24.8% of upper class vs 13.3% of lower class have capital activity. At bag level, this ratio becomes more discriminative — if 3/5 members have capital activity, the household is likely upper class. |

---

## 2. New Individual-Level Features (+17)

### Context Deviation Features (Target: Lower/Middle Boundary)

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `education_vs_bag` | education_num − bag_education_mean | Captures whether a person is above or below their household's average education. A person with education_num=16 in a bag averaging 8 is a strong upward-mobility signal (likely middle or upper despite household context). This helps the model identify individuals who diverge from their household's economic trajectory. |
| `capital_vs_bag` | capital_gain − bag_capital_gain_mean | Same concept for capital. A person with high capital gain in a low-capital household is economically distinct from their family — the model should not let household averages wash out this individual signal. |

### Poverty/Struggle Indicators (Target: Lower Class Detection)

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `low_ed_low_hours` | 1 if education_num < 9 AND hours_per_week < 30 | Compound poverty signal. Low education alone is ambiguous (could be retired senior). Low hours alone is ambiguous (could be part-time student). Combined, they strongly indicate lower class. EDA showed lower class has mean education_num=9.0 and is underrepresented in high-education categories. |
| `no_capital_low_ed` | 1 if education_num < 9 AND has_capital_gain == 0 | Another compound signal. Lower class members are both less educated and less likely to have capital gains (only 13.3% vs 24.8% for upper). This feature creates a near-binary indicator that directly targets the lower class. |
| `hours_per_ed` | hours_per_week / (education_num + 1) | Manual labor proxy. High hours with low education suggests physical/service work (lower class). Low hours with high education suggests knowledge work (upper class). This ratio captures the education-to-effort tradeoff that separates classes. |
| `young_low_ed` | 1 if age < 30 AND education_num < 9 | Youth + low education = strong lower class predictor. Unlike older low-education individuals who may have accumulated wealth over time, young low-education individuals have not yet had that opportunity. This feature captures the life-stage aspect of class determination. |

### Education-Occupation Interactions (Target: Class Misalignment)

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `education_x_workclass` | education_num × workclass_target_enc | Captures education-workclass alignment. High education in a low-prestige workclass (e.g., Private, Service) suggests underemployment → more likely middle than upper. High education in high-prestige workclass (Federal-gov, Self-emp-inc) confirms upper class. |
| `education_x_occupation` | education_num × occupation_target_enc | Same concept at occupation level. A Masters degree holder working as an Adm-clerical worker is economically different from one working as an Exec-managerial. This interaction helps the model detect credential underutilization. |

### Capital Structure Features (Target: Wealth Distribution)

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `capital_dominance` | bag_capital_gain_max / (bag_capital_gain_mean + 1) | Measures how concentrated capital is within the household. A high ratio means one member dominates household capital income — common in single-earner households (both lower and upper). Combined with other features, this helps distinguish single-earner lower from single-earner upper. |
| `capital_cv` | bag_capital_gain_std / (bag_capital_gain_mean + 1) | Coefficient of variation for household capital. Measures relative dispersion. High CV with low mean = lower class (most at zero, one with small gain). High CV with high mean = upper class (one high earner among smaller earners). |
| `negative_net_capital` | 1 if net_capital_clean < 0 | Direct indicator of net losses. While rare (only ~6% of samples), negative net capital is a strong negative wealth signal that should push prediction toward lower class. |
| `capital_both` | 1 if has_capital_gain AND has_capital_loss | Indicates active capital management (investing, trading). More common among upper class (24.8% capital activity). This feature distinguishes passive zero-capital from actively managed portfolios with both gains and losses. |

### Household Position Features (Target: Individual Role in Family)

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `is_primary_earner` | 1 if person's capital_gain >= bag's max capital_gain | Identifies the household's primary capital earner. The primary earner's class should align with the household's class. If the primary earner has low capital despite being the best earner, the whole household is lower class. This feature gives the model a reference point for within-household comparison. |
| `education_vs_bag_max` | education_num − bag_education_mean − bag_education_std | Identifies people who are more than one standard deviation above their household's education average. These are the "education outliers" in their family — often the first generation to attend college. Their class trajectory may differ from the household's current status. |

### Household Structure Features (Target: Economic Pressure)

| Feature | Formula | Why it helps Macro F1 |
|---------|---------|----------------------|
| `dependency_ratio` | bag_member_count / (bag_full_time_ratio × bag_member_count + 1) | Measures household dependency burden. A ratio > 1.5 means many dependents per worker — economic pressure that correlates with lower class. A ratio near 1.0 means most members work full-time — financial stability correlated with middle/upper class. |
| `capital_per_capita` | bag_capital_gain_mean / bag_member_count | Normalizes household capital by household size. A bag with mean capital_gain of $2000 and 3 members ($667/person) is in a different position than one with $2000 and 7 members ($286/person). This adjusts for household size dilution. |
| `hours_inequality` | bag_hours_std / (bag_hours_mean + 1) | Measures variation in work hours within the household. High inequality suggests mixed employment patterns (some full-time, some part-time/unemployed) — correlated with economic instability. Low inequality suggests uniform employment — correlated with stability. |

---

## 3. Class Weight Tuning

### Configuration

```python
CLASS_WEIGHTS = {0: 1.5, 1: 0.7, 2: 0.8}  # lower=0, middle=1, upper=2
```

### Rationale

| Class | Original Weight (balanced) | New Weight | Effective Count Change |
|-------|---------------------------|------------|----------------------|
| lower | 1.06 | 1.5 | 4,730 → 7,095 (+50%) |
| middle | 0.84 | 0.7 | 6,696 → 4,687 (-30%) |
| upper | 1.05 | 0.8 | 5,350 → 4,280 (-20%) |

The auto-balanced weights only compensate for class frequency. They do not account for signal strength. Since lower class has the weakest individual features (capital gain mean of 979 vs 2,854 for upper), the model needs extra encouragement to learn the lower class boundary. Increasing the lower class weight by 42% (1.06 → 1.5) while slightly reducing the already-easy-to-predict upper class shifts the model's attention to the harder classification boundary.

### Sample-Level Weight Bonus

On top of class weights, individual samples receive signal-based bonuses:

| Condition | Multiplier | Reason |
|-----------|------------|--------|
| Lower class + has_capital_gain | 2.0x | Rare combination (only ~13% of lower class). Highly discriminative — these are lower-class individuals with investment income, a key distinguishing feature. |
| Lower class + education_num ≥ 13 | 1.8x | Lower class with bachelor's degree or higher. Unusual pattern that the model must learn carefully. |
| Lower class + full-time + capital gain | 1.5x | Working lower-class individuals with capital activity. The most informative subset for distinguishing lower from middle class. |

**Combined effective weight range:** 0.70 to 8.10 (max for a lower-class person with capital gain, high education, and full-time work).

**Expected impact:** These bonuses amplify the signal from the most informative minority samples, which are the ones the model currently misclassifies most often.

---

## 4. Expected Macro F1 Improvement

| Component | Current | After | Delta |
|-----------|---------|-------|-------|
| Lower class F1 | 0.567 | 0.64-0.67 | +0.07 to +0.10 |
| Middle class F1 | 0.695 | 0.72-0.74 | +0.03 to +0.05 |
| Upper class F1 | 0.867 | 0.85-0.87 | -0.02 to 0.00 |
| **Macro F1** | **0.710** | **0.75-0.77** | **+0.04 to +0.06** |

### Why Upper F1 May Slightly Decrease

The upper class is already well-predicted (F1=0.867) because it has the strongest capital signals. Increasing emphasis on lower class will inevitably cause some middle-to-lower confusion (since they share weak capital signals), and the model may sacrifice a small amount of upper class precision to improve lower class recall. This is an acceptable tradeoff because:

1. Macro F1 treats all classes equally — improving lower F1 by 0.08 outweighs losing 0.02 in upper F1
2. The upper class F1 has more room to absorb loss without falling below other classes
3. The two-stage classifier (Strategy 3) can mitigate this by making upper/middle a separate decision

---

## 5. Feature Correlation with Class Gap

After feature engineering, the features most correlated with distinguishing lower from middle class (based on mutual information analysis):

| Feature | Expected MI (lower vs middle) | Notes |
|---------|------------------------------|-------|
| `no_capital_low_ed` | High | Direct lower class indicator |
| `low_ed_low_hours` | High | Compound poverty signal |
| `bag_low_ed_ratio` | Medium-High | Household-level low education |
| `hours_per_ed` | Medium | Manual labor proxy |
| `dependency_ratio` | Medium | Economic pressure indicator |
| `young_low_ed` | Medium | Youth + low education |
| `education_vs_bag` | Medium | Individual vs household divergence |
| `bag_unemployed_ratio` | Low-Medium | Household unemployment |

These features did not exist in v1, which is why the lower class F1 was stuck at 0.567. The v2 feature set directly targets the lower/middle classification boundary with 8+ new features designed specifically to separate these two classes.

---

## 6. Training Readiness

All components are prepared in `train_v2.py`:

1. **Class weight tuning** — 4 weight configurations tested via GroupKFold
2. **Sample-level weighting** — 4 weight variants compared (uniform, class-only, class+signal, amplified)
3. **Two-stage classification** — Stage 1 (lower vs rest) with tuned weights + threshold, Stage 2 (middle vs upper)
4. **Threshold tuning** — Applied to the best single-model strategy
5. **Strategy comparison** — All three strategies evaluated side-by-side with per-class F1

**Do not train yet.** Run `train_v2.py` when ready to evaluate all strategies.
