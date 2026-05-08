import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

DATA_DIR = Path("code-rush-26-ml-module")
OUT_DIR = Path("eda_plots")
OUT_DIR.mkdir(exist_ok=True)

# ── Load ──
train = pd.read_csv(DATA_DIR / "Coderush-26-ML-Train.csv")
test = pd.read_csv(DATA_DIR / "Coderush-26-ML-test.csv")
print(f"Train: {train.shape} | Test: {test.shape}")
print(f"Train columns ({len(train.columns)}): {list(train.columns)}\n")

# ─────────────────────────────────────────────
# 1. CLASS BALANCE — label distribution
# ─────────────────────────────────────────────
print("=" * 60)
print("1. CLASS BALANCE (label)")
print("=" * 60)
class_counts = train["label"].value_counts().sort_index()
class_pct = train["label"].value_counts(normalize=True).sort_index() * 100
print(f"\n{'Class':<10} {'Count':>7} {'%':>7}")
for cls, cnt in class_counts.items():
    print(f"{cls:<10} {cnt:>7} {class_pct[cls]:>6.1f}%")

minority = class_counts.idxmin()
majority = class_counts.idxmax()
imbalance_ratio = class_counts[majority] / class_counts[minority]
print(f"\nMinority class: {minority} ({class_counts[minority]} samples)")
print(f"Majority class: {majority} ({class_counts[majority]} samples)")
print(f"Imbalance ratio (majority/minority): {imbalance_ratio:.1f}x")
print("⚠ Macro F1 will penalise models that ignore the minority class heavily.\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=train, x="label", ax=axes[0], palette="Set2", order=["lower", "middle", "upper"])
axes[0].set_title("Class Distribution (Train)")
axes[0].set_xlabel("")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontweight="bold")

train["label"].value_counts(normalize=True).plot(kind="bar", ax=axes[1], color=["#e74c3c", "#f39c12", "#2ecc71"])
axes[1].set_title("Class Proportions")
axes[1].set_xlabel("")
axes[1].set_ylabel("Proportion")
axes[1].set_xticklabels(["lower", "middle", "upper"], rotation=0)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_class_balance.png", bbox_inches="tight")
plt.close()
print("Saved: eda_plots/01_class_balance.png\n")

# ─────────────────────────────────────────────
# 2. BAG SIZES — distribution & aggregation strategy
# ─────────────────────────────────────────────
print("=" * 60)
print("2. BAG SIZE ANALYSIS")
print("=" * 60)
bag_stats = train.groupby("bag_id").agg(
    bag_size=("bag_size", "first"),
    n_members=("person_idx", "count"),
    labels=("label", lambda x: ", ".join(sorted(x.unique())))
).reset_index()

print(f"\nTotal bags: {bag_stats.shape[0]}")
print(f"\nBag size distribution (from bag_size column):")
print(bag_stats["bag_size"].value_counts().sort_index().to_string())

print(f"\nActual members per bag (from person_idx count):")
print(bag_stats["n_members"].value_counts().sort_index().head(10).to_string())

# check consistency
mismatch = (bag_stats["bag_size"] != bag_stats["n_members"]).sum()
print(f"\nbag_size vs actual members mismatch: {mismatch}/{len(bag_stats)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(bag_stats["bag_size"], bins=bag_stats["bag_size"].nunique(), ax=axes[0], color="#3498db", kde=False)
axes[0].set_title("Bag Size Distribution")
axes[0].set_xlabel("Bag Size")

sns.countplot(data=bag_stats, x="n_members", ax=axes[1], color="#e67e22")
axes[1].set_title("Actual Members per Bag")
axes[1].set_xlabel("Member Count")

# bag size vs class
bag_class = train.groupby(["bag_id", "label"]).size().unstack(fill_value=0)
bag_class_pct = bag_class.div(bag_class.sum(axis=1), axis=0).stack().reset_index(name="pct")
sns.boxplot(data=bag_stats.merge(train[["bag_id", "label"]].drop_duplicates(), on="bag_id"),
            x="label", y="bag_size", ax=axes[2], palette="Set2", order=["lower", "middle", "upper"])
axes[2].set_title("Bag Size by Class")
axes[2].set_xlabel("")

plt.tight_layout()
plt.savefig(OUT_DIR / "02_bag_sizes.png", bbox_inches="tight")
plt.close()
print("Saved: eda_plots/02_bag_sizes.png\n")

# Check label consistency within bags
mixed_bags = bag_stats[~bag_stats["labels"].str.contains("^[a-z]+$", regex=True)]
print(f"Bags with mixed labels: {len(mixed_bags)} / {len(bag_stats)} ({len(mixed_bags)/len(bag_stats)*100:.1f}%)")
if len(mixed_bags) > 0:
    print("Label combinations in mixed bags:")
    print(mixed_bags["labels"].value_counts().head(10).to_string())

# ─────────────────────────────────────────────
# 3. NULLS & LEAKAGE CHECK
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. NULLS & LEAKAGE ANALYSIS")
print("=" * 60)

null_info = pd.DataFrame({
    "train_nulls": train.isnull().sum(),
    "train_%": train.isnull().mean() * 100,
    "test_nulls": test.isnull().sum(),
    "test_%": test.isnull().mean() * 100,
})
null_info = null_info[null_info["train_nulls"] > 0].sort_values("train_nulls", ascending=False)
if len(null_info) > 0:
    print("\nColumns with NULLs:")
    print(null_info.to_string())
else:
    print("\n✓ No NULL values found in train or test.")

# Leakage / noise candidates
print("\n─ Potential leakage / noise features ─")
leaky_candidates = ["interviewer_id", "processing_flag", "survey_year", "currency_code",
                    "poverty_line_usd", "is_adult_flag", "person_idx", "bag_id", "bag_size"]

for col in leaky_candidates:
    if col in train.columns:
        nuniq = train[col].nunique()
        train_vals = train[col].value_counts(normalize=True).head(3)
        info = f"  {col:<25} unique={nuniq:<8} top3={list(train_vals.items())}"
        
        # check if constant or near-constant
        if nuniq == 1:
            info += " ⚠ CONSTANT — drop (leakage/noise)"
        elif train_vals.iloc[0] > 0.95:
            info += f" ⚠ NEAR-CONSTANT ({train_vals.iloc[0]:.0%}) — consider dropping"
        elif col in ["poverty_line_usd", "is_adult_flag", "survey_year", "currency_code"]:
            info += " ⚠ POTENTIAL LEAKAGE — same for all rows or external info"
        print(info)

# Check poverty_line_usd constancy
if "poverty_line_usd" in train.columns:
    print(f"\n  poverty_line_usd unique values: {train['poverty_line_usd'].nunique()}")
    print(f"  Value counts: {train['poverty_line_usd'].value_counts().to_dict()}")

if "is_adult_flag" in train.columns:
    print(f"\n  is_adult_flag value counts: {train['is_adult_flag'].value_counts().to_dict()}")

if "survey_year" in train.columns:
    print(f"\n  survey_year value counts: {train['survey_year'].value_counts().to_dict()}")

if "currency_code" in train.columns:
    print(f"\n  currency_code value counts: {train['currency_code'].value_counts().to_dict()}")

# interviewer_id cardinality
if "interviewer_id" in train.columns:
    id_vc = train["interviewer_id"].value_counts()
    print(f"\n  interviewer_id: {len(id_vc)} unique, mean bags per interviewer={id_vc.mean():.1f}")
    # check if any interviewer is correlated with a class
    int_class = train.groupby("interviewer_id")["label"].agg(lambda x: x.mode().iloc[0])
    int_entropy = train.groupby("interviewer_id")["label"].apply(lambda x: len(x.unique()))
    print(f"  Interviewers seeing only 1 class: {(int_entropy == 1).sum()}")

# processing_flag
if "processing_flag" in train.columns:
    print(f"\n  processing_flag value counts:\n{train['processing_flag'].value_counts()}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

cols_to_check = ["survey_year", "currency_code", "processing_flag", "is_adult_flag",
                 "interviewer_id", "poverty_line_usd"]
for i, col in enumerate(cols_to_check):
    if col not in train.columns:
        axes[i].axis("off")
        continue
    if train[col].nunique() <= 20:
        sns.countplot(y=train[col], ax=axes[i], palette="muted")
        axes[i].set_title(f"{col} distribution")
    else:
        axes[i].hist(train[col], bins=50, color="#3498db", edgecolor="white")
        axes[i].set_title(f"{col} histogram")

plt.tight_layout()
plt.savefig(OUT_DIR / "03_nulls_leakage.png", bbox_inches="tight")
plt.close()
print("\nSaved: eda_plots/03_nulls_leakage.png")

# ─────────────────────────────────────────────
# 4. CAPITAL GAIN / LOSS — discriminative power
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. CAPITAL GAIN / LOSS ANALYSIS")
print("=" * 60)

for col in ["capital_gain", "capital_loss", "net_capital_asset", "capital_activity_flag"]:
    if col in train.columns:
        print(f"\n{col}:")
        print(f"  Stats:\n{train[col].describe()}")
        zeros = (train[col] == 0).sum()
        print(f"  Zero values: {zeros} ({zeros/len(train)*100:.1f}%)")
        print(f"  By class:")
        by_class = train.groupby("label")[col].agg(["mean", "median", "std", "max"])
        print(by_class.to_string())

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Capital gain distribution by class
for i, (col, title) in enumerate([("capital_gain", "Capital Gain"),
                                   ("capital_loss", "Capital Loss"),
                                   ("net_capital_asset", "Net Capital Asset")]):
    if col not in train.columns:
        axes[0, i].axis("off")
        continue
    df_plot = train[train[col] > 0]  # exclude zeros for clarity
    sns.boxplot(data=df_plot, x="label", y=col, ax=axes[0, i],
                palette="Set2", order=["lower", "middle", "upper"])
    axes[0, i].set_title(f"{title} by Class (non-zero)")
    axes[0, i].set_xlabel("")

# Zero vs non-zero stacked bar
for i, col in enumerate(["capital_gain", "capital_loss", "capital_activity_flag"]):
    if col not in train.columns:
        axes[1, i].axis("off")
        continue
    crosstab = pd.crosstab(train["label"], train[col] > 0, normalize="index")
    crosstab.columns = ["Zero", "Non-Zero"]
    crosstab.plot(kind="bar", stacked=True, ax=axes[1, i], color=["#95a5a6", "#e74c3c"])
    axes[1, i].set_title(f"{col}: Zero vs Non-Zero by Class")
    axes[1, i].set_xlabel("")
    axes[1, i].legend(title=col)
    axes[1, i].set_xticklabels(["lower", "middle", "upper"], rotation=0)

plt.tight_layout()
plt.savefig(OUT_DIR / "04_capital_analysis.png", bbox_inches="tight")
plt.close()
print("Saved: eda_plots/04_capital_analysis.png")

# capital_gain vs capital_loss scatter
if "capital_gain" in train.columns and "capital_loss" in train.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, color in zip(["lower", "middle", "upper"], ["#e74c3c", "#f39c12", "#2ecc71"]):
        sub = train[train["label"] == label]
        axes[0].scatter(sub["capital_gain"], sub["capital_loss"], alpha=0.3, s=8,
                        color=color, label=label, edgecolors="none")
    axes[0].set_xlabel("Capital Gain")
    axes[0].set_ylabel("Capital Loss")
    axes[0].set_title("Capital Gain vs Loss by Class")
    axes[0].legend()
    axes[0].set_xlim(0, train["capital_gain"].quantile(0.99))
    axes[0].set_ylim(0, train["capital_loss"].quantile(0.99))

    # KDE
    for label, color in zip(["lower", "middle", "upper"], ["#e74c3c", "#f39c12", "#2ecc71"]):
        sub = train[train["label"] == label]
        sns.kdeplot(sub["capital_gain"], fill=True, alpha=0.3, ax=axes[1], color=color, label=label)
    axes[1].set_title("Capital Gain KDE by Class")
    axes[1].set_xlabel("Capital Gain")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_capital_scatter.png", bbox_inches="tight")
    plt.close()
    print("Saved: eda_plots/05_capital_scatter.png")

# ─────────────────────────────────────────────
# 5. NUMERICAL FEATURES — overall summary + correlations
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. NUMERICAL FEATURES SUMMARY")
print("=" * 60)

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
# remove ID columns
num_cols = [c for c in num_cols if c not in ["bag_id", "person_idx"]]
print(f"\nNumerical columns ({len(num_cols)}): {num_cols}")
print("\n" + train[num_cols].describe().to_string())

# correlation with label (encode label)
label_enc = train["label"].map({"lower": 0, "middle": 1, "upper": 2})
corrs = []
for col in num_cols:
    corr = train[col].corr(label_enc)
    corrs.append((col, corr))
corrs.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nFeature correlation with label (encoded 0→lower, 1→middle, 2→upper):")
for col, corr in corrs:
    print(f"  {col:<25} r={corr:+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
corr_matrix = train[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0],
            vmin=-1, vmax=1, annot_kws={"size": 7})
axes[0].set_title("Numerical Feature Correlation Matrix")

# correlation with label
corr_df = pd.DataFrame(corrs, columns=["feature", "correlation"]).sort_values("correlation", key=abs, ascending=False)
colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in corr_df["correlation"]]
sns.barplot(data=corr_df, x="correlation", y="feature", ax=axes[1], palette=colors)
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_title("Feature Correlation with Label")
axes[1].set_xlabel("Correlation coefficient")

plt.tight_layout()
plt.savefig(OUT_DIR / "06_numerical_corr.png", bbox_inches="tight")
plt.close()
print("Saved: eda_plots/06_numerical_corr.png")

# ─────────────────────────────────────────────
# 6. CATEGORICAL FEATURES — by class
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. CATEGORICAL FEATURES BY CLASS")
print("=" * 60)

cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "label"]
print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(cat_cols[:9]):
    if col not in train.columns:
        axes[i].axis("off")
        continue
    nuniq = train[col].nunique()
    if nuniq > 15:
        # too many categories — show top 10
        top_cats = train[col].value_counts().head(10).index
        df_sub = train[train[col].isin(top_cats)]
        ctab = pd.crosstab(df_sub[col], df_sub["label"], normalize="index")
    else:
        ctab = pd.crosstab(train[col], train["label"], normalize="index")
    ctab.plot(kind="barh", stacked=True, ax=axes[i], color=["#e74c3c", "#f39c12", "#2ecc71"])
    axes[i].set_title(f"{col} (n={nuniq})")
    axes[i].set_xlabel("Proportion")
    axes[i].legend(title="label", loc="lower right")

plt.tight_layout()
plt.savefig(OUT_DIR / "07_categorical_by_class.png", bbox_inches="tight")
plt.close()
print("Saved: eda_plots/07_categorical_by_class.png")

# ─────────────────────────────────────────────
# 7. KEY NUMERICAL FEATURES — boxplots by class
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. NUMERICAL FEATURES — DISTRIBUTION BY CLASS")
print("=" * 60)

key_numerical = ["education_num", "hours_per_week", "survey_duration_mins",
                 "annual_hours_est", "year_of_birth"]
key_numerical = [c for c in key_numerical if c in train.columns]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(key_numerical):
    if col not in train.columns:
        axes[i].axis("off")
        continue
    sns.boxplot(data=train, x="label", y=col, ax=axes[i],
                palette="Set2", order=["lower", "middle", "upper"])
    axes[i].set_title(f"{col} by Class")
    axes[i].set_xlabel("")

axes[5].axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "08_numerical_boxplots.png", bbox_inches="tight")
plt.close()
print("Saved: eda_plots/08_numerical_boxplots.png")

# ─────────────────────────────────────────────
# 8. EDUCATION TIER — ordered analysis
# ─────────────────────────────────────────────
if "education_tier" in train.columns:
    print("\n" + "=" * 60)
    print("8. EDUCATION TIER ANALYSIS")
    print("=" * 60)
    tier_order = ["Primary", "Secondary", "Higher"]
    tier_present = [t for t in tier_order if t in train["education_tier"].values]
    ctab = pd.crosstab(train["education_tier"], train["label"], normalize="index")
    ctab = ctab.reindex(tier_present)
    print("\nEducation Tier → Class (row proportions):")
    print(ctab.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=train, x="education_tier", hue="label", ax=axes[0],
                  palette="Set2", order=tier_present)
    axes[0].set_title("Education Tier Count by Class")
    axes[0].set_xlabel("")
    ctab.plot(kind="bar", stacked=True, ax=axes[1], color=["#e74c3c", "#f39c12", "#2ecc71"])
    axes[1].set_title("Education Tier → Class Proportions")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Proportion")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_education_tier.png", bbox_inches="tight")
    plt.close()
    print("Saved: eda_plots/09_education_tier.png")

# ─────────────────────────────────────────────
# 9. SUMMARY — feature types & recommendations
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. EDA SUMMARY & RECOMMENDATIONS")
print("=" * 60)

print(f"""
TARGET:
  Classes: {list(class_counts.index)}
  Distribution: {dict(class_pct.round(1))}
  Imbalance: {imbalance_ratio:.1f}x → Use Macro F1 / class weights

BAGS:
  Total bags: {bag_stats.shape[0]}
  Bag sizes: {sorted(bag_stats['bag_size'].unique())}
  Mixed-label bags: {len(mixed_bags)} ({len(mixed_bags)/len(bag_stats)*100:.1f}%)
  → Strategy: {'Aggregate at bag level (mean/mode)' if bag_stats['bag_size'].max() > 1 else 'Row-level prediction'}

FEATURES TO DROP (likely leakage/noise):
  - poverty_line_usd: constant external value
  - is_adult_flag: derived from year_of_birth
  - survey_year: same for all rows
  - currency_code: same for all rows
  - bag_id, person_idx: IDs
  - interviewer_id: high cardinality, potential leakage

STRONG PREDICTORS (likely):
  - capital_gain, capital_loss, net_capital_asset: economic signal
  - education_num, education_tier: education level
  - hours_per_week, annual_hours_est: work intensity
  - occupation, workclass: employment type

NULLS:
  {'None found' if len(null_info) == 0 else f'See null report above'}

RECOMMENDED NEXT STEPS:
  1. Decide aggregation strategy (bag-level vs row-level)
  2. Encode categoricals (target encoding for high-cardinality like occupation)
  3. Engineer features: capital_gain - capital_loss, income proxy = hours × workclass proxy
  4. Use Stratified K-Fold preserving bag membership
  5. Baseline: LightGBM / XGBoost with class_weight='balanced'
""")
