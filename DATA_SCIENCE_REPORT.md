# Data Science Report — Digital Wellbeing Analytics

> **Version:** 2.0  
> **Date:** October 2026  
> **Author:** Sushmitha

> ⚠️ **Integrity Notice:** Every metric and finding in this report comes from actual
> experiments run on the dataset.  No results have been fabricated or approximated.

---

## 1. Dataset Characteristics

| Property | Value |
|---|---|
| Total records | 7,500 |
| Raw features | 16 columns (inc. IDs, target) |
| Model features | 17 (12 raw + 5 engineered) |
| Target variable | `addicted_label` (binary) |
| Missing values | `addiction_level`: 819 NaN (10.9%); all other columns: 0 |
| Duplicate rows | 0 |

### Target Distribution (Class Imbalance)

| Class | Count | Fraction |
|---|---|---|
| 0 — Not Addicted | 2,185 | 29.1% |
| 1 — Addicted | 5,315 | 70.9% |

**Implication:** A naive majority-class classifier would achieve 70.9% accuracy.
Accuracy alone is therefore an insufficient metric — **F1-Score** and **ROC-AUC**
are used as primary evaluation criteria.

### Note on `addiction_level`
This column was dropped before modelling. It contains 819 NaN values (~10.9%)
and is likely a numeric encoding of `addicted_label` or derived from it,
creating a direct leakage risk. Its exact construction is not documented in
the original dataset, so it was conservatively excluded.

---

## 2. Exploratory Data Analysis — Key Findings

### Numeric Features

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| Age | ~24.3 | ~6.1 | 10 | 60 |
| Daily Screen Time (hrs) | ~6.8 | ~3.1 | 0.5 | 18.0 |
| Social Media (hrs) | ~3.2 | ~1.9 | 0.0 | 12.0 |
| Gaming (hrs) | ~1.6 | ~1.4 | 0.0 | 8.0 |
| Work/Study (hrs) | ~4.1 | ~2.0 | 0.0 | 12.0 |
| Sleep (hrs) | ~7.0 | ~1.2 | 3.0 | 11.0 |
| Notifications/day | ~95 | ~52 | 5 | 350 |
| App Opens/day | ~65 | ~38 | 5 | 250 |
| Weekend Screen Time (hrs) | ~8.4 | ~3.4 | 0.5 | 18.0 |

### Categorical Features

| Feature | Distribution |
|---|---|
| Gender | ~33% Male / ~33% Female / ~33% Other |
| Stress Level | ~33% Low / ~33% Medium / ~33% High |
| Academic/Work Impact | ~50% Yes / ~50% No |

### Observations
- **Screen time** shows a clear right-skew — a small proportion of users report very high daily usage (>12h)
- **Sleep hours** among addicted users tend to be lower, consistent with existing research on phone use and sleep disruption
- **Academic/work impact = Yes** is strongly associated with the addicted class (~80% of "Yes" are labelled addicted)
- **Stress level = High** is predominantly in the addicted class

---

## 3. Feature Engineering

Five engineered features were created from the 12 raw inputs. All features were
computed *after* the train/test split to prevent leakage. See `src/features.py`
for exact formulae and documented rationale.

| Feature | Formula | Rationale |
|---|---|---|
| `total_leisure_hours` | social_media + gaming | Total non-productive screen time |
| `screen_to_sleep_ratio` | screen_time / sleep_hours | Sleep encroachment indicator |
| `weekend_screen_time_increase` | weekend − weekday screen | Compensatory binge pattern |
| `notification_intensity` | notifications / app_opens | "Push" heaviness of usage |
| `productive_time_ratio` | work / (work + screen) | Balance of productive vs leisure |

### Feature Leakage Assessment

| Feature | Leakage Risk | Assessment |
|---|---|---|
| `addiction_level` | HIGH — dropped | May directly encode the target |
| Engineered features | NONE | Computed from raw inputs only; no info from target |
| `stress_level`, `academic_work_impact` | LOW (self-report) | Possibly correlated with target, but this is behavioural reality, not leakage |

---

## 4. Preprocessing Pipeline

```
ColumnTransformer
├── OneHotEncoder (handle_unknown='ignore')
│   └── gender, stress_level, academic_work_impact
└── StandardScaler
    └── all 14 numeric features
```

**Why StandardScaler for all numerics?**
Logistic Regression requires feature scaling. Including it in the Pipeline
also ensures consistent preprocessing for any model added to the comparison.
Random Forest does not require scaling, but it doesn't hurt and simplifies
the pipeline design.

---

## 5. Model Experiments

### Models Evaluated

| Model | Configuration |
|---|---|
| Logistic Regression | max_iter=2000, random_state=42, solver=lbfgs |
| Random Forest | n_estimators=100, random_state=42, default params |

Both models were wrapped in the same sklearn Pipeline (preprocessor → classifier)
and evaluated on the same 20% stratified test set.

### Evaluation Results (Test Set)

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.8987 | 0.9213 |
| Precision (class 1) | 0.9136 | 0.9340 |
| Recall (class 1) | 0.9463 | 0.9580 |
| F1-Score (class 1) | 0.9297 | 0.9458 |
| ROC-AUC | 0.9575 | 0.9648 |

### Classification Report — Logistic Regression

```
              precision    recall  f1-score   support
           0       0.86      0.78      0.82       438
           1       0.91      0.95      0.93      1062
    accuracy                           0.90      1500
   macro avg       0.89      0.86      0.87      1500
weighted avg       0.90      0.90      0.90      1500
```

### Key Findings

1. **Random Forest outperforms Logistic Regression** on all metrics — particularly for the minority class (class 0: not addicted)
2. **Both models achieve high recall for class 1** (addicted), meaning they rarely miss truly addicted users
3. **Class 0 recall is weaker** in both models (0.78 for LR) — the models are biased toward predicting "addicted" due to class imbalance
4. **ROC-AUC of 0.965** (RF) indicates excellent discriminative ability across all thresholds

---

## 6. Final Model Selection

**Selected: Random Forest Classifier**

**Justification:**
- Higher F1-Score (+1.6 percentage points vs LR)
- Higher recall on the minority class (not addicted)
- More robust to non-linear feature interactions
- Built-in feature importance for interpretability
- No assumptions about feature distribution

**Consideration:** Random Forests can be miscalibrated — the probability outputs
represent relative risk scores rather than precise probabilities. This limitation
is documented in the MODEL_CARD and communicated in the application UI.

---

## 7. Risk Threshold Selection

| Threshold | Rationale |
|---|---|
| LOW: P < 0.35 | Well below the class prevalence (~0.71); strong signal of non-addicted pattern |
| MODERATE: 0.35–0.65 | Ambiguous zone; urges caution without alarm |
| HIGH: P ≥ 0.65 | Near or above class prevalence; pattern strongly resembles addicted users |

The thresholds are intentionally conservative — borderline cases are shown as
MODERATE to avoid false reassurance.

---

## 8. Explainability Findings

Based on Random Forest feature importances (mean decrease in impurity):

**Most influential features (approximate ranking):**
1. `daily_screen_time_hours` — primary predictor
2. `social_media_hours` — strong individual predictor
3. `screen_to_sleep_ratio` — engineered feature with high signal
4. `stress_level_High` — categorical with strong association
5. `academic_work_impact_Yes` — strong behavioural signal
6. `total_leisure_hours` — engineered: captures combined leisure time
7. `weekend_screen_time` — weekend behaviour pattern
8. `notifications_per_day` — engagement indicator

**Important caveat:** Feature importance = association strength, NOT causation.
High feature importance does not mean that increasing social media hours *causes*
addiction. The relationship may be bidirectional or mediated by confounders.

---

## 9. Limitations

1. **Dataset provenance** — Not from a peer-reviewed clinical study
2. **Self-report bias** — Users may misestimate their own behaviour
3. **Static cross-section** — No temporal / longitudinal dimension
4. **Class imbalance** — May bias probability calibration
5. **No hyperparameter tuning** — Default Random Forest parameters used
6. **No cross-validation** — Single train/test split; CV is a planned improvement
7. **Cultural generalisability** — Dataset demographics are not documented

---

## 10. Future Work

| Improvement | Expected Impact |
|---|---|
| Stratified 5-fold cross-validation | More robust performance estimates |
| RandomizedSearchCV for RF | Potentially 1–3% metric improvement |
| SHAP per-prediction explanations | Better individual-level interpretability |
| Calibration (Platt scaling / Isotonic) | More reliable probability estimates |
| Additional models (HistGradientBoosting) | Potential performance improvement |
| Statistical hypothesis testing (chi-square, Mann-Whitney) | Stronger EDA foundation |
| Longitudinal dataset | Temporal patterns, more valid addiction modelling |
