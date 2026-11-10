# Model Card — Digital Wellbeing Analytics

> **Version:** 2.0  
> **Date:** October 2026  
> **Model Type:** Random Forest Classifier (scikit-learn Pipeline)  
> **Repository:** https://github.com/Sushmitha-328/Digital-Wellbeing-Analytics

---

## Model Purpose

This model estimates the **behavioural risk of smartphone over-reliance** from self-reported
usage patterns.  It is intended as an **awareness tool** to help individuals reflect on their
digital habits — not as a clinical diagnostic instrument.

---

## Training Data

| Property | Value |
|---|---|
| Source | Smartphone Usage Survey Dataset |
| Records | 7,500 |
| Features | 12 raw + 5 engineered = 17 total |
| Target | `addicted_label` (binary: 0 = not addicted, 1 = addicted) |
| Class distribution | ~70.8% addicted / 29.2% not addicted |
| Train / Test split | 80% / 20%, stratified |

### Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | User age in years |
| `gender` | Categorical | Male / Female / Other |
| `daily_screen_time_hours` | Numeric | Average daily screen time (hours) |
| `social_media_hours` | Numeric | Daily social media usage (hours) |
| `gaming_hours` | Numeric | Daily gaming hours |
| `work_study_hours` | Numeric | Daily work or study hours |
| `sleep_hours` | Numeric | Average sleep duration (hours) |
| `notifications_per_day` | Numeric | Daily notification count |
| `app_opens_per_day` | Numeric | Daily app launch count |
| `weekend_screen_time` | Numeric | Weekend screen time (hours) |
| `stress_level` | Categorical | Low / Medium / High |
| `academic_work_impact` | Categorical | Yes / No |
| `total_leisure_hours` | Engineered | social_media + gaming |
| `screen_to_sleep_ratio` | Engineered | screen_time / sleep_hours |
| `weekend_screen_time_increase` | Engineered | weekend - weekday screen time |
| `notification_intensity` | Engineered | notifications / app_opens |
| `productive_time_ratio` | Engineered | work / (work + screen_time) |

---

## Model Architecture

```
sklearn.pipeline.Pipeline
├── preprocessor: ColumnTransformer
│   ├── categorical: OneHotEncoder(handle_unknown='ignore')
│   │   └── applied to: gender, stress_level, academic_work_impact
│   └── numerical: StandardScaler()
│       └── applied to: all remaining 14 features
└── classifier: RandomForestClassifier(random_state=42)
```

---

## Evaluation Metrics (Test Set)

| Metric | Logistic Regression | Random Forest (Final) |
|---|---|---|
| Accuracy | 0.8987 | 0.9213 |
| Precision | 0.9136 | 0.9340 |
| Recall | 0.9463 | 0.9580 |
| F1-Score | 0.9297 | 0.9458 |
| ROC-AUC | 0.9575 | 0.9648 |

**Primary metric:** F1-Score (chosen due to class imbalance)

---

## Risk Thresholds

| Category | Probability Range | Rationale |
|---|---|---|
| LOW | P < 0.35 | Pattern closely resembles non-addicted users |
| MODERATE | 0.35 ≤ P < 0.65 | Borderline — some concerning patterns present |
| HIGH | P ≥ 0.65 | Pattern strongly resembles addicted users in training data |

Thresholds are conservative: borderline cases are classified as MODERATE rather than LOW to avoid false reassurance.

---

## Intended Use

✅ **Appropriate uses:**
- Personal awareness of digital habits
- Educational demonstrations of ML in behavioural analytics
- Portfolio showcase of end-to-end ML engineering

---

## Out-of-Scope Use

❌ **Do NOT use this model for:**
- Clinical diagnosis of smartphone addiction
- Screening employees or students without consent
- Any decision that affects a person's rights, employment, or mental health treatment
- Populations very different from the training data (e.g., children < 10, elderly > 80)

---

## Limitations

1. **Self-report bias** — Users may inaccurately estimate their own usage
2. **No ground truth** — The `addicted_label` target is based on survey criteria, not clinical diagnosis
3. **Dataset provenance** — Not from a peer-reviewed clinical study
4. **Class imbalance** — 71% positive rate may bias probability calibration
5. **Cross-cultural validity** — Not validated across different cultural contexts
6. **Static snapshot** — Does not capture temporal change in behaviour
7. **Model calibration** — Random Forests may be poorly calibrated; probabilities should be interpreted as relative risk scores, not absolute probabilities

---

## Ethical Considerations

- All predictions are probabilistic and should not be presented as definitive
- The model should never be used covertly — users must be informed and consent
- Results should not be shared publicly in a way that identifies individuals
- Association between a feature and prediction ≠ causation
- Users with genuine concerns about compulsive smartphone use should consult a qualified mental health professional

---

## Known Risks

| Risk | Severity | Mitigation |
|---|---|---|
| False positive (person labelled HIGH who is not) | Low (unnecessary concern) | Clear disclaimer; probabilistic framing |
| False negative (person labelled LOW who is at risk) | Moderate | Conservative thresholds; suggest professional advice |
| Misuse as medical diagnosis | High | Prominent disclaimers; out-of-scope documentation |
| Privacy breach in batch prediction | Moderate | No data storage; user responsibility notice |

---

*This model card follows the format proposed by Mitchell et al. (2019), "Model Cards for Model Reporting."*
