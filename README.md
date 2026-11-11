# 📱 Digital Wellbeing Analytics

### Predicting Smartphone Addiction Using Machine Learning

[![CI](https://github.com/Sushmitha-328/Digital-Wellbeing-Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Sushmitha-328/Digital-Wellbeing-Analytics/actions)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b)](https://digital-wellbeing-analytics-9osycluwd9ypkzb6evdwqh.streamlit.app/)

**[🌐 Live Application](https://digital-wellbeing-analytics-9osycluwd9ypkzb6evdwqh.streamlit.app/) · [📓 Notebooks](notebooks/) · [📄 Model Card](MODEL_CARD.md) · [📊 Data Science Report](DATA_SCIENCE_REPORT.md)**

---

## 📌 Problem Statement

Smartphone over-reliance is a growing public health concern. This project builds a
**production-oriented machine learning pipeline** that analyses self-reported smartphone
usage patterns and estimates an individual's behavioural risk profile.

Rather than producing a simple binary label, the system provides:
- A **risk probability** (0–100%)
- A **risk category** (LOW / MODERATE / HIGH) with transparent thresholds
- The **top behavioural risk factors** driving the prediction
- **Personalised digital wellbeing suggestions**

> ⚠️ This is a behavioural risk indicator, not a medical diagnosis.

---

## 🏗️ Architecture

```
User Input
    │
    ▼
Input Validation  ──────────────────── src/validate.py
    │  Range checks · Cross-field logic
    ▼
Feature Engineering  ───────────────── src/features.py
    │  12 raw → 17 features (5 engineered)
    ▼
sklearn Pipeline
    │  ColumnTransformer (OHE + StandardScaler)
    │  Random Forest Classifier
    ▼
Prediction Service  ────────────────── src/predict.py
    │  Probability · Risk Category · Top Factors · Suggestions
    ▼
Streamlit Dashboard  ───────────────── app.py
    │  5-page interactive dashboard
    ▼
Streamlit Cloud (Deployed)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Records | 7,500 smartphone usage surveys |
| Raw features | 12 (demographics, usage patterns, wellbeing) |
| Engineered features | 5 behavioural composites |
| Target | `addicted_label` (binary) |
| Class distribution | 70.9% addicted / 29.1% not addicted |

---

## 🔧 Engineered Features

| Feature | Formula | Rationale |
|---|---|---|
| `total_leisure_hours` | social_media + gaming | Total non-productive screen time |
| `screen_to_sleep_ratio` | screen_time / sleep | Sleep encroachment indicator |
| `weekend_screen_time_increase` | weekend − weekday | Compensatory binge pattern |
| `notification_intensity` | notifications / app_opens | "Push" heaviness of usage |
| `productive_time_ratio` | work / (work + screen) | Productive vs leisure balance |

---

## 🤖 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.899 | 0.930 | 0.958 |
| **Random Forest (Final)** | **0.921** | **0.946** | **0.965** |

Evaluated on a **stratified 20% held-out test set** (1,500 records).  
Primary metric: **F1-Score** (chosen due to class imbalance).

---

## 🗂️ Project Structure

```
Digital-Wellbeing-Analytics/
├── app.py                      # Streamlit dashboard (5 pages)
├── requirements.txt            # Pinned dependencies
├── config.yaml                 # Experiment configuration
├── .gitignore
│
├── src/
│   ├── features.py             # Feature engineering (single source of truth)
│   ├── validate.py             # Input validation
│   └── predict.py              # Inference service
│
├── tests/
│   ├── test_features.py        # Feature engineering unit tests
│   └── test_validate.py        # Input validation unit tests
## Predicting Smartphone Addiction Using Machine Learning

A machine learning project that analyzes smartphone usage and digital wellbeing patterns to predict the likelihood of smartphone addiction.

## 🚀 Live Demo

[Try the deployed Streamlit application](https://digital-wellbeing-analytics-9osycluwd9ypkzb6evdwqh.streamlit.app/)

## 🎯 Objectives

- Analyze smartphone usage patterns.
- Identify behavioral features associated with smartphone addiction.
- Perform exploratory data analysis.
- Engineer meaningful behavioral features.
- Train machine learning classification models.
- Build a prediction system for individual users.
- Deploy the machine learning application using Streamlit.

## 📊 Dataset

The dataset contains **7,500 smartphone usage records** with information including:

- Age
- Gender
- Daily screen time
- Social media usage
- Gaming hours
- Work/study hours
- Sleep hours
- Notifications per day
- App opens per day
- Weekend screen time
- Stress level
- Academic/work impact
- Smartphone addiction label

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- Streamlit

## 🔬 Project Workflow

1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Machine Learning
6. Prediction System
7. Streamlit Deployment

## 🤖 Machine Learning

The project uses a **Random Forest Classifier** to predict smartphone addiction.

The final model achieved approximately **93% accuracy** on the test dataset.

For the test set:

- Class 0 F1-score: **0.89**
- Class 1 F1-score: **0.95**
- Weighted average F1-score: **0.93**

Important features identified by the model include:

- Social media usage
- Daily screen time
- Weekend screen time
- Screen-to-sleep ratio
- Total leisure hours

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

[Open the Live Application](https://digital-wellbeing-analytics-9osycluwd9ypkzb6evdwqh.streamlit.app/)

## 📁 Project Structure

```text
Digital-Wellbeing-Analytics/
│
├── app.py
├── requirements.txt
├── random_forest_final_model.pkl
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_machine_learning.ipynb
│   └── 06_prediction_system.ipynb
│
├── .github/workflows/ci.yml    # GitHub Actions CI
├── MODEL_CARD.md               # Model intended use, limitations, ethics
├── DATA_SCIENCE_REPORT.md      # Statistical findings and methodology
└── PORTFOLIO_READINESS.md      # Portfolio assessment and interview guide
```

---

## 🚀 Setup & Usage

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Sushmitha-328/Digital-Wellbeing-Analytics.git
cd Digital-Wellbeing-Analytics

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Run Tests

```bash
pytest tests/ -v
```

### Reproduce Training

1. Place `smartphone_usage.csv` in `data/raw/`
2. Run notebooks `01` → `05` in order
3. The trained pipeline will be saved as `random_forest_final_model.pkl`

---

## 💡 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Single source of truth for features** | `src/features.py` is used by both training notebooks and the app — eliminates training/serving skew |
| **Input validation before prediction** | Prevents the model from silently producing nonsense results for impossible inputs |
| **F1-Score as primary metric** | Dataset is 71% addicted — accuracy alone is misleading |
| **Conservative risk thresholds** | Borderline cases shown as MODERATE, not LOW |
| **Rule-based suggestions** | Avoids claiming model outputs are recommendations; keeps suggestions evidence-grounded |
| **No MLflow/DVC** | Overkill for a single-dataset project; `config.yaml` provides sufficient reproducibility |

---

## ⚠️ Limitations

- Self-reported data may contain recall bias
- Dataset not from a peer-reviewed clinical study
- Model not cross-validated (single split)
- Random Forest probabilities may be poorly calibrated
- Demographic generalisability not validated

See [MODEL_CARD.md](MODEL_CARD.md) for full ethical considerations.

---

## 🔮 Future Improvements

- [ ] Stratified 5-fold cross-validation
- [ ] Hyperparameter tuning (RandomizedSearchCV)
- [ ] SHAP per-prediction waterfall explanations
- [ ] Model probability calibration (Platt/Isotonic)
- [ ] Statistical hypothesis testing (chi-square, Mann-Whitney)

---

## 🛠️ Tech Stack

`Python 3.9+` · `scikit-learn` · `pandas` · `numpy` · `plotly` · `streamlit` · `pytest` · `GitHub Actions`

---

*Built as a portfolio demonstration of end-to-end ML engineering.*
└── docs/
    └── Digital_Wellbeing_Analytics_Project_Report.docx
