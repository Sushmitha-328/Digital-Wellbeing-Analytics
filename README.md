# 📱 Digital Wellbeing Analytics

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
└── docs/
    └── Digital_Wellbeing_Analytics_Project_Report.docx
