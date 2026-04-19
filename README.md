# 📉 Customer Churn Prediction System

## 🚀 Project Overview
This project is a high-performance **Customer Churn Prediction System** developed as a Data Analytics (DA) Mini Project. It leverages a telecom dataset to predict whether a customer will discontinue their service using **Logistic Regression**.

The system features a premium **Streamlit** dashboard that provides deep insights through Exploratory Data Analysis (EDA) and real-time risk profiling.

## ✨ Features
- **Interactive Dashboard:** Real-time metrics and dataset overview.
- **Exploratory Data Analysis (EDA):** Visual insights into demographics, service usage, and financial metrics using **Plotly**.
- **Predictive Modeling:** Logistic Regression model with performance metrics (Confusion Matrix, ROC-AUC).
- **Risk Profiler:** Input customer data to get an instant churn probability score and AI-driven retention advice.

## 🛠️ Tech Stack
- **Frontend/Dashboard:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations:** [Plotly Express](https://plotly.com/python/plotly-express/), [Seaborn](https://seaborn.pydata.org/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (Logistic Regression)

## 📁 Project Structure
```text
d:/DA MINI BABU/
├── app.py              # Main Streamlit application & ML Logic
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🏃 How to Run
1. **Ensure Python is installed.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset Metadata
- **Source:** IBM Telco Customer Churn (Kaggle)
- **Size:** 7,043 rows
- **Target Variable:** Churn (Yes/No)
- **Features:** Tenure, Monthly Charges, Contract Type, Internet Service, etc.

---
**Developed for DA Mini Project | 2026**
