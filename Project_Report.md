# 📈 Project Report: Customer Churn Prediction System

---

## 👨‍🎓 Project Overview
**Project Title:** Customer Churn Prediction using Predictive Analytics  
**Course:** Data Analytics (DA) Mini Project  
**Implementation:** Machine Learning (Logistic Regression)  
**Tools Tool:** Python, Streamlit, Scikit-learn, Plotly  

---

## 1. Introduction
Customer Churn is defined as the loss of clients or customers. In the highly competitive telecom industry, retaining existing customers is more cost-effective than acquiring new ones. This project develops a predictive model that identifies customers at risk of leaving the service (churning).

## 2. Objective
Objective of the project is to build a robust data analytics system that:
1. Analyzes historical customer behavior.
2. Identifies key factors leading to churn.
3. Uses **Logistic Regression** to predict the probability of future churn.
4. Provides a real-time monitoring dashboard for decision-making.

## 3. Dataset Description
The project uses the **IBM Telco Customer Churn Dataset**. It contains 7,043 customer records with 21 attributes:
- **Demographics:** Gender, Senior Citizen status, Partners, Dependents.
- **Service Details:** Tenure (months), Phone service, Multiple lines, Internet service (DSL/Fiber optic), Tech support.
- **Financial Info:** Monthly Charges, Total Charges, Paperless billing, Payment method.
- **Target Variable:** `Churn` (Yes/No).

## 4. Methodology
The project followed a standard Data Science lifecycle:

### A. Data Preprocessing
- **Type Conversion:** Converted `TotalCharges` to numeric and handled missing values.
- **Feature Encoding:** Used Label Encoding for binary features (Gender, Partner) and One-Hot Encoding for multi-class features (Contract, Payment Method).
- **Scaling:** Applied `StandardScaler` to ensure all numerical features (Tenure, MonthlyCharges) contribute equally to the model.

### B. Exploratory Data Analysis (EDA)
Key insights derived:
- **Tenure:** Shorter tenure (0-6 months) is a high indicator of churn.
- **Contract Type:** "Month-to-month" contracts show significantly higher churn than 2-year contracts.
- **Service:** Fiber optic users have a higher churn rate compared to DSL users.

### C. Predictive Modeling
**Logistic Regression** was chosen for its interpretability and efficiency in binary classification.
- **Split:** 80% Training, 20% Testing.
- **Model Logic:** The sigmoid function is used to output a probability between 0 and 1.

## 5. Implementation Details
The project was deployed using a **Streamlit Dashboard** featuring:
- **Live Operations Command:** Real-time system monitoring simulation.
- **Tactical EDA:** Interactive Plotly histograms and boxplots.
- **AI Prediction Engine:** Visual performance metrics (Confusion Matrix, ROC Curve).
- **Risk Profiler:** A real-time tool where administrators can input customer data to get an instant risk profile.

## 6. Results and Evaluation
The model's performance was evaluated using:
- **Confusion Matrix:** To track True Positives (correctly identified churners).
- **ROC-AUC Curve:** Measuring the model's ability to distinguish between churn and retention.
- **Accuracy:** The model achieved a stable accuracy of approximately **80%**.

## 7. Real-World Application
This system can be used by telecom giants (like Airtel, Jio, or Vodafone) to:
- Identify high-risk customers before they leave.
- Offer targeted discounts only to those likely to churn.
- Optimize marketing spend by focusing on "at-risk" segments.

## 8. Conclusion
The Customer Churn Prediction System successfully demonstrates the power of predictive analytics in business decision-making. By combining Logistic Regression with a modern interactive dashboard, the project provides a practical solution for improving customer retention rates in the telecom industry.

---
**Date:** April 19, 2026  
**Status:** Successfully Completed & Deployed  
