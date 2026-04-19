# 📘 FINAL SEMESTER PROJECT REPORT
## CUSTOMER CHURN PREDICTION SYSTEM USING LOGISTIC REGRESSION

---

### 🎓 COLLEGE PROJECT SUBMISSION
**Project ID:** DA-2026-CHURN  
**Student Name:** [Your Name Here]  
**Register No:** [Your Reg No Here]  
**Course:** [Degree Name - e.g., B.Tech / BCA / B.Sc]  
**Department:** [Department Name]  
**Academic Year:** 2025 - 2026

---

## 📄 CERTIFICATE
This is to certify that the project report entitled **"Customer Churn Prediction System"** is a bona fide record of the work carried out by **[Your Name Here]** under my supervision and guidance. 

<br><br>
**Project Guide** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Head of Department**

---

## 📝 ABSTRACT
Customer churn is one of the most critical metrics for a growing business to evaluate. This project focuses on predicting customer churn in the telecom industry using **Logistic Regression**. By analyzing a dataset of over 7,000 customers, we built a system that identifies patterns in customer behavior (like contract type, tenure, and streaming habits) that lead to cancellation. The final implementation includes an interactive **Streamlit** dashboard that provides real-time risk profiling and visual insights, achieving an accuracy of **~80%**.

---

## 📑 TABLE OF CONTENTS
1. **Introduction**
2. **System Requirements**
3. **System Analysis**
4. **System Design**
5. **Data Preprocessing & EDA**
6. **Implementation**
7. **Testing & Results**
8. **Conclusion & Future Work**
9. **References**

---

## CHAPTER 1: INTRODUCTION
### 1.1 Background
The Telecom industry is one of the fastest-growing sectors globally. However, with multiple service providers like Jio, Airtel, and Vodafone offering competitive plans, retaining customers has become a major challenge. 

### 1.2 Problem Statement
Acquiring a new customer costs 5 to 10 times more than retaining an existing one. Companies lack a real-time tool to predict which customers are unhappy or likely to switch before they actually do.

### 1.3 Objective
To develop a Data Analytics system that predicts churn probability for individual customers and provides actionable insights to management.

### 1.4 Scope
The scope includes cleaning the telecom dataset, performing visual data analysis, training a Logistic Regression model, and deploying a web-based dashboard for easy interaction.

---

## CHAPTER 2: SYSTEM REQUIREMENTS
### 2.1 Hardware Requirements
- **Processor:** Intel i3 or above
- **RAM:** 4GB (Minimum)
- **Storage:** 500MB for project files

### 2.2 Software Requirements
- **Operating System:** Windows 10/11 / MacOS / Linux
- **Language:** Python 3.9+
- **Libraries:** Pandas, NumPy, Scikit-learn, Plotly
- **Framework:** Streamlit (Web Dashboard)

---

## CHAPTER 3: SYSTEM ANALYSIS
### 3.1 Existing System
Currently, many companies manually analyze billing cycles or wait for a customer to request a cancellation (Porting) before offering a discount. This is a "reactive" approach.

### 3.2 Proposed System
The proposed system is "proactive." It uses Machine Learning to score every customer. If a customer is flagged as "High Risk," the company can offer a dynamic discount *before* they think about leaving.

---

## CHAPTER 4: SYSTEM DESIGN
### 4.1 Architecture
The system follows a 3-tier architecture:
1. **Data Tier:** IBM Telco Dataset (CSV).
2. **Logic Tier:** Python-based preprocessing and Logistic Regression training.
3. **Presentation Tier:** Streamlit-based web interface for the end user.

---

## CHAPTER 5: DATA PREPROCESSING & EDA
### 5.1 Cleaning
- Handled empty strings in the `TotalCharges` column.
- Removed 11 rows with null values to maintain data integrity.

### 5.2 Feature Engineering
- **Label Encoding:** Converted Gender (Male/Female) to 0/1.
- **One-Hot Encoding:** Expanded multiclass features like `Contract` into binary columns.

### 5.3 EDA Findings
- **Contract:** Month-to-month users churn the most.
- **Internet:** Fiber optic users are more likely to churn due to high costs or competition.
- **Tenure:** The first 12 months are the most critical for retention.

---

## CHAPTER 6: IMPLEMENTATION
The project is implemented in a modular Python structure:
- **`preprocess_data()`**: Cleans and encodes the raw CSV data.
- **`LogisticRegression()`**: Utilized from Scikit-learn to train the binary classifier.
- **`Streamlit UI`**: Provides a sidebar-driven navigation system for Live Monitoring and Predictions.

---

## CHAPTER 7: TESTING & RESULTS
### 7.1 Accuracy
The model achieved an accuracy of **80.2%** on the testing set.

### 7.2 Confusion Matrix
- **True Negatives:** Correctly identified loyal customers.
- **True Positives:** Successfully identified customers planning to leave.

### 7.3 ROC-AUC
The Area Under the Curve (AUC) is **0.84**, indicating a strong predictive performance.

---

## CHAPTER 8: CONCLUSION & FUTURE WORK
### 8.1 Conclusion
The project successfully bridges the gap between raw data and business decisions. The Logistic Regression model provides clear probabilities, and the UI makes it usable for non-technical managers.

### 8.2 Future Scope
- **Advanced Models:** Incorporating Random Forest or XGBoost for higher accuracy.
- **Auto-Emailer:** Integrating an automated email system to send offers to high-risk customers.
- **SQL Integration:** Connecting the app to a live SQL database for real-time customer data.

---

## CHAPTER 9: REFERENCES
1. Scikit-learn Documentation (Classifier metrics).
2. Streamlit Gallery (UI layout inspirations).
3. Kaggle Telco Dataset Documentation.

---

## APPENDIX: PROJECT CODE (KEY SNIPPETS)
```python
# Model Training Logic
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Prediction Logic
prob = model.predict_proba(sample_scaled)
```
