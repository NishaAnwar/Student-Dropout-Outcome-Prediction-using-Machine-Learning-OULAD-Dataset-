**Project: Student Dropout & Outcome Prediction using Machine Learning (OULAD Dataset)**

🔹 **Overview**

This project focuses on predicting student academic outcomes (Pass, Fail, Withdrawn, Distinction) using demographic information and digital learning activity data from the Open University Learning Analytics Dataset (OULAD).
The goal was to analyze patterns in student behavior, engineer meaningful features, and build a robust machine learning model that can support early intervention strategies for at-risk students.

🔹 **Objectives**

Analyze and preprocess raw educational data (student demographics, assessments, registrations, and online activity).
Handle missing values and engineer domain-specific features such as days registered, missed assessments, and clickstream behavior.
Encode categorical features and scale numeric variables for model compatibility.
Train, evaluate, and compare different machine learning models (Logistic Regression, Random Forest, XGBoost).
Identify the best-performing model and deploy it in a simple Streamlit web application for interactive predictions.

🔹 **Methodology**

->Data Preprocessing & Cleaning

->Merged multiple tables: studentInfo, studentAssessment, studentRegistration, studentVle, and courses.

->Engineered key features:
days_registered (engagement duration in course).
num_missed_assessments (difference between assigned and submitted assessments).
avg_clicks & total_clicks (student LMS activity).

->Imputed missing values based on context (e.g., avg_score = 0 if no assessment attempts).

->Feature Engineering & Encoding

->Encoded categorical variables:
age_band → LabelEncoder
disability → Binary (Y=1, N=0)
final_result → Numeric mapping {Fail=0, Pass=1, Withdrawn=2, Distinction=3}.

->Standardized numeric features with StandardScaler to ensure uniform contribution to models.

->Model Training & Evaluation

->Models Trained: Logistic Regression, Random Forest, XGBoost.

->Used train-test split (80-20) and stratification to maintain class balance.

->Evaluation Metrics: Accuracy, Macro-F1, Weighted-F1.

->XGBoost outperformed other models in terms of overall accuracy and balance across all classes.

->Deployment

Saved trained XGBoost model, LabelEncoder, and Scaler using joblib.
Built a Streamlit web application to allow real-time predictions.
Users can input student demographics and learning activity metrics to get a prediction of academic outcome, along with probability distribution across all classes.

🔹 **Results**

Model Performance:

XGBoost achieved the best performance, outperforming the baseline models (Logistic Regression and Random Forest).
Training Accuracy: 0.896
Test Accuracy: 0.823

Feature importance analysis revealed that average score, total clicks, and days registered were strong predictors of success or withdrawal.

The Streamlit app provides an easy-to-use interface for testing new student data, making the model applicable for early student risk monitoring systems.

🔹 **Tools & Technologies**

Python (Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib)
Streamlit (for model deployment and visualization)
Joblib (for model persistence)
Google Colab / Jupyter Notebook (for development and experimentation)
