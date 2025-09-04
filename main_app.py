import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model + encoders/scaler
# -----------------------------
model = joblib.load("xgb_withdraw_student_model.pkl")
scaler = joblib.load("scaler.pkl")
le_age = joblib.load("le_age.pkl")   # save this when training!

# Define numeric columns (must match training)
num_cols = [
    "num_of_prev_attempts", "studied_credits", "days_registered",
    "avg_score", "num_assessments", "num_missed_assessments",
    "total_clicks", "avg_clicks", "module_presentation_length"
]

# Define categorical mapping for disability
disability_map = {"N": 0, "Y": 1}
outcome_map = {0: "Fail", 1: "Pass", 2: "Withdrawn", 3: "Distinction"}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎓 Student Outcome Prediction App")

st.header("📌 Enter Student Information")
age_band = st.selectbox("Age Band", le_age.classes_.tolist())  # show same labels used in training
num_attempts = st.number_input("Number of Previous Attempts", min_value=0, step=1)
studied_credits = st.number_input("Studied Credits", min_value=0, step=1)
disability = st.selectbox("Disability", ["N", "Y"])
days_registered = st.number_input("Days Registered in Module", min_value=0, step=1)
avg_score = st.number_input("Average Assessment Score (%)", min_value=0.0, max_value=100.0, step=1.0)
num_assessments = st.number_input("Number of Assessments", min_value=0, step=1)
num_missed_assessments = st.number_input("Number of Missed Assessments", min_value=0, step=1)
total_clicks = st.number_input("Total LMS Clicks", min_value=0, step=1)
avg_clicks = st.number_input("Average Clicks per Week", min_value=0.0, step=1.0)
module_presentation_length = st.number_input("Module Presentation Length (days)", min_value=0, step=1)

# -----------------------------
# Preprocess input
# -----------------------------
input_data = pd.DataFrame([{
    "age_band": le_age.transform([age_band])[0],     # ✅ encode using same LabelEncoder
    "num_of_prev_attempts": num_attempts,
    "studied_credits": studied_credits,
    "disability": disability_map[disability],
    "days_registered": days_registered,
    "avg_score": avg_score,
    "num_assessments": num_assessments,
    "num_missed_assessments": num_missed_assessments,
    "total_clicks": total_clicks,
    "avg_clicks": avg_clicks,
    "module_presentation_length": module_presentation_length
}])

# ✅ Scale numeric features with training scaler
input_data[num_cols] = scaler.transform(input_data[num_cols])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Outcome"):
    pred = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]

    st.subheader("📢 Prediction:")
    st.success(outcome_map[pred])

    st.write("### Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Outcome": [outcome_map[i] for i in range(len(probs))],
        "Probability": probs
    })
    st.bar_chart(prob_df.set_index("Outcome"))
