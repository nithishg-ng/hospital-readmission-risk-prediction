import streamlit as st
import pandas as pd
import joblib
 
# ----------------------------------------------------------
# 1. Page setup
# ----------------------------------------------------------
st.set_page_config(page_title="Readmission Risk Predictor", page_icon="🏥", layout="wide")
 
st.title("🏥 Hospital Readmission Risk Prediction")
st.write("Patient details-a podunga, readmission risk-a predict pannidum.")
 
# ----------------------------------------------------------
# 2. Saved model-a load pannradhu (model + scaler + columns)
#    @st.cache_resource -> oru thadava load, apram memory-la vechukkum (fast)
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    data = joblib.load("readmission_model.pkl")
    return data["model"], data["scaler"], data["columns"]
 
model, scaler, model_columns = load_model()
 
# ----------------------------------------------------------
# 3. Risk category function (Step 8-la pannadhu)
# ----------------------------------------------------------
def get_risk_category(prob):
    if prob < 0.40:
        return "Low Risk", "🟢"
    elif prob < 0.70:
        return "Medium Risk", "🟡"
    else:
        return "High Risk", "🔴"
 
# ----------------------------------------------------------
# 4. Recommendations function (Step 8-la pannadhu)
# ----------------------------------------------------------
def get_recommendations(prob, inputs):
    recs = []
    category, _ = get_risk_category(prob)
 
    if category == "High Risk":
        recs.append("🔴 Schedule early review within 7 days")
        recs.append("🔴 Assign dedicated care coordinator")
    elif category == "Medium Risk":
        recs.append("🟡 Schedule follow-up within 14 days")
    else:
        recs.append("🟢 Routine follow-up sufficient")
 
    # Risk factor based (patient values paathu)
    if inputs["previous_admissions"] + inputs["previous_emergency_visits"] > 10:
        recs.append("⚠️ High admission history → Close monitoring needed")
    if inputs["followup_compliance_score"] < 50:
        recs.append("📞 Low compliance → Medication reminder & adherence support")
    if inputs["hospital_stay_days"] > 10:
        recs.append("🏥 Long stay → Post-discharge home care assessment")
 
    return recs
 