import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }

/* ── Header ─────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f2744 0%, #1b4f8a 60%, #2471c8 100%);
    padding: 1.4rem 2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.app-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.01em; }
.app-header p  { margin: 0.25rem 0 0 0; opacity: 0.80; font-size: 0.85rem; }

/* ── Form Sections ──────────────────────── */
.section-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1rem 1.2rem 0.8rem 1.2rem;
    margin-bottom: 0.9rem;
    border-left: 4px solid #1b4f8a;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #1b4f8a;
    margin-bottom: 0.65rem;
}

/* ── Predict Button ─────────────────────── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0f2744 0%, #1b4f8a 100%);
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
    letter-spacing: 0.03em;
    transition: all 0.25s ease;
    box-shadow: 0 3px 10px rgba(27,79,138,0.35);
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(27,79,138,0.45);
}
div[data-testid="stButton"] > button:active { transform: translateY(0); }

/* ── Results Panel ──────────────────────── */
.result-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}
.result-card-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #1b4f8a;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Risk Badge ─────────────────────────── */
.risk-badge {
    border-radius: 40px;
    padding: 0.65rem 1rem;
    font-size: 1.15rem;
    font-weight: 700;
    text-align: center;
    margin: 0.2rem 0 0.6rem 0;
    display: block;
}
.risk-low    { background: #e8f9ef; color: #1a7a3a; border: 2px solid #2ecc71; }
.risk-medium { background: #fff8e6; color: #7d5a00; border: 2px solid #f39c12; }
.risk-high   { background: #fef0f0; color: #8b0000; border: 2px solid #e74c3c; }

/* ── Snapshot Grid ──────────────────────── */
.snap-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
.snap-cell {
    background: #f5f8fc;
    border-radius: 8px;
    padding: 0.7rem 0.8rem;
    text-align: center;
    border: 1px solid #e0e8f4;
}
.snap-value { font-size: 1.2rem; font-weight: 700; color: #0f2744; }
.snap-label { font-size: 0.7rem; color: #7a8fa6; margin-top: 0.15rem; }

/* ── Risk Bars ──────────────────────────── */
.bar-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.45rem; }
.bar-name { font-size: 0.78rem; color: #374151; width: 155px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-bg   { flex: 1; background: #e9eef5; border-radius: 6px; height: 9px; overflow: hidden; }
.bar-fill { border-radius: 6px; height: 9px; transition: width 0.4s ease; }
.bar-pct  { font-size: 0.72rem; color: #7a8fa6; width: 32px; text-align: right; flex-shrink: 0; }

/* ── Recommendations ────────────────────── */
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    padding: 0.55rem 0.7rem;
    border-radius: 8px;
    margin-bottom: 0.45rem;
    font-size: 0.83rem;
    line-height: 1.45;
}
.rec-urgent  { background: #fff5f5; border-left: 3px solid #e53e3e; color: #2d3748; }
.rec-warning { background: #fffbf0; border-left: 3px solid #d97706; color: #2d3748; }
.rec-info    { background: #eff8ff; border-left: 3px solid #2d6ea9; color: #2d3748; }
.rec-success { background: #f0fff5; border-left: 3px solid #38a169; color: #2d3748; }

/* ── Placeholder ────────────────────────── */
.placeholder {
    background: #f5f8fc;
    border: 2px dashed #c3d4e8;
    border-radius: 14px;
    padding: 3.5rem 2rem;
    text-align: center;
}
.placeholder-icon { font-size: 3.2rem; }
.placeholder h3   { color: #5a7a9c; font-size: 1.1rem; margin: 0.8rem 0 0.4rem 0; }
.placeholder p    { color: #8ba5bf; font-size: 0.85rem; line-height: 1.5; margin: 0; }

/* ── Derived Tag ─────────────────────────── */
.derived-tag {
    display: inline-block;
    background: #e8f0fc;
    color: #1b4f8a;
    border-radius: 12px;
    padding: 0.18rem 0.55rem;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 0.3rem;
}

/* Tighten Streamlit widget spacing */
div[data-testid="stVerticalBlock"] > div { gap: 0.25rem; }
.stSlider > div { padding-bottom: 0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    try:
        bundle = joblib.load("models/readmission_model.pkl")
        return bundle["model"], bundle["scaler"], bundle["columns"]
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return None, None, None

model, scaler, model_columns = load_model()


# ──────────────────────────────────────────────────────────────────────────────
# Helper Utilities
# ──────────────────────────────────────────────────────────────────────────────
def bp_category(bp: int) -> str:
    if bp < 90:   return "Low"
    if bp <= 120: return "Normal"
    if bp <= 139: return "Elevated"
    return "High"

def admission_season(month: int) -> str:
    if month in (11, 12, 1, 2): return "Winter"
    if month in (3, 4, 5):      return "Summer"
    if month in (6, 7, 8, 9):   return "Monsoon"
    return "Post-Monsoon"

def risk_meta(prob: float):
    if prob < 0.40: return "Low",    "#2ecc71", "risk-low",    "✅"
    if prob < 0.70: return "Medium", "#f39c12", "risk-medium", "⚠️"
    return              "High",   "#e74c3c", "risk-high",   "🔴"

# All possible category values (sorted) for each categorical feature
# These must match the order used during model training (pd.get_dummies, drop_first=True)
CATEGORY_MAP = {
    "gender":               ["Female", "Male"],
    "marital_status":       ["Divorced", "Married", "Single", "Widowed"],
    "admission_type":       ["Elective", "Emergency", "Urgent"],
    "department":           ["Cardiology", "General Medicine", "Neurology", "Orthopedics", "Pulmonology"],
    "diagnosis":            ["Asthma", "Diabetes", "Heart Disease", "Hypertension", "Infection"],
    # "None" was stored as NaN in training data, so get_dummies only saw 3 real categories
    "secondary_diagnosis":  ["Diabetes", "Hypertension", "Kidney Disease"],
    "insurance_type":       ["Corporate", "Government", "Private", "Self-Paid"],
    "admission_season":     ["Monsoon", "Post-Monsoon", "Summer", "Winter"],
    "bp_category":          ["Elevated", "High", "Low", "Normal"],
}


def preprocess(inp: dict) -> np.ndarray:
    """Build a single-row feature matrix that matches the saved model's column layout."""
    season = admission_season(inp["admission_month"])
    bpc    = bp_category(inp["blood_pressure"])

    row = {
        "patient_age":              inp["patient_age"],
        "blood_pressure":           inp["blood_pressure"],
        "glucose_level":            inp["glucose_level"],
        "heart_rate":               inp["heart_rate"],
        "bmi":                      inp["bmi"],
        "previous_admissions":      inp["previous_admissions"],
        "previous_emergency_visits":inp["previous_emergency_visits"],
        "hospital_stay_days":       inp["hospital_stay_days"],
        "medication_count":         inp["medication_count"],
        "lab_test_count":           inp["lab_test_count"],
        "doctor_rating":            inp["doctor_rating"],
        "followup_compliance_score":inp["followup_compliance_score"],
        "hospital_bill_amount":     inp["hospital_bill_amount"],
        "length_of_stay":           inp["hospital_stay_days"],      # proxy for derived feature
        "followup_delay":           inp["followup_delay"],
        "admission_month":          inp["admission_month"],
        "admission_year":           inp["admission_year"],
        "total_risk_history":       inp["previous_admissions"] + inp["previous_emergency_visits"],
        "has_secondary_diagnosis":  int(inp["secondary_diagnosis"] != "None"),
        "medication_intensity":     inp["medication_count"] / (inp["lab_test_count"] + 1),
        "low_compliance":           int(inp["followup_compliance_score"] < 50),
        # raw categoricals (will be one-hot encoded below)
        "gender":               inp["gender"],
        "marital_status":       inp["marital_status"],
        "admission_type":       inp["admission_type"],
        "department":           inp["department"],
        "diagnosis":            inp["diagnosis"],
        # "None" was NaN in training → get_dummies produces all-zero dummies (= Diabetes reference)
        "secondary_diagnosis":  inp["secondary_diagnosis"] if inp["secondary_diagnosis"] != "None" else np.nan,
        "insurance_type":       inp["insurance_type"],
        "admission_season":     season,
        "bp_category":          bpc,
    }

    df = pd.DataFrame([row])

    # Set explicit Categorical dtype so get_dummies sees all categories even
    # from a single row, then drop_first removes the alphabetically first category
    for col, cats in CATEGORY_MAP.items():
        df[col] = pd.Categorical(df[col], categories=sorted(cats))

    df_enc = pd.get_dummies(df, columns=list(CATEGORY_MAP.keys()), drop_first=True)

    # Convert any boolean dummy columns to int
    for c in df_enc.columns:
        if df_enc[c].dtype == bool:
            df_enc[c] = df_enc[c].astype(int)

    # Align columns to exactly what the model expects (fill missing → 0, drop extras)
    df_enc = df_enc.reindex(columns=model_columns, fill_value=0)

    return scaler.transform(df_enc)


def risk_factors(inp: dict) -> list[tuple[str, int]]:
    """Heuristic risk factor scores from domain knowledge (0–100)."""
    factors = []

    er_score = min(100, inp["previous_emergency_visits"] * 18)
    adm_score = min(100, inp["previous_admissions"] * 12)
    hist_score = min(100, int((er_score * 0.6 + adm_score * 0.4)))
    if hist_score > 0:
        factors.append(("Prior Admission History", hist_score))

    compliance_risk = 100 - inp["followup_compliance_score"]
    if compliance_risk > 5:
        factors.append(("Low Follow-up Compliance", int(compliance_risk)))

    stay_score = min(100, inp["hospital_stay_days"] * 5)
    if stay_score > 0:
        factors.append(("Hospital Stay Duration", stay_score))

    med_score = min(100, inp["medication_count"] * 7)
    if med_score > 0:
        factors.append(("Medication Complexity", med_score))

    if inp["secondary_diagnosis"] != "None":
        factors.append(("Multiple Diagnoses", 72))

    if inp["admission_type"] == "Emergency":
        factors.append(("Emergency Admission", 65))
    elif inp["admission_type"] == "Urgent":
        factors.append(("Urgent Admission", 48))

    bpc = bp_category(inp["blood_pressure"])
    if bpc == "High":
        factors.append(("High Blood Pressure", 55))
    elif bpc == "Low":
        factors.append(("Low Blood Pressure", 40))

    factors.sort(key=lambda x: x[1], reverse=True)
    return factors[:5]


def build_recommendations(inp: dict, risk: str) -> list[tuple[str, str, str]]:
    """Return (type, emoji, text) tuples."""
    recs = []

    if risk == "High":
        recs.append(("urgent",  "🚨", "Schedule an urgent follow-up within 48–72 hours of discharge."))
        recs.append(("urgent",  "👩‍⚕️", "Assign a dedicated care coordinator to monitor this patient closely."))
    elif risk == "Medium":
        recs.append(("warning", "📅", "Schedule a follow-up appointment within 7 days of discharge."))

    if inp["followup_compliance_score"] < 50:
        recs.append(("warning", "💊", "Low compliance detected — set up automated medication reminders and check-in calls."))

    total_hist = inp["previous_admissions"] + inp["previous_emergency_visits"]
    if total_hist >= 4:
        recs.append(("warning", "📋", "High readmission history — enroll patient in a chronic disease management program."))

    if inp["secondary_diagnosis"] != "None":
        recs.append(("info", "🔬", f"Secondary diagnosis ({inp['secondary_diagnosis']}) found — consult a specialist before discharge."))

    if inp["hospital_stay_days"] > 10:
        recs.append(("info", "🏠", "Extended stay — arrange a post-discharge home care assessment."))

    if inp["admission_type"] == "Emergency":
        recs.append(("warning", "🚑", "Emergency admission — confirm stable condition and provide 24/7 emergency contact before discharge."))

    if inp["medication_count"] >= 8:
        recs.append(("info", "💉", "High medication load — review for polypharmacy risks and simplify regimen where possible."))

    if risk == "Low":
        recs.append(("success", "✅", "Low-risk patient — standard discharge with routine 30-day follow-up is appropriate."))
        recs.append(("success", "📞", "Provide patient education materials and emergency contact details for peace of mind."))

    return recs[:6]


# ──────────────────────────────────────────────────────────────────────────────
# Page Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div style="font-size:2.6rem; line-height:1;">🏥</div>
  <div>
    <h1>Hospital Readmission Risk Predictor</h1>
    <p>AI-powered 30-day readmission risk assessment &nbsp;·&nbsp; Powered by XGBoost &nbsp;·&nbsp; 99.3 % accuracy</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Two-column layout
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.15, 0.85], gap="large")

# ─────────────────────────── LEFT : Input Form ───────────────────────────────
with left:

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">👤 Patient Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    patient_age     = c1.slider("Age (years)", 18, 100, 58)
    gender          = c2.selectbox("Gender", ["Male", "Female"])
    marital_status  = c3.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
    c4, c5 = st.columns(2)
    insurance_type  = c4.selectbox("Insurance Type", ["Private", "Government", "Corporate", "Self-Paid"])
    department      = c5.selectbox("Department", ["Cardiology", "General Medicine", "Neurology", "Orthopedics", "Pulmonology"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Vitals & Diagnosis ────────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">🩺 Vitals & Diagnosis</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    blood_pressure  = v1.slider("Blood Pressure (mmHg)", 60, 200, 122)
    glucose_level   = v1.slider("Glucose Level (mg/dL)", 50, 400, 112)
    heart_rate      = v1.slider("Heart Rate (bpm)", 40, 200, 78)
    bmi             = v2.slider("BMI", 10.0, 50.0, 26.5, 0.1)
    diagnosis       = v2.selectbox("Primary Diagnosis", ["Heart Disease", "Diabetes", "Hypertension", "Asthma", "Infection"])
    secondary_diagnosis = v2.selectbox("Secondary Diagnosis", ["None", "Diabetes", "Hypertension", "Kidney Disease"])
    bpc = bp_category(blood_pressure)
    bp_color = {"Low": "#3498db", "Normal": "#2ecc71", "Elevated": "#f39c12", "High": "#e74c3c"}[bpc]
    st.markdown(f'<p style="font-size:0.78rem; color:#6b7280; margin:0.2rem 0 0.4rem 0;">🩸 BP Category: <strong style="color:{bp_color}">{bpc}</strong> &nbsp;(auto-detected)</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Admission Details ─────────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">🏨 Admission Details</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    admission_type  = a1.selectbox("Admission Type", ["Elective", "Emergency", "Urgent"])
    admission_month = a2.selectbox(
        "Admission Month", list(range(1, 13)), index=0,
        format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
    )
    admission_year  = a3.selectbox("Year", [2022, 2023, 2024, 2025], index=2)
    a4, a5 = st.columns(2)
    hospital_stay_days = a4.slider("Hospital Stay (days)", 1, 60, 6)
    followup_delay     = a5.slider("Follow-up Appt (days after discharge)", 1, 90, 14)
    season = admission_season(admission_month)
    st.markdown(f'<p style="font-size:0.78rem; color:#6b7280; margin:0.2rem 0 0.3rem 0;">🗓️ Season: <strong style="color:#1b4f8a">{season}</strong> &nbsp;(auto-detected)</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Treatment & History ───────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">💊 Treatment & History</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    previous_admissions       = t1.slider("Previous Admissions", 0, 20, 1)
    previous_emergency_visits = t1.slider("Previous ER Visits", 0, 20, 0)
    medication_count          = t1.slider("Medications Prescribed", 0, 20, 5)
    doctor_rating             = t2.slider("Doctor's Assessment (1–10)", 1, 10, 7)
    lab_test_count            = t2.slider("Lab Tests Conducted", 0, 30, 6)
    followup_compliance_score = t2.slider("Follow-up Compliance (0–100)", 0, 100, 72)
    t3, _ = st.columns([1, 1])
    hospital_bill_amount = t3.number_input("Hospital Bill Amount (₹)", min_value=0, max_value=1_000_000, value=48_000, step=1_000)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict Button ────────────────────────────────────────────────────────
    predict = st.button("🔍  Predict Readmission Risk", use_container_width=True)


# ─────────────────────────── RIGHT : Results ─────────────────────────────────
with right:

    if not predict:
        st.markdown("""
        <div class="placeholder">
          <div class="placeholder-icon">📊</div>
          <h3>Ready to Analyse</h3>
          <p>Fill in the patient details on the left<br>and click <strong>Predict Readmission Risk</strong><br>to see the AI-driven assessment.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        if model is None:
            st.error("Model unavailable. Please ensure `models/readmission_model.pkl` exists.")
        else:
            inp = dict(
                patient_age=patient_age,
                gender=gender,
                marital_status=marital_status,
                insurance_type=insurance_type,
                department=department,
                blood_pressure=blood_pressure,
                glucose_level=glucose_level,
                heart_rate=heart_rate,
                bmi=bmi,
                diagnosis=diagnosis,
                secondary_diagnosis=secondary_diagnosis,
                admission_type=admission_type,
                admission_month=admission_month,
                admission_year=admission_year,
                hospital_stay_days=hospital_stay_days,
                followup_delay=followup_delay,
                previous_admissions=previous_admissions,
                previous_emergency_visits=previous_emergency_visits,
                medication_count=medication_count,
                doctor_rating=doctor_rating,
                lab_test_count=lab_test_count,
                followup_compliance_score=followup_compliance_score,
                hospital_bill_amount=hospital_bill_amount,
            )

            try:
                X    = preprocess(inp)
                prob = float(model.predict_proba(X)[0][1])
                risk, clr, badge_cls, icon = risk_meta(prob)

                # ── Gauge ──────────────────────────────────────────────
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%", "font": {"size": 38, "color": clr, "family": "Inter"}},
                    title={"text": "Readmission Probability", "font": {"size": 13, "color": "#5a7a9c", "family": "Inter"}},
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickvals": [0, 40, 70, 100],
                            "ticktext": ["0%", "40%", "70%", "100%"],
                            "tickcolor": "#94a3b8",
                            "tickwidth": 1,
                        },
                        "bar": {"color": clr, "thickness": 0.22},
                        "bgcolor": "white",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,  40], "color": "#e8f9ef"},
                            {"range": [40, 70], "color": "#fff8e6"},
                            {"range": [70,100], "color": "#fef0f0"},
                        ],
                        "threshold": {
                            "line": {"color": clr, "width": 3},
                            "thickness": 0.75,
                            "value": round(prob * 100, 1),
                        },
                    },
                ))
                fig.update_layout(
                    height=230,
                    margin=dict(l=25, r=25, t=25, b=5),
                    paper_bgcolor="white",
                )

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(
                    f'<span class="risk-badge {badge_cls}">{icon}&nbsp; {risk} Readmission Risk</span>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Patient Snapshot ───────────────────────────────────
                total_hist = previous_admissions + previous_emergency_visits
                bpc_label  = bp_category(blood_pressure)
                st.markdown(f"""
                <div class="result-card">
                  <div class="result-card-title">📌 Patient Snapshot</div>
                  <div class="snap-grid">
                    <div class="snap-cell">
                      <div class="snap-value">{total_hist}</div>
                      <div class="snap-label">Total Risk History</div>
                    </div>
                    <div class="snap-cell">
                      <div class="snap-value">{followup_compliance_score}%</div>
                      <div class="snap-label">Compliance Score</div>
                    </div>
                    <div class="snap-cell">
                      <div class="snap-value">{bpc_label}</div>
                      <div class="snap-label">BP Category</div>
                    </div>
                    <div class="snap-cell">
                      <div class="snap-value">{hospital_stay_days}d</div>
                      <div class="snap-label">Length of Stay</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Key Risk Drivers ───────────────────────────────────
                factors = risk_factors(inp)
                st.markdown('<div class="result-card"><div class="result-card-title">⚡ Key Risk Drivers</div>', unsafe_allow_html=True)
                for name, score in factors:
                    fill_clr = "#e74c3c" if score > 70 else "#f39c12" if score > 40 else "#2ecc71"
                    st.markdown(f"""
                    <div class="bar-row">
                      <div class="bar-name" title="{name}">{name}</div>
                      <div class="bar-bg"><div class="bar-fill" style="width:{max(4,score)}%; background:{fill_clr};"></div></div>
                      <div class="bar-pct">{score}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Clinical Recommendations ───────────────────────────
                recs = build_recommendations(inp, risk)
                st.markdown('<div class="result-card"><div class="result-card-title">📋 Clinical Recommendations</div>', unsafe_allow_html=True)
                for rtype, emoji, text in recs:
                    st.markdown(f"""
                    <div class="rec-item rec-{rtype}">
                      <span style="font-size:1rem; flex-shrink:0;">{emoji}</span>
                      <span>{text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as exc:
                st.error(f"Prediction error: {exc}")
                with st.expander("Debug info"):
                    st.exception(exc)
