# Hospital Readmission Risk Prediction

An end-to-end machine learning system that predicts 30-day hospital readmission risk and surfaces patient-specific clinical recommendations through a Streamlit web application.

---

## Overview

Hospital readmission within 30 days of discharge is one of the most expensive and preventable problems in healthcare. This project builds an XGBoost classifier trained on 50,000 patient admission records to identify high-risk patients before they leave the hospital. The model is packaged in a clean, interactive web app that clinical staff can use without any data science background.

**Key results:**
- Model: XGBoost classifier
- Accuracy: 99.3%
- Risk tiers: Low / Medium / High (thresholds at 40% and 70% probability)

---

## Features

- **Risk score gauge** — real-time readmission probability (0–100%) with colour-coded tiers
- **Key risk drivers** — top 5 contributing factors visualised as horizontal bar charts
- **Patient snapshot** — at-a-glance summary of total risk history, compliance score, BP category, and length of stay
- **Clinical recommendations** — prioritised, patient-specific action items for the care team
- **Auto-derived fields** — BP category and admission season computed automatically from raw inputs

---

## Project Structure

```
Hospital Readmission Risk Prediction/
├── app.py                          
├── requirements.txt                
├── Data/
│   ├── Raw/
│   │   └── hospital_readmission_dataset.csv
│   └── Processed/                  
├── models/
│   └── readmission_model.pkl       
├── Notebooks/
│   ├── data_understanding.ipynb    
│   ├── data_quality.ipynb          
│   ├── data_preprocessing.ipynb    
│   └── model_building.ipynb        
└── reports/                        
```

---

## Dataset

| Property | Value |
|---|---|
| Source file | `hospital_readmission_dataset.csv` |
| Raw rows | 50,500 |
| Rows after cleaning | 50,000 |
| Features | 28 columns |
| Target | `readmitted` (binary: 1 = readmitted, 0 = not readmitted) |

**Feature categories:**
- **Demographics** — age, gender, marital status, insurance type
- **Clinical vitals** — blood pressure, glucose level, heart rate, BMI
- **Diagnosis** — primary diagnosis, secondary diagnosis
- **Admission details** — admission type, department, admission date, hospital stay days
- **Treatment** — medication count, lab test count, doctor rating
- **Follow-up** — compliance score, follow-up date, follow-up delay
- **History** — previous admissions, previous emergency visits

---

## Data Quality Issues Resolved

| Issue | Detail |
|---|---|
| Missing values | Up to 25.18% missing in `secondary_diagnosis`; 15% in `glucose_level`, `doctor_rating`, `followup_compliance_score` |
| Duplicate records | 500 fully duplicate rows removed |
| Invalid numerics | Out-of-range ages, heart rates, BMI values nulled and imputed |
| Categorical inconsistency | `gender` column had 5 variants (`Male`, `M`, `MALE`, etc.) — normalised to `Male`/`Female` |
| Mixed formats | `hospital_bill_amount` mixed numeric and `₹`-prefixed strings; `followup_date` had two date formats |
| Data leakage | Removed `readmission_notice_sent`, `readmission_confirmed`, `final_readmission_reason` |

---

## Feature Engineering

Derived features added before modelling:

| Feature | Description |
|---|---|
| `total_risk_history` | `previous_admissions` + `previous_emergency_visits` |
| `has_secondary_diagnosis` | Binary flag for presence of a secondary condition |
| `medication_intensity` | `medication_count / (lab_test_count + 1)` |
| `low_compliance` | Binary flag: `followup_compliance_score < 50` |
| `admission_season` | Derived from admission month (Winter / Summer / Monsoon / Post-Monsoon) |
| `bp_category` | Derived from blood pressure (Low / Normal / Elevated / High) |

---

## Setup

### Prerequisites

- Python 3.10+
- A virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/nithishg-ng/hospital-readmission-risk-prediction.git
cd hospital-readmission-risk-prediction

# Create and activate a virtual environment
python -m venv hospitalenv
hospitalenv\Scripts\activate        


# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

> **Note:** The trained model file `models/readmission_model.pkl` must be present before launching the app. Run `Notebooks/model_building.ipynb` end-to-end to generate it if it is missing.

---

## Reproducing the Model

Run the notebooks in order:

1. `Notebooks/data_quality.ipynb` — assess raw data quality
2. `Notebooks/data_understanding.ipynb` — exploratory analysis and visualisations
3. `Notebooks/data_preprocessing.ipynb` — clean the dataset and engineer features
4. `Notebooks/model_building.ipynb` — train, evaluate, and save the model

Each notebook saves its output (cleaned CSVs, plots, the final `.pkl`) so subsequent notebooks can pick up where the previous one left off.

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| Class imbalance | Handled with SMOTE (`imbalanced-learn`) |
| Scaling | `StandardScaler` applied before training |
| Saved bundle | `model`, `scaler`, `columns` list serialised together via `joblib` |

The saved bundle stores the exact column order used during training so the app can safely align any new input to the correct feature layout.

---

## App Input Reference

| Section | Fields |
|---|---|
| Patient Demographics | Age, Gender, Marital Status, Insurance Type, Department |
| Vitals & Diagnosis | Blood Pressure, Glucose Level, Heart Rate, BMI, Primary Diagnosis, Secondary Diagnosis |
| Admission Details | Admission Type, Month, Year, Hospital Stay (days), Follow-up Appointment Delay |
| Treatment & History | Previous Admissions, Previous ER Visits, Medications, Doctor Rating, Lab Tests, Compliance Score, Bill Amount |

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `xgboost` | Gradient boosting classifier |
| `scikit-learn` | Preprocessing and metrics |
| `imbalanced-learn` | SMOTE for class imbalance |
| `shap` | Model explainability |
| `plotly` | Interactive gauge chart |
| `pandas`, `numpy` | Data manipulation |
| `joblib` | Model serialisation |

---

## Author

**Nithish G** — [nithishg.ng@gmail.com](mailto:nithishg.ng@gmail.com)

GitHub: [nithishg-ng/hospital-readmission-risk-prediction](https://github.com/nithishg-ng/hospital-readmission-risk-prediction)
