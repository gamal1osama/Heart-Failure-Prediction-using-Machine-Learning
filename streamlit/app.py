# heart_failure_app.py

import joblib
import numpy as np
import streamlit  as st

# Load the trained model (LightGBM pipeline)
model = joblib.load("lightgbm_pipeline.pkl")  # تأكد إن اسم الملف ده صحيح

# عنوان الصفحة
st.set_page_config(page_title="Heart Failure Prediction", page_icon="🫀")
st.title("🫀 Heart Failure Risk Prediction")
st.write("Enter patient clinical data to estimate the **risk of death** during the follow-up period using a trained LightGBM model.")

# الإدخالات
age = st.number_input("Age", min_value=0, max_value=130, value=60)
anaemia = st.radio("Anaemia (1 = Yes, 0 = No)", [0, 1])
cpk = st.number_input("Creatinine Phosphokinase", min_value=0, value=250)
diabetes = st.radio("Diabetes (1 = Yes, 0 = No)", [0, 1])
ef = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=50)
hbp = st.radio("High Blood Pressure (1 = Yes, 0 = No)", [0, 1])
platelets = st.number_input("Platelets", min_value=0, value=200000)
sc = st.number_input("Serum Creatinine", min_value=0.0, value=1.0)
ss = st.number_input("Serum Sodium", min_value=100, max_value=200, value=140)
sex = st.radio("Sex (1 = Male, 0 = Female)", [0, 1])
smoking = st.radio("Smoking (1 = Yes, 0 = No)", [0, 1])
time = st.number_input("Follow-up Time (Days)", min_value=0, value=250)

# التنبؤ عند الضغط
if st.button("Predict"):
    input_data = np.array([[age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time]])
    pred_prob = model.predict_proba(input_data)[:, 1][0]
    
    if pred_prob < 0.5:
        st.success(f"✅ Likely to Survive — Probability of Death: {pred_prob:.2%}")
    else:
        st.error(f"⚠️ Death Likely — Probability of Death: {pred_prob:.2%}")
