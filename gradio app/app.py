import joblib
import gradio as gr
import numpy as np

# Load the saved pipeline (model only - no scaler inside)
model = joblib.load("lightgbm_pipeline.pkl")  # Use your correct file name

# Input features in the exact same order as training
feature_names = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
    'ejection_fraction', 'high_blood_pressure', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

# Define the prediction function
def predict_heart_failure(age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time):
    # Arrange inputs into correct format
    input_data = np.array([[age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time]])

    # Predict probability
    pred_probs = model.predict_proba(input_data)[:, 1]  # [:, 1] = probability of class 1 (death)

    # Return result
    prob = pred_probs[0]
    label = "✅ Likely to Survive" if prob < 0.5 else "⚠️ Death Likely"
    return f"{label} — Probability of Death: {prob:.2%}"

# Define input fields
inputs = [
    gr.Number(label="Age"),
    gr.Radio([0, 1], label="Anaemia (1=Yes, 0=No)"),
    gr.Number(label="Creatinine Phosphokinase"),
    gr.Radio([0, 1], label="Diabetes (1=Yes, 0=No)"),
    gr.Number(label="Ejection Fraction (%)"),
    gr.Radio([0, 1], label="High Blood Pressure (1=Yes, 0=No)"),
    gr.Number(label="Platelets"),
    gr.Number(label="Serum Creatinine"),
    gr.Number(label="Serum Sodium"),
    gr.Radio([0, 1], label="Sex (1=Male, 0=Female)"),
    gr.Radio([0, 1], label="Smoking (1=Yes, 0=No)"),
    gr.Number(label="Follow-up Time (Days)")
]

# Build the interface
interface = gr.Interface(
    fn=predict_heart_failure,
    inputs=inputs,
    outputs="text",
    title="🫀 Heart Failure Risk Prediction",
    description="Enter patient clinical data to estimate the risk of death during the follow-up period. The model is based on LightGBM trained on real-world heart failure data.",
    allow_flagging="never"
)

# Launch the app
interface.launch()
