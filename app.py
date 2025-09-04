# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Page config & title
st.set_page_config(page_title="Early Diabetes Risk Assessment", layout="centered")
st.title("🩺 Early Diabetes Risk Assessment")

st.markdown(
    "**Disclaimer:** This is a master’s project. The results are for reference only and not medical advice."
)
st.markdown("---")

# 2. Load model, explainer, and template columns
pipe = joblib.load("diabetes_pipeline.pkl")
explainer = joblib.load("shap_explainer.pkl")

# We need the original columns to reindex after get_dummies
df_template = pd.read_csv("diabetes_binary_health_indicators_BRFSS2023.csv")
X_full = df_template.drop(columns=["Diabetes_binary"])
X_cols = pd.get_dummies(X_full, drop_first=True).columns

# 3. Patient Information form
st.header("Patient Information")

# BMI via weight & height
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.1)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.1)

bmi = None
if weight and height:
    bmi = weight / ( (height / 100) ** 2 )
    st.write(f"**Calculated BMI:** {bmi:.1f}")
    if bmi < 18.5:
        st.warning("Underweight (BMI < 18.5). Consider a health checkup.")
    elif bmi < 25:
        st.success("Healthy BMI (18.5–24.9). Keep it up!")
    elif bmi < 30:
        st.info("Overweight (BMI 25–29.9). Maintain a balanced diet and exercise.")
    else:
        st.error("Obese (BMI ≥ 30). Please consult a healthcare professional.")

# Tickboxes with hints
high_bp = st.checkbox(
    "High Blood Pressure",
    help="High BP is typically systolic ≥130 mmHg or diastolic ≥80 mmHg."
)

high_chol = st.checkbox(
    "High Cholesterol",
    help="High cholesterol often means LDL ≥130 mg/dL or total cholesterol ≥200 mg/dL."
)

age = st.number_input(
    "Age (years)", min_value=18, max_value=120, step=1,
    help="Enter your actual age to assign an age group."
)
# Map raw age to 1–13 groups: 18–24→1, 25–29→2, …, 80+→13
group = min((age - 18) // 5 + 1, 13)

diff_walk = st.checkbox("Serious difficulty walking or climbing stairs")

heart_disease = st.checkbox("History of heart disease or heart attack")

# “Physical Health” repurposed as >8 hours activity in last 30 days
phys8 = st.checkbox(
    "Spent more than 8 hours in physical activity (excluding job) in last 30 days"
)

alcohol = st.checkbox(
    "Heavy alcohol consumption",
    help="Men >14 drinks/week or women >7 drinks/week."
)

st.markdown("---")

# 4. Build feature vector
features = {
    "BMI": bmi if bmi is not None else 0,
    "HighBP": 1 if high_bp else 0,
    "HighChol": 1 if high_chol else 0,
    "PhysActivity": 1 if phys_activity else 0,
    "AgeGroup": int(group),
    "DiffWalk": 1 if diff_walk else 0,
    "HeartDiseaseorAttack": 1 if heart_disease else 0,
    "PhysHlth": 1 if phys8 else 0,
    "HvyAlcoholConsump": 1 if alcohol else 0,
}
# Fill missing features with 0
input_df = pd.DataFrame([features])
input_df = pd.get_dummies(input_df, drop_first=True)
input_df = input_df.reindex(columns=X_cols, fill_value=0)

# 5. Prediction & feedback
if st.button("Assess Risk"):
    proba = pipe.predict_proba(input_df)[0, 1]

    # Color‐coded risk score
    if proba < 0.5:
        st.success(f"Diabetes Risk Score: {proba:.2%}")
        st.write("🎉 Congrats! You have a low risk. Continue your healthy lifestyle.")
    else:
        st.error(f"Diabetes Risk Score: {proba:.2%}")
        st.write("⚠️ High risk. Consider a medical checkup and maintain a healthy lifestyle.")

    # SHAP interpretability
    X_scaled = pipe.named_steps["scaler"].transform(input_df)
    shap_vals = explainer.shap_values(X_scaled)[0]
    base_value = explainer.expected_value[1]
    exp = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=X_scaled,
        feature_names=input_df.columns.tolist(),
    )
    st.subheader("Feature Contributions")
    fig, ax = plt.subplots()
    shap.plots.waterfall(exp, max_display=8, show=False)
    st.pyplot(fig)

st.markdown("---")
st.markdown("**Disclaimer:** This assessment is for reference only, not a clinical diagnosis.")
