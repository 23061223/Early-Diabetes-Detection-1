import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and explainer
pipe = joblib.load("diabetes_pipeline.pkl")
explainer = joblib.load("shap_explainer.pkl")

# Load reference dataset for encoding
file_name = "diabetes_binary_health_indicators_BRFSS2023.csv"
df = pd.read_csv(file_name)
X_full = df.drop(columns=["Diabetes_binary"])
X_full_encoded = pd.get_dummies(X_full, drop_first=True)
original_cols = X_full.columns

# Streamlit UI
st.title("🩺 Early Diabetes Risk Assessment")
st.sidebar.header("Patient Information")

# Collect inputs
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0)
phys_health = st.sidebar.slider("Physical Health (days)", 0, 30)
ment_health = st.sidebar.slider("Mental Health (days)", 0, 30)
age_group = st.sidebar.selectbox("Age Group", options=range(1, 13))
kidney_disease = st.sidebar.selectbox("Kidney Disease", options=[1, 2, 9])
high_bp = st.sidebar.selectbox("High Blood Pressure", options=[0, 1])
high_chol = st.sidebar.selectbox("High Cholesterol", options=[0, 1])
chol_check = st.sidebar.selectbox("Cholesterol Check", options=[0, 1])
asthma = st.sidebar.selectbox("Asthma", options=[1, 2, 9])
copd = st.sidebar.selectbox("COPD", options=[1, 2, 9])
smoker = st.sidebar.selectbox("Smoker", options=[0, 1])
stroke = st.sidebar.selectbox("Stroke", options=[0, 1])
heart_disease_or_attack = st.sidebar.selectbox("Heart Disease or Attack", options=[0, 1])
phys_activity = st.sidebar.selectbox("Physical Activity", options=[0, 1])
hvy_alcohol_consump = st.sidebar.selectbox("Heavy Alcohol Consumption", options=[0, 1])
any_healthcare = st.sidebar.selectbox("Any Healthcare", options=[0, 1])
no_docbc_cost = st.sidebar.selectbox("Could Not See Doctor Because of Cost", options=[0, 1])
gen_hlth = st.sidebar.selectbox("General Health", options=range(1, 6))
diff_walk = st.sidebar.selectbox("Difficulty Walking", options=[0, 1])
sex = st.sidebar.selectbox("Sex", options=[0, 1])
education = st.sidebar.selectbox("Education", options=range(1, 7))
income = st.sidebar.selectbox("Income", options=range(1, 12))

# Prepare input
input_data = {col: 0 for col in original_cols}
input_data.update({
    "BMI": bmi,
    "PhysHlth": phys_health,
    "MentHlth": ment_health,
    "AgeGroup": age_group,
    "KidneyDisease": kidney_disease,
    "HighBP": high_bp,
    "HighChol": high_chol,
    "CholCheck": chol_check,
    "Asthma": asthma,
    "COPD": copd,
    "Smoker": smoker,
    "Stroke": stroke,
    "HeartDiseaseorAttack": heart_disease_or_attack,
    "PhysActivity": phys_activity,
    "HvyAlcoholConsump": hvy_alcohol_consump,
    "AnyHealthcare": any_healthcare,
    "NoDocbcCost": no_docbc_cost,
    "GenHlth": gen_hlth,
    "DiffWalk": diff_walk,
    "Sex": sex,
    "Education": education,
    "Income": income
})

if st.sidebar.button("Assess Risk"):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=X_full_encoded.columns, fill_value=0)

    proba = pipe.predict_proba(input_df)[0, 1]
    st.metric("Diabetes Risk Score", f"{proba:.2%}")

    # SHAP explanation
    X_scaled = pipe.named_steps["scaler"].transform(input_df)
    shap_vals = explainer.shap_values(X_scaled)[0]
    base_value = explainer.expected_value[1]

    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=X_scaled,
        feature_names=input_df.columns.tolist()
    )

    st.subheader("Feature Contributions")
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, max_display=10, show=False)
    st.pyplot(fig)
