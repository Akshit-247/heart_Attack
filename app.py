import streamlit as st
import joblib
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# --- Load Model (only once, cached) ---
@st.cache_resource
def load_model():
    model = joblib.load("heart_Attack_model_R.pkl")
    return model

model = load_model()

# --- Header ---
st.title("❤️ Heart Attack Risk Predictor")
st.markdown("Fill in your health details below to check your risk level.")
st.divider()

# --- Personal Info ---
st.subheader("👤 Personal Information")
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Full Name")
    age  = st.number_input("Age", min_value=18, max_value=100, value=30)

with col2:
    sex      = st.radio("Gender", ["Male", "Female"], horizontal=True)
    smoking  = st.selectbox("Smoking", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

st.divider()

# --- Health Metrics ---
st.subheader("🩺 Health Metrics")
col3, col4 = st.columns(2)

with col3:
    LDL = st.slider("LDL (Bad Cholesterol)", 35, 231, 100,
                    help="Low-Density Lipoprotein — lower is better")
    HDL = st.slider("HDL (Good Cholesterol)", 20, 83, 50,
                    help="High-Density Lipoprotein — higher is better")

with col4:
    systolic_bp = st.slider("Systolic Blood Pressure", 74, 165, 120,
                             help="The top number in a blood pressure reading")

st.divider()

# --- Predict Button ---
if st.button("🔍 Predict My Risk", use_container_width=True, type="primary"):

    if not name:
        st.warning("⚠️ Please enter your name first.")
    else:
        # --- Encode inputs exactly as your model was trained ---
        sex_val      = 1 if sex == "Male" else 0
        smoking_val  = 1 if smoking == "Yes" else 0
        diabetes_val = 1 if diabetes == "Yes" else 0

        # Build input array — adjust order to match your training data columns!
        features = np.array([[age, sex_val, LDL, HDL, systolic_bp, smoking_val, diabetes_val]])

        # --- Predict ---
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]  # remove if model doesn't support this

        st.divider()
        st.subheader(f"Results for {name}")

        # Show result
        if prediction == 1:
            st.error("🚨 High Risk of Heart Attack")
        else:
            st.success("✅ Low Risk of Heart Attack")

        # Show probability
        risk_percent = round(probability[1] * 100, 1)
        st.metric("Risk Probability", f"{risk_percent}%")

        # Risk bar
        st.progress(int(risk_percent))

        st.divider()
        st.caption("⚠️ This is an ML prediction, not a medical diagnosis. Please consult a doctor.")