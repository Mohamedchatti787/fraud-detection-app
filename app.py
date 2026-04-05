import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

# 2. Header
st.title("🛡️ FraudShield: Intelligent Transaction Monitoring")
st.write("Real-time fraud detection powered by XGBoost. Developed by Mohamed.")

# 3. Load the Saved Artifacts
@st.cache_resource
def load_assets():
    model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading models: {e}")

# 4. Sidebar Inputs
st.sidebar.header("📍 Transaction Details")
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=120.50)
time_val = st.sidebar.number_input("Seconds since first transaction", min_value=0, value=3600)

st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Advanced System Data (V1-V28)"):
    st.caption("PCA-anonymized features for model input.")
    v_features = []
    for i in range(1, 29):
        v = st.sidebar.slider(f"V{i}", -20.0, 20.0, 0.0)
        v_features.append(v)

# 5. Main Analysis Section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Action Center")
    st.write("Click below to run the AI risk assessment.")
    run_check = st.button("Run Fraud Check")

with col2:
    if run_check:
        # Scale the amount
        scaled_amount = scaler.transform(np.array([[amount]]))[0][0]
        
        # Combine features
        input_data = np.array([[time_val, scaled_amount] + v_features])
        
        # Prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        
        st.subheader("Analysis Result")
        if prediction[0] == 1:
            st.error("### 🚨 HIGH RISK DETECTED")
            st.metric("Risk Level", f"{probability:.2%}", delta="FRAUDULENT", delta_color="inverse")
        else:
            st.success("### ✅ TRANSACTION SECURE")
            st.metric("Risk Level", f"{probability:.2%}", delta="CLEAN")

# 6. Analytics Section
st.markdown("---")
with st.expander("📊 View Model Performance Insights"):
    t1, t2 = st.tabs(["Performance", "Feature Importance"])
    with t1:
        st.image("images/model_comparison_curve.png")
        st.image("images/final_confusion_matrix.png")
    with t2:
        st.image("images/feature_importance.png")