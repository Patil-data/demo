import streamlit as st
from utils.ai_utils import (
    load_tensorflow_model,
    load_pytorch_model,
    predict_disease_tensorflow,
    predict_disease_pytorch,
)

# Load models
tf_model = load_tensorflow_model("models/model.h5")
torch_model = load_pytorch_model("models/model.pth")

# Diagnosis report
if st.button("Generate Diagnosis Report"):
    if all([name, age, weight, height, blood_group, bp, sugar, symptoms, xray_image]):
        # Predict using TensorFlow model
        tf_prediction = predict_disease_tensorflow(tf_model, xray_image)
        
        # Predict using PyTorch model
        torch_prediction = predict_disease_pytorch(torch_model, xray_image)

        st.markdown("## AI-Powered Diagnosis Report")
        st.markdown(f"**TensorFlow Prediction:** {tf_prediction}")
        st.markdown(f"**PyTorch Prediction:** {torch_prediction}")
    else:
        st.error("Please fill in all patient details, symptoms, and upload the X-ray.")
