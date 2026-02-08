import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Load the saved model and encoders
@st.cache_resource
def load_model():
    with open('best_crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, encoder, features

model, encoder, feature_names = load_model()

# Title and description
st.title("🌾 Crop Recommendation System")
st.markdown("""
This intelligent system recommends the best crop to grow based on your soil and climate conditions.
Simply enter your land's parameters below and get instant recommendations!
""")

st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Soil Nutrients")
    N = st.slider("Nitrogen (N)", 0, 140, 50, help="Nitrogen content in soil")
    P = st.slider("Phosphorus (P)", 0, 145, 50, help="Phosphorus content in soil")
    K = st.slider("Potassium (K)", 0, 205, 50, help="Potassium content in soil")
    ph = st.slider("pH Value", 3.5, 10.0, 6.5, 0.1, help="Soil pH level")

with col2:
    st.subheader("🌤️ Climate Conditions")
    temperature = st.slider("Temperature (°C)", 8.0, 45.0, 25.0, 0.1, help="Average temperature")
    humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0, 0.1, help="Relative humidity")
    rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, 1.0, help="Average rainfall")

st.markdown("---")

# Prediction button
if st.button("🔍 Get Crop Recommendation", use_container_width=True):
    # Prepare input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_crop = encoder.inverse_transform(prediction)[0]
    
    # Get prediction probability
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100
    
    # Get top 3 recommendations
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_crops = encoder.inverse_transform(top_3_indices)
    top_3_probs = probabilities[top_3_indices] * 100
    
    # Display results
    st.markdown("---")
    st.success("## 🎯 Recommendation Results")
    
    # Main recommendation
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.metric("🏆 Recommended Crop", predicted_crop.upper())
    with col2:
        st.metric("📊 Confidence", f"{confidence:.1f}%")
    with col3:
        if confidence >= 80:
            st.metric("✅ Reliability", "High")
        elif confidence >= 60:
            st.metric("⚠️ Reliability", "Medium")
        else:
            st.metric("⚠️ Reliability", "Low")
    
    # Top 3 recommendations
    st.markdown("### 📋 Top 3 Crop Recommendations")
    
    for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_probs), 1):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{i}. {crop.capitalize()}**")
        with col2:
            st.progress(prob/100)
            st.write(f"{prob:.1f}%")
    
    # Display input summary
    st.markdown("---")
    st.markdown("### 📝 Input Summary")
    
    input_df = pd.DataFrame({
        'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 
                     'Temperature', 'Humidity', 'pH', 'Rainfall'],
        'Value': [f"{N}", f"{P}", f"{K}", f"{temperature}°C", 
                 f"{humidity}%", f"{ph}", f"{rainfall}mm"]
    })
    
    st.dataframe(input_df, use_container_width=True, hide_index=True)

# Sidebar with additional info
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
**Crop Recommendation System**

This system uses Machine Learning (Random Forest) to predict the best crop based on:
- Soil nutrients (N, P, K, pH)
- Climate conditions (Temperature, Humidity, Rainfall)

**Model Accuracy:** 99.27%

**Supported Crops:** 22 types including rice, wheat, cotton, coffee, and more.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("Accuracy", "99.27%")
st.sidebar.metric("Total Crops", "22")
st.sidebar.metric("Features Used", "7")

st.sidebar.markdown("---")
st.sidebar.success("✅ Model: Random Forest")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ using Streamlit | Powered by Machine Learning </p>
</div>
""", unsafe_allow_html=True)
