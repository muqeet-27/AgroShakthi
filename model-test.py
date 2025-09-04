import streamlit as st
import pandas as pd
import joblib

# =========================
# 1. Load Trained Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("crop_yield_model.pkl")   # 🔹 keep your .pkl file in the repo root

model = load_model()

# =========================
# 2. Streamlit App
# =========================
st.title("🌾 AI-Powered Crop Yield Prediction")

st.markdown("Enter details below to predict crop yield (Kg per ha).")

# Sidebar for crop selection
chosen_crop = st.sidebar.selectbox("Select Crop", ["MAIZE", "WHEAT", "RICE"])

# =========================
# 3. User Input Form
# =========================
with st.form("prediction_form"):
    state = st.text_input("State Name", "Karnataka")
    district = st.text_input("District Name", "Davangere")
    dist_code = st.number_input("District Code", value=123)
    state_code = st.number_input("State Code", value=29)
    year = st.number_input("Year", value=2021)

    st.subheader("Crop Statistics (Areas and Production)")

    rice_area = st.number_input("RICE AREA (1000 ha)", value=150)
    rice_prod = st.number_input("RICE PRODUCTION (1000 tons)", value=400)

    wheat_area = st.number_input("WHEAT AREA (1000 ha)", value=10)
    wheat_prod = st.number_input("WHEAT PRODUCTION (1000 tons)", value=20)

    maize_area = st.number_input("MAIZE AREA (1000 ha)", value=50)
    maize_prod = st.number_input("MAIZE PRODUCTION (1000 tons)", value=120)

    # Submit button
    submitted = st.form_submit_button("🔍 Predict Yield")

# =========================
# 4. Make Prediction
# =========================
if submitted:
    # Prepare input
    user_input = {
        "State Name": state,
        "Dist Name": district,
        "Dist Code": dist_code,
        "Year": year,
        "State Code": state_code,
        "RICE AREA (1000 ha)": rice_area,
        "RICE PRODUCTION (1000 tons)": rice_prod,
        "WHEAT AREA (1000 ha)": wheat_area,
        "WHEAT PRODUCTION (1000 tons)": wheat_prod,
        "MAIZE AREA (1000 ha)": maize_area,
        "MAIZE PRODUCTION (1000 tons)": maize_prod,
    }

    user_df = pd.DataFrame([user_input])

    # Align with training features
    # 🔹 NOTE: You must save `X_features.columns` during training and load it here
    try:
        feature_cols = joblib.load("feature_columns.pkl")  # load saved feature order
        user_df = user_df.reindex(columns=feature_cols, fill_value=0)
    except:
        st.warning("⚠️ Feature columns file not found. Using raw inputs.")

    # Predict
    prediction = model.predict(user_df)[0]
    st.success(f"🌾 Predicted {chosen_crop} Yield: **{round(prediction, 2)} Kg per ha**")
