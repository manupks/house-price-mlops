import streamlit as st
import pandas as pd
import pickle

# Load model and locations
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("locations.pkl", "rb") as f:
    locations = pickle.load(f)

st.title("🏠 House Price Prediction App")

# Inputs
location = st.selectbox("Location", locations)
total_sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=10)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)

# Warning for unrealistic inputs
if total_sqft / bhk < 300:
    st.warning("Total sqft per BHK seems too low. Are you sure about the inputs?")

# Predict
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        'location': location,
        'total_sqft': total_sqft,
        'bath': bath,
        'BHK': bhk
    }])

    try:
        price = model.predict(input_df)[0]
        price = max(price, 0)  # Prevent negative values
        st.success(f"Estimated Price: ₹ {price:,.2f} Lakhs")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
