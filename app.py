# app.py
import streamlit as st
import pickle
import pandas as pd

# Load model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Locations (from your dataset)
locations = [
    'Electronic City Phase II', 'Chikka Tirupathi', 'Uttarahalli',
    'Lingadheeranahalli', 'Kothanur', 'Whitefield',
    'Old Airport Road', 'Marathahalli', 'other'
]

# UI
st.title("🏠 House Price Predictor")

location = st.selectbox("Select Location", locations)
sqft = st.number_input("Enter Total Square Feet", min_value=300, step=50)
bath = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5, 6])
bhk = st.selectbox("Number of BHK", [1, 2, 3, 4, 5, 6])

if st.button("Predict Price"):
    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'BHK'])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")
