import streamlit as st
import numpy as np
import joblib
# Load the trained model
model = joblib.load('rf_model.pkl')
# Define the input fields for the user
st.title("Laptop Price Prediction")
st.divider()
st.write("Enter the specifications of the laptop to predict its price:")
st.divider()
# Input fields for laptop specifications
processor_speed = st.number_input("Processor Speed (GHz)", min_value=0.0, max_value=10.0, step=0.1)
ram_size = st.number_input("RAM Size (GB)", min_value=0, max_value=128, step=4)
storage_capacity = st.number_input("Storage Capacity (GB)", min_value=0, max_value=2000, step=128)
# Convert inputs to a 2D array for prediction
X = [processor_speed, ram_size, storage_capacity]
st.divider()
# Predict the price using the model
prediction = st.button("Predict Price")
st.divider()
if prediction:
    st.balloons()
    # Reshape the input for the model
    X1 = np.array([2.5, 16, 325]).reshape(1, -1)
    predicted_price = model.predict(X1)
    # Make the prediction
    predicted_price = model.predict(X1)
    st.write(f"Price  estimation for the laptop is  {prediction:,.2f}")
    
else:
    st.write("Please use the button for getting a price prediction.")




#['Storage_Capacity', 'Processor_Speed', 'RAM_Size']