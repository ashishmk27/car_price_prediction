# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import datetime

# Set page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load the saved model and encoders
@st.cache_resource
def load_model():
    try:
        with open('car_price_model.pkl', 'rb') as file:
            data = pickle.load(file)
        return data['model'], data['scaler'], data['label_encoders'], data['categorical_columns']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def clean_kms_driven(kms_str):
    """Clean kilometers driven values"""
    if isinstance(kms_str, str):
        kms_str = kms_str.replace('km', '').replace(',', '').strip()
        return int(kms_str)
    return kms_str

def main():
    # Title and description
    st.title("🚗 Used Car Price Predictor")
    st.write("Enter the details of your car to get an estimated price")
    
    # Load model and components
    model, scaler, label_encoders, categorical_columns = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Car Details")
        # Get unique values for each categorical field
        companies = sorted(label_encoders['company'].classes_)
        company = st.selectbox("Select Company", companies)
        
        models = sorted(label_encoders['model'].classes_)
        model_name = st.selectbox("Select Model", models)
        
        fuel_types = sorted(label_encoders['fuel_type'].classes_)
        fuel_type = st.selectbox("Select Fuel Type", fuel_types)
        
    with col2:
        st.subheader("Additional Information")
        cities = sorted(label_encoders['city'].classes_)
        city = st.selectbox("Select City", cities)
        
        kms_driven = st.number_input("Kilometers Driven", 
                                   min_value=0, 
                                   max_value=500000, 
                                   value=50000,
                                   step=1000)
        
        current_year = datetime.datetime.now().year
        year = st.slider("Year of Manufacture", 
                        min_value=current_year-20,
                        max_value=current_year,
                        value=current_year-5)
        
        age = current_year - year
    
    # Create a predict button
    if st.button("Predict Price", type="primary"):
        try:
            # Prepare input data
            input_data = {
                'company': company,
                'model': model_name,
                'fuel_type': fuel_type,
                'city': city,
                'kms_driven': kms_driven,
                'age': age
            }
            
            # Calculate derived features
            input_data['price_per_km'] = 0  # Will be updated after prediction
            input_data['price_per_year'] = 0  # Will be updated after prediction
            input_data['km_per_year'] = input_data['kms_driven'] / (input_data['age'] + 1)
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for column in categorical_columns:
                if column in input_df.columns:
                    input_df[column] = label_encoders[column].transform(input_df[column])
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display prediction with formatting
            st.success(f"### Estimated Price: ₹{prediction:,.2f}")
            
            # Display additional insights
            st.subheader("Price Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                price_per_year = prediction / (age + 1)
                st.metric("Price per Year", f"₹{price_per_year:,.2f}")
            
            with col2:
                price_per_km = prediction / (kms_driven + 1)
                st.metric("Price per KM", f"₹{price_per_km:,.2f}")
            
            with col3:
                kms_per_year = kms_driven / (age + 1)
                st.metric("KMs per Year", f"{kms_per_year:,.0f}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Add information about the model
    with st.expander("About the Model"):
        st.write("""
        This car price prediction model uses Linear Regression to estimate used car prices based on various features:
        
        - **Company and Model**: The manufacturer and specific model of the car
        - **Fuel Type**: The type of fuel used (Petrol, Diesel, etc.)
        - **Location**: The city where the car is being sold
        - **Kilometers Driven**: Total distance covered by the car
        - **Age**: Age of the car in years
        
        The model has been trained on historical car sales data from various Indian cities.
        """)
        
        st.info("""
        **Note**: This is a predictive model and actual prices may vary based on additional factors 
        such as car condition, market demand, and specific features of the car.
        """)

if __name__ == "__main__":
    main()