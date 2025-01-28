# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# 1. Read the data
df = pd.read_csv('used_car.csv')

# 2. Split car names into company and model
def split_car_name(name):
    two_word_companies = ['Land Rover', 'Mercedes Benz', 'Maruti Suzuki', 
                         'Mercedes-Benz', 'Rolls Royce', 'Mini Cooper', 'MINI Cooper']
    name_parts = name.split()
    
    if ' '.join(name_parts[:2]) in two_word_companies:
        company = ' '.join(name_parts[:2])
        model = ' '.join(name_parts[2:])
    else:
        company = name_parts[0]
        model = ' '.join(name_parts[1:])
    
    return pd.Series([company, model], index=['company', 'model'])

# Apply the function and create new columns
df[['company', 'model']] = df['car_name'].apply(split_car_name)

# 3. Clean price data
def clean_price(price_str):
    if isinstance(price_str, str):
        if 'Lakh' in price_str:
            price_str = price_str.replace('₹', '').replace('Lakh', '').strip()
            try:
                return float(price_str) * 100000
            except ValueError:
                return None
        elif 'Crore' in price_str:
            price_str = price_str.replace('₹', '').replace('Crore', '').strip()
            try:
                return float(price_str) * 10000000
            except ValueError:
                return None
    return price_str

df['car_price_in_rupees'] = df['car_price_in_rupees'].apply(clean_price)

# 4. Clean kilometers driven data
def clean_kms_driven(kms_driven_str):
    if isinstance(kms_driven_str, str):
        kms_driven_str = kms_driven_str.replace('km', '').replace(',', '').strip()
        try:
            return int(kms_driven_str)
        except ValueError:
            return None
    return kms_driven_str

df['kms_driven'] = df['kms_driven'].apply(clean_kms_driven)

# 5. Clean fuel type data
df['fuel_type'] = df['fuel_type'].replace({
    'LPG': 'CNG',
    'Diesel + 1': 'Diesel',
    'Petrol + 1': 'Petrol'
})

# 6. Calculate age from year of manufacture
current_year = datetime.datetime.now().year
df['age'] = df['year_of_manufacture'].apply(lambda x: current_year - x)

# 7. Arrange columns in logical order
columns_order = ['company', 'model', 'fuel_type', 'kms_driven', 
                'age', 'car_price_in_rupees', 'city']
df = df[columns_order]

# 8. Create a working copy
df_clean = df.copy()

# 9. Remove companies with low frequency
company_frequency = df_clean['company'].value_counts()
df_clean = df_clean[df_clean['company'].isin(company_frequency[company_frequency >= 15].index)]

# 10. Remove hybrid fuel type if needed
df_clean = df_clean[df_clean['fuel_type'] != 'Hybrid']

# 11. Create dummy variables for categorical columns
fuel_type_dummies = pd.get_dummies(df_clean['fuel_type'], prefix='fuel_type')
city_dummies = pd.get_dummies(df_clean['city'], prefix='city')

# Add dummy variables and drop original columns
df_clean = pd.concat([df_clean, fuel_type_dummies, city_dummies], axis=1)
df_clean = df_clean.drop(['fuel_type', 'city'], axis=1)

# 12. Remove outliers from price
Q1 = df_clean['car_price_in_rupees'].quantile(0.25)
Q3 = df_clean['car_price_in_rupees'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df_clean[~((df_clean['car_price_in_rupees'] < lower_bound) | 
                     (df_clean['car_price_in_rupees'] > upper_bound))]

# 13. Scale kilometers driven
scaler = MinMaxScaler()
df_clean['kms_driven'] = scaler.fit_transform(df_clean[['kms_driven']])

# 14. Encode company and model
company_encoder = LabelEncoder()
model_encoder = LabelEncoder()

df_clean['company'] = company_encoder.fit_transform(df_clean['company'])
df_clean['model'] = model_encoder.fit_transform(df_clean['model'])

# 15. Prepare for linear regression
X = df_clean.drop('car_price_in_rupees', axis=1)
y = df_clean['car_price_in_rupees']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# Optional: Save the cleaned dataset
df_clean.to_csv('cleaned_car_data.csv', index=False)