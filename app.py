import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # Import Pipeline module

# Load the dataset
df = pd.read_csv("C:/DS-recruitment-challenge/08-DS-recruitment-challenge/your_dataset.csv", encoding='utf-8')

# Split the data into features (X) and target (y)
X = df[['BrandName', 'ModelGroup', 'ProductGroup', 'ShopCategory']]
y = df['Returned']

# Get unique values for BrandName and other categorical features
brand_names = df['BrandName'].unique()
model_groups = df['ModelGroup'].unique()
product_groups = df['ProductGroup'].unique()

# Define categories for each feature
categories = [brand_names, model_groups, product_groups, ['Webshop', 'Offline']]  # ShopCategory

# One-hot encode categorical features
encoder = OneHotEncoder(categories=categories, drop='first')
X_encoded = encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize SMOTE object
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Define the pipeline to include SMOTE and Logistic Regression
pipeline = Pipeline([('smote', smote), ('model', LogisticRegression(random_state=42))])

# Fit pipeline to training data
pipeline.fit(X_train, y_train)

# Define a function to make predictions
def predict_return(input_features):
    # Preprocess input features
    input_encoded = encoder.transform(input_features)
    # Make predictions
    predictions = pipeline.predict(input_encoded)
    return predictions

# Create the Streamlit app
def main():
    st.title('Return Prediction')

    # Add input fields for user input
    brand_name = st.selectbox('Brand Name', sorted(df['BrandName'].unique()))
    model_group = st.selectbox('Model Group', sorted(df['ModelGroup'].unique()))
    product_group = st.selectbox('Product Group', sorted(df['ProductGroup'].unique()))
    shop_category = st.selectbox('Shop Category', ['Webshop', 'Offline'])
    
    # Prepare input features for prediction
    input_features = pd.DataFrame({
        'BrandName': [brand_name],
        'ModelGroup': [model_group],
        'ProductGroup': [product_group],
        'ShopCategory': [shop_category]
    })

    # Make prediction on user input
    if st.button('Predict'):
        prediction = predict_return(input_features)
        if prediction[0] == 1:
            st.error('This product is predicted to be returned.')
        else:
            st.success('This product is predicted not to be returned.')

if __name__ == '__main__':
    main()