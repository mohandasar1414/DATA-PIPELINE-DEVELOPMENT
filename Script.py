import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Extract Data (Load the CSV data into a DataFrame)
def extract_data(file_path):
    """Extract data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# 2. Transform Data (Preprocess numerical and categorical columns)
def transform_data(df):
    """Transform the data (impute missing values, scale, and encode)."""
    
    # List the numerical and categorical columns
    numerical_features = ['age', 'income', 'years_of_experience']  # Example columns
    categorical_features = ['gender', 'department']  # Example columns
    
    # Numerical pipeline: impute missing values with mean, scale features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
        ('scaler', StandardScaler())  # Scale features to have mean 0 and std 1
    ])
    
    # Categorical pipeline: impute missing values with the most frequent, encode with OneHotEncoder
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical variables
    ])
    
    # Combine both pipelines into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
    
    return preprocessor

# 3. Load Data into Model (Train a machine learning model on the preprocessed data)
def load_data_into_model(df, preprocessor):
    """Train the model on the preprocessed data."""
    # Extract features and target variable from the DataFrame
    X = df.drop('target', axis=1)  # Features (exclude target)
    y = df['target']  # Target variable
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the machine learning model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Apply preprocessing to the data
        ('classifier', RandomForestClassifier())  # Train a Random Forest Classifier
    ])
    
    # Train the model on the training data
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    test_score = model_pipeline.score(X_test, y_test)
    print(f"Model Test Accuracy: {test_score * 100:.2f}%")
    
    return model_pipeline

# 4. Save the Trained Model for Future Use
def save_model(model_pipeline, model_filename='trained_model_pipeline.pkl'):
    """Save the trained model pipeline to a file."""
    try:
        joblib.dump(model_pipeline, model_filename)
        print(f"Model saved as {model_filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

# Main ETL Workflow
def main():
    # Define file path for the CSV data
    file_path = 'data.csv'  # Replace with the path to your CSV file
    
    # Step 1: Extract Data
    df = extract_data(file_path)
    if df is None:
        return
    
    # Step 2: Transform Data
    preprocessor = transform_data(df)
    
    # Step 3: Load Data into Model (Train the model)
    model_pipeline = load_data_into_model(df, preprocessor)
    
    # Step 4: Save the Model
    save_model(model_pipeline)

# Run the ETL Process
if __name__ == "__main__":
    main()
