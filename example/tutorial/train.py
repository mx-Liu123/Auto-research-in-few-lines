"""
Seoul Bike Sharing Demand training script.
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo

def main():
    print("Fetching Seoul Bike Sharing Demand dataset from UCI...")
    # Fetch dataset (id=560)
    try:
        seoul_bike_data = fetch_ucirepo(id=560)
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        print("Please check your internet connection or UCI ML repository status.")
        return
    
    # The UCI repo sometimes defaults 'Functioning Day' as target and 'Rented Bike Count' as feature.
    # We will explicitly use original data to split them correctly.
    df = seoul_bike_data.data.original
    
    # [Anti-Cheating Guard] Restrict training script access to only the first 70% of the data.
    # The remaining 30% is reserved for the independent evaluator.
    split_idx = int(len(df) * 0.7)
    df = df.iloc[:split_idx].copy()
    
    # Target is Rented Bike Count
    y = df['Rented Bike Count']
    
    # We do NOT drop Date and Functioning Day from X so that the raw data interface is preserved.
    # We only drop the target column.
    X = df.drop(columns=['Rented Bike Count'])
    
    print(f"Data fetched. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    # Identify categorical and numerical columns using all features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    print(f"Categorical features: {categorical_cols}")
    print(f"Numerical features: {numerical_cols}")
    
    # Internal split for training/validation (using 80% of the available 70%)
    train_split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:train_split_idx], X.iloc[train_split_idx:]
    y_train, y_val = y.iloc[:train_split_idx], y.iloc[train_split_idx:]
    
    # Convert y to 1D array
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            # sparse_output=False is compatible with newer scikit-learn, falling back if not supported
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Define the model pipeline using RandomForest
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    # Save the model to local storage
    model_path = "model.joblib"
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    # Load the model back
    print(f"Loading model from {model_path}...")
    loaded_model = joblib.load(model_path)
    
    # Evaluate on the validation set (internal)
    print("Evaluating on the internal validation set...")
    y_pred = loaded_model.predict(X_val)
    
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print("-" * 30)
    print("Internal Validation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 : {r2:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
