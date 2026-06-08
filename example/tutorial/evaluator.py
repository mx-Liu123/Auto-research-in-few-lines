import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo
import os

def main():
    model_path = "model.joblib"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        # Print a high metric to indicate failure but allow the loop to continue
        print("evaluator_metric: 1e18")
        return

    # Fetch dataset (id=560)
    try:
        seoul_bike_data = fetch_ucirepo(id=560)
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        # Return high metric if data fetching fails
        print("evaluator_metric: 1e18")
        return
    
    df = seoul_bike_data.data.original
    
    # Target is Rented Bike Count
    y = df['Rented Bike Count']
    X = df.drop(columns=['Rented Bike Count'])
    
    # Same split as train.py: 70% train, 30% test
    split_idx = int(len(X) * 0.7)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    y_test = y_test.values.ravel()

    # Load the model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("evaluator_metric: 1e18")
        return
    
    # Predict and evaluate
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"evaluator_metric: {mse:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("evaluator_metric: 1e18")

if __name__ == "__main__":
    main()
