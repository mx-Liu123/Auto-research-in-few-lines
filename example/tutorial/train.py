# Modified by Claude
# /// script
# dependencies = [
#   "numpy",
#   "scikit-learn",
#   "joblib",
# ]
# ///
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def get_search_configs():
    """
    Generates 100 random hyperparameter configurations for the strategy.
    """
    # Fix seed for reproducibility across experiment runs
    np.random.seed(42)
    configs = []
    
    # Generate 100 random configurations
    for i in range(100):
        # Randomly choose a model type to make it interesting
        model_type = np.random.choice(["gbm", "ridge", "lasso"])
        
        cfg = {"model_type": model_type}
        
        if model_type == "gbm":
            cfg.update({
                "n_estimators": int(np.random.randint(50, 300)),
                "learning_rate": float(np.random.uniform(0.01, 0.2)),
                "max_depth": int(np.random.randint(2, 6)),
                "subsample": float(np.random.uniform(0.7, 1.0)),
                "min_samples_split": int(np.random.randint(2, 10))
            })
        elif model_type == "ridge":
            cfg.update({
                "alpha": float(10 ** np.random.uniform(-3, 3))
            })
        elif model_type == "lasso":
            cfg.update({
                "alpha": float(10 ** np.random.uniform(-4, 1))
            })
            
        configs.append(cfg)
        
    return configs

class Strategy:
    def __init__(self, params=None):
        """
        Initialize the strategy with hyperparameters.
        """
        self.params = params if params else {}
        self.model = None
        
        model_type = self.params.get("model_type", "gbm")
        
        if model_type == "gbm":
            self.model = GradientBoostingRegressor(
                n_estimators=self.params.get("n_estimators", 100),
                learning_rate=self.params.get("learning_rate", 0.1),
                max_depth=self.params.get("max_depth", 3),
                subsample=self.params.get("subsample", 1.0),
                min_samples_split=self.params.get("min_samples_split", 2),
                random_state=42
            )
        elif model_type == "ridge":
            self.model = Ridge(
                alpha=self.params.get("alpha", 1.0),
                random_state=42
            )
        elif model_type == "lasso":
            self.model = Lasso(
                alpha=self.params.get("alpha", 1.0),
                random_state=42
            )
        else:
            # Fallback
            self.model = GradientBoostingRegressor(random_state=42)

    def fit(self, X, y):
        """
        Fit the underlying model.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict using the underlying model.
        """
        return self.model.predict(X)

if __name__ == "__main__":
    # Generate synthetic data for training
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simple hyperparameter selection (can be modified by agent)
    best_config = get_search_configs()[0]
    strategy = Strategy(best_config)
    
    print(f"Training with config: {best_config}")
    strategy.fit(X_train, y_train)
    
    # Calculate validation MSE to provide a signal to the agent
    y_val_pred = strategy.predict(X_val)
    val_mse = np.mean((y_val - y_val_pred)**2)
    print(f"Validation MSE: {val_mse}")
    
    # Save the raw sklearn model as an artifact for evaluator.py to load
    # This avoids class injection risks and dependency on Strategy class
    # DO NOT CHANGE the artifact filename 'model.joblib'
    joblib.dump(strategy.model, "model.joblib")
    print("Model saved to model.joblib")
    
    # The evaluator will load this file and compute MSE on a hidden test set.
    # Goal: Modify this script (Strategy or hyperparameters) to reduce the final evaluator_metric.