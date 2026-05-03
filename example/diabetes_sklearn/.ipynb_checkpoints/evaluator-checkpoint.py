import sys
import os
import importlib

# Ensure current dir is in path for imports
sys.path.insert(0, os.getcwd())

import estimator
import strategy

def evaluate(strategy_file):
    # Reload in case of changes during agent loop
    importlib.reload(strategy)
    importlib.reload(estimator)
    
    strat_str = strategy.TRANSFORM_STRATEGY
    try:
        cv = estimator.run_analysis(strat_str)
        return cv
    except Exception as e:
        print(f"Evaluation Error: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

if __name__ == "__main__":
    print(evaluate("strategy.py"))
