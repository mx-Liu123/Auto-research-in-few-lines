import os
import sys
import random
import itertools
from contextlib import contextmanager
from evaluator import evaluate

# Ensure we can import strategy
sys.path.insert(0, os.getcwd())

@contextmanager
def suppress_output(suppress=True):
    if not suppress:
        yield
        return
    # Redirect sys.stdout and sys.stderr to devnull for Python level prints
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def main():
    # Define search space
    trend_lbs = list(range(20, 100, 5))
    rv_lbs = list(range(10, 80, 5))
    powers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Generate all combinations
    all_combinations = list(itertools.product(trend_lbs, rv_lbs, powers))
    total_combinations = len(all_combinations)

    # Determine execution plan
    max_iter_env = os.environ.get("MAX_SEARCH_ITER")
    
    if max_iter_env is not None and max_iter_env.strip() != "":
        max_iter = int(max_iter_env)
        print(f"MAX_SEARCH_ITER set to {max_iter}. Sampling {max_iter} random combinations from {total_combinations}.")
        random.shuffle(all_combinations)
        search_space = all_combinations[:max_iter]
    else:
        print(f"MAX_SEARCH_ITER not set. Running ALL {total_combinations} combinations.")
        search_space = all_combinations

    print(f"Starting parameter search for {len(search_space)} iterations...")
    print("Output will be suppressed except for the first iteration and new best results.")
    
    best_cv = float('inf')
    best_params = {}
    
    for i, (t_lb, r_lb, p) in enumerate(search_space):
        # Set env vars
        os.environ["TREND_LB"] = str(t_lb)
        os.environ["RV_LB"] = str(r_lb)
        os.environ["POWER"] = str(p)
        
        # Determine if we should suppress output (keep 1st iter output)
        suppress = (i > 0)
        
        # Evaluate
        try:
            with suppress_output(suppress):
                cv = evaluate("strategy.py")
            
            # Logic to reduce spam:
            # 1. Always check for best
            if cv < best_cv:
                best_cv = cv
                best_params = {
                    "TREND_LB": t_lb,
                    "RV_LB": r_lb,
                    "POWER": p
                }
                # Print NEW BEST immediately
                print(f"[Iter {i+1}/{len(search_space)}] ⭐ New Best! CV={cv:.4f} | Params: {best_params}")
            
            # 2. Heartbeat every 100 iterations (so user knows it's running)
            elif (i + 1) % 100 == 0:
                print(f"[Iter {i+1}/{len(search_space)}] ... Processing ... (Current Best CV: {best_cv:.4f})")
                
        except Exception as e:
            print(f"[Iter {i+1}] Failed: {e}")

    print(f"\nSearch Complete. Best CV: {best_cv:.4f}")
    print(f"Best Params: {best_params}")
    
    # Save best params
    with open("best_param.env", "w") as f:
        for k, v in best_params.items():
            f.write(f"{k}={v}\n")
    print("Saved best_param.env")

if __name__ == "__main__":
    main()
