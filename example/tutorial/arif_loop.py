from arif import AutoResearch, AIAgent
import os

def main():
    # --- Configuration ---
    # AGENT_TIMEOUT: timeout for LLM calls. CMD_TIMEOUT: timeout for code execution.
    AGENT_TIMEOUT, CMD_TIMEOUT = 300, 900 
    
    main_prompt = (
        "This is a regression task on the Seoul Bike Sharing Demand dataset. "
        "The goal is to minimize the Mean Squared Error (MSE) on the test set. "
        "You should modify 'train.py' to improve the model performance. "
        "Potential improvements include:\n"
        "- Feature Engineering: Look at the 'Date' column; it is currently being one-hot encoded, which is suboptimal. Extract day, month, or season.\n"
        "- Model Selection: Explore different regressors beyond the baseline RandomForest.\n"
        "- Hyperparameter Tuning: Optimize the chosen model's parameters.\n\n"
        "SYSTEM CONSTRAINTS:\n"
        "1. Your script must save the final trained model to 'model.joblib' in the current directory. Never run the code, let me run the code.\n"
        "2. The evaluator will load this file and compute the metric on a hidden test set (the last 30% of rows).\n"
        "3. [Anti-Cheating] The 'train.py' script has been restricted to only access the first 70% of the dataset. Do not try to bypass this split. "
        "The 'evaluator_metric:' string is reserved for the evaluator; do not print it in 'train.py'."
    )
    
    # 1. Initialize AutoResearch with file protection
    # We protect the evaluator and setup files to ensure the agent only modifies the strategy/training code.
    log_name = "arif_LLM_response.log"
    ar = AutoResearch(
        project_root="./", 
        protected_files=["evaluator.py", "arif_init.py", "README_for_agent.md", "arif_loop.py"], 
        log_path=log_name
    )
    
    # Initialize the AIAgent with the task instructions
    agent = AIAgent(
        # engine="gemini", 
        engine="agy", 
        system_prompt=main_prompt,
        default_guard=ar.guard,
        default_timeout=AGENT_TIMEOUT,
        log_path=log_name
    )
    
    # Start or continue a research branch
    B, L, S = ar.new_branch()
    best_metric = 171198.739 # We want to minimize MSE

    # The EXACT execution command as specified by the user
    EVAL_CMD = "/home/liumx/.conda/envs/llamacpp/bin/python train.py && /home/liumx/.conda/envs/llamacpp/bin/python evaluator.py"

    for _ in range(10): # Iterate through 20 experiment steps
        with ar.enter_exp(B, L, S):
            # 2. Provide context from previous trials at the same level
            history_text = ar.get_history(B=B, L=L, if_improved=False, limit=3, as_text=True)

            print(f"\n--- Starting Experiment {B}.{L}.{S} ---")
            print("Generating hypothesis based on history...")
            hypothesis_prompt = (
                f"Previous experiment history:\n{history_text}\n"
                "Based on the code and previous results, propose a hypothesis to improve the model and reduce the MSE metric."
            )
            hypothesis = agent.ask(hypothesis_prompt, new_session=True)

            # 3. Modify-Run-Evaluate Cycle
            # ar.modify_and_run_loop handles the internal retries if the code fails to run.
            success, current_metric, _, _ = ar.modify_and_run_loop(
                agent, 
                modify_prompt=". Now modify 'train.py' based on the proposed hypothesis. Ensure it saves the model to 'model.joblib'.",
                eval_cmd=EVAL_CMD,
                metric_extract="evaluator_metric: ",
                best_metric=best_metric,
                max_trials=3,
                timeout=CMD_TIMEOUT,
                smaller_is_better=True # We are minimizing MSE
            )

            print(f"Experiment finished. Success: {success}, Metric (MSE): {current_metric}")
            summary = agent.ask(". Now summarize the modifications made and the resulting performance changes.")

            # 4. Save results to history and evolve the branch
            ar.save_history(metric=current_metric, if_improved=success, hypothesis=hypothesis, summary=summary)
            
            if success:
                # If the metric improved, move to the next Level
                best_metric, L, S = current_metric, L + 1, 1
            else:
                # If it didn't improve, increment the Step for the current Level
                S += 1

if __name__ == "__main__":
    main()
