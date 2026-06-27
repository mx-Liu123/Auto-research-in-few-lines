from arif import AutoResearch, AIAgent

def main():
    # --- Configuration ---
    # AGENT_TIMEOUT, CMD_TIMEOUT = None, 600 # No agent timeout, 10min for command
    AGENT_TIMEOUT, CMD_TIMEOUT = 300, 900
    main_prompt = "This is a machine learning project. Modify train.py to reduce loss. Focus on CWD  and do not read or modify external files. Do not run the code ALWAYS, I will run for you."
    
    # 1. Initialize with Defaults and optional log_path
    log_name = "arif_LLM_response.log" # Log saved inside each exp folder
    ar = AutoResearch(project_root="./", protected_files=["evaluator.py", "evaluator_lib/"], log_path=log_name) # Initialize with protection
    agent = AIAgent(
        engine="agy", # Use agy engine
        model="Gemini 3.5 Flash (Low)", # Model name passed to --model
        system_prompt=main_prompt, # Set global task instructions
        default_guard=ar.guard, # Enable file modification guard
        default_timeout=AGENT_TIMEOUT, # Global timeout for LLM calls
        log_path=log_name # Log full trace to local file
    )
    B, L, S = ar.new_branch() # Start new Branch1 at Level1, Step1
    best_loss = float("inf") # Tracking best metric for evolution

    for _ in range(20): # Run for 20 experiment iterations
        with ar.enter_exp(B, L, S): # Enter isolated workspace folder
            # 2. Simplified History Context (Directly as text)
            history_text = ar.get_history(L=L, if_improved=False, limit=3, as_text=True) # Fetch lessons from failures

            print("Generating experiment hypothesis...")
            hypothesis = agent.ask(f"Previous lessons:\n{history_text}\nFirst observe the code and previous lessons, then propose a hypothesis to improve the code.", new_session=True) # Plan next change

            # 3. High-level Modify-Run Loop
            success, current_loss, _, _ = ar.modify_and_run_loop(
                agent, 
                modify_prompt="Modify the code based on hypothesis. But dont run the code.", # Instructions for agent
                eval_cmd="python train.py && python evaluator.py", # Command to run evaluation
                metric_extract="Loss: ", # Prefix to extract score from stdout
                best_metric=best_loss, # Target to beat
                max_trials=3, # Retry up to 3 times on failure
                timeout=CMD_TIMEOUT # Execution time limit
            )

            summary = agent.ask("Summarize the experiment.") # Get summary for history log

            # 4. Save and Evolve
            ar.save_history(metric=current_loss, if_improved=success, hypothesis=hypothesis, summary=summary) # Persist trial data
            if success:
                best_loss, L, S = current_loss, L + 1, 1 # Increment level on improvement
            else:
                S += 1 # Increment step on failure

if __name__ == "__main__":
    main()
