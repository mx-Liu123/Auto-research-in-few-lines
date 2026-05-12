import os
from arif import AutoResearch, AIAgent

def main():
    # --- Configuration ---
    AGENT_TIMEOUT, CMD_TIMEOUT = None, 600
    main_prompt = (
        "This is a machine learning project. You need to modify strategy.py to reduce Loss (MSE). "
        "Focus on CWD and do not modify external files. Do not modify evaluator.py."
    )

    # Lock project_root to the script's directory for consistency
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ar = AutoResearch(project_root=script_dir, protected_files=["evaluator.py"], log_path="arif.log")
    
    # 1. Initialize Agent with global defaults
    agent = AIAgent(
        engine="gemini", 
        system_prompt=main_prompt, 
        default_guard=ar.guard, 
        default_timeout=AGENT_TIMEOUT
    )

    B, L, S = ar.new_branch()
    best_loss = float("inf")

    # Temporarily set to 2 loops for validation
    for _ in range(2):
        with ar.enter_exp(B, L, S):
            # 2. Simplified History Context (Directly as text)
            history_text = ar.get_history(L=L, if_improved=False, limit=3, as_text=True)

            print("Generating experiment hypothesis...")
            hypothesis = agent.ask(f"Previous lessons:\n{history_text}\nPropose a hypothesis.", new_session=True)

            # 3. High-level Modify-Run Loop
            success, current_loss, _, _ = ar.modify_and_run_loop(
                agent, 
                modify_prompt="Modify strategy.py.", 
                eval_cmd="python evaluator.py strategy.py",
                metric_name="Best metric: ", 
                best_metric=best_loss, 
                max_trials=3,
                timeout=CMD_TIMEOUT
            )

            summary = agent.ask("Summarize the experiment.")

            # 4. Save and Evolve
            ar.save_history(metric=current_loss, if_improved=success, hypothesis=hypothesis, summary=summary)
            if success:
                best_loss, L, S = current_loss, L + 1, 1
            else:
                S += 1
    
    print("\n--- All Experiments Complete ---")
    history = ar.get_history(as_text=True)
    print(history)

if __name__ == "__main__":
    main()
