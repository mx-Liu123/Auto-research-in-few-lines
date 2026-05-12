from arif import AutoResearch, AIAgent

def main():
    # --- Configuration ---
    AGENT_TIMEOUT, CMD_TIMEOUT = None, 600
    main_prompt = "This is a machine learning project. Modify train.py to reduce loss. Focus on CWD  and do not read or modify external files. Do not run the code ALWAYS, I will run for you."
    
    # 1. Initialize with Defaults and optional log_path
    ar = AutoResearch(project_root="./", protected_files=["evaluator.py", "evaluator_lib/"], log_path="arif.log")
    agent = AIAgent(
        engine="claude", 
        system_prompt=main_prompt, 
        default_guard=ar.guard, 
        default_timeout=AGENT_TIMEOUT
    )
    B, L, S = ar.new_branch()
    best_loss = float("inf")

    for _ in range(20):
        with ar.enter_exp(B, L, S):
            # 2. Simplified History Context (Directly as text)
            history_text = ar.get_history(L=L, if_improved=False, limit=3, as_text=True)

            print("Generating experiment hypothesis...")
            hypothesis = agent.ask(f"Previous lessons:\n{history_text}\nFirst observe the code and previous lessons, then propose a hypothesis to improve the code.", new_session=True)

            # 3. High-level Modify-Run Loop
            success, current_loss, _, _ = ar.modify_and_run_loop(
                agent, 
                modify_prompt="Modify the code based on hypothesis. But dont run the code.", 
                eval_cmd="python train.py && python evaluator.py",
                metric_name="Loss: ", 
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

if __name__ == "__main__":
    main()
