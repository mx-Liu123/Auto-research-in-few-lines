import os
import re

from arif import AutoResearch, AIAgent


def main():
    # --- Configuration ---
    AGENT_TIMEOUT = None # Set in seconds, e.g., 300 for 5 minutes. None for no timeout.
    CMD_TIMEOUT = 600    # Set in seconds, e.g., 600 for 10 minutes.

    main_prompt = (
        "This is a machine learning project. You need to modify strategy.py to reduce Loss (MSE). "
        "Focus on CWD and do not modify external files. "
        "Do not modify evaluator.py."
    )

    # Lock project_root to the script's directory for consistency
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ar = AutoResearch(project_root=script_dir, protected_files=["evaluator.py"])
    B, L, S = ar.new_branch()

    agent = AIAgent(engine="claude")
    best_loss = float("inf")

    # Temporarily set to 2 loops for validation
    for i in range(2):
        print(f"\n--- Starting Experiment Loop {i+1}/2 ---")
        with ar.enter_exp(B, L, S):
            print(f"Current Workspace: {ar.current_exp_dir}")
            fails = [h for h in ar.get_history() if h["B"] == B and h["L"] == L and h.get("if_improved") is False][-3:]
            history_text = "\n".join(
                [f"[{h['exp_id']}]: {h.get('error') or h.get('note') or ''}" for h in fails]
            )

            print("Generating experiment hypothesis...")
            hypothesis = agent.execute_safe(
                main_prompt
                + f"Previous lessons:\n{history_text}\nNow you only need to propose an experimental hypothesis, do not modify the code.",
                guard=ar.guard,
                new_session=True,
                timeout=AGENT_TIMEOUT
            )
            print(f"Hypothesis: {hypothesis[:100]}...")

            trails = 0
            current_loss = float("inf")
            if_improved = False

            while trails < 3:
                trails += 1
                print(f"  Trail {trails}/3: Modifying strategy.py...")
                _ = agent.execute_safe(
                    main_prompt + "Now start modifying the code, but do not run it.",
                    guard=ar.guard,
                    timeout=AGENT_TIMEOUT
                )

                print("  Running evaluator.py...")
                status, stdout, stderr = ar.run_cmd("python evaluator.py strategy.py", timeout=CMD_TIMEOUT)

                # evaluator.py prints best metric (MSE) on the last line
                match = re.search(r"Best metric:\s*([-+]?[0-9]*\.?[0-9]+)", stdout)
                current_loss = float(match.group(1)) if match else float("inf")
                print(f"  Current Loss: {current_loss:.4f} (Best so far: {best_loss:.4f})")

                if_improved = current_loss < best_loss
                if if_improved:
                    print(f"  SUCCESS: Improved from {best_loss:.4f} to {current_loss:.4f}!")
                    best_loss = current_loss
                    L, S = L + 1, 1
                    break
                else:
                    print("  No improvement.")

            print("Summarizing experiment...")
            summary = agent.execute_safe(
                main_prompt + "Now you only need to summarize the experiment, do not modify the code.",
                guard=ar.guard,
                timeout=AGENT_TIMEOUT
            )

            ar.save_history(
                metric=current_loss, 
                if_improved=if_improved,
                hypothesis=hypothesis,
                summary=summary
            )
            if not if_improved:
                S += 1
    
    print("\n--- All Experiments Complete ---")
    history = ar.get_history()
    print(f"Total history entries: {len(history)}")
    for h in history:
        print(f"[{h['exp_id']}] Loss: {h.get('metric', 'N/A')}, Improved: {h['if_improved']}")


if __name__ == "__main__":
    main()
