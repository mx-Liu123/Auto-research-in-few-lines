import re

from arif import AutoResearch, AIAgent


def main():
    # --- Configuration ---
    AGENT_TIMEOUT = None # Set in seconds, e.g., 300 for 5 minutes. None for no timeout.
    CMD_TIMEOUT = 600    # Set in seconds, e.g., 600 for 10 minutes.

    main_prompt = "This is a machine learning project. You need to modify train.py to reduce loss. Focus on CWD and do not modify external files."
    ar = AutoResearch(project_root="./", protected_files=["evaluator.py", "evaluator_lib/"])

    B, L, S = ar.new_branch()
    agent = AIAgent(engine="claude")

    best_loss = float("inf")

    for i in range(20):
        print(f"\n--- Starting Experiment Loop {i+1}/20 ---")
        with ar.enter_exp(B, L, S):
            print(f"Current Workspace: {ar.current_exp_dir}")
            fails = [h for h in ar.get_history() if h["L"] == L and h.get("if_improved") is False][-3:]
            history_text = "\n".join([f"[{h['exp_id']}]: {h.get('error') or h.get('note') or ''}" for h in fails])

            print("Generating experiment hypothesis...")
            hypothesis = agent.execute_safe(
                main_prompt + f"Previous lessons:\n{history_text}. Now you only need to propose an experimental hypothesis, do not modify the code.",
                guard=ar.guard,
                new_session=True,
                timeout=AGENT_TIMEOUT
            )

            trails = 0
            current_loss = float("inf")
            if_improved = False
            while trails < 3:
                trails += 1
                print(f"  Trail {trails}/3: Modifying train.py...")
                _ = agent.execute_safe(
                    main_prompt + "Now start modifying the code, but do not run it.",
                    guard=ar.guard,
                    timeout=AGENT_TIMEOUT
                )

                print("  Running evaluation...")
                status, stdout, stderr = ar.run_cmd("python train.py && python evaluator.py", timeout=CMD_TIMEOUT)
                match = re.search(r"Loss:\s*([-+]?[0-9]*\.?[0-9]+)", stdout)
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


if __name__ == "__main__":
    main()
