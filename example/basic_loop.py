import re

from arif import AutoResearch, AIAgent


def main():
    main_prompt = "This is a machine learning project. You need to modify train.py to reduce loss. Focus on CWD and do not modify external files."
    ar = AutoResearch(project_root="./", protected_files=["evaluator.py", "evaluator_lib/"])

    B, L, S = ar.new_branch()
    agent = AIAgent(engine="claude")

    best_loss = float("inf")

    for _ in range(20):
        with ar.enter_exp(B, L, S):
            fails = [h for h in ar.get_history() if h["L"] == L and h.get("if_improved") is False][-3:]
            history_text = "\n".join([f"[{h['exp_id']}]: {h.get('error') or h.get('note') or ''}" for h in fails])

            assumption = agent.execute_safe(
                main_prompt + f"Previous lessons:\n{history_text}. Now you only need to propose an experimental hypothesis, do not modify the code.",
                guard=ar.guard,
                new_session=True,
            )

            trails = 0
            current_loss = float("inf")
            if_improved = False
            while trails < 3:
                trails += 1
                _ = agent.execute_safe(
                    main_prompt + "Now start modifying the code, but do not run it.",
                    guard=ar.guard,
                )

                status, stdout, stderr = ar.run_cmd("python train.py && python evaluator.py")
                match = re.search(r"Loss:\s*([-+]?[0-9]*\.?[0-9]+)", stdout)
                current_loss = float(match.group(1)) if match else float("inf")

                if_improved = current_loss < best_loss
                if if_improved:
                    best_loss = current_loss
                    L, S = L + 1, 1
                    break

            summary = agent.execute_safe(
                main_prompt + "Now you only need to summarize the experiment, do not modify the code.",
                guard=ar.guard,
            )

            ar.save_history(metric=current_loss, assumption=assumption, summary=summary, if_improved=if_improved)
            if not if_improved:
                S += 1


if __name__ == "__main__":
    main()
