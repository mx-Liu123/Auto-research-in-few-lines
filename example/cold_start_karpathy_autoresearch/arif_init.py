import os
import sys
import re
import subprocess

# Ensure arif can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from arif import AIAgent, AutoResearch

# --- User Configuration ---
# let user input info
task_background = 'This is a language model pretraining task on Climbmix-400B. The goal is to minimize val_bpb within a 300s budget on a 50M parameter model. The strategy.py (in this case train.py) should save a model.pth, and evaluator.py should load this weight to run inference on the test set. evaluator.py is a judge that can not be modified by following agents. You should consider the possibility that agents may use strategy py to cheat the metric, and evaluator must be able to hold the standard.'
your_idea_about_loop = 'Basically follows @basic_loop.py, just fill in when necessary, be modified and don\'t delete other parts or comments'
reviewer_prompt = 'Do you agree the current work made by another agent satisfies User\'s prompt? Especially that any way agent can modify other files than evaluator to cheat? It currently the training code can reveal the path to improve in next experiment?'
METRIC_prompt = "evaluator.py metric use val_bpb"
EVAL_SCRIPT = "evaluator.py"
# DIRECTION = "min"  # "min" or "max"
HOW_TO_RUN_YOUR_CODE = "uv run train.py" # Hint to use correct venv python

EVAL_CMD = f"{HOW_TO_RUN_YOUR_CODE} && python {EVAL_SCRIPT}"
LOG_PATH = "arif_init.log"

def run_diagnostic(cmd, key):
    """Verifies that the evaluator runs and the metric is extractable."""
    print(f"\nDiagnostic: Running '{cmd}'...")
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        output = proc.stdout + proc.stderr
        
        # Check metric capture
        pattern = rf"{re.escape(key)}([-+]?[0-9]*\.?[0-9]+)"
        match = re.search(pattern, output, re.IGNORECASE)
        
        if match:
            score = float(match.group(1))
            print(f"  [OK] Found metric: {key}{score}")
            return True, proc.stdout, proc.stderr
        else:
            print(f"  [ERROR] Could not find '{key}' in output.")
            return False, proc.stdout, proc.stderr
    except Exception as e:
        print(f"  [ERROR] Execution failed: {e}")
        return False, "", str(e)

def main():
    # Initialize AutoResearch to provide guard functionality and logging
    ar = AutoResearch(project_root="./", log_path=LOG_PATH)

    # setup_agent: Build loop and evaluator
    setup_agent = AIAgent(
        engine="gemini",
        system_prompt=(
            "You are a research automation expert for the 'arif' library @HOW_TO_USE_Arif.md. "
            "You are helping users to build the loop code and evaluator. Don't run code, just modify."
        ),
        default_guard=ar.guard,
        log_path=LOG_PATH
    )

    # reviewer_agent: Review setup_agent's work
    # Note: protected_files=["*"] ensures reviewer doesn't modify anything
    reviewer_ar = AutoResearch(project_root="./", protected_files=["*"], log_path=LOG_PATH)
    reviewer_agent = AIAgent(
        engine="gemini",
        system_prompt=(
            "You are a research automation expert for the 'arif' library @HOW_TO_USE_Arif.md. "
            "You are reviewing another agent's work about helping users to build the loop code and evaluator."
        ),
        default_guard=reviewer_ar.guard,
        log_path=LOG_PATH
    )

    max_retry = 5

    # Task 1: Build Evaluator
    print("\n[Step 1] Setup Agent is building evaluator.py...")
    ans1 = setup_agent.ask(
        task_background + 
        '. Now build an evaluator that can evaluate the result of ' + HOW_TO_RUN_YOUR_CODE + 
        ' metric prompt: ' + METRIC_prompt + 
        '. It should print out metric explicitly like evaluator_metric: 123.'
    )
    print(f"Agent response length: {len(ans1)}")

    # Task 2: Modify Loop
    print("\n[Step 2] Setup Agent is modifying basic_loop.py...")
    ans2 = setup_agent.ask(
        task_background + your_idea_about_loop + 
        '. Now modify @example/adversarial_init_arif/basic_loop.py based on user\'s prompt, make sure metric prefix is filled correctly'
    )
    print(f"Agent response length: {len(ans2)}")

    # Task 3: Reviewer Critique
    print("\n[Step 3] Reviewer Agent is auditing the generated files...")
    reviewer_ans = reviewer_agent.ask(
        task_background + your_idea_about_loop + reviewer_prompt
    )
    print(f"Reviewer response length: {len(reviewer_ans)}")

    # Task 4: Author Refinement
    print("\n[Step 4] Setup Agent is refining work based on critique...")
    ans4 = setup_agent.ask(
        reviewer_ans + ' Do you think the reviewer\'s opinion is useful? If yes we can improve, if not we just skip it'
    )
    print(f"Agent response length: {len(ans4)}")

    # Task 5: Diagnostic
    is_pass, stdout, stderr = run_diagnostic(EVAL_CMD, "evaluator_metric: ")

    # Task 6: Retry Loop
    for i in range(max_retry):
        if is_pass:
            break
        print(f"\n[Retry {i+1}/{max_retry}] Fix required. Setup Agent is analyzing errors...")
        ans_retry = setup_agent.ask(
            task_background + your_idea_about_loop + 
            ' We got an error, fix: ' + stdout + stderr
        )
        print(f"Retry response length: {len(ans_retry)}")
        is_pass, stdout, stderr = run_diagnostic(EVAL_CMD, "evaluator_metric: ")

    if is_pass:
        print("\n" + "="*50)
        print(" [READY] Initialization successful!")
        print(" 1. Review 'evaluator.py' and 'basic_loop.py'")
        print(" 2. Run: python basic_loop.py")
        print("="*50)
    else:
        print("\n[FAILED] Could not complete setup after max retries.")

if __name__ == "__main__":
    main()
