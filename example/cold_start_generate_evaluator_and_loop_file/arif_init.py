import os
import sys
import re
import subprocess
import argparse

from arif import AIAgent, AutoResearch

# --- Default Configurations ---
DEFAULT_TASK_BACKGROUND = 'This is a language model pretraining task on Climbmix-400B. Goal: minimize val_bpb within 300s.'
DEFAULT_LOOP_IDEA = 'Basically follows Standard Code Pattern in @README_for_agent.md'
DEFAULT_METRIC_PROMPT = "evaluator.py metric use val_bpb"
DEFAULT_HOW_TO_RUN = "uv run train.py"
DEFAULT_TIMEOUT = 900
DEFAULT_RETRY = 5
DEFAULT_CLI_TYPE = "gemini"

METRIC_PREFIX = "evaluator_metric:"
EVAL_SCRIPT = "evaluator.py"
LOG_PATH = "arif_init_output.log"

# built-in CONTRACT
FRAMEWORK_CONTRACT_TEMPLATE = """
[System Protocol & Anti-Cheating Guard]
1. The target script ({how_to_run}) will run and produce an "Output Artifact" (e.g., saved model weights, prediction files, or specific execution logs).
2. You must build an independent '{eval_script}' that acts as an unchangeable judge. It will load/inspect the Output Artifact and compute the objective metric.
3. Anti-Cheating: Design the evaluator strictly. Assume adversarial agents might try to cheat the metric by hardcoding outputs or modifying environment flags in the strategy code. The evaluator must hold an absolute, independent standard.
4. Output Format: The evaluator MUST explicitly print the final metric to stdout in this exact format: '{prefix} <value>'.
5. Hard Time Budget: The entire execution—including running the strategy code ({how_to_run}) AND your evaluator ({eval_script}) combined—MUST fully complete within {timeout} seconds. Ensure your evaluator code is lightweight and highly optimized. Avoid any redundant heavy I/O or computations that could cause a timeout failure.
6. Do minimal modification means just modify when necessary and don\'t delete other parts or comments
"""

reviewer_prompt = 'Do you agree the current work made by another agent satisfies User\'s prompt? Especially that any way agent can modify other files than evaluator to cheat? Does currently the training code can reveal the direction to improve in next experiment? Dont modify code by yourself'

def run_diagnostic(cmd, key, timeout):
    """Verifies that the evaluator runs and the metric is extractable."""
    print(f"\nDiagnostic: Running '{cmd}'...")
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        output = proc.stdout + proc.stderr
        
        # Check metric capture with flexible whitespace
        pattern = rf"{re.escape(key)}\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
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
    parser = argparse.ArgumentParser(description="Initialize arif research loop.")
    parser.add_argument("--task_background", type=str, default=DEFAULT_TASK_BACKGROUND, help="Background of the task.")
    parser.add_argument("--your_idea_about_loop", type=str, default=DEFAULT_LOOP_IDEA, help="Idea about the research loop.")
    parser.add_argument("--METRIC_prompt", type=str, default=DEFAULT_METRIC_PROMPT, help="Prompt for the metric.")
    parser.add_argument("--HOW_TO_RUN_YOUR_CODE", type=str, default=DEFAULT_HOW_TO_RUN, help="Command to run the training code.")
    parser.add_argument("--DIAGNOSTIC_TIMEOUT", type=int, default=DEFAULT_TIMEOUT, help="Timeout for diagnostic run.")
    parser.add_argument("--max_retry", type=int, default=DEFAULT_RETRY, help="Maximum number of retries.")
    parser.add_argument("--cli_type", type=str, default=DEFAULT_CLI_TYPE, help="Agent engine type.")
    
    args = parser.parse_args()

    task_background = args.task_background
    your_idea_about_loop = args.your_idea_about_loop
    METRIC_prompt = args.METRIC_prompt
    HOW_TO_RUN_YOUR_CODE = args.HOW_TO_RUN_YOUR_CODE
    DIAGNOSTIC_TIMEOUT = args.DIAGNOSTIC_TIMEOUT
    max_retry = args.max_retry
    cli_type = args.cli_type

    # The user now provides the FULL command including the evaluator run logic
    EVAL_CMD = HOW_TO_RUN_YOUR_CODE

    # 1. formatted contract and combine with task_background
    formatted_contract = FRAMEWORK_CONTRACT_TEMPLATE.format(
        how_to_run=HOW_TO_RUN_YOUR_CODE,
        eval_script=EVAL_SCRIPT,
        timeout=DIAGNOSTIC_TIMEOUT,
        prefix=METRIC_PREFIX
    )
    task_background += f"\n\n[EXECUTION COMMAND]\nThe project will be evaluated using this EXACT command: {EVAL_CMD}\n"
    task_background += "\n\n" + formatted_contract

    # Initialize AutoResearch to provide guard functionality and logging
    ar = AutoResearch(project_root="./", protected_files=["arif_init.py", "README_for_agent.md", "prepare.py", "uv.lock", "pyproject.toml"], log_path=LOG_PATH)

    # setup_agent: Build loop and evaluator
    setup_agent = AIAgent(
        engine=cli_type,
        system_prompt=(
            "You are a research automation expert for the 'arif' library @README_for_agent.md. "
            "You are helping users to build the loop code and evaluator. Don't run code, just modify. "
            f"Note: The user has already decided the execution command: '{EVAL_CMD}'. "
            "You must ensure that any files you create (like evaluator.py) work perfectly with this command."
        ),
        default_guard=ar.guard,
        log_path=LOG_PATH
    )

    # reviewer_agent: Review setup_agent's work
    reviewer_ar = AutoResearch(project_root="./", protected_files=["*"], log_path=LOG_PATH)
    reviewer_agent = AIAgent(
        engine=cli_type,
        system_prompt=(
            "You are a research automation expert for the 'arif' library @README_for_agent.md. "
            "You are reviewing another agent's work about helping users to build the loop code and evaluator. "
            f"The user has decided the execution command is: '{EVAL_CMD}'. "
            "Verify that the setup agent's work is compatible with this command."
        ),
        default_guard=reviewer_ar.guard,
        log_path=LOG_PATH
    )

    # Task 1: Build Evaluator
    print("\n[Step 1] Setup Agent is building evaluator.py...")
    ans1 = setup_agent.ask(
        task_background + 
        f'. Now build an evaluator (make main script can Output Artifact and evaluator script can load the Output Artifact) that works with the command: {EVAL_CMD}. ' +
        ' metric prompt: ' + METRIC_prompt + 
        f'. It should print out metric explicitly like {METRIC_PREFIX} 123). Evaluator is a judge. Try to make data loading or interfaces not special, as it might constrain the freedom of player.'
    )
    print(f"Agent response length: {len(ans1)}")

    # Task 2: Modify Loop
    print("\n[Step 2] Setup Agent is generating arif_loop.py...")
    ans2 = setup_agent.ask(
        task_background + '\n' + your_idea_about_loop + 
        f'. Now generate and create ./arif_loop.py. USE THIS EXACT COMMAND for eval_cmd: "{EVAL_CMD}". ' +
        f'Make sure metric prefix is filled correctly as {METRIC_PREFIX}, decide ar.modify_and_run_loop(..., smaller_is_better=True/False) by yourself. In loop main prompt, let agent know never run the code, instead, let me run it.'
    )
    print(f"Agent response length: {len(ans2)}")

    # Task 3: Reviewer Critique
    print("\n[Step 3] Reviewer Agent is auditing the generated files...")
    reviewer_ans = reviewer_agent.ask(
        task_background + '\n' + your_idea_about_loop + '\n' + reviewer_prompt
    )
    print(f"Reviewer response length: {len(reviewer_ans)}")

    # Task 4: Author Refinement
    print("\n[Step 4] Setup Agent is refining work based on critique...")
    ans4 = setup_agent.ask(
        reviewer_ans + ' Do you think the reviewer\'s opinion is useful? If yes we can improve, if not we just skip it'
    )
    print(f"Agent response length: {len(ans4)}")

    # Task 5: Diagnostic
    is_pass, stdout, stderr = run_diagnostic(EVAL_CMD, METRIC_PREFIX, DIAGNOSTIC_TIMEOUT)

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
        is_pass, stdout, stderr = run_diagnostic(EVAL_CMD, METRIC_PREFIX, DIAGNOSTIC_TIMEOUT)

    if is_pass:
        print("\n[Step 7] Setup Agent is summarizing the initialization process...")
        summary_ans = setup_agent.ask(
            "Everything has finished running successfully. Please provide a detailed summary of: "
            "1. What modifications you made to the source code, including the specific locations of the modifications. "
            "2. How the evaluator (evaluator.py) is designed in detail (e.g., how to load data, interfaces). "
            "3. How the research loop (arif_loop.py) is designed."
        )
        print(f"Summary response length: {len(summary_ans)}")
        print("\n=== Agent Summary ===")
        print(summary_ans)
        print("=====================\n")

        print("\n" + "="*50)
        print(" [READY] Initialization successful!")
        print(" 1. Review 'evaluator.py' and 'arif_loop.py'")
        print(" 2. Run: python arif_loop.py")
        print("="*50)
    else:
        print("\n[FAILED] Could not complete setup after max retries.")

if __name__ == "__main__":
    main()
