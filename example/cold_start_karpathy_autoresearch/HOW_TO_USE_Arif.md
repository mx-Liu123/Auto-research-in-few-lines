# Guidelines for Coding Agents using `arif`

As a coding agent, you can use the `arif` library to automate research and experimentation. This document provides technical specifications and patterns to help you build and manage research loops effectively.

## Core Concepts for Agents

### 1. State Isolation (Snapshotting)
`arif` handles directory switching and project state isolation automatically. You should never manually `os.chdir` or `shutil.copy`. Always use the `enter_exp` context manager.

```python
with ar.enter_exp(B, L, S):
    # You are now in an isolated folder. 
    # Modifications here won't affect other experiments.
```

### 2. Hierarchical State (B.L.S)
The experiment identifier follows a `Branch.Level.Step` format:
- **Branch (B)**: A major research direction.
- **Level (L)**: Successive improvements within a branch.
- **Step (S)**: Trials at the same level.

### 3. File Protection
Only **evaluator scripts** and their **direct dependencies** should be listed as protected files. 

**Note**: Training scripts or model strategy (e.g., `strategy.py`, `train.py`) should **never** be protected, as these are the primary targets for your modifications.

By convention, generated research tasks use `evaluator.py` as the protected judge. The bootstrap script does not ask for this path; it builds the final evaluation command by running your code first, then `evaluator.py`.

### 4. Bootstrap Configuration
Configure `arif_init.py` by editing the variables at the top of the file:

```python
DIRECTION = "min"
HOW_TO_RUN_YOUR_CODE = "python strategy.py"
EVAL_SCRIPT = "evaluator.py"
EVAL_CMD = f"{HOW_TO_RUN_YOUR_CODE} && python {EVAL_SCRIPT}"
```

Then run `python arif_init.py`. The script writes `arif_run.py` and refreshes `HOW_TO_USE_Arif.md` in the current task folder.

### 5. Advanced API Usage
- **`ar.continue_branch(B)`**: Resumes an existing branch. It automatically detects the last attempt and returns the next `(B, L, S)`.
- **`ar.run_cmd(cmd, timeout)`**: Executes arbitrary shell commands within the current experiment directory. Returns `(returncode, stdout, stderr)`.
- **`ar.modify_and_run_loop(..., smaller_is_better=True)`**: Runs the modify/evaluate/retry cycle. Set `smaller_is_better=False` for metrics that should be maximized.
- **`agent.ask(prompt, model=None, new_session=False)`**: You can dynamically switch to a different LLM model or start a fresh session for summaries or secondary tasks.
- **`AIAgent(..., log_path="arif_LLM_response.log")`**: Exports prompts, CLI output, and parsed responses to a local log file for auditability.

## Recommended Workflow Pattern (Reference: `example/basic_loop.py`)

1. **Initialization**: Configure global timeouts, system prompts, and file protection.
2. **Experiment Isolation**: Use `enter_exp` to create a fresh workspace for every trial.
3. **Context Injection**: Use `ar.get_history(as_text=True)` to let the agent learn from previous failures at the same level.
4. **Adversarial Reasoning**:
   - **Author Agent** proposes a hypothesis.
   - **Reviewer Agent** returns an `APPROVED: true/false` judgment and concrete critique.
   - **Author Agent** refines the plan using the critique, then implements it.
   - **Reviewer Agent** audits stdout/stderr, mentioned plot paths such as `@best_result.png`, and prompt quality after evaluation.
5. **Autonomous Loop**: Use `ar.modify_and_run_loop()` to handle the internal "Modify -> Run -> Self-Correct" cycle.
6. **Evolution Logic**: Update `L` (Level) when the metric improves, otherwise increment `S` (Step).

## Standard Code Pattern

```python
from arif import AutoResearch, AIAgent

# --- Configuration ---
AGENT_TIMEOUT, CMD_TIMEOUT = None, 600 
LOG_NAME = "arif_LLM_response.log" 
SYSTEM_PROMPT = "This is a ML project. Modify train.py to reduce loss. Focus on CWD."

# 1. Initialize
ar = AutoResearch(project_root="./", protected_files=["evaluator.py"], log_path=LOG_NAME)
agent = AIAgent(
    engine="claude", 
    system_prompt=SYSTEM_PROMPT, 
    default_guard=ar.guard, 
    default_timeout=AGENT_TIMEOUT,
    log_path=LOG_NAME
)

B, L, S = ar.new_branch() 
best_metric = float("inf")

# 2. Research Loop
for i in range(20):
    with ar.enter_exp(B, L, S):
        # Fetch lessons from previous failures at current Level
        history_text = ar.get_history(L=L, if_improved=False, limit=3, as_text=True)

        # Propose hypothesis
        hypothesis = agent.ask(f"Lessons:\n{history_text}\nPropose a hypothesis.", new_session=True)

        # High-level Modify-Run-Retry cycle
        success, current_metric, stdout, stderr = ar.modify_and_run_loop(
            agent, 
            modify_prompt=f"Hypothesis: {hypothesis}\nImplement the change.", 
            eval_cmd="python train.py && python evaluator.py",
            metric_extract="Loss: ",
            best_metric=best_metric,
            smaller_is_better=True,
            max_trials=3,
            timeout=CMD_TIMEOUT
        )

        # Summarize and Save
        summary = agent.ask("Summarize the experiment results.")
        ar.save_history(metric=current_metric, if_improved=success, hypothesis=hypothesis, summary=summary)

        # 3. Evolution Logic
        if success:
            best_metric, L, S = current_metric, L + 1, 1 
        else:
            S += 1
```

## Best Practices
- **Self-Correction**: `modify_and_run_loop` provides internal retries. Use up to 5 trials for setup-style dry-run failures so stdout/stderr feedback is returned to the Author before giving up.
- **Atomic Edits**: Focus on one change per experiment.
- **Multi-Agent Collaboration**: Use the Reviewer's feedback to avoid repetitive mistakes, inspect outputs/plots, improve prompts, and look for "cheating" patterns.
- **Log Management**: pass the same `log_path` to `AutoResearch` and every `AIAgent` so all LLM prompts, raw CLI output, and parsed responses are auditable.
- **Environment Verification**: Always verify the evaluator runs correctly in the baseline (`expB.0.0`) before starting a long loop.
