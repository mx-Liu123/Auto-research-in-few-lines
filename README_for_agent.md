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
- **Level (L)**: Successive improvements within a branch (Incremented when results improve).
- **Step (S)**: Trials at the same level (Incremented when results do not improve).

### 3. File Protection
Only **evaluator scripts** and their **direct dependencies** (modules they import) should be listed as protected files. 

**Note**: Training scripts or model strategy (e.g., `strategy.py`, `train.py`) should **never** be protected, as these are the primary targets for your modifications.

## Recommended Workflow Pattern

1. **Initialization**: Initialize `AutoResearch` and `AIAgent` with global defaults (system prompt, guard, etc.).
2. **Reasoning**: Use `ar.get_history(as_text=True)` to fetch past failures and lessons.
3. **Execution**: Use `ar.modify_and_run_loop()` to handle the trial-and-error cycle of modification and evaluation.
4. **Summary**: Use `agent.ask()` to summarize the experiment and save it to history.

## Code Pattern for Agents

```python
from arif import AutoResearch, AIAgent

# Initialize environment with global defaults
ar: AutoResearch = AutoResearch(project_root="./", protected_files=["evaluator.py"], log_path="arif_LLM_response.log")
# Distributed logging: If log_path is relative, a fresh log is created inside each experiment 
# folder (overwritten upon entry to prevent infinite growth).
agent: AIAgent = AIAgent(
    engine="claude", 
    system_prompt="Optimize train.py to reduce loss.",
    default_guard=ar.guard
)

B, L, S = ar.new_branch()

# Research Loop
for i in range(MAX_EXP):
    with ar.enter_exp(B, L, S):
        # 1. Propose hypothesis using lessons as text
        lessons = ar.get_history(L=L, if_improved=False, as_text=True)
        hypo = agent.ask(f"Lessons:\n{lessons}\nPropose hypothesis.", new_session=True)
        
        # 2. Automated Modify-Run-Retry cycle
        improved, score, _, _ = ar.modify_and_run_loop(
            agent, 
            modify_prompt="Implement hypothesis.", 
            eval_cmd="python evaluator.py",
            metric_extract="Loss: "
        )
        
        # 3. Summarize and Save
        summary = agent.ask("Summarize the trial.")
        ar.save_history(metric=score, if_improved=improved, hypothesis=hypo, summary=summary)
        
        # 4. Logic for next L, S
        if improved:
            L += 1; S = 1
        else:
            S += 1
```

## Best Practices
- **Atomic Edits**: Focus on one change per experiment.
- **Minimalism**: Favor `modify_and_run_loop` for the primary optimization cycle.
- **History Retrieval**: Use `ar.get_history(as_text=True)` for quick context injection.
- **Timeout Management**: Use global `default_timeout` or pass `timeout` to `ask` and `run_cmd`.
- **Environment Verification**: Before starting a long research loop, proactively suggest that the user runs the evaluator manually to confirm the environment is correctly configured.
