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
Essential scripts (like evaluators) are protected by a `Guard`. If you modify a protected file during an experiment, `arif` will detect the MD5 change and revert the file to ensure the integrity of the experiment.

## Recommended Workflow Pattern

When you are tasked with optimizing a metric (e.g., reducing loss), follow this structural pattern:

1. **Initialization**: Initialize `AutoResearch` and `AIAgent`.
2. **Branching**: Call `new_branch()` to start a fresh line of inquiry.
3. **Looping**: Iterate through experiments.
4. **Hypothesis**: In the first turn of an experiment, analyze history and propose a change.
5. **Implementation**: Modify the designated files (e.g., `strategy.py` or `model.py`).
6. **Evaluation**: Use `ar.run_cmd()` to execute the evaluation script.
7. **Bookkeeping**: Save results using `ar.save_history(metric=val, if_improved=True/False)`.

## Code Pattern for Agents

```python
from arif import AutoResearch, AIAgent

# Initialize environment
ar = AutoResearch(project_root="./", protected_files=["evaluator.py"])
B, L, S = ar.new_branch()
agent = AIAgent(engine="claude")

# Research Loop
for i in range(MAX_EXP):
    with ar.enter_exp(B, L, S):
        # 1. Propose hypothesis based on ar.get_history()
        # 2. Modify code via agent.execute_safe()
        # 3. Evaluate
        status, stdout, stderr = ar.run_cmd("python evaluator.py")
        # 4. Save history
        ar.save_history(metric=result, if_improved=is_better)
        
        # 5. Logic for next B, L, S
        if is_better:
            L += 1; S = 1
        else:
            S += 1
```

## Best Practices
- **Atomic Edits**: Focus on one change per experiment.
- **CWD Awareness**: Always operate on files in the current working directory.
- **History Retrieval**: Use `ar.get_history()` to learn from previous failures across all levels.
- **Timeout Management**: Pass `timeout` parameters to `execute_safe` and `run_cmd` to prevent hanging during long-running tasks.
