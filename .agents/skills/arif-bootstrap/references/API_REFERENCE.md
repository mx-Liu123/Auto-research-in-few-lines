# arif API Reference

## AutoResearch
Main class for managing experiment branches and isolation.
- `__init__(project_root, protected_files)`
- `new_branch()`: Returns (B, L, S).
- `enter_exp(B, L, S)`: Context manager. Switches to isolated folder.
- `run_cmd(cmd, timeout)`: Runs shell command in the current context.
- `save_history(metric, if_improved)`: Logs metadata.
- `get_history()`: Returns list of past experiments.

## AIAgent
Wrapper for LLM CLI interactions.
- `__init__(engine)`: Engines: "claude", "gemini", "qwen", "opencode".
- `execute_safe(prompt, guard, tools)`: Runs LLM call with MD5 protection.

## Guard
MD5-based file protection.
- `__init__(protected_files)`: List of files to monitor.

## Standard Code Example (arif_run.py)
```python
from arif import AutoResearch, AIAgent

ar = AutoResearch(project_root="./", protected_files=["eval.py"])
B, L, S = ar.new_branch()
agent = AIAgent(engine="claude")

for _ in range(MAX):
    with ar.enter_exp(B, L, S):
        # 1. Reason
        hypo = agent.execute_safe("propose change", guard=ar.guard, tools="")
        # 2. Act
        agent.execute_safe("modify strategy.py", guard=ar.guard)
        # 3. Eval
        s, out, err = ar.run_cmd("python eval.py")
        # 4. Save
        ar.save_history(metric=parse(out), if_improved=is_better)
        # 5. Update B,L,S logic...
```
