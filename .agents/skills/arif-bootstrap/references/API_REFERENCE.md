## AutoResearch
Main class for managing experiment branches and isolation.
- `__init__(project_root: str = "./", protected_files: list[str] | None = None, log_path: str | None = None)`:
    - If `log_path` is a relative filename, it enables **distributed logging**. Now cross-platform (Windows supported). To prevent infinite growth through inheritance, the log is **automatically reset (overwritten)** upon entering each new experiment.
- `new_branch()`: Creates a baseline and returns (B, 1, 1).
- `enter_exp(B, L, S)`: Context manager. Switches to isolated folder. Logs entry to terminal and `log_path`.
- `modify_and_run_loop(agent, modify_prompt, eval_cmd, metric_extract, max_trials=3, best_metric=inf, timeout=None)`: 
    - High-level trial-and-error loop. 
    - Automatically extracts metrics (e.g., `Loss: 0.123`).
    - Provides failure feedback (stdout/stderr) to the agent on retries.
- `get_history(B=None, L=None, S=None, if_improved=None, limit=None, as_text=False)`:
    - Filters history. Supports ranges for B, L, S (e.g., `S=[S-5, S]`).
    - `as_text=True`: Returns a formatted string for LLM context injection.
- `save_history(**kwargs)`: Logs metadata to `history.json`.

## AIAgent
Wrapper for LLM CLI interactions.
- `__init__(engine, model=None, system_prompt="", default_guard=None, default_timeout=None, log_path=None)`:
    - `system_prompt`: Prepended to all calls.
    - `default_guard`: Global security barrier.
    - `default_timeout`: Global execution limit.
    - `log_path`: Optional file to store full LLM execution trace.
- `ask(prompt, guard=IndexError, timeout=IndexError, new_session=False, model=None, **kwargs)`: 
    - Executes LLM call. 
    - Detailed output is sent to `log_path` (terminal remains concise).

## Guard
MD5-based file protection.
- `__init__(arif_instance)`: Monitors `protected_files` from the `AutoResearch` instance.
## Standard Code Example (arif_run.py)
```python
from arif import AutoResearch, AIAgent

ar = AutoResearch(project_root="./", protected_files=["eval.py"], log_path="arif_LLM_response.log")
# Setup Agent with global defaults
...
```

    engine="claude", 
    system_prompt="Optimize strategy.py.", 
    default_guard=ar.guard
)

B, L, S = ar.new_branch()
best_score = float("inf")

for _ in range(20):
    with ar.enter_exp(B, L, S):
...
```

        # 1. Reason with history context
        history = ar.get_history(L=L, if_improved=False, limit=3, as_text=True)
        hypo = agent.ask(f"Lessons:\n{history}\nPropose hypothesis.", new_session=True)
        
        # 2. Modify and Eval Loop
        improved, score, _, _ = ar.modify_and_run_loop(
            agent, 
            modify_prompt="Implement hypothesis.", 
            eval_cmd="python eval.py",
            metric_extract="Score: ", 
            best_metric=best_score
        )
        
        # 3. Save
        summary = agent.ask("Summarize result.")
        ar.save_history(metric=score, if_improved=improved, hypothesis=hypo, summary=summary)
        
        # 4. Evolve
        if improved:
            best_score, L, S = score, L + 1, 1
        else:
            S += 1
```
