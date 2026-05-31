# Arif: Auto-Research In Few-lines

This repository enables users to build customized automated research loops with just a few lines of Python, striking an effective balance between creative freedom and experimental control. Crucially, you can direct your coding agent to use `arif` to autonomously construct and monitor its own research workflows.

A lightweight Python micro-framework for LLM-driven research experiments.

## Core Philosophy
The framework follows a "Minimalist & Ergonomic" approach: the library handles the underlying infrastructure and common patterns, while the user defines the high-level logic:
- **Isolated Workspaces**: Automatic snapshotting of project state for every experiment.
- **Branch Management**: Hierarchical organization (B.L.S) for research directions.
- **Ergonomic LLM Interface**: Global system prompts, security guards, and timeouts.
- **Automated Trials**: Built-in loops for modification, evaluation, and feedback.
- **History Bookkeeping**: Pre-formatted context injection for LLM lessons.

## Installation

```bash
git clone https://github.com/mx-Liu123/Auto-research-in-few-lines.git
cd Auto-research-in-few-lines
pip install -e .
```

## Core Components

### AIAgent
A wrapper for LLM CLI tools with persistent context.
- `__init__(engine: str, system_prompt: str, default_guard: any, default_timeout: int | None, log_path: str | None)`: Set global defaults for all calls.
- `ask(prompt, ...)`: Executes an LLM call. Supports local overrides for guard and timeout. Detailed responses are directed to `log_path`.
- Supported adapters include Gemini, Qwen, Claude, OpenCode, and Codex CLI.

### AutoResearch
Handles experiment lifecycle and data management.
- `__init__(project_root: str, protected_files: list[str] | None, log_path: str | None)`: Initialize with a log filename. Now supports Windows via platform-safe `run_cmd`.
- `modify_and_run_loop(agent, modify_prompt, eval_cmd, metric_extract, ..., smaller_is_better=True)`: A high-level abstraction for the "Modify -> Run -> Extract Metric" cycle with automatic retry feedback. Set `smaller_is_better=False` for metrics that should be maximized.
- `get_history(..., as_text=True)`: Retrieves past experiments, optionally formatted as a single string for direct LLM context injection.
- `enter_exp(B, L, S)`: Context manager for setting up an isolated experiment folder. Automatically logs entry info to terminal and log file.

### Guard
Monitors essential files via MD5 hashing to prevent unauthorized modifications to evaluators or datasets.

## Directory Structure

Experiments are organized into isolated folders to ensure reproducibility:
```text
agent_workspaces/
└── Branch1/
    ├── exp1.0.0/ (Baseline)
    ├── exp1.1.1/ (Success attempt)
    └── exp1.1.2/ (Next attempt based on 1.1.1)
```

## Examples

### Basic Loop
A minimal implementation of the optimization cycle.
```bash
python example/basic_loop.py
```

### Diabetes Model Optimization
A complete example that optimizes a scikit-learn model using the diabetes dataset.
```bash
cd example/diabetes_sklearn
python arif_run.py
```

## License
MIT
