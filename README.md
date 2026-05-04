# arif: Auto-Research In Few-lines

This repository enables users to build customized automated research loops with just a few lines of Python, striking an effective balance between creative freedom and experimental control. Crucially, you can direct your coding agent to use `arif` to autonomously construct and monitor its own research workflows.

A lightweight Python micro-framework for LLM-driven research experiments.

## Core Philosophy
The framework follows a "Like a caveman" approach: the user retains full control over the main loop and logic, while the library handles the underlying infrastructure:
- **Isolated Workspaces**: Automatic snapshotting of project state for every experiment.
- **Branch Management**: Hierarchical organization for different research directions.
- **History Bookkeeping**: Management of history.json across all branches and experiments.
- **File Guard**: MD5 hashing to prevent LLMs from modifying protected files like evaluators.
- **CLI Adapters**: Support for various LLM CLIs (Claude Code, Gemini CLI, Qwen, etc.).

## Installation

```bash
pip install -e .
```

## Core Components

### AutoResearch
Handles the creation and management of experiment branches.
- `new_branch()`: Initializes a new research line.
- `enter_exp(B, L, S)`: Context manager for setting up an isolated experiment folder and managing the working directory.
- `save_history()` / `get_history()`: Functions for JSON-based metadata tracking.

### AIAgent
A wrapper for LLM CLI tools designed for autonomous (YOLO) mode.
- `execute_safe(prompt, guard)`: Executes the LLM call while ensuring protected files remain unchanged via pre-run and post-run hooks.

### Guard
Monitors essential files. It detects modifications to protected files (like changing evaluation metrics) by comparing MD5 hashes and automatically restores them if a change is detected.

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
