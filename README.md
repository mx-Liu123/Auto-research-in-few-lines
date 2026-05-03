# arif: Auto-Research In Few-lines 🦴

A lightweight Python micro-framework for LLM-driven research experiments.

Core Philosophy: **"Like a caveman"** — The user owns the main loop and logic; the library handles only the "dirty work":
- **Isolated Workspaces**: Automatic snapshotting of project state for each experiment.
- **Branch Management**: Clean hierarchy for different research directions.
- **History Bookkeeping**: Automatic `history.json` management across branches.
- **File Guard**: MD5 hashing to prevent LLMs from accidentally modifying "protected" files (like evaluators).
- **Thin CLI Glue**: Native adapters for popular LLM CLIs (Claude Code, Gemini CLI, etc.).

## 🚀 Installation

```bash
# Clone the repo and install in editable mode
pip install -e .
```

## 🛠 Core Components

### 1. `AutoResearch`
Manages branches and experiment snapshots.
- `new_branch()`: Starts a new research line from the project baseline.
- `enter_exp(B, L, S)`: Context manager that creates an isolated folder and switches the CWD to it.
- `save_history()` / `get_history()`: Simple JSON-based bookkeeping.

### 2. `AIAgent`
A wrapper around LLM CLI tools with YOLO (autonomous) mode support.
- `execute_safe(prompt, guard)`: Runs the LLM with "pre-run" and "post-run" file guards.

### 3. `Guard`
Monitors protected files. If an LLM modifies a protected file (e.g., cheats by changing the evaluation metric), the Guard detects the MD5 mismatch and restores the original file.

## 📂 Folder Structure

Each experiment runs in its own isolated environment:
```text
agent_workspaces/
└── Branch1/
    ├── exp1.0.0/ (Baseline)
    ├── exp1.1.1/ (Success attempt)
    └── exp1.1.2/ (Next attempt based on 1.1.1)
```

## 🧪 Examples

### 1. Basic Loop
A minimal implementation of an optimization loop.
```bash
python example/basic_loop.py
```

### 2. Diabetes Optimization (Recommended)
A full-featured example optimizing a scikit-learn model.
```bash
cd example/diabetes_sklearn
python arif_run.py
```

## 📜 License
MIT
