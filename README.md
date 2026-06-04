# Arif: Auto-Research In Few-lines

This repository allows users to build customized auto-research loops in just a few lines of Python, orchestrating your coding CLIs and shell commands, striking an effective balance between creative freedom and experimental control. Crucially, you can direct your coding agent to use `arif` to autonomously construct and monitor its own research workflows.

A lightweight Python micro-framework for LLM-driven research experiments.

## Core Philosophy
The framework follows a minimalist approach: the library manages CLI interfacing and common patterns, while the user defines the high-level logic:
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

---

## Quick Start

Once installed, you can quickly initialize a research loop in your project directory using the following commands. We explicitly list all parameters to make it easy for you to fine-tune them for your specific task:

```bash
# 1. Copy the initialization script and agent guidelines to your project
cp example/cold_start_generate_evaluator_and_loop_file/{arif_init.py,README_for_agent.md} ./

# 2. Run the initialization script (with full default parameters)
python arif_init.py \
  --task_background "This is a language model pretraining task on Climbmix-400B." \
  --your_idea_about_loop "Basically follows Standard Code Pattern in @README_for_agent.md" \
  --METRIC_prompt "evaluator.py metric use val_bpb" \
  --HOW_TO_RUN_YOUR_CODE "uv run train.py" \
  --DIAGNOSTIC_TIMEOUT 900 \
  --max_retry 5 \
  --cli_type "gemini"
```


Parameter Descriptions:
 * `--task_background`: Description of the task, telling the Agent what the optimization goal is.
 * `--your_idea_about_loop`: Expectations for the experiment loop (e.g., which template to reference).
 * `--METRIC_prompt`: Natural language description of the metric, used to guide the generation of the evaluator.
 * `--HOW_TO_RUN_YOUR_CODE`: The command to run your original training code.
 * `--DIAGNOSTIC_TIMEOUT`: Maximum wait time (in seconds) for the diagnostic run. Recommended: training time + compilation overhead.
 * `--max_retry`: Maximum number of retries when automatically generating the evaluator and loop scripts.
 * `--cli_type`: The model engine type to use (default is `gemini`).

After the script finishes, it will generate `evaluator.py` and `arif_loop.py`. You can then start the automated research process by running `python arif_loop.py`.

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
