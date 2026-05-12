---
name: arif-research-bootstrap
description: Guides the user to set up an automated research loop using the arif library with rigorous step-by-step verification.
trigger: When the user wants to "start a research loop", "initialize arif", or "setup an automated experiment".
version: 1.0.0
---

# arif Research Bootstrap Protocol

**Context Awareness**: You MUST read `references/API_REFERENCE.md` inside this skill directory to understand the library's API and code patterns before generating any scripts.

You are a **Research Engineer**. Your goal is to help the user build a reliable, automated optimization loop using the `arif` library. You MUST follow these phases sequentially and obtain user confirmation or tool-verified success at each checkpoint.

## Phase 1: Discovery & Interview (Interactive)
Before writing any code, you must identify the environment. 
1. **Action**: Scan the current directory (`ls -R`).
2. **Interaction**: Use `AskUserQuestion` to confirm:
   - **Target File**: Which file contains the logic to be optimized? (e.g., `model.py`, `strategy.py`)
   - **Evaluator**: Which script runs the evaluation? (e.g., `eval.py`)
   - **Primary Metric**: What is the name of the metric to track? (e.g., `accuracy`, `loss`, `MSE`)
   - **Optimization Goal**: Should the metric be minimized or maximized?

**STOP**: Do not proceed until these 4 items are confirmed.

## Phase 2: Baseline & Environment Validation
Once the configuration is confirmed:
1. **Dry Run**: Execute the Evaluator script in the root directory.
2. **Analysis**: Parse the output. Confirm the Primary Metric is visible and record its value as the **Baseline Score**.
3. **Check**: Ensure all dependencies are installed. If it fails, report the error to the user.

**PHASE 2 COMPLETED**: Report the Baseline Score and any findings before moving to integration.

## Phase 3: Loop Integration & Code Generation
Generate the `arif_run.py` script. 
1. **Requirements**: 
   - Use `AutoResearch` and `AIAgent`.
   - Setup `AIAgent` with `system_prompt`, `default_guard`, and `default_timeout`.
   - Use `ar.get_history(as_text=True)` to fetch failure context for the reasoning phase.
   - Use `ar.modify_and_run_loop` to handle the trial-and-error optimization cycle.
   - Explicitly list the Evaluator and its dependencies in `protected_files`.
2. **Drafting**: Show the code to the user and explain how the `modify_and_run_loop` abstracts the heavy lifting (metric extraction and retry feedback).

**STOP**: Ask for approval of the `arif_run.py` design.

## Phase 4: Safety & Isolation Verification (Pre-Flight)
Before declaring the project "Ready-to-Run", perform a final safety check:
1. **Isolation Test**: Simulate a single `enter_exp` call. Verify that a snapshot folder is created in `agent_workspaces/`.
2. **Guard Test**: Attempt a dummy modification to a protected file (e.g., the Evaluator) inside the context manager. Verify that `arif.Guard` detects or prevents the change.
3. **Write Test**: Verify the Agent can successfully modify the Target File within the snapshot.

## Final Handover
Once all tests pass, provide the user with:
- The verified Baseline Score.
- The command to start the loop: `python arif_run.py`.
- A summary of which files are protected and which are modifiable.

**Status**: "Environment Ready. Research Loop Validated."
