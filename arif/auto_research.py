"""
AutoResearch core module for managing experiment branches and snapshots.
"""

import contextlib
import json
import os
import shutil
import subprocess

from .guard import Guard


class AutoResearch:
    def __init__(self, project_root="./", protected_files=None):
        self.project_root = os.path.abspath(project_root)
        self.protected_files = protected_files or []
        self.workspace_root = os.path.join(self.project_root, "agent_workspaces")
        self.current_exp_dir = None
        self.current_metadata = {}

        os.makedirs(self.workspace_root, exist_ok=True)
        self.guard = Guard(self)

    def new_branch(self):
        """Create Branch{B} and baseline exp{B}.0.0, return (B, 1, 1)."""
        B = max(self._get_existing_branches(), default=0) + 1
        branch_dir = os.path.join(self.workspace_root, f"Branch{B}")
        os.makedirs(branch_dir, exist_ok=True)

        baseline_dir = os.path.join(branch_dir, f"exp{B}.0.0")
        self._copy_project_to(baseline_dir)

        with open(os.path.join(baseline_dir, "history.json"), "w") as f:
            json.dump(
                {"B": B, "L": 0, "S": 0, "if_improved": True, "note": "Initial baseline"},
                f,
            )

        return B, 1, 1

    def continue_branch(self, B):
        """Continue Branch{B} and return next (B, L, S)."""
        branch_dir = os.path.join(self.workspace_root, f"Branch{B}")
        if not os.path.isdir(branch_dir):
            raise ValueError(f"Branch{B} does not exist. Use new_branch() to create it.")

        _, last_L, last_S, last_if = self._find_latest_attempt(B)
        if last_L is None:
            return B, 1, 1

        if last_if is True:
            return B, last_L + 1, 1
        return B, last_L, last_S + 1

    @contextlib.contextmanager
    def enter_exp(self, B, L, S):
        branch_dir = os.path.join(self.workspace_root, f"Branch{B}")
        exp_id = f"exp{B}.{L}.{S}"
        exp_dir = os.path.join(branch_dir, exp_id)

        print(f"\n>>> Entering Experiment: {exp_id}")
        
        base_dir = self._find_best_base(B)
        if not os.path.exists(exp_dir):
            shutil.copytree(base_dir, exp_dir)

        old_cwd = os.getcwd()
        os.chdir(exp_dir)
        self.current_exp_dir = exp_dir
        self.current_metadata = {"B": B, "L": L, "S": S, "exp_id": exp_id}
        try:
            yield self
        finally:
            os.chdir(old_cwd)
            self.current_exp_dir = None
            self.current_metadata = {}

    def get_history(self, B=None, L=None, S=None, if_improved=None, limit=None, as_text=False):
        """
        Get experiment history with optional filtering.
        ... (rest of docstring) ...
        If as_text is True, returns a formatted string of hypothesis/summary/metrics.
        """
        def _check(val, filter_val):
            if filter_val is None:
                return True
            if isinstance(filter_val, (list, tuple)) and len(filter_val) == 2:
                return filter_val[0] < val < filter_val[1]
            if isinstance(filter_val, (list, tuple)):
                return val in filter_val
            return val == filter_val

        history = []
        if not os.path.isdir(self.workspace_root):
            return "" if as_text else history

        # ... (rest of filtering logic) ...

        # Get and sort branches numerically
        b_dirs = []
        for d in os.listdir(self.workspace_root):
            if d.startswith("Branch"):
                try:
                    b_num = int(d.replace("Branch", ""))
                    if not _check(b_num, B):
                        continue
                    b_dirs.append((b_num, d))
                except ValueError:
                    pass
        b_dirs.sort()

        for _, b_dir in b_dirs:
            b_path = os.path.join(self.workspace_root, b_dir)

            # Get and sort experiments numerically
            e_dirs = []
            for d in os.listdir(b_path):
                curr_B, curr_L, curr_S = self._parse_exp_dirname(d)
                if curr_B is not None:
                    if not _check(curr_L, L) or not _check(curr_S, S):
                        continue
                    e_dirs.append(((curr_B, curr_L, curr_S), d))
            e_dirs.sort()

            for (curr_B, curr_L, curr_S), e_dir in e_dirs:
                exp_path = os.path.join(b_path, e_dir)
                h_path = os.path.join(exp_path, "history.json")
                if not os.path.isfile(h_path):
                    continue
                try:
                    with open(h_path, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue

                if if_improved is not None and data.get("if_improved") != if_improved:
                    continue

                # FORCE trust folder name metadata over JSON content
                data["B"] = curr_B
                data["L"] = curr_L
                data["S"] = curr_S
                data["exp_id"] = e_dir
                history.append(data)

        if limit is not None:
            history = history[-limit:]
            
        if as_text:
            text_blocks = []
            for h in history:
                block = f"[{h['exp_id']}] "
                if h.get('hypothesis'): block += f"Hypothesis: {h['hypothesis']} "
                if h.get('metric'): block += f"Metric: {h['metric']} "
                if h.get('summary'): block += f"Summary: {h['summary']} "
                if h.get('error'): block += f"Error: {h['error']} "
                text_blocks.append(block.strip())
            return "\n".join(text_blocks)
            
        return history

    def modify_and_run_loop(
        self,
        agent,
        modify_prompt,
        eval_cmd,
        metric_name="Loss",
        max_trials=3,
        best_metric=float("inf"),
        timeout=None,
        max_print_chars=IndexError,
    ):
        """
        Encapsulate the Modify -> Run -> Compare loop.
        Provides detailed feedback (stdout/stderr) to the agent on failure/retry.
        """
        import re

        trials_history = []
        current_metric = float("inf")
        if_improved = False
        final_stdout, final_stderr = "", ""

        for trial in range(1, max_trials + 1):
            prompt = modify_prompt
            if trial > 1:
                feedback = f"\n\n[RETRY FEEDBACK] Your previous attempt(s) failed. Here is the execution history:\n"
                for i, h in enumerate(trials_history):
                    feedback += f"--- Trial {i+1} ---\nSTDOUT:\n{h['stdout']}\nSTDERR:\n{h['stderr']}\n"
                feedback += "\nPlease analyze these results and try a different approach to improve the metric."
                prompt += feedback

            print(f"  Trial {trial}/{max_trials}: Modifying code...")
            # agent.ask will use its default_guard, default_timeout, and default_max_print_chars if not provided
            agent.ask(prompt, max_print_chars=max_print_chars)

            print(f"  Trial {trial}/{max_trials}: Running evaluation...")
            status, stdout, stderr = self.run_cmd(eval_cmd, timeout=timeout)
            
            # Extract metric (handling optional sign and decimals)
            # No assumptions: user must provide the exact prefix (including colons/spaces) in metric_name
            pattern = rf"{metric_name}([-+]?[0-9]*\.?[0-9]+)"
            match = re.search(pattern, stdout, re.IGNORECASE)
            current_metric = float(match.group(1)) if match else float("inf")
            
            trials_history.append({"stdout": stdout, "stderr": stderr, "metric": current_metric})
            final_stdout, final_stderr = stdout, stderr

            print(f"  Current {metric_name}: {current_metric:.4f} (Best: {best_metric:.4f})")

            if_improved = current_metric < best_metric
            if if_improved:
                print(f"  SUCCESS: Improved from {best_metric:.4f} to {current_metric:.4f}!")
                break
            else:
                print(f"  No improvement.")

        return if_improved, current_metric, final_stdout, final_stderr

    def run_cmd(self, cmd, timeout=None):
        with open("stdout.log", "w") as f_out, open("stderr.log", "w") as f_err:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out_chunks, err_chunks = [], []
            import time
            start_time = time.time()
            
            try:
                while True:
                    # Check timeout manually if provided
                    if timeout and (time.time() - start_time) > timeout:
                        proc.kill()
                        raise subprocess.TimeoutExpired(cmd, timeout)

                    # We use a small sleep to avoid busy waiting and allow pipes to fill
                    time.sleep(0.1)
                    
                    # Read available output
                    while True:
                        import select
                        # Use select for non-blocking check if possible, or just read line by line
                        # Simplified for cross-platform compatibility
                        out = proc.stdout.readline() if select.select([proc.stdout], [], [], 0)[0] else None
                        err = proc.stderr.readline() if select.select([proc.stderr], [], [], 0)[0] else None
                        
                        if out:
                            f_out.write(out); f_out.flush()
                            out_chunks.append(out)
                        if err:
                            f_err.write(err); f_err.flush()
                            err_chunks.append(err)
                        
                        if not out and not err:
                            break

                    if proc.poll() is not None:
                        break
            except subprocess.TimeoutExpired:
                proc.kill()
                err_msg = f"TIMEOUT_ERROR: Command timed out after {timeout}s"
                f_err.write(err_msg)
                return 124, "".join(out_chunks), "".join(err_chunks) + "\n" + err_msg
            except Exception as e:
                proc.kill()
                err_msg = f"Command failed: {str(e)}"
                f_err.write(err_msg)
                return 1, "".join(out_chunks), "".join(err_chunks) + "\n" + err_msg

        return proc.returncode, "".join(out_chunks), "".join(err_chunks)

    def save_history(self, **kwargs):
        if not self.current_exp_dir:
            raise RuntimeError("Not inside an experiment context (enter_exp).")

        h_path = os.path.join(self.current_exp_dir, "history.json")
        data = self.current_metadata.copy()
        data.update(kwargs)
        with open(h_path, "w") as f:
            json.dump(data, f, indent=4)

    def _copy_project_to(self, dst):
        if os.path.exists(dst):
            shutil.rmtree(dst)

        def ignore_func(directory, contents):
            return ["agent_workspaces", ".git", "__pycache__", ".ipynb_checkpoints"]

        shutil.copytree(self.project_root, dst, ignore=ignore_func)

    def _get_existing_branches(self):
        branches = []
        if not os.path.isdir(self.workspace_root):
            return branches
        for d in os.listdir(self.workspace_root):
            if not d.startswith("Branch"):
                continue
            try:
                branches.append(int(d.replace("Branch", "")))
            except ValueError:
                pass
        return branches

    def _parse_exp_dirname(self, name):
        if not name.startswith("exp"):
            return None, None, None
        parts = name[3:].split(".")
        if len(parts) != 3:
            return None, None, None
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return None, None, None

    def _find_best_base(self, B):
        """Pick latest exp with history.if_improved==True; else baseline expB.0.0."""
        branch_dir = os.path.join(self.workspace_root, f"Branch{B}")
        improved = []
        if os.path.isdir(branch_dir):
            for d in os.listdir(branch_dir):
                exp_path = os.path.join(branch_dir, d)
                h_path = os.path.join(exp_path, "history.json")
                if not os.path.isfile(h_path):
                    continue
                b2, l2, s2 = self._parse_exp_dirname(d)
                if b2 != B:
                    continue
                try:
                    with open(h_path, "r") as f:
                        h = json.load(f)
                    # VALIDATION: Ensure JSON content matches directory name (detect zombie)
                    if h.get("exp_id") != d:
                        continue
                except Exception:
                    continue
                if h.get("if_improved") is True:
                    improved.append((l2, s2, exp_path))

        if improved:
            improved.sort(key=lambda x: (x[0], x[1]))
            return improved[-1][2]

        baseline = os.path.join(branch_dir, f"exp{B}.0.0")
        if os.path.isdir(baseline):
            return baseline
        return self.project_root

    def _find_latest_attempt(self, B):
        """Return (path, L, S, if_improved) for latest attempt by (L,S)."""
        branch_dir = os.path.join(self.workspace_root, f"Branch{B}")
        latest = None
        if not os.path.isdir(branch_dir):
            return None, None, None, None

        for d in os.listdir(branch_dir):
            exp_path = os.path.join(branch_dir, d)
            h_path = os.path.join(exp_path, "history.json")
            if not os.path.isfile(h_path):
                continue
            b2, l2, s2 = self._parse_exp_dirname(d)
            if b2 != B:
                continue
            try:
                with open(h_path, "r") as f:
                    h = json.load(f)
                # VALIDATION: Ensure JSON content matches directory name (detect zombie)
                if h.get("exp_id") != d:
                    continue
            except Exception:
                continue
            if_improved = h.get("if_improved")
            cand = (l2, s2, exp_path, if_improved)
            if latest is None or (cand[0], cand[1]) > (latest[0], latest[1]):
                latest = cand

        if latest is None:
            return None, None, None, None
        return latest[2], latest[0], latest[1], latest[3]
