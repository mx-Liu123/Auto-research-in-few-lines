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

    def get_history(self):
        history = []
        if not os.path.isdir(self.workspace_root):
            return history

        for b_dir in sorted(os.listdir(self.workspace_root)):
            if not b_dir.startswith("Branch"):
                continue
            b_path = os.path.join(self.workspace_root, b_dir)

            for e_dir in sorted(os.listdir(b_path)):
                exp_path = os.path.join(b_path, e_dir)
                h_path = os.path.join(exp_path, "history.json")
                if not os.path.isfile(h_path):
                    continue
                B, L, S = self._parse_exp_dirname(e_dir)
                if B is None:
                    continue
                try:
                    with open(h_path, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue

                data.setdefault("B", B)
                data.setdefault("L", L)
                data.setdefault("S", S)
                data.setdefault("exp_id", e_dir)
                history.append(data)
        return history

    def run_cmd(self, cmd):
        with open("stdout.log", "w") as f_out, open("stderr.log", "w") as f_err:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out_chunks, err_chunks = [], []
            while True:
                out = proc.stdout.readline()
                err = proc.stderr.readline()
                if out == "" and err == "" and proc.poll() is not None:
                    break
                if out:
                    f_out.write(out)
                    f_out.flush()
                    out_chunks.append(out)
                if err:
                    f_err.write(err)
                    f_err.flush()
                    err_chunks.append(err)

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
            except Exception:
                continue
            if_improved = h.get("if_improved")
            cand = (l2, s2, exp_path, if_improved)
            if latest is None or (cand[0], cand[1]) > (latest[0], latest[1]):
                latest = cand

        if latest is None:
            return None, None, None, None
        return latest[2], latest[0], latest[1], latest[3]
