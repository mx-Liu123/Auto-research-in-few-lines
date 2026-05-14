"""
Guard module for protecting files from modification.
"""

import os
import shutil
import hashlib


class Guard:
    """
    Protects specified files from modification by AI agents.

    Before AI operations, calculates hashes of protected files.
    After operations, verifies hashes and restores from project root if modified.
    """

    def __init__(self, parent):
        self.parent = parent
        self.hashes = {}

    def _calculate_hashes(self):
        """Calculate hashes for all protected files in current working directory."""
        import glob
        hashes = {}
        
        # Internal arif ignore list to prevent self-protection of workspaces or heavy git folders
        ignores = {"agent_workspaces", ".git", "__pycache__", ".ipynb_checkpoints"}
        if self.parent.log_path:
            ignores.add(os.path.basename(self.parent.log_path))
        
        expanded_files = []
        for f in self.parent.protected_files:
            if "*" in f:
                # Expand glob pattern
                matches = glob.glob(f)
                expanded_files.extend([m for m in matches if m not in ignores])
            else:
                expanded_files.append(f)

        for f in set(expanded_files):
            path = os.path.join(os.getcwd(), f)
            if os.path.exists(path):
                if os.path.isfile(path):
                    hashes[f] = self._file_hash(path)
                elif os.path.isdir(path):
                    hashes[f] = self._dir_hash(path)
        return hashes

    def _file_hash(self, path):
        """Calculate MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _dir_hash(self, path):
        """Calculate MD5 hash of a directory (recursive)."""
        hasher = hashlib.md5()
        for root, dirs, files in os.walk(path):
            for f in sorted(files):
                f_path = os.path.join(root, f)
                hasher.update(f.encode())
                hasher.update(self._file_hash(f_path).encode())
        return hasher.hexdigest()

    def before(self):
        """Hook called before AI operation. Snapshots current file hashes."""
        self.hashes = self._calculate_hashes()

    def after(self):
        """Hook called after AI operation. Restores modified protected files."""
        current_hashes = self._calculate_hashes()
        for f, h in self.hashes.items():
            if current_hashes.get(f) != h:
                print(f"[Guard] Warning: {f} was modified! Restoring from project root.")
                src = os.path.join(self.parent.project_root, f)
                dst = os.path.join(os.getcwd(), f)

                if os.path.exists(dst):
                    if os.path.isfile(dst):
                        os.remove(dst)
                    else:
                        shutil.rmtree(dst)

                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    shutil.copytree(src, dst)
