import unittest
import os
import shutil
from arif.auto_research import AutoResearch

class TestGuard(unittest.TestCase):
    def setUp(self):
        # Use an absolute path for the test project root to avoid confusion when chdir happens
        self.test_root = os.path.abspath("test_guard_project")
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)
        os.makedirs(self.test_root, exist_ok=True)
        
        self.protected_file = os.path.join(self.test_root, "important.txt")
        with open(self.protected_file, "w") as f:
            f.write("original content")
        
        self.old_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.old_cwd)
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)
        if os.path.exists("agent_workspaces"):
            shutil.rmtree("agent_workspaces")

    def test_file_protection_and_restoration(self):
        # Change to a subdirectory to simulate an experiment folder
        exp_dir = os.path.join(self.test_root, "exp1.1.1")
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy2(os.path.join(self.test_root, "important.txt"), os.path.join(exp_dir, "important.txt"))
        
        os.chdir(exp_dir)
        
        ar = AutoResearch(project_root=self.test_root, protected_files=["important.txt"])
        ar.guard.before()
        
        # Tamper
        with open("important.txt", "w") as f:
            f.write("TAMPERED")
            
        ar.guard.after()
        
        with open("important.txt", "r") as f:
            self.assertEqual(f.read(), "original content")

    def test_directory_protection(self):
        secret_dir = os.path.join(self.test_root, "secret_dir")
        os.makedirs(secret_dir, exist_ok=True)
        with open(os.path.join(secret_dir, "data.bin"), "w") as f:
            f.write("12345")
            
        exp_dir = os.path.join(self.test_root, "exp1.1.2")
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copytree(secret_dir, os.path.join(exp_dir, "secret_dir"))
        
        os.chdir(exp_dir)
        
        ar = AutoResearch(project_root=self.test_root, protected_files=["secret_dir"])
        ar.guard.before()
        
        # Tamper
        with open("secret_dir/data.bin", "w") as f:
            f.write("MODIFIED")
            
        ar.guard.after()
        
        with open("secret_dir/data.bin", "r") as f:
            self.assertEqual(f.read(), "12345")

if __name__ == "__main__":
    unittest.main()
