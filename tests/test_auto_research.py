import os
import shutil
import unittest
from arif import AutoResearch

class TestAutoResearch(unittest.TestCase):
    def setUp(self):
        self.test_root = "test_project"
        os.makedirs(self.test_root, exist_ok=True)
        with open(os.path.join(self.test_root, "evaluator.py"), "w") as f:
            f.write("print('Score: 0.95')")

    def tearDown(self):
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)
        if os.path.exists("agent_workspaces"):
            shutil.rmtree("agent_workspaces")

    def test_new_branch(self):
        ar = AutoResearch(project_root=self.test_root)
        B, L, S = ar.new_branch()
        self.assertEqual(B, 1)
        self.assertEqual(L, 1)
        self.assertEqual(S, 1)
        self.assertTrue(os.path.exists(os.path.join(ar.workspace_root, "Branch1", "exp1.0.0")))

    def test_enter_exp(self):
        ar = AutoResearch(project_root=self.test_root)
        B, L, S = ar.new_branch()
        with ar.enter_exp(B, L, S):
            self.assertTrue(os.getcwd().endswith(f"exp{B}.{L}.{S}"))
            self.assertTrue(os.path.exists("evaluator.py"))

if __name__ == "__main__":
    unittest.main()
