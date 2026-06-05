import unittest
from unittest.mock import patch, MagicMock
import os
import io
import subprocess
from arif.agent import AIAgent

class TestAIAgent(unittest.TestCase):
    def setUp(self):
        self.log_path = "test_agent.log"
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def tearDown(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    @patch("subprocess.Popen")
    def test_ask_basic_success(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = io.StringIO('{"type": "init", "session_id": "sess_123"}\n{"type": "message", "role": "assistant", "content": "Hello world"}')
        mock_proc.stderr = io.StringIO('')
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        agent = AIAgent(engine="gemini", log_path=self.log_path)
        response = agent.ask("Hi")

        self.assertEqual(response, "Hello world")
        self.assertEqual(agent.session_id, "sess_123")

    @patch("subprocess.Popen")
    def test_ask_retry_on_congestion(self, mock_popen):
        # 1st call fails with 429, 2nd succeeds
        mock_proc_fail = MagicMock()
        mock_proc_fail.returncode = 1
        mock_proc_fail.stdout = io.StringIO('')
        mock_proc_fail.stderr = io.StringIO('Error: 429 Too Many Requests')
        mock_proc_fail.wait.return_value = None

        mock_proc_ok = MagicMock()
        mock_proc_ok.returncode = 0
        mock_proc_ok.stdout = io.StringIO('{"type": "message", "role": "assistant", "content": "Success"}')
        mock_proc_ok.stderr = io.StringIO('')
        mock_proc_ok.wait.return_value = None

        mock_popen.side_effect = [mock_proc_fail, mock_proc_ok]

        agent = AIAgent(engine="gemini")
        with patch("arif.agent.time.sleep") as mock_sleep:
            response = agent.ask("Hi")
        
        self.assertEqual(response, "Success")
        self.assertEqual(mock_popen.call_count, 2)
        mock_sleep.assert_called()

    @patch("subprocess.Popen")
    def test_timeout_handling(self, mock_popen):
        mock_proc = MagicMock()
        # CRITICAL: Must provide EOF-able streams to prevent infinite loop in reader threads
        mock_proc.stdout = io.StringIO('')
        mock_proc.stderr = io.StringIO('')
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=0.1)
        mock_popen.return_value = mock_proc

        agent = AIAgent(engine="gemini", default_timeout=0.1)
        response = agent.ask("Hi")
        
        self.assertTrue("TIMEOUT_ERROR" in response)

if __name__ == "__main__":
    unittest.main()
