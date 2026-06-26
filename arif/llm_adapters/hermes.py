import re
from .base import BaseAdapter

class HermesAdapter(BaseAdapter):
    def build_command(self, prompt, session_id, model, yolo, **kwargs):
        # hermes chat -q "PROMPT" -Q --yolo [--resume SESSION_ID / --continue] [--model MODEL]
        cmd = ["hermes", "chat", "-q", prompt, "-Q"]
        
        if yolo:
            cmd.append("--yolo")
            
        cmd.append("--pass-session-id")
            
        if session_id:
            if session_id == "AUTO_RESUME":
                # State 3: Fallback to resume the latest session
                cmd.append("--continue")
            else:
                # State 2: Resume specific session ID
                cmd.extend(["--resume", str(session_id)])
        # State 1: New session (session_id is None)
            
        if model and model not in ["hermes", "hermes-agent"]:
            cmd.extend(["--model", model])
            
        return cmd

    def parse_output(self, stdout, original_session_id):
        """
        Hermes -Q output format:
        ↻ Resumed session ... (optional banner)
        [Response Text]
        
        session_id: [ID]
        """
        lines = stdout.strip().splitlines()
        if not lines:
            return {"text": "", "session_id": original_session_id, "is_error": False}

        session_id = original_session_id
        text_lines = []
        
        for line in lines:
            # Look for "session_id: XXX" at the end to capture the ID
            match = re.search(r"session_id:\s*(\S+)", line, re.IGNORECASE)
            if match:
                session_id = match.group(1)
            else:
                # Filter out the decorative resume banner to keep the text clean
                if not line.startswith("↻ Resumed session"):
                    text_lines.append(line)

        full_text = "\n".join(text_lines).strip()
        
        return {
            "text": full_text,
            "session_id": session_id,
            "is_error": False,
            "error_detail": None
        }

    def get_run_kwargs(self, prompt, env):
        # hermes chat -q passes the prompt as a shell argument.
        # We explicitly set input to None to avoid double-feeding if the CLI tries to read stdin.
        return {
            "capture_output": True,
            "text": True,
            "env": env,
            "input": None 
        }
