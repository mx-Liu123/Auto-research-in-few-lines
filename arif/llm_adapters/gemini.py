import json
from .base import BaseAdapter

class GeminiAdapter(BaseAdapter):
    def build_command(self, prompt, session_id, model, yolo, **kwargs):
        # cmd = ["gemini", "-o", "stream-json", "--skip-trust"]
        # 
        # # Explicitly use -p - to read prompt from stdin
        # cmd.extend(["-p", "-"])
        # cmd = ["agy", "--print"]
        cmd = ["agy", "-p", prompt]

        # if model and model != "gemini" and not model.startswith("auto-gemini"):
        #      cmd.extend(["-m", model])
        # elif model and model.startswith("auto-gemini"):
        #      cmd.extend(["-m", model])
        # if model and model != "gemini" and not model.startswith("auto-gemini"):
        #      cmd.extend(["--model", model])
        # elif model and model.startswith("auto-gemini"):
        #      cmd.extend(["--model", model])
        if model and model != "agy" and not model.startswith("auto-agy"):
             cmd.extend(["--model", model])
        elif model and model.startswith("auto-agy"):
             cmd.extend(["--model", model])
            
        # if session_id:
        #     if session_id == "AUTO_RESUME":
        #         cmd.extend(["--resume", "latest"])
        #     else:
        #         cmd.extend(["--resume", session_id])
        # if session_id:
        #     if session_id == "AUTO_RESUME":
        #         cmd.append("--continue")
        #     else:
        #         cmd.extend(["--conversation", session_id])
        if session_id:
            cmd.append("--continue")
                
        # if yolo:
        #     cmd.append("-y")
        if yolo:
            cmd.append("--dangerously-skip-permissions")
            
        return cmd

    def get_run_kwargs(self, prompt, env):
        # We explicitly set input to None since prompt is passed via command line argument -p
        return {
            "capture_output": True,
            "text": True,
            "env": env,
            "input": None
        }

    def parse_output(self, stdout, original_session_id):
        new_session_id = None
        full_response = ""
        is_json_stream = False
        
        for line in stdout.splitlines():
            line = line.strip()
            if not line: continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    continue
                is_json_stream = True
                event_type = event.get("type")
                
                if event_type == "init":
                    new_session_id = event.get("session_id")
                elif event_type == "message" and event.get("role") == "assistant":
                    content = event.get("content", "")
                    if content:
                        full_response += content
                elif event_type == "item.completed":
                    # Handle potential structured items
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        content = item.get("text") or item.get("message")
                        if content:
                            full_response += content
                    
            except json.JSONDecodeError:
                pass

        # if not new_session_id and original_session_id:
        #     new_session_id = original_session_id
        if not new_session_id:
            new_session_id = original_session_id or "CONTINUE"

        # if not full_response.strip():
        #     full_response = stdout
        if not is_json_stream or not full_response.strip():
            full_response = stdout

        return full_response.strip(), new_session_id
