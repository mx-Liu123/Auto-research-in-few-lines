import json
from .base import BaseAdapter

class GeminiAdapter(BaseAdapter):
    def build_command(self, prompt, session_id, model, yolo, **kwargs):
        cmd = ["gemini", "-o", "stream-json", "--skip-trust"]
        
        # Explicitly use -p - to read prompt from stdin
        cmd.extend(["-p", "-"])

        if model and model != "gemini" and not model.startswith("auto-gemini"):
             cmd.extend(["-m", model])
        elif model and model.startswith("auto-gemini"):
             cmd.extend(["-m", model])
            
        if session_id:
            if session_id == "AUTO_RESUME":
                cmd.extend(["--resume", "latest"])
            else:
                cmd.extend(["--resume", session_id])
                
        if yolo:
            cmd.append("-y")
            
        return cmd

    def parse_output(self, stdout, original_session_id):
        new_session_id = None
        full_response = ""
        
        for line in stdout.splitlines():
            line = line.strip()
            if not line: continue
            try:
                event = json.loads(line)
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

        if not new_session_id and original_session_id:
            new_session_id = original_session_id

        return full_response.strip(), new_session_id
