from .base import BaseAdapter

class OpenCodeAdapter(BaseAdapter):
    def build_command(self, prompt, session_id, model, yolo, **kwargs):
        # opencode run -m model --format json
        cmd = ["opencode", "run", "--format", "json"]
        
        if model:
            cmd.extend(["-m", model])
            
        if session_id:
            if session_id == "AUTO_RESUME":
                cmd.append("-c")
            else:
                cmd.extend(["-s", session_id])
            
        return cmd

    def parse_output(self, stdout, original_session_id):
        texts = []
        session_id = original_session_id

        # OpenCode JSON output is JSONL
        for line in stdout.splitlines():
            line = line.strip()
            if not line: continue
            try:
                event = json.loads(line)
                
                # Capture Session ID
                if event.get("session_id"):
                    session_id = event["session_id"]
                
                # Extract text
                if event.get("type") == "message" and event.get("role") == "assistant":
                    content = event.get("content", "")
                    if content: texts.append(content)
                elif event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        content = item.get("text") or item.get("message")
                        if content: texts.append(content)

            except json.JSONDecodeError:
                pass

        if texts:
            return "\n\n".join(texts).strip(), session_id
        
        # Fallback to raw text if no JSON parsing worked
        if not stdout.strip().startswith("{"):
             return stdout.strip(), session_id
        return "", session_id
