import json

from .base import BaseAdapter


class CodexAdapter(BaseAdapter):
    def build_command(self, prompt, session_id, model, yolo, **kwargs):
        if session_id:
            # Resume existing session
            cmd = [
                "codex",
                "exec",
                "resume",
            ]
            if session_id == "AUTO_RESUME":
                cmd.append("--last")
            else:
                cmd.append(session_id)
            
            cmd.extend([
                "--json",
                "--skip-git-repo-check",
            ])
        else:
            # New session
            cmd = [
                "codex",
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write" if yolo else "read-only",
            ]

        if model and model != "codex":
            cmd.extend(["--model", model])

        cmd.append("-")
        return cmd

    def parse_output(self, stdout, original_session_id):
        texts = []
        session_id = original_session_id
        non_json_lines = []

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                non_json_lines.append(line)
                continue

            # Capture Session/Thread ID
            tid = event.get("thread_id") or event.get("session_id")
            if tid:
                session_id = tid

            # Extract Agent Messages
            # Modern Format: type='item.completed', item.type='agent_message'
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    t = item.get("text") or item.get("message")
                    if t: texts.append(t)
            
            # Legacy/Alternate Format: type='agent_message'
            elif event.get("type") in {"agent_message", "final_answer"}:
                t = event.get("message") or event.get("text")
                if t: texts.append(t)

            # Nested format: message.type='agent_message'
            msg = event.get("msg") or event.get("message")
            if isinstance(msg, dict) and msg.get("type") in {"agent_message", "final_answer"}:
                t = msg.get("message") or msg.get("text")
                if t: texts.append(t)

        if texts:
            return "\n\n".join(texts).strip(), session_id
        
        # If no explicit agent messages found, return non-json lines or raw stdout
        if non_json_lines:
            return "\n".join(non_json_lines).strip(), session_id
            
        # Last resort: if everything was JSON but no message found, return empty or raw
        # (Usually means the agent didn't say anything or it's an error)
        return "", session_id
