import json
from .base import BaseAdapter

class ClaudeAdapter(BaseAdapter):
    def build_command(self, prompt, session_id, model, yolo):
        # Claude CLI construction
        # Use --output-format json for structured results
        cmd = ["claude", "-p", "--output-format", "json"]
        
        cmd.append("--dangerously-skip-permissions")
        
        if session_id and session_id != "AUTO_RESUME":
             cmd.extend(["-r", session_id])
             
        # If the user passed a specific model (not just "claude"), pass it via --model
        if model and model not in ["claude", "claude-cli"]:
             cmd.extend(["--model", model])
             
        return cmd

    def parse_output(self, stdout, original_session_id):
        # Claude CLI in JSON mode may return multiple JSON objects 
        # either concatenated or separated by newlines.
        texts = []
        session_id = original_session_id
        is_error = False
        error_detail = None

        raw_stdout = stdout.strip()
        if not raw_stdout:
            return {"text": "", "session_id": original_session_id, "is_error": False, "error_detail": None}

        # Robust JSON stream parsing
        objs = []
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(raw_stdout):
            # Skip whitespace/junk between objects
            while pos < len(raw_stdout) and raw_stdout[pos] not in '{[':
                pos += 1
            if pos >= len(raw_stdout):
                break
            try:
                obj, json_len = decoder.raw_decode(raw_stdout[pos:])
                objs.append(obj)
                pos += json_len
            except json.JSONDecodeError:
                # If we hit a decode error, try to skip to the next possible object
                pos += 1

        # If no valid JSON objects found, fallback to raw text
        if not objs:
            return {"text": raw_stdout, "session_id": original_session_id, "is_error": False, "error_detail": None}

        for data in objs:
            if not isinstance(data, dict):
                continue
                
            if "session_id" in data:
                session_id = data["session_id"]
            
            msg_type = data.get("type")
            if msg_type == "result":
                res = data.get("result", "")
                if res: texts.append(res)
                # Success in result means the TASK was completed, but is_error might refer to CLI error
                is_error = data.get("is_error", False)
                error_detail = data.get("error_detail")
            elif msg_type == "text":
                content = data.get("content", "")
                if content: texts.append(content)
            elif data.get("subtype") == "success":
                 res = data.get("result", "")
                 if res: texts.append(res)
            elif msg_type == "error":
                 is_error = True
                 error_detail = data.get("message") or data.get("content")

        # Join all captured text chunks
        full_text = "\n\n".join(texts)

        return {
            "text": full_text,
            "session_id": session_id,
            "is_error": is_error,
            "error_detail": error_detail
        }
