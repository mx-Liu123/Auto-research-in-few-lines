"""
AIAgent module for executing LLM CLI commands with guard protection.
"""

import os
import subprocess

from .llm_adapters import get_adapter


class AIAgent:
    """Wrapper for LLM CLI tools with session + guard hooks."""

    def __init__(self, engine="claude", model=None):
        self.engine = engine
        self.model = model or engine
        self.adapter = get_adapter(self.model)
        self.session_id = None

    def execute_safe(self, prompt, guard, new_session=False, timeout=None, model=None):
        if new_session:
            self.session_id = None

        # Determine target model and adapter
        target_model = model or self.model
        current_adapter = self.adapter
        if model and model != self.model:
            # If a different model is requested, check if we need a different adapter
            current_adapter = get_adapter(model)

        guard.before()

        cmd = current_adapter.build_command(
            prompt=prompt,
            session_id=self.session_id,
            model=target_model,
            yolo=True,
        )
        kwargs = current_adapter.get_run_kwargs(prompt, os.environ.copy())

        # Use Popen to stream output in real-time
        kwargs.pop("capture_output", None) # Popen doesn't use capture_output
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        # If input was in kwargs, we need to pass it to communicate
        stdin_input = kwargs.pop("input", None)

        print(f"[AIAgent] Executing {target_model}...")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin_input else None, **kwargs)

        stdout_chunks = []
        stderr_chunks = []

        import threading
        def stream_reader(pipe, chunks, prefix):
            for line in iter(pipe.readline, ""):
                if line:
                    print(f"[{prefix}] {line.strip()}")
                    chunks.append(line)

        t_out = threading.Thread(target=stream_reader, args=(proc.stdout, stdout_chunks, "LLM STDOUT"))
        t_err = threading.Thread(target=stream_reader, args=(proc.stderr, stderr_chunks, "LLM STDERR"))
        t_out.start()
        t_err.start()

        if stdin_input:
            proc.stdin.write(stdin_input)
            proc.stdin.close()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"\n[AIAgent ERROR] Execution timed out after {timeout} seconds.")
            raise RuntimeError(f"LLM Agent Execution Timed Out after {timeout}s")
        finally:
            t_out.join()
            t_err.join()

        full_stdout = "".join(stdout_chunks)
        # 增加错误检测逻辑
        parsed = current_adapter.parse_output(full_stdout, self.session_id)
        # 兼容旧版本 adapter 返回 (text, session_id) 或新版本返回 dict 的情况
        if isinstance(parsed, dict):
            response_text = parsed.get("text", "")
            self.session_id = parsed.get("session_id")

            if parsed.get("is_error") or proc.returncode != 0:
                error_msg = parsed.get("error_detail") or parsed.get("text") or "Process failed"
                print(f"\n[AIAgent ERROR] {error_msg}")
                # We don't necessarily raise here to allow the loop to try next trail
                # but for validation we'll keep it
                raise RuntimeError(f"LLM Agent Execution Failed: {error_msg}")

            response_text = parsed.get("text", "")
        else:
            # Fallback for simple (text, session_id) tuple
            response_text, self.session_id = parsed

        guard.after()
        return response_text
