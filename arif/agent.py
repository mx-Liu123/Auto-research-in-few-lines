"""
AIAgent module for executing LLM CLI commands with guard protection.
"""

import os
import subprocess
import time

from .llm_adapters import get_adapter


class AIAgent:
    """Wrapper for LLM CLI tools with session + guard hooks."""

    def __init__(self, engine: str = "claude", model: str | None = None, delay: int = 0, system_prompt: str = "", default_guard: any = None, default_timeout: int | None = None, log_path: str | None = None):
        self.engine = engine
        self.model = model or engine
        self.delay = delay
        self.system_prompt = system_prompt
        self.default_guard = default_guard
        self.default_timeout = default_timeout
        self.log_path = log_path
        self.adapter = get_adapter(self.model)
        self.session_id = None

    def _log(self, message):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def ask(self, prompt: str, guard: any = IndexError, timeout: int | None = IndexError, new_session: bool = False, model: str | None = None, **kwargs) -> str:
        """
        Execute an LLM command. 
        - guard: If IndexError (default), use default_guard. If None, skip guard.
        - timeout: If IndexError (default), use default_timeout.
        """
        # Resolve defaults: local override > global default
        effective_guard = self.default_guard if guard is IndexError else guard
        effective_timeout = self.default_timeout if timeout is IndexError else timeout

        max_retries = 3
        # Prepend system_prompt if set
        full_prompt = self.system_prompt + "\n" + prompt if self.system_prompt else prompt

        for attempt in range(max_retries):
            try:
                if self.delay > 0:
                    print(f"[AIAgent] Waiting for {self.delay}s to prevent rate limiting...")
                    time.sleep(self.delay)

                if new_session:
                    self.session_id = None

                # Determine target model and adapter
                target_model = model or self.model
                current_adapter = self.adapter
                if model and model != self.model:
                    current_adapter = get_adapter(model)

                if effective_guard:
                    effective_guard.before()

                cmd = current_adapter.build_command(
                    prompt=full_prompt,
                    session_id=self.session_id,
                    model=target_model,
                    yolo=True,
                    **kwargs
                )
                run_kwargs = current_adapter.get_run_kwargs(full_prompt, os.environ.copy())

                # Use Popen to stream output in real-time
                run_kwargs.pop("capture_output", None)
                run_kwargs["stdout"] = subprocess.PIPE
                run_kwargs["stderr"] = subprocess.PIPE
                stdin_input = run_kwargs.pop("input", None)

                msg = f"[AIAgent] Executing {target_model}..."
                print(msg)
                self._log(f"\n{msg}\nPROMPT: {full_prompt}")

                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin_input else None, **run_kwargs)

                stdout_chunks = []
                stderr_chunks = []

                import threading
                def stream_reader(pipe, chunks, prefix):
                    for line in iter(pipe.readline, ""):
                        if line:
                            # Log every line, but don't print to terminal
                            self._log(f"[{prefix}] {line.strip()}")
                            chunks.append(line)

                t_out = threading.Thread(target=stream_reader, args=(proc.stdout, stdout_chunks, "LLM STDOUT"))
                t_err = threading.Thread(target=stream_reader, args=(proc.stderr, stderr_chunks, "LLM STDERR"))
                t_out.start()
                t_err.start()

                if stdin_input:
                    proc.stdin.write(stdin_input)
                    proc.stdin.close()

                try:
                    proc.wait(timeout=effective_timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    err_msg = f"[AIAgent WARNING] Execution timed out after {effective_timeout} seconds."
                    print(err_msg)
                    self._log(err_msg)
                    return f"TIMEOUT_ERROR: Agent execution timed out after {effective_timeout}s"
                finally:
                    t_out.join()
                    t_err.join()

                full_stdout = "".join(stdout_chunks)
                full_stderr = "".join(stderr_chunks)

                # Check for session resume errors
                if proc.returncode != 0:
                    err_low = full_stderr.lower()
                    if "invalid session" in err_low or "error resuming" in err_low or "searched for sessions" in err_low:
                        print(f"[AIAgent] Session {self.session_id} not found. Falling back to latest session...")
                        self.session_id = "AUTO_RESUME"
                        if effective_guard: effective_guard.after()
                        continue

                parsed = current_adapter.parse_output(full_stdout, self.session_id)
                
                if isinstance(parsed, dict):
                    response_text = parsed.get("text", "")
                    self.session_id = parsed.get("session_id")
                    
                    self._log(f"[AIAgent Response]:\n{response_text}")

                    if parsed.get("is_error") or proc.returncode != 0:
                        error_msg = (parsed.get("error_detail") or parsed.get("text") or "Process failed").replace("\n", " ")
                        
                        # Terminal gets a clean single-line error
                        print(f"[AIAgent ERROR] {error_msg}")
                        
                        if attempt < max_retries - 1 and ("429" in error_msg or "capacity" in error_msg.lower()):
                            wait_time = (attempt + 1) * 60
                            print(f"[AIAgent RETRY] API congestion. Waiting {wait_time}s...")
                            time.sleep(wait_time)
                            if effective_guard: effective_guard.after()
                            continue
                        raise RuntimeError(f"LLM Agent Execution Failed: {error_msg}")

                    response_text = parsed.get("text", "")
                else:
                    response_text, self.session_id = parsed
                    self._log(f"[AIAgent Response]:\n{response_text}")

                if effective_guard:
                    effective_guard.after()
                return response_text

            except Exception as e:
                if attempt < max_retries - 1 and ("429" in str(e) or "capacity" in str(e).lower()):
                    wait_time = (attempt + 1) * 60
                    print(f"\n[AIAgent RETRY] Exception ({e}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise e
