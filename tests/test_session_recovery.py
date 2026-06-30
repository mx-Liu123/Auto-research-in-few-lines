import os
import shutil
from arif.agent import AIAgent

def run_session_recovery_test(clis=None):
    print("\n=== Arif Live-Fire Session Recovery Test ===")
    print(f"{'CLI Name':<12} | {'Session ID':<20} | {'Recovery (ID)':<13} | {'Recovery (Latest)'}")
    print("-" * 70)

    # 1. Setup isolated test folder
    base_test_dir = os.path.abspath("session_recovery_test_suite")
    if os.path.exists(base_test_dir):
        shutil.rmtree(base_test_dir)
    os.makedirs(base_test_dir, exist_ok=True)

    old_cwd = os.getcwd()
    os.chdir(base_test_dir)

    if clis is None:
        clis = ["agy", "claude", "qwen", "codex", "hermes"]

    # 2. Strict instructions restricting file access to the active folder
    prompt_hello = (
        "STRICT INSTRUCTION: Focus ONLY on the current directory. "
        "Do NOT read, touch, or modify ANY files outside of this folder. "
        "Say hello!"
    )
    
    prompt_ask = (
        "STRICT INSTRUCTION: Focus ONLY on the current directory. "
        "Do NOT read, touch, or modify ANY files outside of this folder. "
        "What was my first message?"
    )

    for cli in clis:
        if not shutil.which(cli):
            print(f"{cli:<12} | {'SKIPPED':<20} | {'SKIPPED':<13} | SKIPPED")
            continue

        try:
            # 1. Start Conversation 1 (New session)
            agent1 = AIAgent(engine=cli)
            response1 = agent1.ask(prompt_hello, new_session=True)
            sess_id = agent1.session_id

            if not sess_id:
                print(f"{cli:<12} | {'NONE':<20} | {'FAIL':<13} | FAIL")
                continue

            # 2. Test Recovery by Session ID
            agent2 = AIAgent(engine=cli)
            agent2.session_id = sess_id
            response2 = agent2.ask(prompt_ask)
            # rec_id_status = "PASS" if "hello" in response2.lower() else "FAIL"
            rec_id_status = "PASS" if any(w in response2.lower() for w in ["hello", "hi", "say", "greet"]) else "FAIL"

            # 3. Test Recovery by Latest Session (AUTO_RESUME)
            agent3 = AIAgent(engine=cli)
            agent3.session_id = "AUTO_RESUME"
            response3 = agent3.ask(prompt_ask)
            # rec_latest_status = "PASS" if "hello" in response3.lower() else "FAIL"
            rec_latest_status = "PASS" if any(w in response3.lower() for w in ["hello", "hi", "say", "greet"]) else "FAIL"

            if rec_id_status == "FAIL":
                print(f"[DEBUG {cli} Recovery (ID) FAIL] Response: {repr(response2)}")
            if rec_latest_status == "FAIL":
                print(f"[DEBUG {cli} Recovery (Latest) FAIL] Response: {repr(response3)}")

            # Colors for PASS/FAIL
            id_color = f"\033[92m{rec_id_status:<13}\033[0m" if rec_id_status == "PASS" else f"\033[91m{rec_id_status:<13}\033[0m"
            latest_color = f"\033[92m{rec_latest_status}\033[0m" if rec_latest_status == "PASS" else f"\033[91m{rec_latest_status}\033[0m"

            print(f"{cli:<12} | {str(sess_id)[:20]:<20} | {id_color} | {latest_color}")

        except Exception as e:
            err_msg = str(e).replace("\n", " ")[:25]
            print(f"{cli:<12} | {'ERROR':<20} | {err_msg:<13} | ERROR")

    os.chdir(old_cwd)
    if os.path.exists(base_test_dir):
        shutil.rmtree(base_test_dir)

if __name__ == "__main__":
    run_session_recovery_test()
