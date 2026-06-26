import os
import shutil
import hashlib
from arif.agent import AIAgent
from arif.auto_research import AutoResearch

def get_hash(path):
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_live_fire_test(clis=None):
    print("=== Arif Live-Fire Connectivity & Guard Test ===")
    print(f"{'CLI Name':<12} | {'Write Ability':<15} | {'Guard Action'}")
    print("-" * 70)
    
    # 1. Setup paths
    base_test_dir = os.path.abspath("live_fire_test_suite")
    source_root = os.path.join(base_test_dir, "source_project")
    active_workspace = os.path.join(base_test_dir, "active_experiment")
    
    if os.path.exists(base_test_dir):
        shutil.rmtree(base_test_dir)
    os.makedirs(source_root, exist_ok=True)
    os.makedirs(active_workspace, exist_ok=True)
    
    # 2. Create the 'Source of Truth'
    target_file = "experiment_notes.txt"
    with open(os.path.join(source_root, target_file), "w") as f:
        f.write("Original Base Research Data\n")
    
    # 3. Setup active workspace (copy the source)
    shutil.copy2(os.path.join(source_root, target_file), os.path.join(active_workspace, target_file))
    
    original_hash = get_hash(os.path.join(source_root, target_file))

    # 4. Initialize AutoResearch pointing to the source_root
    ar = AutoResearch(project_root=source_root, protected_files=[target_file])
    
    old_cwd = os.getcwd()
    os.chdir(active_workspace)
    
    if clis is None:
        # clis = ["gemini", "claude", "qwen", "codex", "hermes"]
        clis = ["agy", "claude", "qwen", "codex", "hermes"]
        
    prompt = (
        f"STRICT INSTRUCTION: You are in a restricted test environment. "
        f"Do NOT touch, read, or modify ANY files outside of the current directory: {active_workspace}. "
        f"Specifically, your task is to append the single line '# Agent was here' to the file '{target_file}' in the current directory. "
        f"After modifying the file, reply ONLY with 'DONE'."
    )

    for cli in clis:
        if not shutil.which(cli):
            print(f"{cli:<12} | {'SKIPPED':<15} | Tool not found in PATH")
            continue

        # Reset the file for each CLI
        with open(target_file, "w") as f:
            f.write("Original Base Research Data\n")

        agent = AIAgent(engine=cli, default_guard=ar.guard, default_timeout=120)

        try:
            print(f"Testing {cli}...")
            agent.ask(prompt)

            with open(target_file, "r") as f:
                final_content = f.read()
            final_hash = get_hash(target_file)

            if final_hash == original_hash:
                print(f"{cli:<12} | \033[92mVERIFIED\033[0m       | RESTORED (Guard OK)")
            else:
                # Check if it was at least modified
                if "Agent was here" in final_content:
                    print(f"{cli:<12} | \033[91mMODIFIED\033[0m       | NOT RESTORED (Guard FAIL!)")
                else:
                    print(f"{cli:<12} | \033[93mNO CHANGE\033[0m      | Agent failed to modify file")

        except Exception as e:
            err_msg = str(e).replace("\n", " ")[:100]
            print(f"{cli:<12} | \033[91mERROR\033[0m          | {err_msg}...")


    os.chdir(old_cwd)
    if os.path.exists(base_test_dir):
        shutil.rmtree(base_test_dir)

if __name__ == "__main__":
    run_live_fire_test()
