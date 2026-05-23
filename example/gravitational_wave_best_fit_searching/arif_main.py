import os
import re
import json
from arif import AutoResearch, AIAgent

def main():
    # --- Configuration ---
    # Global Baseline Control
    BASELINE_UNFINISHED_COUNT = 4.057084139683834
    
    # PBS walltime is 30m, so we wait up to 40m for safety
    AGENT_TIMEOUT = 600  # 10 minutes for the agent
    CMD_TIMEOUT = 9000   # 2.5 hours for the PBS job
    
    # Project Root and Protected Files
    log_name = "arif_LLM_response.log"
    ar = AutoResearch(
        project_root="./", 
        protected_files=["trusted_evaluator.py", "auto_run.sh", "run_all.pbs", "utils.py"],
        log_path=log_name
    )
    
    main_prompt = (
        "You are an expert in Gravitational Wave data analysis and optimization. "
        "Your task is to modify 'modular_optimization-Fisher.py' to minimize the Mismatch of ALL 25 signals. "
        "Each PBS experiment now has a 1-hour limit for the optimizer, so the script is designed to optimize ONE UNFINISHED INDEX at a time and then exit for evaluation. "
        f"The current goal is to finish the remaining {int(BASELINE_UNFINISHED_COUNT)} unfinished indexes (no-maximization mismatch > 0.1) by improving the 'physical phase' search. "
        "The 'trusted_evaluator.py' now returns the 'unfinished_count' as the primary metric. Your goal is to reduce this count to 0. "
        "Search Strategy Context:\n"
        "- Instead of a Fisher-based prior, we now use an empirical prior derived from Layer 1.\n"
        "- In Layer 1, we run multiple low-precision Nelder-Mead optimizations to find a cluster of candidate points.\n"
        "- The covariance of these points is calculated and magnified (10x on width, 100x on covariance) to define the search space for Layer 2.\n"
        "- This empirical ellipsoid defines the affine prior for the PARIS sampler in Layer 2.\n"
        "- NOTE: We abandoned the initial Fisher matrix because it is often UNRELIABLE for *MRI; it predicts identical narrow biases for m1 and m2 despite their vastly different magnitudes.\n"
        "- By default, we enable Time Maximization and Phase (PH) Maximization during the optimization search.\n"
        "- CRITICAL: The 'trusted_evaluator.py' checks the end product WITHOUT any maximization (No Time/Phase shift). "
        "Your optimized parameters must be accurate enough to match the signal directly.\n\n"
        "Constraints:\n"
        "1. Do not modify trusted_evaluator.py, utils.py, or the bash runners.\n"
        "2. Only modify 'modular_optimization-Fisher.py'.\n"
        "3. The goal is to reduce the total number of unfinished indices (mismatch > 0.1). \n"
        f"4. Current baseline unfinished count is {int(BASELINE_UNFINISHED_COUNT)}.\n"
        "5. TIME CONSTRAINT: You MUST keep 'num_iterations' for PARIS fixed at 2000. Do not increase it, or the PBS job will timeout.\n"
        "6. PARIS ALGORITHM CONTEXT: PARIS is a Parallel Adaptive Importance Sampler. It works as follows:\n"
        "   - It first initializes a set of points using Latin Hypercube Sampling (LHS) across the prior space.\n"
        "   - It then selects the top-performing LHS points as seeds to start parallel importance sampling processes.\n"
        "   - Each process adaptively updates its covariance matrix every 'gamma' iterations and uses a sliding window of 'alpha' samples for importance weighting.\n"
        "   - This allows it to climb local peaks and refine the posterior distribution.\n"
        "   - CRITICAL TECHNICAL NOTE: You MUST set 'use_pool=False' in the SamplerConfig. Using multiprocessing ('use_pool=True') causes a pickling error that crashes the experiment.\n"
        "7. LOADING PREVIOUS RESULTS: You can (and should) load previously optimized parameters from 'optimized_params_*MRI_0PA.npy' (or the 1PA version) as your starting point to continue refinement across experiments.\n"
        "8. RECOMMENDED STRATEGY: A successful workflow involves: 1. Loading previous best points. 2. Running a global search (PARIS) or local search (Nelder-Mead) WITH Time/Phase maximization to find the physical neighborhood. 3. Performing a final high-precision polish WITHOUT any maximization to lock the phase and satisfy the trusted_evaluator.\n"
        "9. THREE-LAYER OPTIMIZATION PIPELINE:  the first experiment (u may looking at no first experiment code but i told u first experiment anyway) use following structure and get mismatch (no max) 0.1169, u r free to tune and add and delete layers.:\n"
        "   - Layer 1 (Prior Generation): Run 10 low-precision Nelder-Mead optimizations WITH Time/Phase maximization. Initialize from the true point and 9 perturbed points. No formal prior transform is used here; it is a local search to find the physical cluster.\n"
        "   - Layer 2 (Global Refinement): Run PARIS WITHOUT maximization (2000 iterations) using the magnified covariance of Layer 1 results as the empirical prior. This identifies the 'true mountain'.\n"
        "   - Layer 3 (Precision Polish): Run Nelder-Mead WITHOUT maximization (500 calls) on the best result from Layer 2 to lock the phase for the evaluator.\n"
        "10. TUNING FLEXIBILITY: Except for iterations, you are encouraged to tune all other aspects: the magnification factor of the empirical prior ellipsoid, the seed cloud size (seed_cloud), the adaptation parameters (alpha, gamma), maximization logic/strategies, and Nelder-Mead polishing parameters.the numer of NM to get empirical prior ellipsoid and how u cal can modify also.\n"
        "11. Refer to the source code in 'parismc/' for algorithm details if needed.\n"
    )

    # Initialize Agent with Gemini
    agent = AIAgent(
        engine="gemini", 
        delay=60,
        system_prompt=main_prompt,
        default_guard=ar.guard,
        log_path=log_name
    )

    # Branch Management
    history = ar.get_history()
    if not history:
        B, L, S = ar.new_branch()
    else:
        B = history[-1]['B']
        B, L, S = ar.continue_branch(B)

    # Get best mismatch so far from history
    valid_metrics = [h['metric'] for h in history if 'metric' in h and h.get('metric') is not None]
    best_mismatch = min(valid_metrics) if valid_metrics else BASELINE_UNFINISHED_COUNT

    print(f"Starting Autonomous Research Branch {B}, starting at {L}.{S}")
    print(f"Best Mismatch so far: {best_mismatch}")

    # Research Loop (Run for 10 major levels)
    for _ in range(10):
        print(f"\n--- Starting Level {L} Step {S} ---")
        with ar.enter_exp(B, L, S):
            # 1. Hypothesis Generation
            history_text = ar.get_history(B=B, L=L, limit=3, as_text=True) or "No previous history in this level."

            print("Agent is proposing a hypothesis...")
            hypothesis = agent.ask(
                f"TECHNICAL HISTORY & LESSONS LEARNED:\n{history_text}\n\n"
                "Based on the history above, propose a specific code change to improve the optimizer performance. "
                "Ensure you don't repeat the same mistakes found in the failures. Do not output code yet.",
                new_session=True,
                timeout=AGENT_TIMEOUT
            )
            print(f"Hypothesis: {hypothesis[:200]}...")

            # 2. Implementation and Trial (High-level Modify-Run Loop)
            success, current_mismatch, stdout, stderr = ar.modify_and_run_loop(
                agent,
                modify_prompt=(
                    f"Hypothesis: {hypothesis}\n"
                    "Now modify 'modular_optimization-Fisher.py' to implement your plan.\n"
                    "CRITICAL GUIDELINES:\n"
                    "1. Prefer SURGICAL EDITS over complete file rewrites.\n"
                    "2. Do NOT delete comments, unrelated utility functions, or the 'if __name__ == \"__main__\":' block.\n"
                    "3. Ensure the 'argparse' section remains intact and functional.\n"
                    "4. Maintain consistent indentation and coding style."
                ),
                eval_cmd="bash -l -c './auto_run.sh'",
                metric_extract='"metric": ',
                best_metric=best_mismatch,
                max_trials=3,
                timeout=CMD_TIMEOUT
            )

            # 3. Summary
            print("Agent is summarizing the experiment...")
            summary = agent.ask(
                "Summarize the results of this experiment. Why did it " + ("succeed" if success else "fail") + "?"
            )

            ar.save_history(
                metric=current_mismatch, 
                if_improved=success,
                hypothesis=hypothesis,
                summary=summary
            )
            
            if success:
                best_mismatch = current_mismatch
                L += 1
                S = 1
            else:
                S += 1

    print("\n--- Research Cycle Complete ---")

if __name__ == "__main__":
    main()
