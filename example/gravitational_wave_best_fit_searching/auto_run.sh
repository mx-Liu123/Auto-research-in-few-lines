#!/bin/bash

# Function to submit and watch a job
run_task() {
    local PBS_SCRIPT=$1
    local LOG_PREFIX=$2
    local EXTRA_ARGS=$3
    
    echo "========================================================"
    echo "🚀 Submitting $PBS_SCRIPT..."
    
    # Pass the SAVE_LOCAL flag if it exists
    QSUB_OUTPUT=$(qsub $EXTRA_ARGS $PBS_SCRIPT)
    if [ $? -ne 0 ]; then
        echo "❌ Submit failed: $QSUB_OUTPUT"
        return 1
    fi
    
    JOB_ID=$(echo "$QSUB_OUTPUT" | sed 's/\..*//' | sed 's/[^0-9]*//g')
    echo "✅ Job ID: $JOB_ID"
    
    # Wait for output files to appear
    echo "⏳ Waiting for job start..."
    STDOUT_FILE="${LOG_PREFIX}.${JOB_ID}" # Modified to match PBS default or captured output
    STDERR_FILE="${LOG_PREFIX/stdout/stderr}.${JOB_ID}"
    
    # In some PBS setups, stdout doesn't appear until completion or periodic flush.
    # But since run_all.pbs redirects inside singularity, we check for those files.
    # Actually, let's use the PBS default names if we don't redirect in the script.
    # In run_all.pbs I didn't redirect stdout/stderr to specific files, so they will be $PBS_JOBNAME.o$JOB_ID
    
    STDOUT_FILE=$(ls *.o${JOB_ID} 2>/dev/null | head -n 1)
    
    while true; do
        STDOUT_FILE=$(ls *.o${JOB_ID} 2>/dev/null | head -n 1)
        if [ -n "$STDOUT_FILE" ] && [ -e "$STDOUT_FILE" ]; then break; fi
        if ! qstat $JOB_ID >/dev/null 2>&1; then 
            echo "⚠️ Job finished/died before creating stdout."
            sleep 2
            STDOUT_FILE=$(ls *.o${JOB_ID} 2>/dev/null | head -n 1)
            if [ -n "$STDOUT_FILE" ] && [ -e "$STDOUT_FILE" ]; then break; fi
            return 1
        fi
        sleep 5
    done
    
    echo "📺 Streaming Output ($STDOUT_FILE)..."
    tail -f $STDOUT_FILE &
    TAIL_PID=$!
    
    # Wait for finish
    while qstat $JOB_ID >/dev/null 2>&1; do
        sleep 10
    done
    
    kill $TAIL_PID 2>/dev/null
    wait $TAIL_PID 2>/dev/null
    
    echo "✅ Job $JOB_ID finished."
    
    # Find stderr file too
    STDERR_FILE=$(ls *.e${JOB_ID} 2>/dev/null | head -n 1)
    
    # Print logs
    echo "--- FULL STDOUT ---"
    cat $STDOUT_FILE
    echo "-------------------"
    if [ -s "$STDERR_FILE" ]; then
        echo "--- FULL STDERR ---"
        cat $STDERR_FILE
        echo "-------------------"
    fi
    
    if grep -q "Traceback" $STDERR_FILE; then
        echo "❌ Job seemed to fail (Traceback found)."
        return 1
    fi
    
    return 0
}

# --- Main Flow ---
run_task "run_all.pbs" "stdout" "" ""

if [ $? -ne 0 ]; then
    echo "⛔ Task failed. Aborting."
    exit 1
fi

echo "🎉 All tasks completed successfully!"
