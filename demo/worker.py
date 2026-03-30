"""
Demo worker script for SLURM parallel job test.
Each job runs this script with a unique SLURM_ARRAY_TASK_ID.
"""
import os
import time
import socket
import math

job_id = os.environ.get("SLURM_JOB_ID", "local")
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
node = socket.gethostname()
time.sleep(5)  # Simulate some startup time
print(f"[Task {task_id}] Starting on node={node}, job_id={job_id}")

# Simulate a workload proportional to task_id so runtimes differ slightly
workload = 10_000_000 + task_id * 5_000_000
result = sum(math.sqrt(i) for i in range(1, workload))
time.sleep(10)  # Simulate some cleanup time
print(f"[Task {task_id}] Computed sum-of-sqrt up to {workload:,} → {result:.4f}")


print(f"[Task {task_id}] Done.")
