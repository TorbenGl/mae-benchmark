# Demo: Running the Worker on SLURM

Two ways to run `demo/worker.py` on a compute node.

---

## Option 1 — sbatch (fire and forget) 

Submit the job array and let SLURM schedule it:

```bash
# from the repo root
mkdir -p demo/logs
sbatch demo/submit_demo.sh
```

Watch progress:

```bash
squeue --me                          # see your jobs in the queue
tail -f demo/logs/task_<JOBID>_0.out # follow output of task 0
```

Each of the 6 array tasks runs `worker.py` independently on the cluster. Logs land in `demo/logs/task_<JOBID>_<TASKID>.out`.

---

## Option 2 — interactive session + /bin/bash

Request an interactive shell directly on a compute node, then run the worker yourself:

```bash
# 1. Allocate an interactive node (adjust partition name as needed)
srun --partition=compute-node \
     --nodes=1 \
     --cpus-per-task=4 \
     --mem=2G \
     --time=00:10:00 \
     --pty /bin/bash
```

You are now inside a shell **on the compute node**. From there:

```bash
# 2. Navigate to the repo and run the worker directly
cd /path/to/mae-benchmark

# Run as a plain Python script (no SLURM array — task_id defaults to 0)
python demo/worker.py

# Or simulate a specific array task by setting the env var manually
SLURM_ARRAY_TASK_ID=3 python demo/worker.py
```

Exit the interactive session when done:

```bash
exit
```

---

## What the worker does

`demo/worker.py` reads `SLURM_JOB_ID` and `SLURM_ARRAY_TASK_ID` from the environment, then computes a sum-of-square-roots workload scaled to the task ID so runtimes differ slightly across tasks. In an interactive session these env vars are absent (or can be set manually), so it behaves like task 0 by default.
