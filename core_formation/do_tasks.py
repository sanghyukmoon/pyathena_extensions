import argparse
import subprocess
from pathlib import Path
import uuid

from dask.distributed import Client
from dask_jobqueue.slurm import SLURMRunner

from core_formation import config, tasks, models, load_sim

jobid = uuid.uuid4().hex[:8]
SCRIPT_PATH = f"./job{jobid}.slurm"

# --ntasks-per-node=32 and --cpus-per-task=3 uses all 96 cores in stellar
# for radial profile, due to large memory requirement, we cannot use
# all 32 tasks per node; we have to reduce ntasks-per-node and use 2 nodes.
def write_slurm_script(model, tasks, overwrite):
    jobname = f"{jobid}{model}"
    tasks_str = " ".join(tasks)
    overwrite_flag = "--overwrite" if overwrite else ""
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=3
#SBATCH --mem=742G
#SBATCH --time=24:00:00
#SBATCH --output={model}_%j.out
#SBATCH --error={model}_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=sanghyuk.moon@princeton.edu

eval "$(/home/sm69/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate pyathena

echo Launching $SLURM_NTASKS tasks
srun -n $SLURM_NTASKS python do_tasks.py {model} {tasks_str} --runbyslurm {overwrite_flag}
    """
    if Path(SCRIPT_PATH).exists():
        raise FileExistsError(f"SLURM script path {SCRIPT_PATH} already exists.")
    with open(SCRIPT_PATH, 'w') as f:
        f.write(slurm_script)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model to process")
    parser.add_argument("tasks", nargs='+', type=str, help="Tasks to do")
    parser.add_argument("--runbyslurm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num", type=int)

    args = parser.parse_args()

    if args.runbyslurm:
        sa = load_sim.LoadSimAll(models.models)
        # memory_limit must be (742 GiB) / (# worker), where
        # (# worker) = (ntasks_per_node) - 2 (1 for scheduler, 1 for client)
        # Well, actually that's not true! scheduler and client also use memory,
        # especially when graphs are large.
        with SLURMRunner(
            scheduler_options={"interface": "ib0"},
            worker_options={
                "interface": "ib0",
                "nthreads": 3,
                "memory_limit": "23.1875 GiB", # See comment above
            }
        ) as runner:
            # The runner object contains the scheduler address info and can be used to construct a client.
            with Client(runner) as client:
                # Wait for all the workers to be ready before continuing.
                client.wait_for_workers(runner.n_workers)

                for task in args.tasks:
                    if task == 'radial_profile':
                        s = sa.set_model(args.model, override_cores=True,
                                         load_derived_cores=False)
                    else:
                        s = sa.set_model(args.model, override_all=True)
                    tasks.__dict__[task](s, overwrite=args.overwrite)
    else:
        write_slurm_script(args.model, args.tasks, args.overwrite)
        subprocess.run(["sbatch", SCRIPT_PATH])
