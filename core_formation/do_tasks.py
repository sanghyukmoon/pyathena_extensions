import argparse
import subprocess
from pathlib import Path
import uuid

from dask.distributed import Client
from dask_jobqueue.slurm import SLURMRunner

from core_formation import config, tasks, models, load_sim

jobid = uuid.uuid4().hex[:8]
SCRIPT_PATH = f"./job{jobid}.slurm"

def write_slurm_script(model, tasks, overwrite):
    jobname = f"{jobid}{model}"
    tasks_str = " ".join(tasks)
    overwrite_flag = "--overwrite" if overwrite else ""
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=3
#SBATCH --mem=740G
#SBATCH --time=24:00:00
#SBATCH --output={model}_%j.out
#SBATCH --error={model}_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=sanghyuk.moon@princeton.edu

eval "$(/home/sm69/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate pyathena

srun python do_tasks.py {model} {tasks_str} --runbyslurm {overwrite_flag}
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
        with SLURMRunner(scheduler_options={"interface": "ib0"},
                         worker_options={"interface": "ib0"}) as runner:
            # The runner object contains the scheduler address info and can be used to construct a client.
            with Client(runner) as client:
                # Wait for all the workers to be ready before continuing.
                client.wait_for_workers(runner.n_workers)

                for task in args.tasks:
                    s = sa.set_model(args.model, force_override=True)
                    tasks.__dict__[task](s, overwrite=args.overwrite)
    else:
        write_slurm_script(args.model, args.tasks, args.overwrite)
        subprocess.run(["sbatch", SCRIPT_PATH])
