#!/home/sm69/miniforge3/envs/pyathena/bin/python
import argparse
from shlex import quote
from subprocess import run

from core_formation.models import models
from pyathena.find_files import FindFiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=models, metavar='MODEL')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Rebuild existing outputs for all three stages')
    args = parser.parse_args()
    model = args.model
    overwrite = ' --overwrite' if args.overwrite else ''
    files = FindFiles(models[model])
    ncpu = len(set(getattr(files, 'pids', [])))
    if ncpu == 0:
        parser.exit(1, f'No particle histories found for {model}\n')

    script = f"""#!/bin/bash
#SBATCH --job-name=pp{model}
#SBATCH --cpus-per-task={ncpu}
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=cascade
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH --mail-user=sanghyuk.moon@princeton.edu
#SBATCH --mail-type=end

eval "$(/home/sm69/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate pyathena

srun --unbuffered --cpus-per-task="$SLURM_CPUS_PER_TASK" \\
    python -m core_formation.do_tasks_old {quote(model)} \\
    --track-cores --critical-tes --collapse-history --np "$SLURM_CPUS_PER_TASK"{overwrite}
"""
    run(['sbatch'], input=script, text=True, check=True)
