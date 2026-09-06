"""Exercise the actual CLI with a test-only model mapping and two workers."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--test-root', type=Path, required=True)
args = parser.parse_args()
root = args.test_root.resolve()
work = root/'task-driven'
source = root/'M10.J4.B2.P0.N256'
destination = work/'runner'
for path in (source, destination):
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Unsafe path: {path}')
shutil.copytree(work/'onthefly/on_the_fly', destination/'on_the_fly', dirs_exist_ok=True)
code = f'''
import runpy
import sys
from core_formation import models, load_sim
models.models = {{'validation': {str(source)!r}}}
original = load_sim.LoadSim
def load_test(*args, **kwargs):
    kwargs['savdir'] = {str(destination)!r}
    return original(*args, **kwargs)
load_sim.LoadSim = load_test
sys.argv = ['do_tasks_old.py', 'validation', '--no-legacy', '--critical-tes',
            '--lagrangian-props', '--methods', 'virial0', '--pids', '1', '2',
            '--np', '2', '--overwrite']
runpy.run_module('core_formation.do_tasks_old', run_name='__main__', alter_sys=True)
'''
subprocess.run([sys.executable, '-c', code], check=True)
for pid in (1, 2):
    for name in (f'critical_tes.par{pid}.nc', f'core_props.virial0.par{pid}.nc'):
        assert (destination/'cores'/name).exists(), name
print('Actual runner: two-worker TES and Lagrangian tasks passed')
