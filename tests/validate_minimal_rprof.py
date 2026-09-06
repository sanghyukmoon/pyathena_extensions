"""Opt-in real-data parity test using the unchanged task runner and safe savdirs.

Run from the extension repository with the pyathena environment. All artifacts
are confined to the approved test root. Requires the existing 15-core fixture.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from core_formation.load_sim import LoadSim


def run_tasks(source, destination, legacy):
    code = f'''
import runpy, sys
from core_formation import models, load_sim
models.models = {{'validation': {str(source)!r}}}
original = load_sim.LoadSim
def load_test(*args, **kwargs):
    kwargs['savdir'] = {str(destination)!r}
    return original(*args, **kwargs)
load_sim.LoadSim = load_test
sys.argv = ['do_tasks_old.py', 'validation', {'--legacy' if legacy else '--no-legacy'!r},
            '--track-cores', '--critical-tes', '--lagrangian-props', '--np', '2', '--overwrite']
runpy.run_module('core_formation.do_tasks_old', run_name='__main__', alter_sys=True)
'''
    with (destination/'runner.log').open('w') as log:
        subprocess.run([sys.executable, '-c', code], stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-tasks', action='store_true', help='Compare previously generated test products')
    args = parser.parse_args()
    root = Path('/scratch/gpfs/sm69/onthefly-rprof-test')
    source = root/'M10.J4.B2.P0.N256'
    work = root/'minimal-rebuild'
    sims = {}
    for label, legacy in (('legacy', True), ('onthefly', False)):
        destination = work/label
        destination.mkdir(parents=True, exist_ok=True)
        if legacy:
            (destination/'GRID').mkdir(exist_ok=True)
            shutil.copy2(source/'GRID/minima.p', destination/'GRID/minima.p')
            (destination/'radial_profile').mkdir(exist_ok=True)
            # This is an input profile history, not an old derived/TES cache.
            shutil.copy2(source/'radial_profile/radial_profile.concatenated.p',
                         destination/'radial_profile/radial_profile.concatenated.p')
        if not args.skip_tasks:
            run_tasks(source, destination, legacy)
        # The baseline caches core-property tables before Lagrangian tasks run.
        # Refresh that derived cache explicitly to join the newly saved histories.
        sims[label] = LoadSim(str(source), savdir=str(destination), legacy=legacy,
                              override_derived_cores=not args.skip_tasks)
        print(label, 'loaded', len(sims[label].cores), 'cores', flush=True)

    a, b = sims['legacy'], sims['onthefly']
    assert len(a.pids) == len(b.pids) == 15
    pd.testing.assert_frame_equal(a.tcoll_cores, b.tcoll_cores)
    for pid in a.pids:
        assert a.times[a.tcoll_cores.loc[pid].num] < a.tcoll_cores.loc[pid].time
    raw_fields = set(b.load_rprof(b.nums[0]).data_vars)
    discrepancies = []
    nonfinite_discrepancies = []
    report = {'cores': len(a.pids), 'tracks_equal': True, 'raw_profiles_equal': True,
              'critical_numbers_equal': True, 'lagrangian_files': {}}
    for pid in a.pids:
        left = pd.read_pickle(work/'legacy/cores'/f'cores.par{pid}.p')
        right = pd.read_pickle(work/'onthefly/cores'/f'cores.par{pid}.p')
        pd.testing.assert_frame_equal(left, right)
        nums = a.cores[pid].index
        left, right = a.rprofs[pid].sel(num=nums), b.rprofs[pid].sel(num=nums)
        for name in raw_fields & set(left.data_vars):
            values = left[name].values
            scale = max(1., np.nanmax(np.abs(values)))
            np.testing.assert_allclose(right[name], values, rtol=1e-10, atol=1e-12*scale,
                                       equal_nan=True, err_msg=f'pid={pid}, field={name}')
        for name in set(left.data_vars) & set(right.data_vars) - raw_fields:
            values = left[name].values
            finite = np.isfinite(values)
            other_finite = np.isfinite(right[name].values)
            for row, col in np.argwhere(finite != other_finite):
                nonfinite_discrepancies.append(dict(
                    pid=int(pid), field=name, num=int(nums[row]), radius=float(left.r[col]),
                    legacy=float(values[row, col]), onthefly=float(right[name].values[row, col])))
            finite = finite & other_finite
            scale = max(1., np.max(np.abs(values[finite]))) if finite.any() else 1.
            if not np.allclose(right[name], values, rtol=1e-10, atol=1e-12*scale, equal_nan=True):
                error = float(np.max(np.abs(right[name].values[finite]-values[finite]))) if finite.any() else 0.
                discrepancies.append(dict(pid=int(pid), field=name, absolute_error=error,
                                          error_over_field_scale=error/scale))
        for method in ('empirical', 'virial', 'virial0', 'virial1'):
            np.testing.assert_equal(a.cores_dict[method][pid].attrs['numcrit'],
                                    b.cores_dict[method][pid].attrs['numcrit'])
    for label in sims:
        report['lagrangian_files'][label] = len(list((work/label/'cores').glob('lprops_tcrit_*.p')))
    report['derived_source_discrepancies'] = discrepancies
    report['derived_nonfinite_discrepancies'] = nonfinite_discrepancies
    report['derived_profiles_equal_at_raw_tolerance'] = not (discrepancies or nonfinite_discrepancies)
    (work/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if not k.startswith('derived_')}, indent=2))
    print('Derived fields beyond raw tolerance:', len(discrepancies))
    print('Derived finite/nonfinite discrepancies:', len(nonfinite_discrepancies))


if __name__ == '__main__':
    main()
