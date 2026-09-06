"""Opt-in N128 end-to-end test; never writes to the simulation input directory.

Run with the pyathena environment on a compute node:
  python tests/validate_pipeline.py --work SAFE_ROOT/pipeline-validation/N128
Resume with the same --work; completed stage markers are reused. For a fresh
experiment choose a new empty --work inside SAFE_ROOT/pipeline-validation.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import runpy
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

from pipeline_comparison import EPS32, compare_products

SAFE_ROOT = Path('/scratch/gpfs/sm69/onthefly-rprof-test')
SOURCE = SAFE_ROOT / 'M10.J4.B2.P0.N128'
METHODS = ('empirical', 'virial', 'virial0', 'virial1')
STAGES = ('minima', 'track', 'profiles', 'tes', 'lagrangian', 'reload', 'plots')


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + '\n')


def load(mode, destination, **kwargs):
    from core_formation.load_sim import LoadSim
    return LoadSim(str(SOURCE), savdir=str(destination), legacy=mode == 'legacy', **kwargs)


def run_standard(mode, destination, flag, workers):
    from core_formation import models, load_sim
    models.models = {'validation': str(SOURCE)}
    original = load_sim.LoadSim

    def load_test(*args, **kwargs):
        kwargs['savdir'] = str(destination)
        return original(*args, **kwargs)

    load_sim.LoadSim = load_test
    sys.argv = ['do_tasks_old.py', 'validation',
                '--legacy' if mode == 'legacy' else '--no-legacy', flag,
                '--np', str(workers)]
    try:
        runpy.run_module('core_formation.do_tasks_old', run_name='__main__', alter_sys=True)
    finally:
        load_sim.LoadSim = original


def products(s, destination):
    """Coverage expectations come from independently loaded tracks in each mode."""
    inventory = {'pids': list(map(int, s.pids)), 'methods': list(s.cores_dict),
                 'profiles': list(map(int, s.rprofs)), 'expected': {}, 'missing': []}
    required = []
    for pid in s.pids:
        required.append(f'cores/cores.par{pid}.p')
        if pid not in s.rprofs:
            inventory['missing'].append(f'rprofs/{pid}')
    for method in METHODS:
        if method not in s.cores_dict:
            inventory['missing'].append(f'cores_dict/{method}')
            continue
        s.select_cores(method)
        inventory['missing'].extend(f'cores_dict/{method}/{pid}'
                                    for pid in s.pids if pid not in s.cores)
        inventory['expected'][method] = {'eligible_lagrangian': list(map(int, s.good_cores(0))),
                                       'tracked_snapshots': {str(pid): list(map(int, c.index))
                                                             for pid, c in s.cores.items()}}
        for pid, cores in s.cores.items():
            required.extend(f'cores/critical_tes.par{pid}.{num:05d}.p' for num in cores.index)
            if pid in s.rprofs and not np.array_equal(s.rprofs[pid].num, cores.index):
                inventory['missing'].append(f'incomplete profile history/{pid}/{method}')
        required.extend(f'cores/lprops_tcrit_{method}.par{pid}.p' for pid in s.good_cores(0))
    inventory['missing'].extend(name for name in sorted(set(required))
                                if not (destination / name).exists())
    inventory['files'] = sorted(str(p.relative_to(destination)) for p in destination.rglob('*')
                                if p.is_file())
    return inventory


def stage(mode, name, destination, workers):
    import dask
    from core_formation import tasks
    # Forked task-runner workers must not inherit a running Dask thread pool.
    dask.config.set(scheduler='synchronous')
    if name == 'minima':
        s = load(mode, destination, load_derived_cores=False)
        if mode == 'legacy':
            tasks.save_minima(s)
    elif name in ('track', 'tes', 'lagrangian'):
        flag = {'track': '--track-cores', 'tes': '--critical-tes',
                'lagrangian': '--lagrangian-props'}[name]
        run_standard(mode, destination, flag, workers)
    elif name == 'profiles':
        s = load(mode, destination, load_derived_cores=False, override_cores=True)
        if mode == 'legacy':
            with dask.config.set(scheduler='threads', num_workers=workers):
                tasks.radial_profile(s)
            s.concat_radial_profiles()
        else:
            assert set(s.rprofs) == set(s.pids), 'Incomplete on-the-fly profile assembly'
    elif name == 'reload':
        s = load(mode, destination, override_all=True)
        inventory = products(s, destination)
        write_json(destination / 'products.json', inventory)
        assert not inventory['missing'], inventory['missing']
    elif name == 'plots':
        s = load(mode, destination)
        s.select_cores('virial0')
        requested = [('sink', None, int(n)) for n in s.nums]
        requested += [('core', int(pid), int(n)) for pid, c in s.cores.items() for n in c.index]
        # A failed figure must not prevent the remaining requested figures.
        global plot_one

        def plot_one(item):
            kind, pid, num = item
            try:
                if kind == 'sink':
                    tasks.plot_sink_history(s, num)
                    filename = f'sink_history.{num:05d}.png'
                else:
                    tasks.plot_core_evolution(s, pid, num, method='virial0')
                    filename = f'core_evolution.par{pid}.tcrit_virial0.{num:05d}.png'
                path = destination / 'figures' / filename
                assert path.is_file() and path.stat().st_size > 0
                return dict(kind=kind, pid=pid, num=num, file=str(path), passed=True)
            except Exception:
                return dict(kind=kind, pid=pid, num=num, passed=False, error=traceback.format_exc())

        with mp.get_context('fork').Pool(workers) as pool:
            inventory = list(pool.imap(plot_one, requested, chunksize=1))
        write_json(destination / 'plots.json', inventory)
        if not all(item['passed'] for item in inventory):
            raise RuntimeError('Some figures failed; see plots.json')


def provenance(work):
    import h5py
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
                'python': sys.version, 'eps32': EPS32, 'packages': {}, 'repositories': {}, 'outputs': {}}
    for package in ('numpy', 'pandas', 'xarray', 'dask', 'scipy', 'h5py', 'matplotlib'):
        manifest['packages'][package] = importlib.metadata.version(package)
    for name in ('pyathena_extensions', 'pyathena', 'tigris'):
        repository = Path('/home/sm69') / name
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repository), *args], text=True)
        manifest['repositories'][name] = {'head': git('rev-parse', 'HEAD').strip(),
                                          'status': git('status', '--short'), 'diff': git('diff')}
    for mode in ('legacy', 'onthefly'):
        s = load(mode, work / mode, load_derived_cores=False)
        manifest['outputs'][mode] = {'nums': list(map(int, s.nums)),
                                     'times': {str(n): float(t) for n, t in s.times.items()},
                                     'pids': list(map(int, s.pids))}
    with h5py.File(next(SOURCE.glob('*.athdf')), 'r') as f:
        manifest['hdf5_dtypes'] = {k: str(v.dtype) for k, v in f.items()}
    write_json(work / 'manifest.json', manifest)


def compare(work):
    sims, errors = {}, {}
    for mode in ('legacy', 'onthefly'):
        try:
            sims[mode] = load(mode, work / mode)
        except Exception:
            # Preserve the failed full-load result, but still compare completed
            # profiles/tracks/TES. Never label missing core properties as parity.
            errors[mode] = traceback.format_exc()
            sims[mode] = load(mode, work / mode, load_derived_cores=False)
    a, b = sims.values()
    records = []
    compare_products({n: np.sort(ids) for n, ids in a.minima.items()},
                     {n: np.sort(ids) for n, ids in b.minima.items()}, 'minima', records)
    compare_products(a.tcoll_cores, b.tcoll_cores, 'collapse', records)
    if errors:
        records.append({'path': 'cores_dict/availability', 'passed': False,
                        'reason': 'Full initialization failed', 'errors': errors})
    else:
        compare_products(a.cores_dict, b.cores_dict, 'cores_dict', records)
    compare_products(a.rprofs, b.rprofs, 'rprofs', records)
    for prefix, pattern in (('tracks', 'cores.par*.p'), ('tes', 'critical_tes.par*.p'),
                            ('lagrangian', 'lprops_tcrit_*.p')):
        data = [{p.name: pd.read_pickle(p) for p in (work / mode / 'cores').glob(pattern)}
                for mode in sims]
        compare_products(*data, prefix, records)
    # Label raw vs. derived profile comparisons without filtering either group.
    raw_fields = set(b.load_rprof(b.nums[0]).data_vars)
    for record in records:
        parts = record['path'].split('/')
        if parts[0] == 'rprofs' and len(parts) > 2:
            record['profile_kind'] = 'raw' if parts[2] in raw_fields else 'derived_or_metadata'
    groups = {}
    for group in ('minima', 'collapse', 'tracks', 'rprofs', 'tes', 'lagrangian', 'cores_dict'):
        selected = [r for r in records if r['path'].split('/')[0] == group]
        groups[group] = {'checks': len(selected), 'failed_checks': sum(not r['passed'] for r in selected)}
    report = {'eps32': EPS32, 'criterion': 'abs(a-b) <= eps32 * max(abs(a),abs(b)); no atol',
              'passed': all(r['passed'] for r in records), 'groups': groups,
              'initialization_errors': errors, 'records': records}
    write_json(work / 'comparison' / 'comparison.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2), flush=True)
    return report


def pdf_report(work, report, stages):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from textwrap import wrap
    destination = work / 'comparison' / 'validation-report.pdf'
    with PdfPages(destination) as pdf:
        fig, ax = plt.subplots(figsize=(8.3, 11.7))
        ax.axis('off')
        ax.text(.02, .97, 'N128 pipeline validation', fontsize=18, va='top')
        for x, title, steps in ((.04, 'Legacy', 'HDF5\nMinima + tracking\nPython radial profiles'),
                                 (.55, 'On-the-fly', '.rprof\nMinima + tracking\nAssembled radial profiles')):
            ax.text(x, .87, title, fontsize=14)
            ax.text(x, .76, steps, bbox=dict(boxstyle='round', facecolor='#edf3f8'), fontsize=11)
            ax.annotate('', xy=(x+.12, .65), xytext=(x+.12, .72), arrowprops=dict(arrowstyle='->'))
        ax.text(.08, .60, 'Same critical TES and Lagrangian tasks\nIndependent caches; all four critical-time methods', fontsize=12)
        failures = [s for s in stages if s['returncode']]
        lines = [f'Pipeline/plot stages with errors: {len(failures)}',
                 f'Single-precision comparison: {"PASS" if report and report["passed"] else "FAIL / unavailable"}',
                 f'Tolerance: eps32={EPS32:.9g}, symmetric relative; no absolute tolerance.',
                 'HDF5 fields are float32. Scientific calculations were not changed.',
                 'A failed numerical test is distinct from a successfully executed pipeline.']
        if report:
            lines += [f'{k}: {v["failed_checks"]} failed / {v["checks"]} checks' for k,v in report['groups'].items()]
        ax.text(.02, .52, '\n\n'.join(lines), fontsize=10, va='top')
        fig.savefig(work / 'comparison' / 'report-summary.png', dpi=120)
        pdf.savefig(fig); plt.close(fig)
        if report:
            bad = [r for r in report['records'] if not r['passed']]
            fig, ax = plt.subplots(figsize=(8.3, 11.7)); ax.axis('off')
            lines = ['Representative discrepancies (complete results in comparison.json)', '']
            for r in bad[:12]:
                lines += wrap(r['path'], width=95)
                lines += wrap(str({k:v for k,v in r.items() if k not in ('path','examples')}), width=95)
                if r.get('examples'):
                    lines += wrap(str(r['examples'][0]), width=95)
                lines.append('')
            ax.text(.01, .98, '\n'.join(lines[:76]), va='top', fontsize=8)
            pdf.savefig(fig); plt.close(fig)
        for pattern in ('sink_history.*.png', 'core_evolution.*.png'):
            paths = sorted((work / 'legacy' / 'figures').glob(pattern))
            paired = [p for p in paths if (work / 'onthefly' / 'figures' / p.name).exists()]
            if not paired:
                continue
            selected = paired[-1]
            fig, axes = plt.subplots(2, 1, figsize=(11.7, 12))
            for ax, mode in zip(axes, ('legacy', 'onthefly')):
                ax.imshow(plt.imread(work / mode / 'figures' / selected.name))
                ax.set_title(mode + ': ' + selected.name, fontsize=10); ax.axis('off')
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=SAFE_ROOT / 'pipeline-validation' / 'N128')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--mode', choices=('legacy', 'onthefly'))
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--compare-only', action='store_true')
    args = parser.parse_args()
    work = args.work.resolve()
    if not work.is_relative_to(SAFE_ROOT / 'pipeline-validation') or work == SAFE_ROOT / 'pipeline-validation':
        parser.error('--work must be a child of the approved pipeline-validation directory')
    if bool(args.mode) != bool(args.stage):
        parser.error('--mode and --stage must be given together')
    for subdir in ('legacy', 'onthefly', 'logs', 'comparison'):
        (work / subdir).mkdir(parents=True, exist_ok=True)
    if args.stage:
        stage(args.mode, args.stage, work / args.mode, args.workers)
        return 0
    if not (work / 'manifest.json').exists():
        provenance(work)
    statuses = []
    if not args.compare_only:
        for mode in ('legacy', 'onthefly'):
            for name in STAGES:
                marker = work / 'logs' / f'{mode}-{name}.done.json'
                if marker.exists():
                    statuses.append(json.loads(marker.read_text()))
                    continue
                logpath = work / 'logs' / f'{mode}-{name}.log'
                command = [sys.executable, '-u', __file__, '--work', str(work), '--mode', mode,
                           '--stage', name, '--workers', str(args.workers)]
                print(f'{datetime.now(timezone.utc).isoformat()} {mode}/{name}', flush=True)
                with logpath.open('a') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                status = dict(mode=mode, stage=name, returncode=result.returncode,
                              finished_utc=datetime.now(timezone.utc).isoformat())
                statuses.append(status)
                write_json(work / 'logs' / 'stages.json', statuses)
                if result.returncode == 0:
                    write_json(marker, status)
                else:
                    break
    else:
        statuses = json.loads((work / 'logs' / 'stages.json').read_text())
    report = None
    try:
        report = compare(work)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        write_json(work / 'comparison' / 'comparison-error.json', {'error': error})
    pdf_report(work, report, statuses)
    return 0 if report and report['passed'] and all(s['returncode'] == 0 for s in statuses) else 1


if __name__ == '__main__':
    raise SystemExit(main())
