"""Compare one on-the-fly profile with tools.radial_profile on matching HDF5.

Run in the pyathena environment: python tests/compare_radial_profile.py
Optional selection: --num 100 --center-id 12345. No pipeline caches are reused.
Exit 0 means all checks pass; exit 1 means a numerical/structural mismatch.
"""
import argparse
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from core_formation.load_sim import LoadSim
from core_formation import tools

ROOT = Path('/scratch/gpfs/sm69/onthefly-rprof-test')
EPS = float(np.finfo(np.float32).eps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--center-id', type=int)
    args = parser.parse_args()
    output_root = ROOT / 'single-profile-comparison'
    output_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'num{args.num:05d}-', dir=output_root))
    print(f'Output directory: {work}', flush=True)
    s = LoadSim(str(ROOT / 'M10.J4.B2.P0.N128'), savdir=str(work / 'cache'),
                legacy=True, load_derived_cores=False)
    header = s.load_rprof(args.num, metadata_only=True)
    center_id = int(header.center_id.min()) if args.center_id is None else args.center_id
    onthefly = s.load_rprof(args.num, center_ids=[center_id]).isel(center_id=0, drop=True)
    ds = s.load_hdf5(args.num)
    time_match = abs(float(ds.Time) - onthefly.attrs['time']) <= EPS * max(
        abs(float(ds.Time)), abs(onthefly.attrs['time']))
    center = dict(zip(('x', 'y', 'z'), s.flatindex_to_cartesian(center_id)))
    print(f'num={args.num}, center_id={center_id}, center={center}', flush=True)
    ds, center, _ = tools.recenter_dataset(ds, center)
    python = tools.radial_profile(s, ds, list(center.values()),
                                 rmax=onthefly.attrs['rmax'], nsub=onthefly.attrs['nsub'],
                                 compute_flux=True).compute()
    onthefly.to_netcdf(work / 'onthefly.nc')
    python.to_netcdf(work / 'python.nc')

    missing_python = sorted(set(onthefly.data_vars) - set(python.data_vars))
    missing_onthefly = sorted(set(python.data_vars) - set(onthefly.data_vars))
    passed = time_match and not missing_python and not missing_onthefly
    lines = [f'num={args.num}, center_id={center_id}',
             f'HDF5 time={float(ds.Time):.17g}; rprof time={onthefly.attrs["time"]:.17g}',
             f'rmax={onthefly.attrs["rmax"]}; nsub={onthefly.attrs["nsub"]}',
             f'Criterion: abs(a-b) <= {EPS:.9g} * max(abs(a),abs(b)); no atol',
             f'Missing Python fields: {missing_python}',
             f'Missing on-the-fly fields: {missing_onthefly}',
             'field                      max_abs       max_rel   failed / total']
    fields = ['r'] + sorted(set(onthefly.data_vars) | set(python.data_vars))
    results = {}
    for name in fields:
        if name not in onthefly or name not in python:
            continue  # Missing fields are already reported as failures above.
        a, b = onthefly[name], python[name]
        if a.dims != b.dims or a.shape != b.shape:
            lines.append(f'{name}: dimensions differ: {a.dims} {a.shape} vs {b.dims} {b.shape}')
            passed = False
            continue
        av, bv = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        finite = np.isfinite(av) & np.isfinite(bv)
        same_special = ((np.isnan(av) & np.isnan(bv)) |
                        (np.isposinf(av) & np.isposinf(bv)) |
                        (np.isneginf(av) & np.isneginf(bv)))
        delta = np.abs(av[finite] - bv[finite])
        scale = np.maximum(np.abs(av[finite]), np.abs(bv[finite]))
        relative = np.divide(delta, scale, out=np.zeros_like(delta), where=scale != 0)
        failures = int(np.count_nonzero(delta > EPS * scale) + np.count_nonzero(~finite & ~same_special))
        passed &= failures == 0
        lines.append(f'{name:26s} {delta.max(initial=0):12.4e} {relative.max(initial=0):12.4e} '
                     f'{failures:6d} / {av.size}')
        results[name] = (av, bv, failures)
    lines.append('PASS' if passed else 'FAIL')
    report = '\n'.join(lines) + '\n'
    print(report, flush=True)
    (work / 'comparison.txt').write_text(report)

    with PdfPages(work / 'comparison.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(8.3, 11.7)); ax.axis('off')
        ax.text(.03, .98, 'Single radial-profile consistency test', fontsize=16, va='top')
        ax.text(.03, .89, 'HDF5 -> recenter -> tools.radial_profile\n'
                          '.rprof -> select the same minimum\n'
                          'Compare raw fields on the same radial grid', fontsize=11, va='top')
        ax.text(.03, .77, '\n'.join(lines[:7]) + '\n\n' + lines[-1], fontsize=9, va='top')
        ax.text(.03, .43, 'No tracking, TES, or Lagrangian calculations.\n'
                          'HDF5 inputs are float32; differences are evaluated in float64.\n'
                          'NaN locations and signed infinities must match.\n'
                          'No interpolation, bin truncation, or tolerance relaxation.\n\n'
                          'Getting started: run tests/compare_radial_profile.py\n'
                          'Optional: --num 100 --center-id ID\n'
                          'Each invocation creates a fresh output directory.\n\n'
                          'Limitation: one profile does not validate every minimum\n'
                          'or snapshot. Near-zero values can fail a relative-only test.', fontsize=10, va='top')
        fig.savefig(work / 'summary.png', dpi=120)
        pdf.savefig(fig); plt.close(fig)
        for start in range(7, len(lines), 45):
            fig, ax = plt.subplots(figsize=(8.3, 11.7)); ax.axis('off')
            ax.text(.01, .98, '\n'.join(lines[start:start+45]), fontfamily='monospace', fontsize=8, va='top')
            pdf.savefig(fig); plt.close(fig)
        names = [name for name in results if name != 'r']
        for start in range(0, len(names), 4):
            fig, axes = plt.subplots(4, 2, figsize=(11.7, 10), squeeze=False)
            for row, name in enumerate(names[start:start+4]):
                a, b, failures = results[name]
                xa = onthefly.r.values if a.ndim else [0]
                xb = python.r.values if b.ndim else [0]
                axes[row, 0].plot(xa, np.atleast_1d(a), 'o-', ms=3, label='on-the-fly')
                axes[row, 0].plot(xb, np.atleast_1d(b), 'x--', ms=3, label='Python')
                axes[row, 0].set_title(f'{name}: {failures} failing values', fontsize=10)
                axes[row, 0].legend(fontsize=8)
                axes[row, 1].plot(xa, np.atleast_1d(a-b), 'o-', ms=3)
                axes[row, 1].axhline(0, color='gray', linewidth=.5)
                axes[row, 1].set_title('on-the-fly minus Python', fontsize=10)
                for ax in axes[row]:
                    ax.set_xlabel('r' if a.ndim else 'scalar')
            for row in range(len(names[start:start+4]), 4):
                for ax in axes[row]:
                    ax.axis('off')
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
