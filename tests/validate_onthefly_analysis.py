"""Opt-in real-data regression. All artifacts stay within --workdir."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from core_formation.load_sim import LoadSim
from core_formation.rprof_derived import add_rprof_derived
from pyathena.io.read_radial_profile import read_radial_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--test-root', type=Path, required=True,
                        help='Explicitly permitted parent directory for all test data')
    args = parser.parse_args()
    permitted = args.test_root.resolve()
    for path in (args.basedir, args.workdir):
        if not path.resolve().is_relative_to(permitted):
            raise ValueError('This regression must remain inside the permitted test root')
    fixture = args.workdir/'hdf5-free'
    fixture.mkdir(exist_ok=True)
    for pattern in ('athinput.runtime', '*.rprof', '*.parbin', '*.csv', '*.hst'):
        for source in args.basedir.glob(pattern):
            if not source.resolve().is_relative_to(permitted):
                raise ValueError(f'Unsafe source link: {source}')
            target = fixture/source.name
            if not target.exists():
                shutil.copy2(source, target)
    assert not list(fixture.rglob('*.athdf'))
    assert not list(fixture.rglob('*.p'))
    start = perf_counter()
    with patch.object(LoadSim, 'load_hdf5', side_effect=AssertionError('HDF5 accessed')):
        s = LoadSim(fixture, legacy=False, savdir=str(args.workdir/'hdf5-free-analysis'))
        assert len(s.cores) == len(s.pids) == len(s.rprofs)
        assert not s.load_errors, s.load_errors
        assert all(s.tcoll_cores.output_time <= s.tcoll_cores.time)
        assert not s.nums_hdf5
        assert all(len(value) == len(s.pids) for value in s.cores_dict.values())
    report = dict(cores=len(s.pids), initial_cache_aware_seconds=perf_counter()-start,
                  no_hdf5=True)
    start = perf_counter()
    with patch.object(LoadSim, 'load_hdf5', side_effect=AssertionError('HDF5 accessed')):
        cached = LoadSim(fixture, legacy=False, savdir=s.savdir)
    report['warm_seconds'] = perf_counter()-start
    for pid in s.pids:
        xr.testing.assert_identical(s.rprofs[pid], cached.rprofs[pid])
        for method in s.cores_dict:
            left, right = s.cores_dict[method][pid], cached.cores_dict[method][pid]
            for column in left:
                np.testing.assert_allclose(np.asarray(left[column], dtype=float),
                                           np.asarray(right[column], dtype=float),
                                           rtol=0, atol=0, equal_nan=True)

    # Compare against source captured before the scientific refactor.
    baseline = args.workdir/'baseline'
    raw = xr.load_dataset(baseline/'legacy-raw.nc')
    expected = xr.load_dataset(baseline/'legacy-derived.nc')
    actual = add_rprof_derived(raw, cs=s.cs, gconst=s.gconst, mhd=s.mhd)
    xr.testing.assert_equal(actual, expected)
    report['derived_radial_exact_variables'] = len(expected.data_vars)
    report['raw_profile_comparisons'] = []
    for num, center in ((80, 10525457), (109, 3966648), (129, 997181)):
        expected_raw = xr.load_dataset(args.basedir/'radial_profile'/
                         f'radial_profile.{center}.{num:05d}.nc').isel(t=0)
        actual_raw = read_radial_profile(args.basedir/f'GMTF.rprof.{num:05d}.rprof',
                                         center_ids=[center]).isel(center_id=0)
        for name in expected_raw:
            value = expected_raw[name].values
            scale = max(1., np.nanmax(np.abs(value)))
            np.testing.assert_allclose(actual_raw[name].values, value,
                                       rtol=1e-10, atol=1e-12*scale, equal_nan=True)
        report['raw_profile_comparisons'].append(dict(num=num, center=center,
                                                     variables=len(expected_raw.data_vars)))

    # Existing saved products must remain readable without rewriting originals.
    legacy_cache = args.workdir/'legacy-cache-compatibility'
    for subdir, patterns in (('cores', ('cores.p', 'tcoll_cores.p', 'cores_tcrit_virial*.p')),
                             ('radial_profile', ('radial_profile.p',))):
        target_dir = legacy_cache/subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            for source in (args.basedir/subdir).glob(pattern):
                shutil.copy2(source, target_dir/source.name)
    legacy = LoadSim(str(args.basedir), savdir=str(legacy_cache), load_derived_cores=False)
    assert legacy.legacy
    assert len(legacy.cores) == len(legacy.rprofs) == len(s.pids)
    for method in ('virial', 'virial0', 'virial1'):
        loaded = legacy.update_core_props(method, prefix=f'cores_tcrit_{method}',
                                          savdir=legacy_cache/'cores')
        assert len(loaded) == len(s.pids)
    report['legacy_saved_products_cores'] = len(legacy.cores)
    spec = importlib.util.spec_from_file_location('core_formation._reference_load_sim',
                                                  baseline/'load_sim.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    old = reference.LoadSim.__new__(reference.LoadSim)
    old.__dict__.update(s.__dict__)
    old.cores = {}
    tes_columns = ['center_density', 'sonic_radius', 'pindex', 'critical_contrast',
                   'critical_radius', 'critical_mass']
    for pid, track in s._core_tracks.items():
        old.cores[pid] = track.join(s.cores_dict['empirical'][pid][tes_columns])
        old.cores[pid].attrs = track.attrs.copy()
    old.load_par = s.load_par
    empty = args.workdir/'no-legacy-products'
    empty.mkdir(exist_ok=True)
    report['derived_core_comparisons'] = {}
    for method in s.cores_dict:
        expected = reference.LoadSim.update_core_props.__wrapped__(old, method, savdir=empty)
        for pid in s.pids:
            actual = s.cores_dict[method][pid]
            for column in actual:
                np.testing.assert_allclose(np.asarray(actual[column], dtype=float),
                                           np.asarray(expected[pid][column], dtype=float),
                                           rtol=0, atol=0, equal_nan=True)
            for key, value in expected[pid].attrs.items():
                if isinstance(value, (float, int, np.number)):
                    np.testing.assert_allclose(actual.attrs[key], value,
                                               rtol=0, atol=0, equal_nan=True)
                else:
                    assert actual.attrs[key] == value
        report['derived_core_comparisons'][method] = len(expected)
    with (args.workdir/'analysis-validation.json').open('w') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
