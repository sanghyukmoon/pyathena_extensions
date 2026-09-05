"""Compare before/after analysis caches and prove warm loading uses them."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import xarray as xr

from core_formation import load_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.test_root.resolve()
    work = root/'refactor-validation'
    before = work/'before/hdf5-free-analysis/on_the_fly'
    after = work/'real-data/hdf5-free-analysis/on_the_fly'
    for path in (before, after, work/'real-data/hdf5-free'):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe fixture path: {path}')
    names = sorted(path.name for path in before.glob('*.nc'))
    assert names and names == sorted(path.name for path in after.glob('*.nc'))
    for name in names:
        left, right = xr.load_dataset(before/name), xr.load_dataset(after/name)
        # Sources were copied into separate fixture directories. All other
        # attributes, coordinates, integer IDs, and data must be identical.
        left.attrs.pop('fingerprint', None)
        right.attrs.pop('fingerprint', None)
        xr.testing.assert_identical(left, right)
    stamps = {name: (after/name).stat().st_mtime_ns for name in names}
    start = perf_counter()
    with patch.object(load_sim.LoadSim, 'load_hdf5', side_effect=AssertionError('HDF5 read')):
        with patch.object(load_sim.LoadSim, '_compute_core_props',
                          side_effect=AssertionError('Unexpected property cache miss')):
            with patch.object(load_sim, 'read_radial_profile',
                              side_effect=AssertionError('Unexpected profile cache miss')):
                s = load_sim.LoadSim(work/'real-data/hdf5-free', legacy=False,
                                    savdir=str(after.parent))
    assert not s.load_errors, s.load_errors
    assert stamps == {name: (after/name).stat().st_mtime_ns for name in names}
    report = dict(identical_datasets=len(names), cores=len(s.pids),
                  warm_seconds=perf_counter()-start, warm_timings=s.load_timings,
                  no_hdf5=True, no_recomputation=True, no_cache_rewrites=True)
    (work/'refactor-comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
