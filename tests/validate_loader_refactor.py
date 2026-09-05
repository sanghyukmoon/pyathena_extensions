"""Opt-in parity check for the loader simplification, within the test root."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import xarray as xr

from core_formation.load_sim import LoadSim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.test_root.resolve()
    work = root/'loader-refactor'
    source = root/'M10.J4.B2.P0.N256'
    for path in (work, source):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe path: {path}')
    start = perf_counter()
    s = LoadSim(source, legacy=False, savdir=str(work/'new-cache'), overwrite=True)
    assert not s.load_errors, s.load_errors
    assert s.nums == s.nums_rprof
    assert len(s.cores) == len(s.rprofs) == 15
    assert all(s.tcoll_cores.output_time <= s.tcoll_cores.time)
    baseline = work/'baseline/on_the_fly'
    after = work/'new-cache/on_the_fly'
    names = sorted(p.name for p in baseline.glob('*.nc'))
    assert len(names) == 75
    for name in names:
        expected = xr.load_dataset(baseline/name).drop_vars('cycle', errors='ignore')
        actual = xr.load_dataset(after/name)
        xr.testing.assert_equal(actual, expected)
        if name.startswith('core_rprof'):
            for key in ('cs', 'gconst', 'mhd', 'pid', 'source'):
                assert actual.attrs[key] == expected.attrs[key]
        else:
            left = json.loads(actual.attrs['core_attributes'])
            right = json.loads(expected.attrs['core_attributes'])
            assert left.keys() == right.keys()
            for key in left:
                if isinstance(left[key], (float, int)):
                    np.testing.assert_equal(left[key], right[key])
                else:
                    assert left[key] == right[key], key
    report = dict(cores=15, identical_datasets=75, seconds=perf_counter()-start)
    stamps = {name:(after/name).stat().st_mtime_ns for name in names}
    with patch.object(LoadSim, '_compute_core_props', side_effect=AssertionError('cache miss')):
        with patch.object(LoadSim, 'load_par', side_effect=AssertionError('particle read')):
            cached = LoadSim(source, legacy=False, savdir=str(work/'new-cache'))
    assert not cached.load_errors, cached.load_errors
    assert stamps == {name:(after/name).stat().st_mtime_ns for name in names}
    report['cache_reuse_without_particle_reads'] = True
    (work/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
