"""Opt-in task-driven regression; all outputs stay below the permitted test root."""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from core_formation import load_sim, tasks, tools
from pyathena.load_sim import LoadSim as BaseLoadSim


def reference_properties():
    source = subprocess.check_output(
        ['git', 'show', 'a9d530169a2f143eb57be3:core_formation/load_sim.py'], text=True)
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LoadSim')
    func = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'update_core_props')
    func.decorator_list = []
    namespace = vars(load_sim).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])), '<baseline>', 'exec'), namespace)
    return namespace['update_core_props']


def assert_table_values(left, right):
    assert set(left.columns) == set(right.columns)
    np.testing.assert_array_equal(left.index, right.index)
    for name in left:
        np.testing.assert_allclose(np.asarray(left[name], dtype=float),
                                   np.asarray(right[name], dtype=float), rtol=0, atol=0, equal_nan=True)
    for name, value in right.attrs.items():
        if isinstance(value, (int, float, np.number)):
            np.testing.assert_equal(left.attrs[name], value)
        else:
            assert left.attrs[name] == value, name


def compare_products(work):
    report = {'products': 0, 'differences_at_raw_tolerance': []}
    def compare(left, right, product, field):
        left, right = np.asarray(left), np.asarray(right)
        np.testing.assert_array_equal(np.isfinite(left), np.isfinite(right))
        mask = np.isfinite(left)
        np.testing.assert_array_equal(left[~mask], right[~mask])
        finite = np.abs(left[np.isfinite(left)])
        scale = max(1., finite.max()) if finite.size else 1.
        if not np.allclose(right, left, rtol=1e-10, atol=1e-12*scale, equal_nan=True):
            delta = np.abs(right[mask]-left[mask])
            error = float(delta.max())
            report['differences_at_raw_tolerance'].append(
                dict(product=product, field=field, max_absolute_error=error,
                     error_over_field_scale=error/scale))
    left_dir, right_dir = work/'legacy/cores', work/'onthefly/cores'
    for path in sorted(left_dir.glob('*.nc')):
        left = xr.load_dataset(path)
        right = xr.load_dataset(right_dir/path.name)
        assert set(left.variables) == set(right.variables), path.name
        np.testing.assert_array_equal(left['num'], right['num'])
        for name in left:
            compare(left[name].values, right[name].values, path.name, name)
        la, ra = json.loads(left.attrs['core_attributes']), json.loads(right.attrs['core_attributes'])
        assert la.keys() == ra.keys()
        for name, value in la.items():
            if isinstance(value, (int, float)):
                if name in ('numcrit', 'numcoll', 'num_start'):
                    np.testing.assert_equal(ra[name], value)
                else:
                    compare(value, ra[name], path.name, 'attr:'+name)
            else:
                assert ra[name] == value
        report['products'] += 1
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.test_root.resolve()
    source = root/'M10.J4.B2.P0.N256'
    work = root/'task-driven'
    for path in (source, work):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe test path: {path}')
    legacy_cache = work/'legacy/radial_profile'
    legacy_cache.mkdir(parents=True, exist_ok=True)
    for name in ('radial_profile.p', 'radial_profile.concatenated.p'):
        if not (legacy_cache/name).exists():
            shutil.copy2(source/'radial_profile'/name, legacy_cache/name)

    sims = {}
    for label, legacy in (('legacy', True), ('onthefly', False)):
        s = load_sim.LoadSim(source, legacy=legacy, savdir=str(work/label), load_derived_cores=False)
        assert not s.load_errors, s.load_errors
        assert len(s.rprofs) == len(s.pids) == 15
        assert all(s.tcoll_cores.output_time < s.tcoll_cores.time)
        sims[label] = s
        print(label, 'profiles loaded:', len(s.rprofs), flush=True)
    a, b = sims['legacy'], sims['onthefly']
    raw_names = set(BaseLoadSim.load_rprof(b, b.nums[0]).data_vars)
    for pid in a.pids:
        pd.testing.assert_frame_equal(a._core_tracks[pid], b._core_tracks[pid])
        left, right = a.rprofs[pid], b.rprofs[pid]
        np.testing.assert_array_equal(left.num, right.num)
        for name in raw_names & set(left.data_vars):
            expected = left[name].values
            scale = max(1., np.nanmax(np.abs(expected)))
            np.testing.assert_allclose(right[name].values, expected,
                                       rtol=1e-10, atol=1e-12*scale, equal_nan=True, err_msg=f'{pid}: {name}')
    print('All tracks and raw profiles agree', flush=True)

    methods = ('empirical', 'virial', 'virial0', 'virial1')
    report = {'cores': 15, 'raw_profile_parity': True, 'tasks': {}}
    for label, s in sims.items():
        report['tasks'][label] = {}
        for pid in s.pids:
            tasks.critical_tes(s, pid, overwrite=True)
            s.load_critical_tes(pid)
        print(label, 'TES complete', flush=True)
        tes_stamps = {p.name:p.stat().st_mtime_ns for p in (work/label/'cores').glob('critical_tes.*.nc')}
        for method in methods:
            failures = {}
            for pid in s.pids:
                try:
                    tasks.lagrangian_props(s, pid, method=method, overwrite=True)
                except (ValueError, KeyError) as error:
                    failures[str(pid)] = str(error)
            report['tasks'][label][method] = failures
            print(label, method, 'failures:', failures, flush=True)
        assert tes_stamps == {p.name:p.stat().st_mtime_ns for p in (work/label/'cores').glob('critical_tes.*.nc')}

    # Exact scientific comparison on identical inputs, not merely cache reuse.
    reference = reference_properties()
    s = sims['onthefly']
    old = copy.copy(s)
    old.cores = s._core_tracks.copy()
    report['reference_matches'] = {}
    for method in methods:
        expected = reference(old, method, savdir=work/'empty-legacy-products')
        actual = s.load_core_props(method)
        matched = 0
        for pid, table in expected.items():
            if np.isfinite(table.attrs['numcrit']) and np.isfinite(table.attrs['rcore']):
                try:
                    lagrangian = tools.lagrangian_property(s, table, s.rprofs[pid])
                except (ValueError, KeyError):
                    assert str(pid) in report['tasks']['onthefly'][method]
                    continue
                attrs = {**table.attrs, **lagrangian.attrs}
                table = table.join(lagrangian)
                table.attrs = attrs
                force = table.Fthm + table.Ftrb + table.Fcen + table.Fani - table.Fgrv
                if s.mhd:
                    force += table.Fmag
                table['Fnet'] = force / table.Fgrv
            assert_table_values(actual[pid], table)
            matched += 1
        report['reference_matches'][method] = matched
        print('baseline', method, 'exact matches:', matched, flush=True)

    # Loading must not calculate missing products, even with overwrite requested.
    with patch.object(tools, 'critical_tes_property', side_effect=AssertionError('TES during loading')):
        with patch.object(tools, 'lagrangian_property', side_effect=AssertionError('Lagrangian during loading')):
            with patch.object(load_sim.LoadSim, '_compute_core_props', side_effect=AssertionError('core calculation during loading')):
                for label, legacy in (('legacy', True), ('onthefly', False)):
                    loaded = load_sim.LoadSim(source, legacy=legacy, savdir=str(work/label))
                    assert len(loaded.cores) == 15
    report['read_only_initialization'] = True
    report['source_comparison'] = compare_products(work)
    (work/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
