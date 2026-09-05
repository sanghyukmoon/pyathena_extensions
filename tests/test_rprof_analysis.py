"""Small tracking, dependency, and persistence fixtures; no simulation required."""
import ast
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from core_formation import rprof_analysis as analysis
from core_formation import load_sim as loader
from core_formation.load_sim import LoadSim

class SyntheticSimulation(LoadSim):
    @property
    def rprof_outputs(self):
        return self._fixture_outputs


def simulation_state(**kwargs):
    s = SyntheticSimulation(None, legacy=False)
    s.cores_dict = {}
    s.rprofs = {}
    for name, value in kwargs.items():
        if name == 'rprof_outputs':
            name = '_fixture_outputs'
        elif isinstance(getattr(type(s), name, None), property):
            name = '_'+name
        setattr(s, name, value)
    return s

from test_rprof_derived import raw_profiles


def simulation(directory, positions=(.90, .95, .00, .05), first=0):
    headers, rows = {}, []
    for offset, position in enumerate(positions):
        num = first+offset
        source = raw_profiles().expand_dims(center_id=[np.uint64(num+10)])
        source = source.assign_coords(x1=('center_id', [position]),
                                      x2=('center_id', [0.]), x3=('center_id', [0.]))
        source.attrs.update(num=num, time=float(offset), cycle=num*10)
        headers[num] = source
        path = Path(directory, f'{num}.rprof')
        path.touch()
        rows.append(dict(num=num, time=float(offset), cycle=num*10, path=path))
    s = simulation_state(rprof_outputs=pd.DataFrame(rows).set_index('num'),
                        _rprof_headers=headers, Lbox=1., dx=.01,
                        cs=1., gconst=np.pi, mhd=True, savdir=directory,
                        logger=logging.getLogger('test'), load_errors={},
                        tcoll_cores=pd.DataFrame([dict(num=first+len(positions)-1,
                            time=float(len(positions)-1)+.1,
                            x1=positions[-1], x2=0., x3=0.)], index=[1]))
    s._core_tracks = {1: s._track_core(1)}
    return s


class TestTracking(unittest.TestCase):
    def test_periodic_and_preformation(self):
        with tempfile.TemporaryDirectory() as directory:
            s = simulation(directory)
            self.assertEqual(list(s._core_tracks[1].index), [0, 1, 2, 3])
            self.assertEqual(s._precollapse_num(2.5), 2)
            self.assertEqual(s._precollapse_num(2.), 2)
            with self.assertRaisesRegex(ValueError, 'before collapse'):
                s._precollapse_num(-.1)

    def test_continuity_excludes_failed_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            s = simulation(directory, (.4, .95, .0, .05), first=17)
            track = s._core_tracks[1]
            self.assertEqual(list(track.index), [18, 19, 20])
            self.assertEqual(track.attrs['stop_reason'], 'continuity')

    def test_short_and_empty_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            s = simulation(directory, (.1,))
            self.assertTrue(s._core_tracks[1].attrs['track_failed'])
            s._rprof_headers[0] = s._rprof_headers[0].isel(center_id=[])
            with self.assertRaisesRegex(ValueError, 'No minimum'):
                s._track_core(1)


class TestProfileCache(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.s = simulation(self.directory.name)
        def reader(path, center_ids):
            return self.s._rprof_headers[int(Path(path).stem)].sel(center_id=center_ids)
        self.reader = patch.object(loader, 'read_radial_profile', side_effect=reader)
        self.mock_reader = self.reader.start()
        self.addCleanup(self.reader.stop)
        self.path = Path(self.directory.name, 'on_the_fly/core_rprof.par1.nc')

    def test_roundtrip_reuse_and_overwrite(self):
        original = self.s._load_core_rprof_onthefly(1)
        self.assertEqual(self.mock_reader.call_count, 4)
        self.assertEqual(original.rho.dims, ('t', 'r'))
        self.assertEqual(original.sel(num=2).center_id.item(), 12)
        cached = self.s._load_core_rprof_onthefly(1)
        xr.testing.assert_identical(original, cached)
        self.assertEqual(self.mock_reader.call_count, 4)
        self.s._load_core_rprof_onthefly(1, overwrite=True)
        self.assertEqual(self.mock_reader.call_count, 8)

    def test_cache_false_never_writes(self):
        self.s._load_core_rprof_onthefly(1, cache=False, overwrite=True)
        self.assertFalse(self.path.exists())

    def test_raw_does_not_replace_derived(self):
        self.s._load_core_rprof_onthefly(1)
        stamp = self.path.stat().st_mtime_ns
        raw = self.s._load_core_rprof_onthefly(1, derived=False, overwrite=True)
        self.assertNotIn('menc', raw)
        self.assertEqual(stamp, self.path.stat().st_mtime_ns)

    def test_invalidation(self):
        self.s._load_core_rprof_onthefly(1)
        self.s.cs = 2.
        self.s._load_core_rprof_onthefly(1)
        self.assertEqual(self.mock_reader.call_count, 8)
        Path(self.directory.name, '0.rprof').touch()
        self.s._load_core_rprof_onthefly(1)
        self.assertEqual(self.mock_reader.call_count, 12)

    def test_corrupt_metadata_rebuilt(self):
        self.s._load_core_rprof_onthefly(1)
        data = xr.load_dataset(self.path)
        data['cycle'][0] = -100
        analysis.write_netcdf(data, self.path)
        result = self.s._load_core_rprof_onthefly(1)
        self.assertEqual(result.cycle[0].item(), 0)
        self.assertEqual(self.mock_reader.call_count, 8)

    def test_atomic_failure_preserves_existing(self):
        self.s._load_core_rprof_onthefly(1)
        before = self.path.read_bytes()
        with patch.object(xr.Dataset, 'to_netcdf', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.s._load_core_rprof_onthefly(1, overwrite=True)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob('*.tmp')), [])


class TestParticlesAndTables(unittest.TestCase):
    def test_explicit_hdf5_time_matching(self):
        from core_formation.load_sim import LoadSim
        s = LoadSim(None, legacy=False)
        s.nums = [100, 101]
        s.core_num_to_time = lambda num: 1.
        s.num_to_time = lambda num: {100: 0., 101: 1.}[num]
        self.assertEqual(s.hdf5_num_for_core(15), 101)
        s.nums = None
        with self.assertRaisesRegex(ValueError, 'found 0'):
            s.hdf5_num_for_core(15)

    def test_table_roundtrip(self):
        frame = pd.DataFrame({'leaf_id': [2**54+1], 'time': [np.array(np.nan)]},
                             index=pd.Index([5], name='num'), dtype=object)
        frame.attrs = dict(track_failed=False, stop_reason='continuity', numcoll=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'core.nc')
            analysis.write_netcdf(analysis.frame_to_dataset(frame), path)
            result = analysis.dataset_to_frame(xr.load_dataset(path))
            self.assertEqual(result.leaf_id.iloc[0], 2**54+1)
            self.assertTrue(np.isnan(result.time.iloc[0]))
            self.assertEqual(result.attrs, frame.attrs)

    def test_missing_peer_gates_all_dependents(self):
        s = simulation_state(pids=[1, 2], _core_tracks={}, load_errors={},
                            logger=logging.getLogger('test'))
        self.assertEqual(s._update_core_props_onthefly('virial0'), {})
        self.assertIn('Peer trajectories', s.load_errors[1]['derived:virial0'])


    def test_valid_core_continues_with_missing_particle_file_for_another(self):
        with tempfile.TemporaryDirectory() as directory:
            s = simulation(directory)
            s.pids = [1, 2]
            s._core_tracks[2] = s._core_tracks[1].iloc[:2].copy()
            s._core_tracks[2].attrs['pid'] = 2
            s._get_fparhst = lambda pid: Path(directory, '0.rprof')
            def reader(path, center_ids):
                return s._rprof_headers[int(Path(path).stem)].sel(center_id=center_ids)
            with patch.object(loader, 'read_radial_profile', side_effect=reader):
                s.rprofs = {pid: s._load_core_rprof_onthefly(pid, cache=False) for pid in s.pids}
            def particle_path(num):
                if num > 1:
                    raise FileNotFoundError(f'Particle snapshot {num} not found')
                return Path(directory, f'{num}.rprof')
            s._particle_path = particle_path
            computed = []
            def compute(method, savdir, pids):
                computed.extend(pids)
                return {pid: s._core_tracks[pid].copy() for pid in pids}
            s._compute_core_props = compute
            result = s._update_core_props_onthefly('virial0', cache=False)
            self.assertEqual(list(result), [2])
            self.assertEqual(computed, [2])
            self.assertIn('Particle snapshot', s.load_errors[1]['derived:virial0'])

    def test_hdf5_to_core_mapping(self):
        s = simulation_state(rprof_outputs=pd.DataFrame({'time': [0., 1.]}, index=[14, 15]))
        s.num_to_time = lambda num: 1.
        self.assertEqual(s.core_num_for_hdf5(101), 15)
        s.num_to_time = lambda num: 2.
        with self.assertRaisesRegex(ValueError, 'found 0'):
            s.core_num_for_hdf5(101)
        s._fixture_outputs.loc[16] = [1.]
        s.num_to_time = lambda num: 1.
        with self.assertRaisesRegex(ValueError, 'found 2'):
            s.core_num_for_hdf5(101)
        s.legacy = True
        self.assertEqual(s.core_num_for_hdf5(101), 101)

    def test_particle_resolver_and_reader_use_same_number(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind in ('parbin', 'partab'):
                with self.subTest(kind=kind):
                    path = Path(directory, f'test.{kind}')
                    path.touch()
                    s = simulation_state(files={kind: {'par0': [str(path)]}})
                    setattr(s, kind+'_outid', 3)
                    with patch.object(s, '_get_f'+kind, return_value=str(path)) as resolve:
                        self.assertEqual(s._particle_path(15), path)
                        resolve.assert_called_once_with(3, 'par0', num=15)
                    empty = pd.DataFrame(columns=['x1', 'x2', 'x3', 'mass'])
                    with patch.object(s, 'load_'+kind, return_value=empty) as read:
                        self.assertTrue(s.load_par(15).empty)
                        read.assert_called_once_with(15)
                    with patch.object(s, '_get_f'+kind, return_value=str(path)+'missing'):
                        with self.assertRaisesRegex(FileNotFoundError, 'snapshot 15'):
                            s._particle_path(15)
                    with patch.object(s, 'load_'+kind, side_effect=ValueError('truncated')):
                        with self.assertRaisesRegex(ValueError, 'truncated'):
                            s.load_par(15)

    def test_core_calculation_requests_profile_number(self):
        s = simulation_state(_core_tracks={1: pd.DataFrame(index=[15])}, rprofs={1: None})
        s._core_tracks[1].attrs['track_failed'] = False
        with patch.object(s, 'load_par', side_effect=ValueError('stop after read')) as read:
            with self.assertRaisesRegex(ValueError, 'stop after read'):
                s._compute_core_props('virial0', None, pids=[1])
            read.assert_called_once_with(15)


class TestDerivedDependencies(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.s = s = simulation(self.directory.name)
        s.pids = [1, 2]
        s._core_tracks[2] = s._core_tracks[1].copy()
        s._core_tracks[2].attrs['pid'] = 2
        self.particle = Path(self.directory.name, 'particle.parbin')
        self.particle.touch()
        s._get_fparhst = lambda pid: Path(self.directory.name, '0.rprof')
        s._particle_path = lambda num: self.particle
        def reader(path, center_ids):
            return s._rprof_headers[int(Path(path).stem)].sel(center_id=center_ids)
        read_patch = patch.object(loader, 'read_radial_profile', side_effect=reader)
        read_patch.start()
        self.addCleanup(read_patch.stop)
        for pid in s.pids:
            s.load_core_rprof(pid, cache=False)
        self.compute = patch.object(s, '_compute_core_props', side_effect=lambda method, savdir, pids:
                                    {pid: s._core_tracks[pid].copy() for pid in pids})
        self.computed = self.compute.start()
        self.addCleanup(self.compute.stop)

    def test_operation_reuse_and_explicit_refresh(self):
        s = self.s
        shared = {'profiles': {}, 'particles': {}}
        with patch.object(s, '_profile_fingerprint', wraps=s._profile_fingerprint) as profiles:
            with patch.object(s, '_particle_path', wraps=s._particle_path) as particles:
                for method in ('empirical', 'virial', 'virial0', 'virial1'):
                    s._update_core_props_onthefly(method, cache=False, dependencies=shared)
                self.assertEqual(profiles.call_count, 2)
                self.assertEqual(particles.call_count, 4)
                s.update_core_props('virial0')
                self.assertEqual(profiles.call_count, 4)
                self.assertEqual(particles.call_count, 8)

    def test_initialization_shares_dependencies(self):
        s = self.s
        tracks = s._core_tracks.copy()
        with patch.object(s, '_load_tcoll_cores', return_value=s.tcoll_cores):
            with patch.object(s, '_track_core', side_effect=lambda pid: tracks[pid]):
                with patch.object(s, '_profile_fingerprint', wraps=s._profile_fingerprint) as profiles:
                    with patch.object(s, '_particle_path', wraps=s._particle_path) as particles:
                        s._initialize_onthefly(method='virial0', load_rprofs=True,
                                              load_derived_cores=True, overwrite=False)
                        self.assertEqual(profiles.call_count, 4)  # two profile loads + two dependencies
                        self.assertEqual(particles.call_count, 4)
        self.assertEqual(len(s.cores_dict), 4)
        self.assertFalse(hasattr(s, 'load_timings'))

    def test_cached_properties_require_existing_particles(self):
        s = self.s
        self.assertEqual(len(s.update_core_props('virial0')), 2)
        calls = self.computed.call_count
        self.assertEqual(len(s.update_core_props('virial0')), 2)
        self.assertEqual(self.computed.call_count, calls)
        self.particle.unlink()
        self.assertEqual(s.update_core_props('virial0'), {})
        self.assertEqual(self.computed.call_count, calls)
        self.assertIn('derived:virial0', s.load_errors[1])
        self.particle.write_bytes(b'replaced particle fixture')
        self.assertEqual(len(s.update_core_props('virial0')), 2)
        self.assertEqual(self.computed.call_count, calls+2)
        self.assertNotIn('derived:virial0', s.load_errors[1])

    def test_unreadable_particles_gate_properties(self):
        self.computed.side_effect = ValueError('Truncated particle data')
        self.assertEqual(self.s.update_core_props('virial0'), {})
        self.assertIn('Truncated', self.s.load_errors[1]['derived:virial0'])

    def test_missing_profile_never_computes_dependent_properties(self):
        del self.s.rprofs[1]
        result = self.s.update_core_props('virial0')
        self.assertEqual(list(result), [2])
        self.computed.assert_called_once_with('virial0', self.s.savdir, pids=[2])


class TestLegacyTaskDriver(unittest.TestCase):
    def test_script_has_no_new_mode_or_scheduler_side_effect(self):
        from core_formation import tasks
        self.assertFalse(hasattr(tasks, 'core_profiles'))
        self.assertFalse(hasattr(tasks, 'core_properties'))
        path = Path(loader.__file__).with_name('do_tasks.py')
        source = path.read_text()
        self.assertNotIn('--on-the-fly', source)
        self.assertNotIn('--legacy', source)
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'write_slurm_script')
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory, 'test.slurm')
            namespace = dict(Path=Path, jobid='test', SCRIPT_PATH=str(script))
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
            namespace['write_slurm_script']('test', ['save_minima'], False)
            text = script.read_text()
            self.assertIn('save_minima --runbyslurm', text)
            self.assertNotIn('--on-the-fly', text)
            self.assertNotIn('--legacy', text)


if __name__ == '__main__':
    unittest.main()
