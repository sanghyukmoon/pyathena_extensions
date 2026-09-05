"""Small tracking, dependency, and persistence fixtures; no simulation required."""
import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from core_formation import rprof_analysis as analysis
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
    s = SimpleNamespace(rprof_outputs=pd.DataFrame(rows).set_index('num'),
                        _rprof_headers=headers, Lbox=1., dx=.01,
                        cs=1., gconst=np.pi, mhd=True, savdir=directory,
                        logger=logging.getLogger('test'), load_errors={},
                        tcoll_cores=pd.DataFrame([dict(num=first+len(positions)-1,
                            time=float(len(positions)-1)+.1,
                            x1=positions[-1], x2=0., x3=0.)], index=[1]))
    s._core_tracks = {1: analysis.track_core(s, 1)}
    return s


class TestTracking(unittest.TestCase):
    def test_periodic_and_preformation(self):
        with tempfile.TemporaryDirectory() as directory:
            s = simulation(directory)
            self.assertEqual(list(s._core_tracks[1].index), [0, 1, 2, 3])
            self.assertEqual(analysis.precollapse_num(s, 2.5), 2)
            self.assertEqual(analysis.precollapse_num(s, 2.), 2)
            with self.assertRaisesRegex(ValueError, 'before collapse'):
                analysis.precollapse_num(s, -.1)

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
                analysis.track_core(s, 1)


class TestProfileCache(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.s = simulation(self.directory.name)
        def reader(path, center_ids):
            return self.s._rprof_headers[int(Path(path).stem)].sel(center_id=center_ids)
        self.reader = patch.object(analysis, 'read_radial_profile', side_effect=reader)
        self.mock_reader = self.reader.start()
        self.addCleanup(self.reader.stop)
        self.path = Path(self.directory.name, 'on_the_fly/core_rprof.par1.nc')

    def test_roundtrip_reuse_and_overwrite(self):
        original = analysis.load_core_rprof(self.s, 1)
        self.assertEqual(self.mock_reader.call_count, 4)
        self.assertEqual(original.rho.dims, ('t', 'r'))
        self.assertEqual(original.sel(num=2).center_id.item(), 12)
        cached = analysis.load_core_rprof(self.s, 1)
        xr.testing.assert_identical(original, cached)
        self.assertEqual(self.mock_reader.call_count, 4)
        analysis.load_core_rprof(self.s, 1, overwrite=True)
        self.assertEqual(self.mock_reader.call_count, 8)

    def test_cache_false_never_writes(self):
        analysis.load_core_rprof(self.s, 1, cache=False, overwrite=True)
        self.assertFalse(self.path.exists())

    def test_raw_does_not_replace_derived(self):
        analysis.load_core_rprof(self.s, 1)
        stamp = self.path.stat().st_mtime_ns
        raw = analysis.load_core_rprof(self.s, 1, derived=False, overwrite=True)
        self.assertNotIn('menc', raw)
        self.assertEqual(stamp, self.path.stat().st_mtime_ns)

    def test_invalidation(self):
        analysis.load_core_rprof(self.s, 1)
        self.s.cs = 2.
        analysis.load_core_rprof(self.s, 1)
        self.assertEqual(self.mock_reader.call_count, 8)
        Path(self.directory.name, '0.rprof').touch()
        analysis.load_core_rprof(self.s, 1)
        self.assertEqual(self.mock_reader.call_count, 12)

    def test_corrupt_metadata_rebuilt(self):
        analysis.load_core_rprof(self.s, 1)
        data = xr.load_dataset(self.path)
        data['cycle'][0] = -100
        analysis.write_netcdf(data, self.path)
        result = analysis.load_core_rprof(self.s, 1)
        self.assertEqual(result.cycle[0].item(), 0)
        self.assertEqual(self.mock_reader.call_count, 8)

    def test_atomic_failure_preserves_existing(self):
        analysis.load_core_rprof(self.s, 1)
        before = self.path.read_bytes()
        with patch.object(xr.Dataset, 'to_netcdf', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                analysis.load_core_rprof(self.s, 1, overwrite=True)
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

    def test_time_matching_not_number_matching(self):
        s = SimpleNamespace(rprof_outputs=pd.DataFrame({'time': [1.]}, index=[15]),
                            particle_outputs=pd.DataFrame({'time': [1.]}, index=[104]))
        self.assertEqual(analysis.match_particles(s, 15).name, 104)
        s.particle_outputs.loc[104, 'time'] = 1.000001
        with self.assertRaisesRegex(ValueError, 'found 0'):
            analysis.match_particles(s, 15)

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
        s = SimpleNamespace(pids=[1, 2], _core_tracks={}, load_errors={},
                            logger=logging.getLogger('test'))
        with patch.object(analysis, 'particle_outputs', return_value=pd.DataFrame()):
            self.assertEqual(analysis.update_core_props(s, 'virial0'), {})
        self.assertIn('Peer trajectories', s.load_errors[1]['derived:virial0'])

    def test_particle_gap_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            s = SimpleNamespace(basedir=directory, problem_id='test')
            for number in (1, 3):
                Path(directory, f'test.out3.{number:05d}.par0.parbin').touch()
            with self.assertRaisesRegex(ValueError, 'Gap'):
                analysis.particle_outputs(s)
            Path(directory, 'test.out3.00002.par0.parbin').touch()
            Path(directory, 'parbin').mkdir()
            Path(directory, 'parbin/test.out3.00002.par0.parbin').touch()
            with patch.object(analysis, 'read_parbin', return_value={'time': 1.}):
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    analysis.particle_outputs(s)

    def test_obsolete_tasks_do_not_write(self):
        from core_formation import tasks
        s = SimpleNamespace(legacy=False)
        with self.assertRaisesRegex(ValueError, 'already exist'):
            tasks.save_minima(s)
        with self.assertRaisesRegex(ValueError, 'already exist'):
            tasks.radial_profile(s)

    def test_valid_core_continues_with_missing_particle_time_for_another(self):
        with tempfile.TemporaryDirectory() as directory:
            s = simulation(directory)
            s.pids = [1, 2]
            s._core_tracks[2] = s._core_tracks[1].iloc[:2].copy()
            s._core_tracks[2].attrs['pid'] = 2
            s._get_fparhst = lambda pid: Path(directory, '0.rprof')
            def reader(path, center_ids):
                return s._rprof_headers[int(Path(path).stem)].sel(center_id=center_ids)
            with patch.object(analysis, 'read_radial_profile', side_effect=reader):
                s.rprofs = {pid: analysis.load_core_rprof(s, pid, cache=False) for pid in s.pids}
            particle_index = pd.DataFrame(dict(time=[0., 1.], paths=[[], []],
                                               kind=['parbin', 'parbin']), index=[100, 101])
            computed = []
            def compute(method, savdir, pids):
                computed.extend(pids)
                return {pid: s._core_tracks[pid].copy() for pid in pids}
            s._compute_core_props = compute
            with patch.object(analysis, 'particle_outputs', return_value=particle_index):
                result = analysis.update_core_props(s, 'virial0', cache=False)
            self.assertEqual(list(result), [2])
            self.assertEqual(computed, [2])
            self.assertIn('particle snapshot', s.load_errors[1]['derived:virial0'])


if __name__ == '__main__':
    unittest.main()
