"""Small tracking and explicit-cache fixtures; no simulation required."""
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from core_formation.load_sim import LoadSim, LoadSimBase
from core_formation import rprof_analysis as analysis
from test_rprof_derived import raw_profiles


def simulation(directory, positions=(.90, .95, .00, .05), first=1):
    s = LoadSim(None, legacy=False)
    s.nums = s.nums_rprof = list(range(first, first+len(positions)))
    s.times = {num:float(i) for i,num in enumerate(s.nums)}
    s.minima = {num:np.array([num+10], dtype=np.uint64) for num in s.nums}
    s.Lbox, s.dx, s.cs, s.gconst, s.mhd = 1., .01, 1., np.pi, True
    s.u = SimpleNamespace(Myr=1.)
    s.dt_output = {'rprof': 1., 'hdf5': 1.}
    s._basedir = directory or ''
    s.savdir = directory
    s.logger = logging.getLogger('test')
    s.cores_dict, s.rprofs = {}, {}
    s.pids = [1]
    s.flatindex_to_cartesian = lambda lid: (positions[lid-10-first], 0., 0.)
    s.tcoll_cores = pd.DataFrame([dict(num=s.nums[-1], time=len(positions)-1+.1,
                                     x1=positions[-1], x2=0., x3=0.)], index=[1])
    s._core_tracks = {1:s._track_core(1)}
    return s


class TestTracking(unittest.TestCase):
    def test_periodic_and_continuity(self):
        s = simulation(None)
        self.assertEqual(list(s._core_tracks[1].index), [1, 2, 3, 4])
        self.assertNotIn('cycle', s._core_tracks[1])
        s = simulation(None, (.4, .95, .0, .05), first=17)
        self.assertEqual(list(s._core_tracks[1].index), [18, 19, 20])
        self.assertEqual(s._core_tracks[1].attrs['stop_reason'], 'continuity')

    def test_short_empty_and_preformation(self):
        s = simulation(None, (.1,))
        self.assertTrue(s._core_tracks[1].attrs['track_failed'])
        s.minima[1] = np.array([], dtype=np.uint64)
        with self.assertRaisesRegex(ValueError, 'No minimum'):
            s._track_core(1)
        s = simulation(None)
        history = pd.DataFrame([dict(time=2., age=0., x1=0., x2=0., x3=0.,
                                     v1=0., v2=0., v3=0.)])
        with patch.object(s, 'load_parhst', return_value=history):
            self.assertEqual(s._load_tcoll_cores().loc[1, 'num'], 2)
            history.loc[0, 'time'] = 2.5
            self.assertEqual(s._load_tcoll_cores().loc[1, 'num'], 3)
            history.loc[0, 'time'] = -.1
            self.assertTrue(s._load_tcoll_cores().empty)
            self.assertIn('collapse', s.load_errors[1])

    def test_modes_share_cutoff_and_strict_selection(self):
        s = simulation(None)
        expected = s._track_core(1)
        s.legacy = True
        pd.testing.assert_frame_equal(s._track_core(1), expected)
        s.tcoll_cores.loc[1, 'time'] = 4.6
        self.assertEqual(list(s._track_core(1).index), [3, 4])
        history = pd.DataFrame([dict(time=2., age=0., x1=0., x2=0., x3=0.,
                                     v1=0., v2=0., v3=0.)])
        for mode in (True, False):
            s.legacy = mode
            with patch.object(s, 'load_parhst', return_value=history):
                self.assertEqual(s._load_tcoll_cores().loc[1, 'num'], 2)


class TestProfileCache(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.s = s = simulation(self.directory.name)
        def reader(num, center_ids=None, metadata_only=False):
            data = raw_profiles().expand_dims(center_id=s.minima[num])
            pos = s.flatindex_to_cartesian(int(s.minima[num][0]))
            data = data.assign_coords(**{c:('center_id',[p]) for c,p in zip(('x1','x2','x3'),pos)})
            data.attrs.update(num=num, time=s.times[num], cycle=num*10)
            return data if center_ids is None else data.sel(center_id=center_ids)
        self.reader_patch = patch.object(LoadSimBase, 'load_rprof', side_effect=reader)
        self.reader = self.reader_patch.start()
        self.addCleanup(self.reader_patch.stop)
        self.path = Path(self.directory.name, 'on_the_fly/core_rprof.par1.nc')

    def test_roundtrip_and_explicit_overwrite(self):
        original = self.s.load_core_rprof(1)
        self.assertEqual(self.reader.call_count, 4)
        self.assertIn('menc', original)
        self.assertNotIn('cycle', original)
        xr.testing.assert_identical(original, self.s.load_core_rprof(1))
        self.assertEqual(self.reader.call_count, 4)
        self.s.cs = 2.  # Explicit refresh, not automatic invalidation.
        xr.testing.assert_identical(original, self.s.load_core_rprof(1))
        self.s.load_core_rprof(1, overwrite=True)
        self.assertEqual(self.reader.call_count, 8)

    def test_cache_false(self):
        self.s.load_core_rprof(1, cache=False, overwrite=True)
        self.assertFalse(self.path.exists())
        self.s.load_core_rprof(1)
        stamp = self.path.stat().st_mtime_ns
        self.s.load_core_rprof(1, cache=False)
        self.assertEqual(stamp, self.path.stat().st_mtime_ns)

    def test_corrupt_cache_is_not_rebuilt(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b'not netcdf')
        with self.assertRaisesRegex(ValueError, 'overwrite=True'):
            self.s.load_core_rprof(1)
        self.assertEqual(self.reader.call_count, 0)
        self.s.load_core_rprof(1, overwrite=True)
        self.assertEqual(self.reader.call_count, 4)

    def test_atomic_failure_preserves_existing(self):
        self.s.load_core_rprof(1)
        before = self.path.read_bytes()
        with patch.object(xr.Dataset, 'to_netcdf', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.s.load_core_rprof(1, overwrite=True)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob('*.tmp')), [])

    def test_no_legacy_or_raw_selector(self):
        with self.assertRaises(TypeError):
            self.s.load_core_rprof(1, derived=False)
        with self.assertRaises(TypeError):
            self.s.load_rprof(0, derived=False)
        self.s.legacy = True
        with self.assertRaisesRegex(ValueError, 's.rprofs'):
            self.s.load_core_rprof(1)

    def test_property_cache_has_no_source_dependencies(self):
        s = self.s
        s.load_core_rprof(1, cache=False)
        with patch.object(s, '_compute_core_props', return_value={1:s._core_tracks[1].copy()}) as compute:
            original = s.update_core_props('virial0')[1]
            self.assertEqual(compute.call_count, 1)
            with patch.object(s, 'load_par', side_effect=AssertionError('particle read')):
                s._core_tracks, s.rprofs = {}, {}
                pd.testing.assert_frame_equal(original.astype(float), s.update_core_props('virial0')[1].astype(float), check_index_type=False)
                self.assertEqual(compute.call_count, 1)
            self.assertEqual(s.update_core_props('virial0', overwrite=True), {})
            self.assertEqual(compute.call_count, 1)

    def test_missing_profile_gates_computation(self):
        with patch.object(self.s, '_compute_core_props') as compute:
            self.assertEqual(self.s.update_core_props('virial0'), {})
            compute.assert_not_called()

    def test_property_cache_false_and_overwrite(self):
        s = self.s
        s.load_core_rprof(1, cache=False)
        path = Path(s.savdir, 'on_the_fly/core_props.virial0.par1.nc')
        with patch.object(s, '_compute_core_props', return_value={1:s._core_tracks[1].copy()}) as compute:
            s.cache = False
            s.update_core_props('virial0')
            self.assertFalse(path.exists())
            s.cache = True
            s.update_core_props('virial0')
            stamp = path.stat().st_mtime_ns
            s.cache = False
            s.update_core_props('virial0', overwrite=True)
            self.assertEqual(path.stat().st_mtime_ns, stamp)
            s.cache = True
            s.update_core_props('virial0', overwrite=True)
            self.assertEqual(compute.call_count, 4)

    def test_corrupt_property_cache_needs_explicit_overwrite(self):
        s = self.s
        s.load_core_rprof(1, cache=False)
        path = Path(s.savdir, 'on_the_fly/core_props.virial0.par1.nc')
        path.parent.mkdir()
        path.write_bytes(b'not netcdf')
        with patch.object(s, '_compute_core_props', return_value={1:s._core_tracks[1].copy()}) as compute:
            self.assertEqual(s.update_core_props('virial0'), {})
            self.assertIn('overwrite=True', s.load_errors[1]['derived:virial0'])
            compute.assert_not_called()
            self.assertIn(1, s.update_core_props('virial0', overwrite=True))
            self.assertNotIn('derived:virial0', s.load_errors[1])
            compute.assert_called_once()


class TestSerialization(unittest.TestCase):
    def test_large_integer_and_attributes(self):
        frame = pd.DataFrame({'leaf_id':[2**54+1], 'time':[np.nan]}, index=pd.Index([5], name='num'))
        frame.attrs = {'track_failed':False}
        result = analysis.dataset_to_frame(analysis.frame_to_dataset(frame))
        self.assertEqual(result.leaf_id.iloc[0], 2**54+1)
        self.assertEqual(result.attrs, frame.attrs)


if __name__ == '__main__':
    unittest.main()
