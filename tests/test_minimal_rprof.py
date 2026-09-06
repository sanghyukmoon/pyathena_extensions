"""Regression tests for the baseline-preserving profile source integration."""
import logging
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from core_formation.load_sim import LoadSim, LoadSimBase, TimingReader
from core_formation import tools

TEST_ROOT = Path('/scratch/gpfs/sm69/onthefly-rprof-test/minimal-rebuild')


def raw_profile():
    r = np.linspace(0, 1, 9)
    fields = {'rho': np.ones(9), 'gacc1_mw': -r,
              'xgx_mw': -r*r/3, 'ygy_mw': -r*r/3, 'zgz_mw': -r*r/3,
              'phi_B': r*r, 'vshell': np.ones(9)}
    for axis in ('x', 'y', 'z'):
        fields['Ldens_'+axis] = r
        fields['bhat_'+axis] = np.ones(9)/np.sqrt(3)
    for axis in (1, 2, 3):
        fields[f'b{axis}_sq'] = np.ones(9)
    for axis in (1, 2, 3, 'x', 'y', 'z'):
        fields[f'vel{axis}_mw'] = np.zeros(9)
        fields[f'vel{axis}_sq_mw'] = np.ones(9)
    return xr.Dataset({name: ('r', data) for name, data in fields.items()}, coords={'r': r})


class Timelines(unittest.TestCase):
    def initialize(self, *, legacy=True, hdf5=(0, 1, 2), particle='parbin',
                   nums=(0, 1, 2), multiple=False):
        def base(s, *args, **kwargs):
            s.ff = SimpleNamespace(nums_hdf5={'cons': list(hdf5)})
            s._par = {'problem': {'Mach': 10},
                     'configure': {'Magnetic_fields': 'ON'},
                     'output1': {'file_type': 'hdf5', 'variable': 'cons', 'dt': .1}}
            if multiple:
                s.par['output2'] = s.par['output1'].copy()
            s._basedir = s._savdir = '/scratch/gpfs/sm69/onthefly-rprof-test/minimal-rebuild'
            s._basename = s._problem_id = 'synthetic'
            s._files = {particle: {}}
            setattr(s, 'nums_'+particle, {'par0': list(nums)})
            s.nums_rprof = list(nums)
            s._domain = {'Lx': np.ones(3), 'dx': np.ones(3)/16}
            s.logger = logging.getLogger('minimal-test')
            s.pids = []
        def metadata(s, num, **kwargs):
            self.assertEqual(kwargs, {'metadata_only': True})
            return xr.Dataset(coords={'center_id': np.array([num+10], dtype=np.uint64)},
                              attrs={'time': num*.1+.001})
        with patch.object(LoadSimBase, '__init__', base), \
             patch.object(TimingReader, '__init__', return_value=None), \
             patch.object(LoadSim, 'load_hdf5', side_effect=lambda n, **kw: {'Time': n*.1+.001}), \
             patch.object(LoadSim, 'load_par', side_effect=lambda n, **kw: {'time': n*.1+.001}), \
             patch.object(LoadSim, 'load_rprof', metadata):
            return LoadSim('synthetic', legacy=legacy, load_derived_cores=False)

    def test_hdf5(self):
        s = self.initialize()
        self.assertEqual(s.nums, [0, 1, 2])
        self.assertEqual(s.times, s.times_hdf5)

    def test_particle_fallbacks(self):
        for particle in ('parbin', 'partab'):
            with self.subTest(particle=particle):
                s = self.initialize(hdf5=(), particle=particle)
                self.assertEqual(s.nums_hdf5, [])
                self.assertEqual(s.times_hdf5, {})
                self.assertAlmostEqual(s.times[2], .201)

    def test_rprof_minima(self):
        s = self.initialize(legacy=False)
        np.testing.assert_array_equal(s.minima[2], [12])
        self.assertEqual(s.minima[2].dtype, np.uint64)
        self.assertAlmostEqual(s.times[2], .201)

    def test_gaps(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy), self.assertRaisesRegex(ValueError, 'Gap'):
                self.initialize(legacy=legacy, hdf5=(0, 2), nums=(0, 2))

    def test_multiple_hdf5(self):
        with self.assertRaisesRegex(ValueError, 'one cons'):
            self.initialize(multiple=True)

    def test_strict_collapse(self):
        s = LoadSim()
        s.pids = [1]
        s.nums = [0, 1, 2]
        s.times = {0: .001, 1: .101, 2: .201}
        row = dict(x1=0, x2=0, x3=0, v1=0, v2=0, v3=0, time=.201, age=0.)
        s.load_parhst = lambda pid: pd.DataFrame([row])
        self.assertEqual(s._load_tcoll_cores().loc[1].num, 1)
        row['time'] = .2
        self.assertEqual(s._load_tcoll_cores().loc[1].num, 1)
        row['time'] = .001
        with self.assertRaisesRegex(ValueError, 'strictly before'):
            s._load_tcoll_cores()


class Profiles(unittest.TestCase):
    def test_assembly_derived_parity_and_cache(self):
        for mhd in (False, True):
            with self.subTest(mhd=mhd), tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
                root = Path(tmp)
                legacy_dir = root/'legacy'
                legacy_dir.mkdir()
                nums, times = [1, 2], [.101, .201]
                raw = raw_profile().expand_dims(t=times).assign_coords(num=('t', nums))
                with (legacy_dir/'radial_profile.concatenated.p').open('wb') as f:
                    pickle.dump({7: raw}, f)
                s = LoadSim()
                s.logger = logging.getLogger('minimal-test')
                s._savdir = tmp
                s.mhd = mhd
                s.cores = {7: pd.DataFrame({'leaf_id': [123, 456]}, index=nums)}
                s.times = dict(zip(nums, times))
                s.legacy = True
                expected = s._load_radial_profiles(savdir=legacy_dir)[7]
                s.legacy = False
                def snapshot(num, center_ids):
                    self.assertEqual(center_ids, [s.cores[7].loc[num].leaf_id])
                    return raw_profile().expand_dims(center_id=center_ids)
                with patch.object(s, 'load_rprof', side_effect=snapshot) as reader:
                    actual = s._load_radial_profiles(savdir=root/'onthefly')[7]
                    self.assertEqual(reader.call_count, 2)
                    xr.testing.assert_identical(actual, expected)
                    np.testing.assert_allclose(actual.menc.isel(t=0), 4*np.pi*raw.r**3/3, atol=1e-15)
                    self.assertEqual(actual.sel(num=1).sizes['r'], 9)
                    cached = s._load_radial_profiles(savdir=root/'onthefly')[7]
                    self.assertEqual(reader.call_count, 2)
                    xr.testing.assert_identical(cached, actual)
                    s._load_radial_profiles(savdir=root/'onthefly', force_override=True)
                    self.assertEqual(reader.call_count, 4)


class Tracking(unittest.TestCase):
    def test_shared_tracker_uses_minima_and_times(self):
        s = LoadSim()
        s._basename = 'synthetic'
        s.nums = [1, 2, 3]
        s.times = {1: .101, 2: .201, 3: .301}
        s.minima = {1: [1, 9], 2: [2, 9], 3: [3, 9]}
        s.tcoll_cores = pd.DataFrame({'num': [3]}, index=[1])
        s.distance_between = lambda a, b: abs(a-b)
        with patch.object(tools, 'find_tcoll_core', return_value=3):
            legacy = tools.track_cores(s, 1)
            s.legacy = False
            nonlegacy = tools.track_cores(s, 1)
        pd.testing.assert_frame_equal(legacy, nonlegacy)
        self.assertEqual(legacy.leaf_id.tolist(), [1, 2, 3])
        self.assertEqual(legacy.time.tolist(), [.101, .201, .301])

    def test_gradient_is_periodic_before_cropping(self):
        n = 16
        x = (np.arange(n)+.5)/n
        phi = np.broadcast_to(np.sin(2*np.pi*x), (n, n, n)).copy()
        ds = xr.Dataset({'phi': (('z', 'y', 'x'), phi)},
                        coords={'x': x, 'y': x, 'z': x})
        s = SimpleNamespace(dx=1/n)
        class Cropped(Exception):
            pass
        with patch.object(xr.Dataset, 'sel', side_effect=Cropped), self.assertRaises(Cropped):
            tools.radial_profile(s, ds, (.5, .5, .5), rmax=.2)
        expected = -(np.roll(phi, -1, axis=2)-np.roll(phi, 1, axis=2))*n/2
        np.testing.assert_array_equal(ds.gaccx, expected)
        np.testing.assert_array_equal(ds.gaccy, np.zeros_like(phi))
        np.testing.assert_array_equal(ds.gaccz, np.zeros_like(phi))


if __name__ == '__main__':
    unittest.main()
