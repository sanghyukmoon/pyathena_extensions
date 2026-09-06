"""Regression tests for the baseline-preserving profile source integration."""
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from core_formation.load_sim import LoadSim, LoadSimBase, TimingReader


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


if __name__ == '__main__':
    unittest.main()
