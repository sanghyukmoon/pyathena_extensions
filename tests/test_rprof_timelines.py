import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr

from core_formation.load_sim import LoadSim, LoadSimBase


def simulation(legacy=False, hdf5=True):
    s = LoadSim(None, legacy=legacy)
    s._par = {'output1': {'file_type': 'rprof', 'dt': .1}}
    if hdf5:
        s._par['output2'] = {'file_type': 'hdf5', 'variable': 'cons', 'dt': .2}
    s.dt_output = {output['file_type']: output['dt'] for output in s._par.values()}
    s.ff = SimpleNamespace(nums_hdf5={'cons': [0, 1, 2]})
    s.nums_rprof = [0, 1, 2, 3, 4]
    return s


class TestTimelines(unittest.TestCase):
    def initialize(self, s):
        def header(num, metadata_only):
            self.assertTrue(metadata_only)
            return xr.Dataset(coords={'center_id': [num+10]}, attrs={'num':num, 'time':num*.1})
        with patch.object(s, 'load_hdf5', side_effect=lambda num, **kwargs:{'Time':num*.2}):
            with patch.object(s, 'load_rprof', side_effect=header) as reader:
                s._initialize_timelines()
                return reader.call_count

    def test_nonlegacy(self):
        s = simulation()
        self.assertEqual(self.initialize(s), 5)
        self.assertEqual(s.nums, s.nums_rprof)
        self.assertEqual(s.nums_hdf5, [0, 1, 2])
        self.assertEqual(s.times[3], .1*3)
        self.assertEqual(s.times_hdf5[2], .4)
        np.testing.assert_array_equal(s.minima[2], [12])
        self.assertEqual(s.hdf5_stride, 2)
        self.assertFalse(hasattr(s, '_rprof_headers'))

    def test_legacy(self):
        s = simulation(legacy=True)
        self.assertEqual(self.initialize(s), 0)
        self.assertEqual(s.nums, [0, 1, 2])
        self.assertEqual(s.times, s.times_hdf5)

    def test_no_hdf5(self):
        s = simulation(hdf5=False)
        self.initialize(s)
        self.assertEqual(s.nums_hdf5, [])
        self.assertEqual(s.times_hdf5, {})
        with self.assertRaises(FileNotFoundError):
            s.load_hdf5(rprof_num=0)
        s = simulation()
        s.ff.nums_hdf5 = {'cons': []}
        s.dt_output['hdf5'] = -1  # Disabled HDF5 is also profile-only.
        self.initialize(s)
        self.assertEqual(s.nums_hdf5, [])
        self.assertEqual(s.times_hdf5, {})

    def test_hdf5_config_errors(self):
        s = simulation()
        s._par['output2']['variable'] = 'prim'
        with self.assertRaisesRegex(ValueError, 'cons'):
            self.initialize(s)
        s._par['output2']['variable'] = 'cons'
        s._par['output3'] = dict(s._par['output2'])
        with self.assertRaisesRegex(ValueError, 'cons'):
            self.initialize(s)

    def test_gap(self):
        s = simulation()
        s.nums_rprof = [0, 2]
        with self.assertRaisesRegex(ValueError, 'Gap'):
            self.initialize(s)

    def test_no_radial_profiles(self):
        s = simulation()
        s.nums_rprof = []
        with self.assertRaisesRegex(FileNotFoundError, 'No radial-profile'):
            self.initialize(s)

    def test_cadence_ratio(self):
        s = simulation()
        s.dt_output['hdf5'] = .25
        with self.assertRaisesRegex(ValueError, 'integer multiple'):
            self.initialize(s)
        s.dt_output['hdf5'] = .2 + np.finfo(float).eps
        self.initialize(s)
        self.assertEqual(s.hdf5_stride, 2)
        s.dt_output['hdf5'] = .1 - np.finfo(float).eps
        self.initialize(s)
        self.assertEqual(s.hdf5_stride, 1)

    def test_header_consistency(self):
        s = simulation(hdf5=False)
        with patch.object(s, 'load_rprof', return_value=xr.Dataset(
                coords={'center_id': [10]}, attrs={'num': 8, 'time': 0})):
            with self.assertRaisesRegex(ValueError, 'Inconsistent'):
                s._initialize_timelines()
        with patch.object(s, 'load_rprof', side_effect=lambda num, **kw: xr.Dataset(
                coords={'center_id': [10]}, attrs={'num': num, 'time': 0})):
            with self.assertRaisesRegex(ValueError, 'must increase'):
                s._initialize_timelines()

    def test_selectors(self):
        s = simulation()
        self.initialize(s)
        with patch.object(LoadSimBase, 'load_hdf5', return_value='dataset') as reader:
            self.assertEqual(s.load_hdf5(rprof_num=4), 'dataset')
            reader.assert_called_once_with(s, 2)
            self.assertEqual(s.load_hdf5(1), 'dataset')
            with self.assertRaisesRegex(ValueError, 'No HDF5'):
                s.load_hdf5(rprof_num=3)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                s.load_hdf5(2, rprof_num=4)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                s.load_hdf5()
            s.legacy = True
            with self.assertRaisesRegex(ValueError, 'nonlegacy'):
                s.load_hdf5(rprof_num=4)


if __name__ == '__main__':
    unittest.main()
