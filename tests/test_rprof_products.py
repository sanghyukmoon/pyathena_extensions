"""Shared TES and core-product readers never perform task calculations."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core_formation import rprof_analysis
from test_rprof_analysis import simulation


class TestProducts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.s = simulation(self.directory.name)
        self.s.cores = self.s._core_tracks.copy()
        self.path = Path(self.directory.name, 'cores')

    def test_read_only_in_both_modes(self):
        s = self.s
        tes = pd.DataFrame({'critical_radius': [.1]*4, 'pindex': [.5]*4},
                           index=s._core_tracks[1].index)
        table = s._core_tracks[1].assign(radius=.2)
        table.attrs['derived_available'] = True
        rprof_analysis.write_netcdf(rprof_analysis.frame_to_dataset(tes), self.path/'critical_tes.par1.nc')
        rprof_analysis.write_netcdf(rprof_analysis.frame_to_dataset(table), self.path/'core_props.virial0.par1.nc')
        stamps = {p.name:p.stat().st_mtime_ns for p in self.path.iterdir()}
        for legacy in (True, False):
            s.legacy = legacy
            with patch.object(s, '_compute_core_props', side_effect=AssertionError('calculation')):
                with patch.object(rprof_analysis, 'write_netcdf', side_effect=AssertionError('write')):
                    pd.testing.assert_frame_equal(s.load_critical_tes(1), tes, check_index_type=False)
                    self.assertIn('critical_radius', s._core_tracks[1])
                    result = s.load_core_props('virial0')[1]
                    self.assertEqual(result.attrs, table.attrs)
                    self.assertTrue((result.radius == .2).all())
                    s.select_cores('virial0')
                    self.assertIn('radius', s.cores[1])
        self.assertEqual(stamps, {p.name:p.stat().st_mtime_ns for p in self.path.iterdir()})

    def test_missing_does_not_compute(self):
        s = self.s
        with patch.object(s, '_compute_core_props', side_effect=AssertionError('calculation')):
            with self.assertRaisesRegex(FileNotFoundError, '--critical-tes'):
                s.load_critical_tes(1)
            self.assertEqual(s.load_core_props('virial0'), {})
            self.assertIn('--lagrangian-props', s.load_errors[1]['derived:virial0'])
            s.select_cores('virial0')
            self.assertIn('leaf_id', s.cores[1])
        self.assertFalse(self.path.exists())

    def test_corrupt_is_not_rebuilt(self):
        self.path.mkdir()
        (self.path/'core_props.virial0.par1.nc').write_bytes(b'invalid')
        with patch.object(self.s, '_compute_core_props', side_effect=AssertionError('calculation')):
            self.assertEqual(self.s.load_core_props('virial0'), {})
        self.assertEqual((self.path/'core_props.virial0.par1.nc').read_bytes(), b'invalid')

    def test_initialization_retains_tracks_without_computing_missing_products(self):
        s = self.s
        tracks, collapse = s._core_tracks.copy(), s.tcoll_cores.copy()
        for legacy in (True, False):
            s.legacy = legacy
            with patch.object(s, '_load_tcoll_cores', return_value=collapse):
                with patch.object(s, '_load_cores', return_value=tracks):
                    with patch.object(s, '_compute_core_props', side_effect=AssertionError('calculation')):
                        with patch.object(rprof_analysis, 'write_netcdf', side_effect=AssertionError('write')):
                            s._initialize_analysis('virial0', False, True, False)
            self.assertIn('leaf_id', s.cores[1])
            self.assertIn('tes', s.load_errors[1])
            self.assertIn('derived:virial0', s.load_errors[1])
        self.assertFalse(self.path.exists())


if __name__ == '__main__':
    unittest.main()
