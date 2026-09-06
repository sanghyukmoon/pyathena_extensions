"""Task-controlled TES and Lagrangian persistence, shared by both sources."""
from pathlib import Path
import multiprocessing as mp
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from core_formation import tasks, tools
from test_rprof_analysis import simulation


def parallel_tes(pid):
    tasks.critical_tes(_parallel_sim, pid, overwrite=True)


def parallel_lagrangian(pid):
    tasks.lagrangian_props(_parallel_sim, pid, method='virial0', overwrite=True)


class TestTasks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.s = s = simulation(self.directory.name)
        s.cores = s._core_tracks.copy()
        s.rprofs[1] = xr.Dataset(coords={'num': [1, 2, 3, 4]})
        self.path = Path(s.savdir, 'cores')
        self.tes = dict(critical_radius=.2, critical_mass=.3, pindex=.5)

    def calculate_tes(self):
        with patch.object(tools, 'critical_tes_property', return_value=self.tes):
            tasks.critical_tes(self.s, 1)

    def preliminary(self):
        core = self.s._core_tracks[1].copy()
        core.attrs.update(numcrit=2, rcore=.1, mcore=.2)
        return {1: core}

    def lprops(self):
        frame = pd.DataFrame(dict(radius=[.1]*4, Fthm=[1.]*4, Ftrb=[2.]*4,
                                  Fcen=[3.]*4, Fani=[4.]*4, Fgrv=[5.]*4, Fmag=[6.]*4),
                             index=self.s._core_tracks[1].index)
        frame.attrs['sigma_r'] = .4
        return frame

    def test_source_modes_and_explicit_overwrite(self):
        self.calculate_tes()
        tes_path = self.path/'critical_tes.par1.nc'
        stamp = tes_path.stat().st_mtime_ns
        with patch.object(tools, 'critical_tes_property', side_effect=AssertionError('TES recomputed')):
            for mode in (True, False):
                self.s.legacy = mode
                with patch.object(self.s, '_compute_core_props', side_effect=lambda *a, **k:self.preliminary()) as compute:
                    with patch.object(tools, 'lagrangian_property', return_value=self.lprops()) as lagrangian:
                        tasks.lagrangian_props(self.s, 1, method='virial0', overwrite=True)
                        result = self.s.load_core_props('virial0')[1]
                        np.testing.assert_allclose(result.Fnet, 11/5)
                        self.assertEqual(result.attrs['sigma_r'], .4)
                        tasks.lagrangian_props(self.s, 1, method='virial0')
                        self.assertEqual(compute.call_count, 1)
                        self.assertEqual(lagrangian.call_count, 1)
        self.assertEqual(tes_path.stat().st_mtime_ns, stamp)

    def test_missing_tes_and_failed_calculation_do_not_publish(self):
        with self.assertRaisesRegex(FileNotFoundError, '--critical-tes'):
            tasks.lagrangian_props(self.s, 1, method='virial0')
        self.calculate_tes()
        with patch.object(self.s, '_compute_core_props', side_effect=lambda *a, **k:self.preliminary()):
            with patch.object(tools, 'lagrangian_property', side_effect=ValueError('bad profile')):
                with self.assertRaisesRegex(ValueError, 'bad profile'):
                    tasks.lagrangian_props(self.s, 1, method='virial0')
        self.assertFalse((self.path/'core_props.virial0.par1.nc').exists())

    def test_undefined_critical_time_skips_lagrangian(self):
        self.calculate_tes()
        table = self.preliminary()
        table[1].attrs.update(numcrit=np.nan, rcore=np.nan)
        with patch.object(self.s, '_compute_core_props', return_value=table):
            with patch.object(tools, 'lagrangian_property', side_effect=AssertionError('undefined core')):
                tasks.lagrangian_props(self.s, 1, method='virial0')
        self.assertTrue(np.isnan(self.s.load_core_props('virial0')[1].attrs['numcrit']))

    def test_serial_parallel_tes_agree(self):
        global _parallel_sim
        s = self.s
        s.pids = [1, 2]
        s._core_tracks[2] = s._core_tracks[1].copy()
        s._core_tracks[2].attrs['pid'] = 2
        s.rprofs[2] = s.rprofs[1]
        _parallel_sim = s
        with patch.object(tools, 'critical_tes_property', return_value=self.tes):
            for pid in s.pids:
                tasks.critical_tes(s, pid)
            before = {pid: xr.load_dataset(self.path/f'critical_tes.par{pid}.nc') for pid in s.pids}
            with mp.get_context('fork').Pool(2) as pool:
                pool.map(parallel_tes, s.pids)
        for pid in s.pids:
            xr.testing.assert_identical(before[pid], xr.load_dataset(self.path/f'critical_tes.par{pid}.nc'))
        def preliminary(*args, pids, **kwargs):
            result = {}
            for pid in pids:
                table = s._core_tracks[pid].copy()
                table.attrs.update(numcrit=2, rcore=.1, mcore=.2)
                result[pid] = table
            return result
        with patch.object(s, '_compute_core_props', side_effect=preliminary):
            with patch.object(tools, 'lagrangian_property', return_value=self.lprops()):
                for pid in s.pids:
                    tasks.lagrangian_props(s, pid, method='virial0')
                before = {pid: xr.load_dataset(self.path/f'core_props.virial0.par{pid}.nc') for pid in s.pids}
                with mp.get_context('fork').Pool(2) as pool:
                    pool.map(parallel_lagrangian, s.pids)
        for pid in s.pids:
            xr.testing.assert_identical(before[pid], xr.load_dataset(self.path/f'core_props.virial0.par{pid}.nc'))


if __name__ == '__main__':
    unittest.main()
