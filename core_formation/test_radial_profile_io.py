import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

import radial_profile_io


def _dataset(number, time, center_ids):
    center_ids = np.asarray(center_ids, dtype=np.uint64)
    ncenter = center_ids.size
    return xr.Dataset(
        {
            "rho": (
                ("center_id", "r"),
                np.arange(ncenter*2, dtype=float).reshape(ncenter, 2) + 1.0,
            ),
            "velx_origin": ("center_id", np.arange(ncenter, dtype=float)),
        },
        coords={
            "center_id": center_ids,
            "r": [0.0, 0.25],
            "x1": ("center_id", np.zeros(ncenter)),
            "x2": ("center_id", np.zeros(ncenter)),
            "x3": ("center_id", np.zeros(ncenter)),
        },
        attrs={"num": number, "cycle": 10*number, "time": time},
    )


class TestRadialProfileTrack(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.files = [root/"a.rprof", root/"b.rprof"]
        for path in self.files:
            path.touch()
        self.datasets = {
            self.files[0]: _dataset(3, 0.5, [10, 20]),
            self.files[1]: _dataset(4, 0.75, [21, 30]),
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _read(self, path):
        return self.datasets[Path(path)]

    def test_minimum_ids(self):
        with patch.object(radial_profile_io, "read_radial_profile", self._read):
            ids = radial_profile_io.read_radial_profile_minima(self.files[0])
        np.testing.assert_array_equal(ids, [10, 20])

    def test_changing_center_track(self):
        with patch.object(radial_profile_io, "read_radial_profile", self._read):
            result = radial_profile_io.load_radial_profile_track(
                self.files, {3: 20, 4: 21}
            )
        np.testing.assert_array_equal(result.center_id, [20, 21])
        np.testing.assert_array_equal(result.num, [3, 4])
        np.testing.assert_array_equal(result.cycle, [30, 40])
        np.testing.assert_allclose(result.t, [0.5, 0.75])
        self.assertEqual(result.rho.dims, ("t", "r"))

    def test_missing_id_strict_and_permissive(self):
        with patch.object(radial_profile_io, "read_radial_profile", self._read):
            with self.assertRaisesRegex(KeyError, "99"):
                radial_profile_io.load_radial_profile_track(
                    self.files, {3: 99, 4: 21}
                )
            result = radial_profile_io.load_radial_profile_track(
                self.files, {3: 99, 4: 21}, permissive=True
            )
        self.assertTrue(np.isnan(result.rho.isel(t=0)).all())
        self.assertTrue(np.isfinite(result.rho.isel(t=1)).all())


if __name__ == "__main__":
    unittest.main()
