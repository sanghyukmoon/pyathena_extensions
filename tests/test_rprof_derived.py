"""Synthetic checks for the shared radial calculations (no simulation files)."""
import unittest
import numpy as np
import xarray as xr
from core_formation.rprof_derived import add_rprof_derived


def raw_profiles():
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


class TestDerivedProfiles(unittest.TestCase):
    def test_dimensions_and_raw_preservation(self):
        raw = raw_profiles()
        original = raw.copy(deep=True)
        for mhd in (False, True):
            single = add_rprof_derived(raw, cs=1., gconst=np.pi, mhd=mhd)
            history = add_rprof_derived(raw.expand_dims(t=[0., 1.]),
                                       cs=1., gconst=np.pi, mhd=mhd)
            snapshot = add_rprof_derived(raw.expand_dims(center_id=[1, 2]),
                                        cs=1., gconst=np.pi, mhd=mhd)
            xr.testing.assert_equal(single, history.isel(t=0, drop=True))
            xr.testing.assert_equal(single, snapshot.isel(center_id=0, drop=True))
            np.testing.assert_allclose(single.menc, 4*np.pi*raw.r**3/3, atol=1e-15)
            if not mhd:
                self.assertTrue((single.Omega_M == 0).all())
        xr.testing.assert_identical(raw, original)


if __name__ == '__main__':
    unittest.main()
