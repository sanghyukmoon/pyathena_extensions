import numpy as np
import pandas as pd
import xarray as xr

from pipeline_comparison import EPS32, compare_arrays, compare_products


def test_tolerance_boundary_and_no_rounding():
    assert compare_arrays([1.0], [1.0 + EPS32])['passed']
    assert not compare_arrays([1.0], [1.0 + 2 * EPS32])['passed']
    assert not compare_arrays([1.0], [1.0 + 1.4 * EPS32])['passed']


def test_zero_and_special_values():
    assert compare_arrays([0., np.nan, np.inf, -np.inf],
                          [0., np.nan, np.inf, -np.inf])['passed']
    assert not compare_arrays([0.], [1.e-30])['passed']
    assert compare_arrays([np.nan, np.inf], [0., -np.inf])['nonfinite_mismatches'] == 2


def test_exact_identifiers_and_shapes():
    assert not compare_arrays([2**60], [2**60+1])['passed']
    assert not compare_arrays([True], [False])['passed']
    assert not compare_arrays([1, 2], [[1, 2]])['passed']


def test_missing_keys_and_attributes():
    a = pd.DataFrame({'density': [1.]}, index=[7])
    b = a.copy()
    a.attrs['numcrit'], b.attrs['numcrit'] = 3, 4
    records = compare_products({'virial0': {1: a}}, {'virial0': {1: b, 2: b}})
    assert any(r.get('missing_left') == ['2'] for r in records)
    assert any(r['path'].endswith('/attrs/numcrit') and not r['passed'] for r in records)


def test_coordinates_not_silently_aligned():
    a = xr.Dataset({'rho': ('t', [1., 2.])}, coords={'t': [0., 1.], 'num': ('t', [0, 1])})
    b = a.assign_coords(t=[0., 2.])
    records = compare_products(a, b)
    assert any(r['path'] == 'root/t' and not r['passed'] for r in records)
    assert all(r['passed'] for r in compare_products(a, a.copy(deep=True)))


def test_object_numeric_values():
    result = compare_arrays(np.array([1., np.nan, 3], dtype=object),
                            np.array([1. + EPS32, np.nan, 3], dtype=object))
    assert result['passed']


def test_empty_and_mixed_arrays():
    assert compare_arrays([], [])['passed']
    assert compare_arrays(np.array(['missing', np.nan], dtype=object),
                          np.array(['missing', np.nan], dtype=object))['passed']


def test_paths_are_reported_separately_from_scientific_attributes():
    a = xr.Dataset(attrs={'path': 'legacy/input', 'rmax': 0.53})
    b = xr.Dataset(attrs={'path': 'onthefly/input', 'rmax': 0.53})
    records = compare_products(a, b)
    assert all(r['passed'] for r in records)
    assert any(r.get('comparison') == 'path provenance only' for r in records)
    b.attrs['rmax'] = 0.54
    assert any(not r['passed'] for r in compare_products(a, b))


if __name__ == '__main__':
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                               for name, value in list(globals().items())
                               if name.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
