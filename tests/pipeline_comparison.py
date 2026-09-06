"""Strict float32-relative comparison of complete analysis products."""
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd
import xarray as xr

EPS32 = float(np.finfo(np.float32).eps)


def compare_arrays(left, right):
    """No alignment, rounding, broadcasting, or near-zero absolute tolerance."""
    a, b = np.asarray(left), np.asarray(right)
    result = {'shape_left': list(a.shape), 'shape_right': list(b.shape)}
    if a.shape != b.shape:
        return dict(result, passed=False, reason='shape mismatch')
    av, bv = a.ravel(), b.ravel()
    numeric = all(isinstance(x, Real) for x in (*av, *bv))
    integral = numeric and all(isinstance(x, (Integral, np.bool_)) for x in (*av, *bv))
    if numeric and not integral:
        av, bv = av.astype(np.float64), bv.astype(np.float64)
        finite = np.isfinite(av) & np.isfinite(bv)
        same_special = ((np.isnan(av) & np.isnan(bv)) |
                        (np.isposinf(av) & np.isposinf(bv)) |
                        (np.isneginf(av) & np.isneginf(bv)))
        passed = same_special.copy()
        delta = np.abs(av[finite] - bv[finite])
        scale = np.maximum(np.abs(av[finite]), np.abs(bv[finite]))
        relative = np.divide(delta, scale, out=np.zeros_like(delta), where=scale != 0)
        passed[finite] = delta <= EPS32 * scale
        result.update(max_absolute_error=float(delta.max(initial=0)),
                      max_relative_error=float(relative.max(initial=0)),
                      nonfinite_mismatches=int((~finite & ~same_special).sum()))
    else:
        passed = np.array([bool(x == y) or
                           (isinstance(x, Real) and isinstance(y, Real) and
                            np.isnan(x) and np.isnan(y))
                           for x, y in zip(av, bv)], dtype=bool)
    bad = np.flatnonzero(~passed)
    result.update(passed=not bad.size, count=int(a.size), failed_count=int(bad.size),
                  examples=[{'index': list(map(int, np.unravel_index(int(i), a.shape))),
                             'left': repr(av[i]), 'right': repr(bv[i])}
                            for i in bad[:5]])
    return result


def compare_products(left, right, path='root', records=None):
    """Compare scientific values AND metadata; keep missing-key failures visible."""
    if records is None:
        records = []

    def record(result, suffix=''):
        records.append(dict(path=path + suffix, **result))

    def keys(a, b, suffix):
        missing_left, missing_right = set(b) - set(a), set(a) - set(b)
        record({'passed': not (missing_left or missing_right),
                'missing_left': sorted(map(str, missing_left)),
                'missing_right': sorted(map(str, missing_right))}, suffix)

    if isinstance(left, xr.Dataset) and isinstance(right, xr.Dataset):
        compare_products(dict(left.sizes), dict(right.sizes), path + '/sizes', records)
        keys(left.variables, right.variables, '/variables')
        keys(left.coords, right.coords, '/coordinates')
        compare_products(left.attrs, right.attrs, path + '/attrs', records)
        for name in left.variables:
            if name in right.variables:
                a, b = left[name], right[name]
                compare_products(a.dims, b.dims, path + f'/{name}/dims', records)
                result = compare_arrays(a.values, b.values)
                for example in result.get('examples', []):
                    example['coordinates'] = {
                        dim: repr(a[dim].values[i]) for dim, i in zip(a.dims, example['index'])
                        if dim in a.coords and a[dim].ndim == 1}
                    if 't' in a.dims and 'num' in left.coords:
                        example['num'] = int(left.num.values[example['index'][a.dims.index('t')]])
                records.append(dict(path=path + f'/{name}', **result))
                compare_products(a.attrs, b.attrs, path + f'/{name}/attrs', records)
    elif isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        compare_products(left.index.to_numpy(), right.index.to_numpy(), path + '/index', records)
        compare_products(left.index.names, right.index.names, path + '/index_names', records)
        compare_products(left.columns.to_numpy(), right.columns.to_numpy(), path + '/columns', records)
        compare_products(left.attrs, right.attrs, path + '/attrs', records)
        for name in left.columns:
            if name in right.columns:
                result = compare_arrays(left[name].to_numpy(), right[name].to_numpy())
                for example in result.get('examples', []):
                    example['row'] = repr(left.index[example['index'][0]])
                records.append(dict(path=path + f'/{name}', **result))
    elif isinstance(left, Mapping) and isinstance(right, Mapping):
        if path.endswith('/attrs'):
            path_keys = {'path', 'basedir', 'savdir', 'filename'} & (set(left) | set(right))
            for key in sorted(path_keys):
                record({'passed': True, 'comparison': 'path provenance only',
                        'left': str(left.get(key)), 'right': str(right.get(key))}, '/' + key)
            left = {k: v for k, v in left.items() if k not in path_keys}
            right = {k: v for k, v in right.items() if k not in path_keys}
        keys(left, right, '/keys')
        for key in left:
            if key in right:
                compare_products(left[key], right[key], path + f'/{key}', records)
    else:
        record(compare_arrays(left, right))
    return records
