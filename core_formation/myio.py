import h5py
import numpy as np
import xarray as xr
# Bottleneck does not use stable sum.
# See xarray #1346, #7344 and bottleneck #193, #462 and more.
# Let's disable third party softwares to go conservative.
# Accuracy is more important than performance.
xr.set_options(use_bottleneck=False, use_numbagg=False)
import dask.array as da

def load_dataframe(filename):
    """Read a table sorted by num, rejecting missing or duplicate snapshots.

    Intended as a analogous reader to xr.load_dataset, but for pandas dataframe

    A sequence may start at any snapshot. Empty and single-row tables are valid;
    no floating-point time-spacing checks are performed.
    """
    dataset = xr.load_dataset(filename, engine='netcdf4')
    frame = dataset.to_dataframe().sort_index()
    if frame.index.hasnans:
        raise ValueError(f"Missing snapshot number in {filename}")
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate snapshot numbers in {filename}")
    if np.any(np.diff(frame.index.to_numpy()) != 1):
        raise ValueError(f"Gap in snapshot number sequence in {filename}")
    frame.attrs = dataset.attrs.copy()
    return frame


def save_dataframe(frame, filename):
    """Write a typed table with a named index, preserving its attributes.

    Sorting, output directories and overwrite decisions belong to the caller.
    """
    dataset = frame.to_xarray()
    dataset.attrs = frame.attrs.copy()
    dataset.to_netcdf(filename, engine='netcdf4')
