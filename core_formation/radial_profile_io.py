"""Opt-in adapters for Athena++ on-the-fly radial-profile files."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr

from pyathena.io.read_radial_profile import read_radial_profile


def read_radial_profile_minima(path):
    """Return the packed potential-minimum IDs stored in one rprof file."""
    return read_radial_profile(path).center_id.to_numpy().copy()


def _track_center_id(track, output_number, file_index):
    if hasattr(track, "loc"):
        try:
            entry = track.loc[output_number]
        except KeyError as error:
            raise KeyError(
                f"Track has no entry for output number {output_number}"
            ) from error
        if hasattr(entry, "index") and "leaf_id" in entry.index:
            entry = entry["leaf_id"]
        return int(entry)
    if isinstance(track, Mapping):
        try:
            return int(track[output_number])
        except KeyError as error:
            raise KeyError(
                f"Track has no entry for output number {output_number}"
            ) from error
    try:
        return int(track[file_index])
    except IndexError as error:
        raise IndexError(f"Track has no entry for file index {file_index}") from error


def _missing_center_row(dataset, center_id):
    data_vars = {}
    for name, variable in dataset.data_vars.items():
        if "r" in variable.dims:
            data_vars[name] = ("r", np.full(dataset.sizes["r"], np.nan))
        else:
            data_vars[name] = np.nan
    row = xr.Dataset(data_vars, coords={"r": dataset.r})
    return row.assign_coords(
        center_id=np.uint64(center_id), x1=np.nan, x2=np.nan, x3=np.nan
    )


def load_radial_profile_track(files, track, permissive=False):
    """Load one possibly changing center ID per output into a (t, r) Dataset.

    Mappings and pandas objects are indexed by output number; DataFrames use
    their leaf_id column. Sequences align one-to-one with files. When
    permissive is true, an absent center becomes a NaN row. Missing files are
    always reported because they lack trustworthy time metadata.
    """
    rows = []
    for file_index, path in enumerate(files):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Radial-profile track file does not exist: {path}")
        dataset = read_radial_profile(path)
        output_number = int(dataset.attrs["num"])
        center_id = _track_center_id(track, output_number, file_index)
        matches = np.flatnonzero(dataset.center_id.to_numpy() == np.uint64(center_id))
        if matches.size == 0:
            if not permissive:
                raise KeyError(
                    f"Center ID {center_id} is absent from {path} "
                    f"(output {output_number})"
                )
            row = _missing_center_row(dataset, center_id)
        else:
            row = dataset.isel(center_id=int(matches[0]), drop=True)
            row = row.assign_coords(center_id=np.uint64(center_id))

        time = float(dataset.attrs["time"])
        row = row.expand_dims(t=[time])
        row = row.assign_coords(
            num=("t", [output_number]),
            cycle=("t", [int(dataset.attrs["cycle"])]),
            center_id=("t", [np.uint64(center_id)]),
        )
        rows.append(row)

    if not rows:
        raise ValueError("No radial-profile files were supplied")
    reference_radius = rows[0].r.to_numpy()
    for row in rows[1:]:
        if not np.array_equal(row.r.to_numpy(), reference_radius):
            raise ValueError("Track files do not share identical radial coordinates")
    return xr.concat(rows, dim="t", combine_attrs="drop_conflicts")
