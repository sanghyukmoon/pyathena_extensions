"""On-the-fly core tracking and explicit NetCDF profile persistence."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import xarray as xr


SCHEMA_VERSION = 1


def fingerprint(content):
    """Hash explicit provenance without depending on simulation state."""
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def frame_to_dataset(frame):
    # Existing core tables use object dtype even for numeric columns.
    numeric = frame.map(lambda value: np.asarray(value).item()).apply(pd.to_numeric)
    result = xr.Dataset.from_dataframe(numeric)
    result.attrs['core_attributes'] = json.dumps(frame.attrs, default=lambda x: x.item())
    return result


def dataset_to_frame(dataset):
    result = dataset.to_dataframe()
    result.attrs = json.loads(dataset.attrs['core_attributes'])
    return result


def periodic_displacement(displacement, length):
    return (np.asarray(displacement) + length/2) % length - length/2


def file_stamp(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def write_netcdf(dataset, path):
    """Publish a complete file atomically; preserve integer coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        encoding = {name: {'_FillValue': None} for name, value in dataset.variables.items()
                    if np.issubdtype(value.dtype, np.integer)}
        dataset.to_netcdf(temporary, engine='netcdf4', encoding=encoding)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
