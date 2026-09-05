"""On-the-fly core tracking and explicit NetCDF profile persistence."""
import hashlib
import json
import os
import re
from time import perf_counter
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

from pyathena.io.read_radial_profile import read_radial_profile
from pyathena.io.read_particles import read_parbin, read_partab
from .rprof_derived import add_rprof_derived, CALCULATION_VERSION

SCHEMA_VERSION = 1


def report_failure(s, pid, stage, error):
    s.load_errors.setdefault(pid, {})[stage] = str(error)
    s.logger.warning(f'On-the-fly pid {pid}, {stage}: {error}')


def particle_outputs(s):
    """Index one contiguous particle stream; do not equate its counter to rprof."""
    pattern = re.compile(re.escape(s.problem_id) +
                         r'(?:\.block(\d+))?\.(out\d+)\.(\d+)\.par0\.(parbin|tab)$')
    groups, streams = {}, set()
    for directory in (Path(s.basedir), Path(s.basedir, 'parbin'),
                      Path(s.basedir, 'partab')):
        for path in sorted(directory.glob(s.problem_id+'*.par0.*')):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            block, stream, number, kind = match.groups()
            streams.add((stream, kind))
            groups.setdefault(int(number), []).append((block, path))
    if len(streams) != 1:
        raise ValueError(f'Expected one particle output stream, found {sorted(streams)}')
    numbers = sorted(groups)
    if np.any(np.diff(numbers) != 1):
        raise ValueError('Gap in particle output numbering')
    kind = next(iter(streams))[1]
    reader = read_parbin if kind == 'parbin' else read_partab
    records = []
    for num in numbers:
        entries = groups[num]
        blocks = [block for block, _ in entries]
        if blocks == [None]:
            pass
        elif None in blocks or len(set(blocks)) != len(blocks):
            raise ValueError(f'Duplicate/ambiguous particle output {num}')
        else:
            nblocks = int(np.prod([s.par['mesh'][f'nx{i}']//
                                  s.par['meshblock'][f'nx{i}'] for i in (1, 2, 3)]))
            if set(map(int, blocks)) != set(range(nblocks)):
                raise ValueError(f'Incomplete particle MeshBlock coverage at {num}')
        paths = [path for _, path in entries]
        times = [reader(path, header_only=True).get('time', np.nan) for path in paths]
        if not np.all(np.isfinite(times)) or not np.all(np.asarray(times) == times[0]):
            raise ValueError(f'Missing or inconsistent particle time at output {num}')
        records.append(dict(num=num, time=times[0], paths=paths, kind=kind))
    result = pd.DataFrame(records).set_index('num')
    if np.any(np.diff(result.time) <= 0):
        raise ValueError('Particle output times must increase')
    return result


def match_particles(s, num):
    """Match recorded times within 32 float64 eps * max(1, |time|)."""
    time = float(s.rprof_outputs.loc[num, 'time'])
    tolerance = 32*np.finfo(float).eps*max(1., abs(time))
    matches = s.particle_outputs.loc[np.abs(s.particle_outputs.time-time) <= tolerance]
    if len(matches) != 1:
        raise ValueError(f'Expected one particle snapshot at rprof {num}, time {time}; '
                         f'found {len(matches)}')
    return matches.iloc[0]


def load_particles(s, num):
    output = match_particles(s, num)
    reader = read_parbin if output.kind == 'parbin' else read_partab
    frames = [reader(path) for path in output.paths]
    result = pd.concat(frames).sort_index()
    if not result.index.is_unique:
        raise ValueError(f'Duplicate particle IDs at rprof {num}')
    return result


def initialize(s, *, method, load_rprofs, load_derived_cores, overwrite):
    """Eagerly populate independent results, gating each dependent calculation."""
    s.cores, s.rprofs, s.cores_dict, s._core_tracks = {}, {}, {}, {}
    s.load_timings = {}
    start = perf_counter()
    outputs = s.rprof_outputs  # Global numbering errors are not recoverable per core.
    s.logger.info(f'Radial-profile coverage: {outputs.index.min()}..{outputs.index.max()}')
    s.load_timings['index_seconds'] = perf_counter()-start
    start = perf_counter()
    s.pids = list(getattr(s, 'pids', []))
    s.tcoll_cores = s._load_tcoll_cores()
    s.load_timings['collapse_seconds'] = perf_counter()-start
    start = perf_counter()
    for pid in s.pids:
        if 'collapse' in s.load_errors.get(pid, {}):
            continue
        try:
            track = track_core(s, pid)
            s._core_tracks[pid] = track
            if track.attrs['track_failed']:
                report_failure(s, pid, 'tracking', track.attrs['stop_reason'])
        except (ValueError, OSError) as error:
            report_failure(s, pid, 'tracking', error)
    s.cores = s._core_tracks.copy()
    s.load_timings['tracking_seconds'] = perf_counter()-start
    try:
        s.particle_outputs = particle_outputs(s)
    except (ValueError, OSError) as error:
        s.particle_outputs = None
        for pid in s.pids:
            report_failure(s, pid, 'particles', error)
    if load_rprofs:
        start = perf_counter()
        for pid, track in s._core_tracks.items():
            if track.attrs['track_failed']:
                continue
            try:
                s.rprofs[pid] = load_core_rprof(s, pid, cache=s.cache, overwrite=overwrite)
            except (ValueError, OSError, KeyError) as error:
                report_failure(s, pid, 'profiles', error)
        s.load_timings['profiles_seconds'] = perf_counter()-start
    if load_derived_cores and load_rprofs:
        for name in ('empirical', 'virial', 'virial0', 'virial1'):
            start = perf_counter()
            s.cores_dict[name] = update_core_props(s, name, cache=s.cache, overwrite=overwrite)
            s.load_timings[f'{name}_seconds'] = perf_counter()-start
        s.select_cores(method)


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


def update_core_props(s, method, *, cache=True, overwrite=False):
    """Explicit per-core dependency checks and mode-separated NetCDF caches."""
    if method not in ('empirical', 'virial', 'virial0', 'virial1'):
        raise ValueError(f'Unknown critical-time method {method}')
    result = {}
    s.particle_outputs = None
    try:
        s.particle_outputs = particle_outputs(s)
    except (ValueError, OSError) as error:
        for pid in s.pids:
            report_failure(s, pid, f'derived:{method}', error)
        return result
    peer_missing = [pid for pid in s.pids if pid not in s._core_tracks or
                    s._core_tracks[pid].attrs['track_failed']]
    for pid in s.pids:
        stage = f'derived:{method}'
        try:
            if peer_missing:
                raise ValueError(f'Peer trajectories incomplete for {peer_missing}')
            if pid not in s.rprofs:
                raise ValueError('Radial profiles have not loaded successfully')
            track = s._core_tracks[pid]
            validate_profile(s, track, s.rprofs[pid])
            current_fingerprint = profile_fingerprint(s, track, s.rprof_outputs)
            if s.rprofs[pid].attrs.get('fingerprint') != current_fingerprint:
                raise ValueError('In-memory profiles are stale; reload load_core_rprof first')
            matched = [match_particles(s, num) for num in track.index]
            content = dict(schema=SCHEMA_VERSION, method=method,
                           profiles=profile_fingerprint(s, track, s.rprof_outputs),
                           peers={int(cid): profile_fingerprint(s, other, s.rprof_outputs)
                                  for cid, other in s._core_tracks.items()},
                           track_attributes=track.attrs,
                           particles=[file_stamp(path) for row in matched for path in row.paths],
                           history=file_stamp(s._get_fparhst(pid)))
            fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
            path = Path(s.savdir, 'on_the_fly', f'core_props.{method}.par{pid}.nc')
            if cache and not overwrite and path.exists():
                try:
                    stored = xr.load_dataset(path, engine='netcdf4')
                    if stored.attrs.get('fingerprint') == fingerprint:
                        result[pid] = dataset_to_frame(stored)
                        s.load_errors.get(pid, {}).pop(stage, None)
                        continue
                except (OSError, ValueError, KeyError):
                    pass
            computed = s._compute_core_props(method, s.savdir, pids=[pid])
            if pid not in computed:
                raise ValueError('Core calculation did not return a usable result')
            frame = computed[pid]
            frame.attrs['derived_available'] = True
            if cache:
                dataset = frame_to_dataset(frame)
                dataset.attrs['fingerprint'] = fingerprint
                write_netcdf(dataset, path)
            result[pid] = frame
            s.load_errors.get(pid, {}).pop(stage, None)
        except (ValueError, OSError, KeyError, RuntimeError) as error:
            report_failure(s, pid, stage, error)
    return result


def precollapse_num(s, time):
    outputs = s.rprof_outputs
    eligible = outputs.index[outputs.time <= time]
    if len(eligible) == 0:
        raise ValueError(f"No radial-profile output at or before collapse time {time}")
    return int(eligible[-1])


def periodic_displacement(displacement, length):
    return (np.asarray(displacement) + length/2) % length - length/2


def track_core(s, pid, *, f_mul=3):
    """Follow nearest periodic minima backward until continuity fails."""
    outputs = s.rprof_outputs
    collapse = s.tcoll_cores.loc[pid]
    numcoll = int(collapse.num)
    position = collapse[['x1', 'x2', 'x3']].to_numpy(dtype=float)
    rows, positions = [], []
    reason = 'coverage_start'
    for num in outputs.loc[:numcoll].index[::-1]:
        header = s._rprof_headers[int(num)]
        candidates = np.column_stack([header[c].values for c in ('x1', 'x2', 'x3')])
        if not len(candidates):
            reason = 'empty_minima'
            break
        distances = np.linalg.norm(periodic_displacement(candidates-position, s.Lbox), axis=1)
        chosen = int(np.argmin(distances))
        candidate = candidates[chosen]
        if len(positions) >= 2:
            displacement = periodic_displacement(positions[-1]-positions[-2], s.Lbox)
            prediction = positions[-1] + displacement
            error = np.linalg.norm(periodic_displacement(candidate-prediction, s.Lbox))
            if error > f_mul*max(np.linalg.norm(displacement), s.dx):
                reason = 'continuity'
                break
        position = candidate
        positions.append(position)
        rows.append(dict(num=int(num), time=float(header.attrs['time']),
                         cycle=int(header.attrs['cycle']),
                         leaf_id=int(header.center_id.values[chosen])))
    if not rows:
        raise ValueError(f"No minimum at the pre-collapse output for pid {pid}")
    cores = pd.DataFrame(rows, dtype=object).set_index('num').sort_index()
    cores.attrs.update(pid=int(pid), numcoll=numcoll, tcoll=float(collapse.time),
                       source='rprof', f_mul=float(f_mul), stop_reason=reason,
                       num_start=int(cores.index[0]),
                       track_failed=len(cores) < 2 or reason == 'empty_minima')
    return cores


def file_stamp(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def profile_fingerprint(s, track, outputs):
    records = [[int(num), int(row.leaf_id), float(row.time),
                file_stamp(outputs.loc[num, 'path'])]
               for num, row in track.iterrows()]
    content = dict(schema=SCHEMA_VERSION, calculation=CALCULATION_VERSION,
                   source='rprof', cs=float(s.cs), gconst=float(s.gconst),
                   mhd=bool(s.mhd), records=records)
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def validate_profile(s, track, dataset):
    """Check provenance against the current trajectory and file headers."""
    numbers = np.asarray(track.index, dtype=np.int64)
    if not np.array_equal(dataset.num, numbers):
        raise ValueError('Profile output numbers differ from the core trajectory')
    if not np.array_equal(dataset.center_id, track.leaf_id.to_numpy(dtype=np.uint64)):
        raise ValueError('Profile center IDs differ from the core trajectory')
    if not np.array_equal(dataset.t, track.time.to_numpy(dtype=float)):
        raise ValueError('Profile times differ from the core trajectory')
    for index, (num, row) in enumerate(track.iterrows()):
        header = s._rprof_headers[int(num)]
        if float(row.time) != header.attrs['time']:
            raise ValueError(f'Trajectory time disagrees with rprof output {num}')
        if 'cycle' in track and int(row.cycle) != header.attrs['cycle']:
            raise ValueError(f'Trajectory cycle disagrees with rprof output {num}')
        if not np.array_equal(dataset.r, header.r):
            raise ValueError(f'Radial coordinates differ at output {num}')
        if int(row.leaf_id) not in header.center_id.values:
            raise ValueError(f'Trajectory center missing at output {num}')
        if int(dataset.cycle.values[index]) != header.attrs['cycle']:
            raise ValueError(f'Cached cycle disagrees with rprof output {num}')
        center = header.sel(center_id=int(row.leaf_id))
        for coordinate in ('x1', 'x2', 'x3'):
            if float(dataset[coordinate].values[index]) != float(center[coordinate]):
                raise ValueError(f'Cached center position differs at output {num}')


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


def load_core_rprof(s, pid, *, derived=True, cache=True, overwrite=False):
    track = s._core_tracks[pid]
    if track.empty or track.attrs.get('track_failed', False):
        raise ValueError(f'Core {pid} has no valid prestellar trajectory')
    outputs = s.rprof_outputs
    fingerprint = profile_fingerprint(s, track, outputs)
    path = Path(s.savdir, 'on_the_fly', f'core_rprof.par{pid}.nc')
    dataset = None
    if cache and not overwrite and path.exists():
        try:
            candidate = xr.load_dataset(path, engine='netcdf4')
            if (candidate.attrs.get('fingerprint') == fingerprint and
                    (not derived or candidate.attrs.get('derived_version') == CALCULATION_VERSION)):
                validate_profile(s, track, candidate)
                dataset = candidate
        except (OSError, ValueError, KeyError) as error:
            s.logger.warning(f'Rebuilding invalid profile cache for pid {pid}: {error}')
    if dataset is None:
        rows, radius = [], None
        for num, core in track.iterrows():
            source = read_radial_profile(outputs.loc[num, 'path'],
                                         center_ids=[int(core.leaf_id)])
            if radius is not None and not np.array_equal(source.r, radius):
                raise ValueError(f'Radial coordinates differ at output {num}')
            radius = source.r.values
            row = source.isel(center_id=0, drop=True).expand_dims(t=[source.attrs['time']])
            row = row.assign_coords(
                num=('t', [int(num)]), cycle=('t', [int(source.attrs['cycle'])]),
                center_id=('t', [np.uint64(core.leaf_id)]),
                **{coord: ('t', [float(source[coord].values[0])])
                   for coord in ('x1', 'x2', 'x3')})
            rows.append(row)
        dataset = xr.concat(rows, dim='t', join='exact', combine_attrs='drop_conflicts')
        raw_variables = list(dataset.data_vars)
        validate_profile(s, track, dataset)
        if derived:
            dataset = add_rprof_derived(dataset, cs=s.cs, gconst=s.gconst, mhd=s.mhd)
        dataset.attrs.update(fingerprint=fingerprint, source='rprof', pid=int(pid),
                             schema_version=SCHEMA_VERSION,
                             derived_version=CALCULATION_VERSION if derived else 0,
                             raw_variables=json.dumps(raw_variables),
                             cs=float(s.cs), gconst=float(s.gconst), mhd=int(s.mhd))
        if cache:
            # A raw-only refresh must not replace a valid full derived cache.
            preserve = False
            if not derived and path.exists():
                try:
                    with xr.open_dataset(path, engine='netcdf4') as previous:
                        preserve = (previous.attrs.get('fingerprint') == fingerprint and
                                    previous.attrs.get('derived_version') == CALCULATION_VERSION)
                except (OSError, ValueError):
                    pass
            if not preserve:
                write_netcdf(dataset, path)
    if not derived:
        dataset = dataset[json.loads(dataset.attrs['raw_variables'])]
    return dataset.set_xindex('num')
