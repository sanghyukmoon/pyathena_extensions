import json
from dataclasses import asdict

"""Module containing functions that are not generally reusable"""
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter
import xarray as xr
# Bottleneck does not use stable sum.
# See xarray #1346, #7344 and bottleneck #193, #462 and more.
# Let's disable third party softwares to go conservative.
# Accuracy is more important than performance.
xr.set_options(use_bottleneck=False, use_numbagg=False)
import dask.array as da
import subprocess
import pickle
import h5py
import glob
import logging
from pyathena.util import uniform, transform
from scipy import fft

from . import plots, tools, config, stats, myio


def combine_partab(s, ns=None, ne=None, partag="par0", remove=False,
                   include_last=False):
    """Combine particle .tab output files.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    ns : int, optional
        Starting snapshot number.
    ne : int, optional
        Ending snapshot number.
    partag : str, optional
        Particle tag (<particle?> in the input file).
    remove : str, optional
        If True, remove the block? per-core outputs after joining.
    include_last : bool
        If false, do not process last .tab file, which might being written
        by running Athena++ process.
    """
    script = "/home/sm69/tigris/vis/tab/combine_partab.sh"
    outid = "out{}".format(s.partab_outid)
    block0_pattern = '{}/{}.block0.{}.?????.{}.tab'.format(s.basedir,
                                                           s.problem_id, outid,
                                                           partag)
    file_list0 = sorted(glob.glob(block0_pattern))
    if not include_last:
        file_list0 = file_list0[:-1]
    if len(file_list0) == 0:
        print("Nothing to combine", flush=True)
        return
    if ns is None:
        ns = int(file_list0[0].split('/')[-1].split('.')[3])
    if ne is None:
        ne = int(file_list0[-1].split('/')[-1].split('.')[3])
    nblocks = 1
    for axis in [1, 2, 3]:
        nblocks *= ((s.par['mesh'][f'nx{axis}']
                    // s.par['meshblock'][f'nx{axis}']))
    if partag not in s.partags:
        raise ValueError("Particle {} does not exist".format(partag))
    subprocess.run([script, s.problem_id, outid, partag, str(ns), str(ne)],
                   cwd=s.basedir)

    if remove:
        joined_pattern = '{}/{}.{}.?????.{}.tab'.format(s.basedir,
                                                        s.problem_id, outid,
                                                        partag)
        joined_files = set(glob.glob(joined_pattern))
        block0_files = {f.replace('block0.', '') for f in file_list0}
        if block0_files.issubset(joined_files):
            print("All files are joined. Remove block* files", flush=True)
            file_list = []
            for fblock0 in block0_files:
                for i in range(nblocks):
                    file_list.append(fblock0.replace(
                        outid, "block{}.{}".format(i, outid)))
            file_list.sort()
            for f in file_list:
                Path(f).unlink()
        else:
            print("Not all files are joined", flush=True)


def output_sparse_hdf5(s, gids, num):
    """Read Athena++ hdf5 file and remove all the MeshBlocks
    except the selected ones.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    gids : list of int
        List of global ids of the selected MeshBlocks.
    num : int
        Snapshot number.

    TODO
    ----
    Refactor to use load_hdf5(raw=True)
    """
    s.load_hdf5(num, file_only=True)
    filename = s.fhdf5
    native_hdf5_num = num if s.legacy else num // s._hdf5_stride
    fsrc = h5py.File(filename, 'r')
    # Read Mesh information
    block_size = fsrc.attrs['MeshBlockSize']
    mesh_size = fsrc.attrs['RootGridSize']
    num_blocks = mesh_size // block_size  # Assuming uniform grid

    if num_blocks.prod() != fsrc.attrs['NumMeshBlocks']:
        raise ValueError("Number of blocks does not match the attribute")
    # Array of logical locations, arranged by Z-ordering
    # (lx1, lx2, lx3)
    # (  0,   0,   0)
    # (  1,   0,   0)
    # (  0,   1,   0)
    # ...
    logical_loc = fsrc['LogicalLocations']

    # lazy load from HDF5
    ds = dict()
    for dsetname in fsrc.attrs['DatasetNames']:
        darr = da.from_array(fsrc[dsetname], chunks=(1, 1, *block_size))
        if len(darr.shape) != 5:
            # Expected shape: (nvar, nblock, z, y, x)
            raise ValueError("Invalid shape of the dataset")
        ds[dsetname] = darr

    for k, v in ds.items():
        ds[k] = v[:, gids, ...]
    ofname = Path(
        s.basedir, "sparse", f"{s.problem_id}.{native_hdf5_num:05d}.athdf"
    )
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists():
        ofname.unlink()
    # Dump the raw dask array into HDF5 file
    da.to_hdf5(ofname, ds)

    # Read in the fresh HDF5 file and copy back the other properties and attributes
    fdst = h5py.File(ofname, 'a')
    fdst.attrs.update(fsrc.attrs)

    dataset_names = set(name.decode() for name in fsrc.attrs['DatasetNames'])
    for k in set(fsrc.keys()) - dataset_names:
        fsrc.copy(k, fdst)
    fdst.create_dataset("gids", data=gids)
    fsrc.close()
    fdst.close()


def critical_tes(s, pid, overwrite=False):
    """Calculates and saves critical tes associated with each core.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    pid : int
        Particle id.
    overwrite : bool, optional
        If true, recomputes all snapshots and overwrites the core NetCDF file.

    Raises
    ------
    KeyError
        A required radial profile is missing. No output is written until all
        snapshots have been calculated successfully.
    """
    # Check if file exists
    ofname = Path(s.savdir, config.CORE_DIR,
                  f'critical_tes.par{pid}.nc')
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[critical_tes] file already exists. Skipping...')
        return

    results = []
    for core in s.cores[pid].itertuples():
        num = core.Index
        print(f'[critical_tes] processing model {s.basename} pid {pid} num {num}')
        rprf = s.rprofs[pid].sel(num=num)
        result = tools.critical_tes_property(s, rprf, core)
        result['num'] = num
        results.append(result)

    if not results:
        logging.warning(f'No critical TES results for pid={pid}; not writing a file.')
        return
    frame = pd.DataFrame(results).set_index('num').sort_index()
    myio.save_dataframe(frame, ofname)
    ofname.with_name("cores.p").unlink(missing_ok=True)


def core_tracking(s, pids=None, overwrite=False):
    """Loops over all sink particles and find their progenitor cores

    Finds a unique minima at each snapshot that is going to collapse.
    For each sink particle, back-traces the evolution of its progenitor cores.
    Saves the resulting trajectories as NetCDF.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    pid : int
        Particle ID
    overwrite : str, optional
        If true, overwrites the existing trajectory NetCDF file.
    """
    if pids is None:
        pids = s.pids

    for pid in pids:
        # Check if file exists
        ofname = Path(s.savdir, config.CORE_DIR, f'core_trajectories.par{pid}.nc')
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            print('[core_tracking] file already exists. Skipping...')
            continue

        cores = tools.track_cores(s, pid)
        myio.save_dataframe(cores, ofname)
        ofname.with_name("cores.p").unlink(missing_ok=True)


def radial_profile(s, nums=None, pids=None, overwrite=False, all_minima=False):
    """Calculates and pickles radial profiles of all cores.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    nums : array of int
        Snapshot numbers
    pids : list of int
        Particle ids to process.
    overwrite : str, optional
        If true, overwrites the existing pickle file.
    """

    if pids is None:
        pids = s.pids

    if nums is None:
        nums = s.nums_with_hdf5

    for num in nums:
        print(f"[radial_profile] Start reading snapshot at num = {num}.")

        # Load the snapshot
        # ds0 should not be modified in the following loop.
        ds0 = s.load_hdf5(num, chunks=config.CHUNKSIZE)

        if all_minima:
            unique_leaves = set(s.minima[num])
        else:
            # Loop through cores and find unique node ids
            unique_leaves = set()
            for pid in s.pids:
                cores = s.cores[pid]
                if num not in cores.index:
                    # This snapshot `num` does not contain any image of the core `pid`
                    # Continue to the next core.
                    continue
                unique_leaves.add(cores.at[num, 'leaf_id'])

        for lid in unique_leaves:
            # Create directory and check if a file already exists
            ofname = Path(s.savdir, config.RPROF_DIR,
                          f'radial_profile.{lid}.{num:05d}.nc')
            ofname.parent.mkdir(exist_ok=True)
            if ofname.exists() and not overwrite:
                msg = (f"[radial_profile] A file already exists for lid = {lid} "
                       f", num = {num}. Continue to the next core")
                print(msg)
                continue

            msg = (f"[radial_profile] processing model {s.basename}, "
                   f"lid {lid}, num {num}")
            print(msg)

            # Find the location of the core
            center = s.flatindex_to_cartesian(lid)
            center = dict(zip(['x', 'y', 'z'], center))

            # Roll the data such that the core is at the center of the domain
            ds, center, _ = tools.recenter_dataset(ds0, center)

            # Calculate radial profile
            rprf = tools.radial_profile(s, ds, list(center.values()),
                                        rmax=0.53, compute_flux=True)
            rprf = rprf.expand_dims(dict(t=[ds.Time,]))

            # write to file
            if ofname.exists():
                ofname.unlink()
            rprf.to_netcdf(ofname)


def prj_radial_profile(s, num, pids, overwrite=False):
    pids_skip = []
    for pid in pids:
        cores = s.cores[pid]
        if num not in cores.index:
            pids_skip.append(pid)
            continue
        ofname = Path(s.savdir, config.RPROF_DIR,
                      'prj_radial_profile.par{}.{:05d}.nc'.format(pid, num))
        if ofname.exists() and not overwrite:
            pids_skip.append(pid)

    pids_to_process = sorted(set(pids) - set(pids_skip))

    if len(pids_to_process) == 0:
        msg = ("[prj_radial_profile] Every core alreay has radial profiles at "
               f"num = {num}. Skipping...")
        print(msg)
        return

    msg = ("[prj_radial_profile] Start reading snapshot at "
           f"num = {num}.")
    print(msg)

    # Loop through cores
    for pid in pids_to_process:
        cores = s.cores[pid]
        if num not in cores.index:
            # This snapshot `num` does not contain any image of the core `pid`
            # Continue to the next core.
            continue

        # Create directory and check if a file already exists
        ofname = Path(s.savdir, config.RPROF_DIR,
                      f'prj_radial_profile.par{pid}.{num:05d}.nc')
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            msg = (f"[prj_radial_profile] A file already exists for pid = {pid} "
                   f", num = {num}. Continue to the next core")
            print(msg)
            continue

        msg = (f"[prj_radial_profile] processing model {s.basename}, "
               f"pid {pid}, num {num}")
        print(msg)

        core = cores.loc[num]

        # Find the location of the core
        center = s.flatindex_to_cartesian(cores.at[num, 'leaf_id'])

        # Calculate radial profile
        rprf = tools.radial_profile_projected(s, num, center)
        rprf = rprf.expand_dims(dict(t=[core.time,]))

        # write to file
        if ofname.exists():
            ofname.unlink()
        rprf.to_netcdf(ofname)


def power_spectrum(s, nums=None, overwrite=False):
    if nums is None:
        nums = s.nums_with_hdf5
    for num in nums:
        ofname = Path(s.savdir, config.FOURIER_DIR, f'power_spectrum.{num:05d}.nc')
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            print('[power_spectrum] file already exists. Skipping...')
            continue

        msg = '[power_spectrum] processing model {} num {}'
        print(msg.format(s.basename, num))

        ds = s.load_hdf5(num, chunks=config.CHUNKSIZE)
        ds['mom1'] /= ds.dens
        ds['mom2'] /= ds.dens
        ds['mom3'] /= ds.dens
        ds['log_dens'] = np.log(ds.dens)
        ds = ds.rename({f'mom{i}':f'vel{i}' for i in [1,2,3]})
        fields = ['dens', 'log_dens', 'vel1', 'vel2', 'vel3', 'phi']
        ps = []
        for f in fields:
            ps.append(stats.power_spectrum(ds[f], s.domain['Nx'][0], s.Lbox,
                      nbin=s.domain['Nx'][0]))
        ps = xr.Dataset(dict(zip(fields, ps)))
        ps = ps.expand_dims(dict(t=[ds.Time,]))
        # write to file
        if ofname.exists():
            ofname.unlink()
        ps.to_netcdf(ofname)


def collapse_history(s, cores, onset_definition, *, overwrite=False):
    """Calculate and save one core's intrinsic collapse and Lagrangian history."""
    pid = cores.attrs['pid']
    ofname = Path(s.savdir, config.CORE_DIR,
                  f'collapse_history_{onset_definition.filename_token}.par{pid}.nc')
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        return

    cores = cores.copy()
    cores.attrs["onset_definition"] = json.dumps(asdict(onset_definition), sort_keys=True)
    rprofs = s.rprofs[pid]

    min_dst, mw_dst, min_dst_to_core = [], [], []
    for core in cores.itertuples():
        num = core.Index
        pds = s.load_par(num)
        dst, mass = [], []
        if len(pds) == 0:
            min_dst.append(np.nan)
            mw_dst.append(np.nan)
        else:
            for par in pds.itertuples():
                dst.append(s.distance_between(core.leaf_id,
                                                 s.cartesian_to_flatindex(par.x1, par.x2, par.x3))[()])
                mass.append(par.mass)
            min_dst.append(min(dst))
            mw_dst.append(np.average(dst, weights=mass))
        for cid in s.pids:
            other_cores = s.cores[cid]
            if num not in other_cores.index:
                continue
            other_leaf_id = other_cores.at[num, 'leaf_id']
            if other_leaf_id == core.leaf_id:
                continue
            dst.append(s.distance_between(core.leaf_id, other_leaf_id))
        if len(dst) == 0:
            min_dst_to_core.append(np.inf)
        else:
            min_dst_to_core.append(min(dst))
    cores['min_dst_to_star'] = np.asarray(min_dst, dtype=np.float64)
    cores['mw_dst_to_star'] = np.asarray(mw_dst, dtype=np.float64)
    cores['min_dst_to_pscore'] = np.asarray(min_dst_to_core, dtype=np.float64)

    if onset_definition.rcrit_from == 'tes':
        cores['rcrit'] = cores['rtes']
    else:
        cores['rcrit'] = tools.virial_radius(rprofs, onset_definition)

    # Find critical time
    ncrit, rcrit = tools.critical_time(s, cores)
    cores.attrs['numcrit'] = ncrit
    if np.isnan(ncrit):
        cores.attrs['tcrit'] = np.nan
        cores.attrs['rcore'] = np.nan
        cores.attrs['mcore'] = np.nan
        cores.attrs['mean_density'] = np.nan
        cores.attrs['tff_crit'] = np.nan
    else:
        core = cores.loc[ncrit]
        rprf = rprofs.sel(num=ncrit)
        rcore = rcrit
        if np.isnan(rcore):
            raise ValueError("Critical radius at t_crit is NaN: "
                             f"Model {s.basename}, par {pid}, ncrit = {ncrit}"
                             f" definition {onset_definition}")
        if rcore > rprf.r.max()[()]:
            msg = (
                f"Core radius exceeds the maximum rprof radius for "
                f"model {s.basename}, par {pid}. ncrit = {ncrit}, "
                f"rcore = {rcore:.2f}; rprf_max = {rprf.r.max().data[()]:.2f}"
            )
            raise ValueError(msg)
        if np.isfinite(rcore):
            mcore = rprf.menc.interp(r=rcore).data[()]
            mean_density = mcore / (4*np.pi*rcore**3/3)
            tff_crit = tools.tfreefall(mean_density, s.gconst)
        else:
            mcore = np.nan
            mean_density = np.nan
            tff_crit = np.nan
        cores.attrs['tcrit'] = core.time
        cores.attrs['rcore'] = rcore
        cores.attrs['mcore'] = mcore
        cores.attrs['mean_density'] = mean_density
        cores.attrs['tff_crit'] = tff_crit

    # Calculate Lagrangian properties after determining the onset.
    if np.isfinite(ncrit):
        lprops = tools.lagrangian_property(s, cores)
        # Save attributes before performing join, which will drop them.
        attrs = cores.attrs.copy()
        attrs.update(lprops.attrs)
        cores = cores.join(lprops)
        # Reattach attributes
        cores.attrs = attrs

        # Net force
        Fnet = cores.Fthm + cores.Ftrb + cores.Fcen + cores.Fani - cores.Fgrv
        if s.mhd:
            Fnet += cores.Fmag
        cores['Fnet'] = Fnet / cores.Fgrv

    mcore = cores.attrs['mcore']
    rcore = cores.attrs['rcore']

    # Building time
    if np.isnan(ncrit):
        cores.attrs['dt_build'] = np.nan
    else:
        rprf = rprofs.sel(num=ncrit)
        mdot = (-4*np.pi*rcore**2*rprf.rho*rprf.vel1_mw).interp(r=rcore).data[()]
        cores.attrs['dt_build'] = mcore / mdot

    # Collapse time
    cores.attrs['dt_coll'] = cores.attrs['tcoll'] - cores.attrs['tcrit']

    # Infall time
    if np.isnan(mcore):
        tf = np.nan
    else:
        phst = s.load_parhst(pid)
        idx = phst.mass.sub(mcore).abs().argmin()
        if idx == phst.index[-1]:
            tf = np.nan
        else:
            tf = phst.loc[idx].time
    cores.attrs['tinfall_end'] = tf
    cores.attrs['dt_infall'] = tf - cores.attrs['tcoll']

    # Calculate normalized times
    cores.insert(1, 'tnorm1',
                 (cores.time - cores.attrs['tcoll'])
                  / cores.attrs['tff_crit'])
    cores.insert(2, 'tnorm2',
                 (cores.time - cores.attrs['tcrit']) / cores.attrs['dt_coll'])
    cores.insert(3, 'tnorm3',
                 (cores.time - cores.attrs['tcoll']) / cores.attrs['dt_coll'])


    cores.attrs = {k: cores.attrs[k] for k in sorted(cores.attrs)}
    myio.save_dataframe(cores, ofname)
    ofname.with_name(f'collapse_history_{onset_definition.filename_token}.p').unlink(
        missing_ok=True)


def projections(s, nums=None, overwrite=False):
    if nums is None:
        nums = s.nums_with_hdf5
    nums = np.atleast_1d(nums)

    def threshold(ds, ncrit, method):
        if method == 'tophat':
            return ds.where((ds.dens >= ncrit) &
                            (ds.dens < 10*ncrit), other=0)
        if method == 'step':
            return ds.where(ds.dens >= ncrit, other=0)
        raise ValueError(f'Unknown method {method}')

    def weighted_mean(data, weight, dim):
        return (data*weight).sum(dim) / weight.sum(dim)

    axtoi = dict(x=0, y=1, z=2)

    for num in nums:
        ofname = Path(s.savdir, config.PROJ_DIR,
                      f'projection.{num:05d}.nc')
        ofname.parent.mkdir(parents=True, exist_ok=True)
        if ofname.exists() and not overwrite:
            print(f'[projections] {ofname} already exists. Skipping...')
            continue

        print(f'[projections] processing model {s.basename} num {num}')
        ds = s.load_hdf5(num, chunks=config.CHUNKSIZE)
        data_vars = {}

        for ax, i in axtoi.items():
            dx = s.domain['dx'][i]
            ds[f'vel{i+1}'] = ds[f'mom{i+1}'] / ds.dens

            data_vars[f'{ax}_Sigma_gas'] = (ds.dens*dx).sum(ax)
            for method in ['tophat', 'step']:
                for ncrit in [10, 30, 100]:
                    d = threshold(ds, ncrit, method)
                    # Surface density
                    name = f'{ax}_Sigma_gas_mtd{method}_nc{ncrit}'
                    data_vars[name] = (d.dens*dx).sum(ax)

                    # Velocity and velocity dispersion
                    vel = d[f'vel{i+1}']
                    vel_los = weighted_mean(vel, d.dens, ax)
                    vdisp_los = np.sqrt(
                        weighted_mean(vel**2, d.dens, ax) - vel_los**2
                    )
                    name = f'{ax}_vel_mtd{method}_nc{ncrit}'
                    data_vars[name] = vel_los
                    name = f'{ax}_veldisp_mtd{method}_nc{ncrit}'
                    data_vars[name] = vdisp_los
        prj = xr.Dataset(data_vars)
        prj = prj.expand_dims(dict(t=[ds.Time,]))

        if ofname.exists():
            ofname.unlink()
        prj.to_netcdf(ofname)

def observables(s, pid, num, overwrite=False):
    # Check if file exists
    ofname = Path(s.savdir, config.CORE_DIR,
                  'observables.par{}.{:05d}.p'.format(pid, num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[observables] file already exists. Skipping...')
        return

    if num not in s.rprofs[pid].num:
        msg = (f"Radial profile for num={num} does not exist. "
                "Cannot calculate observables. Skipping...")
        logging.warning(msg)
        return

    msg = '[observables] processing model {} pid {} num {}'
    print(msg.format(s.basename, pid, num))

    # Calculate observables
    observables = tools.observable(s, pid, num)

    # write to file
    if ofname.exists():
        ofname.unlink()
    with open(ofname, 'wb') as handle:
        pickle.dump(observables, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ofname.with_name("observables.p").unlink(missing_ok=True)


def save_minima(s, overwrite=False):
    """Save indices of the potential minima

    Parameters
    ----------
    s : LoadSim
        Simulation metadata.
    num : int
        Snapshot number.
    """
    # Check if file exists
    ofname = Path(s.savdir, 'GRID', 'minima.p')
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[save_minima] file already exists. Skipping...')
        return

    minima = dict()
    for num in s.nums:
        print('[save_minima] processing model {} num {}'.format(s.basename, num))
        ds = s.load_hdf5(num, chunks=config.CHUNKSIZE)
        arr = ds.phi.data
        arr_min_filtered = arr.map_overlap(
            minimum_filter, depth=1, boundary='periodic', size=3, mode='wrap'
        ).flatten()
        arr = arr.flatten()
        minima[num] = ((arr == arr_min_filtered).nonzero()[0]).compute()

    with open(ofname, 'wb') as handle:
        pickle.dump(minima, handle, protocol=pickle.HIGHEST_PROTOCOL)

def resample_hdf5(s, level=0):
    """Resamples AMR output into uniform resolution.

    Reads a HDF5 file with a mesh refinement and resample it to uniform
    resolution amounting to a given refinement level.

    Resampled HDF5 file will be written as
        {basedir}/uniform/{problem_id}.level{level}.?????.athdf

    Args:
        s: LoadSim instance
        level: Refinement level to resample. root level=0.
    """
    if not s.legacy:
        raise ValueError('Uniform HDF5 resampling is a legacy-only utility')
    ifname = Path(s.basedir, '{}.out2'.format(s.problem_id))
    odir = Path(s.basedir, 'uniform')
    odir.mkdir(exist_ok=True)
    ofname = odir / '{}.level{}'.format(s.problem_id, level)
    kwargs = dict(start=s.nums[0],
                  end=s.nums[-1],
                  stride=1,
                  input_filename=ifname,
                  output_filename=ofname,
                  level=level,
                  m=None,
                  x=None,
                  quantities=None)
    uniform.main(**kwargs)


def plot_core_evolution(s, cores, num, overwrite=False):
    """Creates multi-panel plot for t_coll core properties

    Parameters
    ----------
    s : LoadSim
        Simulation metadata.
    cores : pandas.DataFrame
        Selected trajectory with onset-definition metadata.
    num : int
        Snapshot number.
    overwrite : str, optional
        If true, overwrite output files.
    """
    pid = cores.attrs['pid']
    onset_def = tools.CollapseOnsetDefinition(
        **json.loads(cores.attrs["onset_definition"]))

    fname = Path(s.savdir, 'figures', "{}.par{}.tcrit_{}.{:05d}.png".format(
                 config.PLOT_PREFIX_CORE_EVOLUTION, pid, onset_def.filename_token, num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_core_evolution] file already exists. Skipping...')
        return
    print(f'[plot_core_evolution] processing model {s.basename} pid: {pid} num: {num}, onset definition: {onset_def}')
    fig = plots.plot_core_evolution(s, cores, num)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_mass_radius(s, cores, overwrite=False):
    pid = cores.attrs['pid']
    onset_def = tools.CollapseOnsetDefinition(
        **json.loads(cores.attrs["onset_definition"]))
    fig = plt.figure()
    ax = fig.add_subplot()
    for num in cores.index:
        msg = '[plot_mass_radius] processing model {} pid {} num {}'
        msg = msg.format(s.basename, pid, num)
        print(msg)
        fname = Path(s.savdir, 'figures', "{}.par{}.tcrit_{}.{:05d}.png".format(
            config.PLOT_PREFIX_MASS_RADIUS, pid, onset_def.filename_token, num))
        fname.parent.mkdir(exist_ok=True)
        if fname.exists() and not overwrite:
            print('[plot_mass_radius] file already exists. Skipping...')
            return
        plots.mass_radius(s, cores, num, ax=ax)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        ax.cla()


def plot_sink_history(s, num, overwrite=False):
    """Creates multi-panel plot for sink particle history

    Args:
        s: LoadSim instance
    """
    fname = Path(s.savdir, 'figures', "{}.{:05d}.png".format(
                 config.PLOT_PREFIX_SINK_HISTORY, num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_sink_history] file already exists. Skipping...')
        return
    print(f'[plot_sink_history] processing model {s.basename} num: {num}')
    fig = plots.plot_sinkhistory(s, num)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_diagnostics(s, cores, overwrite=False):
    """Creates diagnostics plots for a given model

    Save projections in {basedir}/figures for all snapshots.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance
    cores : pandas.DataFrame
        Selected trajectory with onset-definition metadata
    overwrite : bool, optional
        Flag to overwrite
    """
    pid = cores.attrs['pid']
    onset_def = tools.CollapseOnsetDefinition(
        **json.loads(cores.attrs["onset_definition"]))
    fname = Path(s.savdir, 'figures',
                 f'diagnostics_normalized.par{pid}.tcrit_{onset_def.filename_token}.png')
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_diagnostics] file already exists. Skipping...')
        return

    msg = '[plot_diagnostics] model {} pid {}'
    print(msg.format(s.basename, pid))

    fig = plots.plot_diagnostics(s, cores, normalize_time=True)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)

    fname = Path(s.savdir, 'figures', f'diagnostics.par{pid}.tcrit_{onset_def.filename_token}.png')
    if fname.exists() and not overwrite:
        return
    fig = plots.plot_diagnostics(s, cores, normalize_time=False)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_radial_profile_at_tcrit(s, all_cores, nrows=5, ncols=6, overwrite=False):
    """Plot resolved cores from one selected population."""
    if not all_cores:
        return
    onset_def = tools.CollapseOnsetDefinition(
        **json.loads(next(iter(all_cores.values())).attrs["onset_definition"]))
    fname = Path(s.savdir, 'figures',
                 f'radial_profile_at_tcrit.tcrit_{onset_def.filename_token}.png')
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_radial_profile_at_tcrit] file already exists. Skipping...')
        return

    msg = '[plot_radial_profile_at_tcrit] Processing model {}'
    print(msg.format(s.basename))

    pids = s.good_cores(all_cores)
    if len(pids) > nrows*ncols:
        raise ValueError("Number of good cores {} exceeds the number of panels.".format(len(pids)))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), sharex=True, squeeze=False,
                            gridspec_kw={'hspace':0.05, 'wspace':0.12})
    for pid, ax in zip(pids, axs.flat):
        cores = all_cores[pid]
        plots.radial_profile_at_tcrit(s, cores, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.text(0.6, 0.86, f"pid {pid}", transform=ax.transAxes)
        nc = cores.attrs['numcrit']
        ax.text(0.6, 0.73, "{:.2f} tff".format(cores.at[nc, 'tnorm1']),
                transform=ax.transAxes)
    for ax in axs[:, 0]:
        ax.set_ylabel(r'$\rho/\rho_c$')
    for ax in axs[-1, :]:
        ax.set_xlabel(r'$r/r_\mathrm{TES}$')
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def calculate_linewidth_size(s, num, seed=None, pid=None, overwrite=False, ds=None):
    if seed is not None and pid is not None:
        raise ValueError("Provide either seed or pid, not both")
    elif seed is not None:
        # Check if file exists
        ofname = Path(s.savdir, 'linewidth_size',
                      'linewidth_size.{:05d}.{}.nc'.format(num, seed))
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            print('[linewidth_size] file already exists. Skipping...')
            return

        msg = '[linewidth_size] processing model {} num {} seed {}'
        print(msg.format(s.basename, num, seed))

        if ds is None:
            ds = s.load_hdf5(num, quantities=['dens', 'mom1', 'mom2', 'mom3'])
            ds['vel1'] = ds.mom1/ds.dens
            ds['vel2'] = ds.mom2/ds.dens
            ds['vel3'] = ds.mom3/ds.dens

        if len(np.unique(s.domain['Nx'])) > 1:
            raise ValueError("Cubic domain is assumed, but the domain is not cubic")
        Nx = s.domain['Nx'][0]  # Assume cubic domain
        rng = np.random.default_rng(seed)
        i, j, k = rng.integers(low=0, high=Nx-1, size=(3))
        origin = (ds.x.isel(x=i).data[()],
                  ds.y.isel(y=j).data[()],
                  ds.z.isel(z=k).data[()])
    elif pid is not None:
        if num not in s.cores[pid].index:
            print(f'[linewidth_size] {num} is not in the snapshot list of core {pid}')
            return
        elif num > s.cores[pid].attrs['numcoll']:
            print(f'[linewidth_size] core {pid} is protostellar at snapshot {num}')
            return

        # Check if file exists
        ofname = Path(s.savdir, 'linewidth_size',
                      'linewidth_size.{:05d}.par{}.nc'.format(num, pid))
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            print('[linewidth_size] file already exists. Skipping...')
            return

        msg = '[linewidth_size] processing model {} num {} pid {}'
        print(msg.format(s.basename, num, pid))

        lid = s.cores[pid].at[num, 'leaf_id']
        origin = s.flatindex_to_cartesian(lid)

        if ds is None:
            ds = s.load_hdf5(num, quantities=['dens', 'mom1', 'mom2', 'mom3'])
            ds['vel1'] = ds.mom1/ds.dens
            ds['vel2'] = ds.mom2/ds.dens
            ds['vel3'] = ds.mom3/ds.dens
    else:
        raise ValueError("Provide either seed or pid")

    ds, origin, _ = tools.recenter_dataset(ds, dict(x=origin[0], y=origin[1], z=origin[2]))
    ds.coords['r'] = np.sqrt((ds.z - origin['z'])**2 + (ds.y - origin['y'])**2 + (ds.x - origin['x'])**2)

    rmax = s.Lbox/2

    nbin = int(np.ceil(rmax/s.dx))
    ledge = 0.5*s.dx
    redge = (nbin + 0.5)*s.dx

    # Convert density and velocities to spherical coord.
    vel = {}
    for dim, axis in zip(['x', 'y', 'z'], [1, 2, 3]):
        # Recenter velocity
        vel_ = ds['vel{}'.format(axis)]
        dvel_ = vel_ - vel_.sel(x=origin['x'], y=origin['y'], z=origin['z'])
        ds[f'vel{axis}'] = dvel_
        vel[dim] = dvel_

    _, (ds['vels1'], ds['vels2'], ds['vels3'])\
        = transform.to_spherical(vel.values(), origin.values())

    rprf = {}
    for cum_flag, suffix in zip([True, False], ['', '_sh']):
        rprf['rho'+suffix] = transform.fast_groupby_bins(ds.dens, 'r', ledge, redge, nbin, cumulative=cum_flag)
        for k in ['vel1', 'vel2', 'vel3', 'vels1', 'vels2', 'vels3']:
            rprf[k+suffix] = transform.fast_groupby_bins(ds[k], 'r', ledge, redge, nbin, cumulative=cum_flag)
            rprf[f'{k}_sq'+suffix] = transform.fast_groupby_bins(ds[k]**2, 'r', ledge, redge, nbin, cumulative=cum_flag)
            rprf[f'd{k}'+suffix] = np.sqrt(rprf[f'{k}_sq'+suffix] - rprf[k+suffix]**2)
            # Mass weighted
            rprf[k+suffix+'_mw'] = transform.fast_groupby_bins(ds.dens*ds[k], 'r', ledge, redge, nbin, cumulative=cum_flag) / rprf['rho'+suffix]
            rprf[f'{k}_sq'+suffix+'_mw'] = transform.fast_groupby_bins(ds.dens*ds[k]**2, 'r', ledge, redge, nbin, cumulative=cum_flag) / rprf['rho'+suffix]
            rprf[f'd{k}'+suffix+'_mw'] = np.sqrt(rprf[f'{k}_sq'+suffix+'_mw'] - rprf[k+suffix+'_mw']**2)
    rprf = xr.Dataset(rprf)

    # write to file
    if ofname.exists():
        ofname.unlink()
    rprf.to_netcdf(ofname)


def plot_pdfs(s, num, overwrite=False):
    """Creates density PDF and velocity power spectrum for a given model

    Save figures in {basedir}/figures for all snapshots.

    Args:
        s: LoadSim instance
    """
    fname = Path(s.savdir, 'figures', "{}.{:05d}.png".format(
        config.PLOT_PREFIX_PDF_PSPEC, num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_pdfs] file already exists. Skipping...')
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax1_twiny = axs[1].twiny()

    ds = s.load_hdf5(num, quantities=['dens', 'mom1', 'mom2', 'mom3'],
                     load_method='xarray')
    plots.plot_PDF(s, ds, axs[0])
    plots.plot_Pspec(s, ds, axs[1], ax1_twiny)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
