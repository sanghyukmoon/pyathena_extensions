"""Module containing functions that are not generally reusable"""
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import numpy as np
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
from grid_dendro import dendrogram

from . import plots, tools, config


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
    """
    outid = s._hdf5_outid_def
    outvar = s._hdf5_outvar_def
    filename = s._get_fhdf5(outid, outvar, num, None)
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
        s.basedir, "sparse", f"{s.problem_id}.{num:05d}.athdf"
    )
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists():
        ofname.unlink()
    da.to_hdf5(ofname, ds)
    fdst = h5py.File(ofname, 'a')
    fdst.attrs.update(fsrc.attrs)

    dataset_names = set(name.decode() for name in fsrc.attrs['DatasetNames'])
    for k in set(fsrc.keys()) - dataset_names:
        fsrc.copy(k, fdst)
    fdst.create_dataset("gids", data=gids)
    fsrc.close()
    fdst.close()


def critical_tes(s, pid, num, overwrite=False):
    """Calculates and saves critical tes associated with each core.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    pid : int
        Particle id.
    num : int
        Snapshot number
    overwrite : str, optional
        If true, overwrites the existing pickle file.
    """
    # Check if file exists
    ofname = Path(s.savdir, config.CORE_DIR,
                  'critical_tes.par{}.{:05d}.p'.format(pid, num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[critical_tes] file already exists. Skipping...')
        return

    if num not in s.rprofs[pid].num:
        msg = (f"Radial profile for pid={pid}, num={num} does not exist. "
                "Cannot calculate critical_tes. Skipping...")
        logging.warning(msg)
        return

    msg = '[critical_tes] processing model {} pid {} num {}'
    print(msg.format(s.basename, pid, num))

    # Load the radial profile
    rprf = s.rprofs[pid].sel(num=num)
    core = s.cores[pid].loc[num]

    # Calculate critical TES
    critical_tes = tools.critical_tes_property(s, rprf, core)
    critical_tes['num'] = num

    # write to file
    if ofname.exists():
        ofname.unlink()
    with open(ofname, 'wb') as handle:
        pickle.dump(critical_tes, handle, protocol=pickle.HIGHEST_PROTOCOL)


def core_tracking(s, pid, overwrite=False):
    """Loops over all sink particles and find their progenitor cores

    Finds a unique grid-dendro leaf at each snapshot that is going to collapse.
    For each sink particle, back-traces the evolution of its progenitor cores.
    Pickles the resulting data.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    pid : int
        Particle ID
    overwrite : str, optional
        If true, overwrites the existing pickle file.
    """
    # Check if file exists
    ofname = Path(s.savdir, config.CORE_DIR, 'cores.par{}.p'.format(pid))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[core_tracking] file already exists. Skipping...')
        return

    cores = tools.track_cores(s, pid)
    cores.to_pickle(ofname, protocol=pickle.HIGHEST_PROTOCOL)


def radial_profile(s, num, pids, overwrite=False, full_radius=False, days_overwrite=30):
    """Calculates and pickles radial profiles of all cores.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    num : int
        Snapshot number
    pids : list of int
        Particle ids to process.
    overwrite : str, optional
        If true, overwrites the existing pickle file.
    full_radius : bool, optional
        If true, use the full domain size as the outer radius.
    """

    pids_skip = []
    for pid in pids:
        cores = s.cores[pid]
        if num not in cores.index:
            pids_skip.append(pid)
            continue
        ofname = Path(s.savdir, config.RPROF_DIR,
                      'radial_profile.par{}.{:05d}.nc'.format(pid, num))
        if ofname.exists():
            if overwrite:
                creation_time = ofname.stat().st_ctime
                creation_date = datetime.datetime.fromtimestamp(creation_time)
                current_time = datetime.datetime.now()
                if (current_time - creation_date).days < days_overwrite:
                    pids_skip.append(pid)
                else:
                    pass
            else:
                pids_skip.append(pid)

    pids_to_process = sorted(set(pids) - set(pids_skip))

    if len(pids_to_process) == 0:
        msg = ("[radial_profile] Every core alreay has radial profiles at "
               f"num = {num}. Skipping...")
        print(msg)
        return

    msg = ("[radial_profile] Start reading snapshot at "
           f"num = {num}.")
    print(msg)

    # Load the snapshot
    # ds0 should not be modified in the following loop.
    ds0 = s.load_hdf5(num, chunks=config.CHUNKSIZE)

    # Loop through cores
    for pid in pids_to_process:
        cores = s.cores[pid]
        if num not in cores.index:
            # This snapshot `num` does not contain any image of the core `pid`
            # Continue to the next core.
            continue

        # Create directory and check if a file already exists
        ofname = Path(s.savdir, config.RPROF_DIR,
                      f'radial_profile.par{pid}.{num:05d}.nc')
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            msg = (f"[radial_profile] A file already exists for pid = {pid} "
                   f", num = {num}. Continue to the next core")
            print(msg)
            continue

        msg = (f"[radial_profile] processing model {s.basename}, "
               f"pid {pid}, num {num}")
        print(msg)

        core = cores.loc[num]

        if full_radius:
            rmax = 0.5*s.Lbox
        else:
            rmax = min(0.5*s.Lbox, 3*cores.loc[:cores.attrs['numcoll']].tidal_radius.max())

        # Find the location of the core
        center = s.flatindex_to_cartesian(core.leaf_id)
        center = dict(zip(['x', 'y', 'z'], center))

        # Roll the data such that the core is at the center of the domain
        ds, center, _ = tools.recenter_dataset(ds0, center)

        # Workaround for xarray being unable to chunk IndexVariable
        # see https://github.com/pydata/xarray/issues/6204
        # The workaround is provided by _chunk_like helper function introduced in
        # xclim. See https://github.com/Ouranosinc/xclim/pull/1542
        x, y, z = transform._chunk_like(ds.x, ds.y, ds.z, chunks=ds.chunksizes)

        # Calculate the angular momentum vector within the tidal radius.
        x = x - center['x']
        y = y - center['y']
        z = z - center['z']
        r = np.sqrt(z**2 + y**2 + x**2)
        lx = (y*ds.mom3 - z*ds.mom2).where(r <= max(core.tidal_radius, 1.1*s.dx)).sum().data[()]*s.dV
        ly = (z*ds.mom1 - x*ds.mom3).where(r <= max(core.tidal_radius, 1.1*s.dx)).sum().data[()]*s.dV
        lz = (x*ds.mom2 - y*ds.mom1).where(r <= max(core.tidal_radius, 1.1*s.dx)).sum().data[()]*s.dV
        lvec = (lx, ly, lz)

        # Calculate radial profile
        rprf = tools.radial_profile(s, ds, list(center.values()), rmax, lvec)
        rprf = rprf.expand_dims(dict(t=[ds.Time,]))
        rprf['lx'] = xr.DataArray(np.atleast_1d(lx), dims='t')
        rprf['ly'] = xr.DataArray(np.atleast_1d(ly), dims='t')
        rprf['lz'] = xr.DataArray(np.atleast_1d(lz), dims='t')

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
        center = s.flatindex_to_cartesian(core.leaf_id)

        # Calculate radial profile
        rprf = tools.radial_profile_projected(s, num, center)
        rprf = rprf.expand_dims(dict(t=[core.time,]))

        # write to file
        if ofname.exists():
            ofname.unlink()
        rprf.to_netcdf(ofname)


def lagrangian_props(s, pid, method='empirical', overwrite=False):
    # Check if file exists
    ofname = Path(s.savdir, config.CORE_DIR, f'lprops_tcrit_{method}.par{pid}.p')
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[lagrangian_props] file already exists. Skipping...')
        return

    s.select_cores(method)
    cores = s.cores[pid]
    rprofs = s.rprofs[pid]
    print(f'[lagrangian_props] Calculate Lagrangian props for core {pid} with version {method}')
    lprops = tools.lagrangian_property(s, cores, rprofs)
    lprops.to_pickle(ofname, protocol=pickle.HIGHEST_PROTOCOL)


def projections(s, num, overwrite=True):
    msg = '[projections] processing model {} num {}'
    print(msg.format(s.basename, num))
    s.read_prj(num, force_override=overwrite)


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

    # Load the radial profile
    rprf = s.rprofs[pid].sel(num=num)
    core = s.cores[pid].loc[num]

    # Calculate observables
    observables = tools.observable(s, core, rprf)

    # write to file
    if ofname.exists():
        ofname.unlink()
    with open(ofname, 'wb') as handle:
        pickle.dump(observables, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_minima(s, overwrite=False):
    """Run GRID-dendro

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
        # Load data and construct dendrogram
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


def run_grid(s, num, overwrite=False):
    """Run GRID-dendro

    Parameters
    ----------
    s : LoadSim
        Simulation metadata.
    num : int
        Snapshot number.
    """
    # Check if file exists
    ofname = Path(s.savdir, 'GRID',
                  'dendrogram.{:05d}.p'.format(num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[run_grid] file already exists. Skipping...')
        return

    # Load data and construct dendrogram
    print('[run_grid] processing model {} num {}'.format(s.basename, num))
    ds = s.load_hdf5(num, quantities=['phi',],
                     load_method='xarray').transpose('z', 'y', 'x')
    phi = ds.phi.to_numpy()
    gd = dendrogram.Dendrogram(phi, verbose=False)
    gd.construct()

    # Write to file
    with open(ofname, 'wb') as handle:
        pickle.dump(gd, handle, protocol=pickle.HIGHEST_PROTOCOL)


def prune(s, num, overwrite=False):
    """Prune GRID-dendro

    Parameters
    ----------
    s : LoadSim
        Simulation metadata.
    num : int
        Snapshot number.
    """
    # Check if file exists
    ofname = Path(s.savdir, 'GRID',
                  'dendrogram.pruned.{:05d}.p'.format(num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[prune] file already exists. Skipping...')
        return

    # Load original dendrogram and prune it
    print('[prune] processing model {} num {}'.format(s.basename, num))
    gd = s.load_dendro(num, pruned=False)
    gd.prune()

    # Write to file
    with open(ofname, 'wb') as handle:
        pickle.dump(gd, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


def plot_core_evolution(s, pid, num, method='empirical', overwrite=False, rmax=None):
    """Creates multi-panel plot for t_coll core properties

    Parameters
    ----------
    s : LoadSim
        Simulation metadata.
    pid : int
        Unique ID of a selected particle.
    num : int
        Snapshot number.
    overwrite : str, optional
        If true, overwrite output files.
    """
    fname = Path(s.savdir, 'figures', "{}.par{}.tcrit_{}.{:05d}.png".format(
                 config.PLOT_PREFIX_CORE_EVOLUTION, pid, method, num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_core_evolution] file already exists. Skipping...')
        return
    print(f'[plot_core_evolution] processing model {s.basename} pid: {pid} num: {num}, tcrit_method: {method}')
    s.select_cores(method)
    fig = plots.plot_core_evolution(s, pid, num, rmax=rmax)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_mass_radius(s, pid, overwrite=False):
    fig = plt.figure()
    ax = fig.add_subplot()
    for num in s.cores[pid].index:
        msg = '[plot_mass_radius] processing model {} pid {} num {}'
        msg = msg.format(s.basename, pid, num)
        print(msg)
        fname = Path(s.savdir, 'figures', "{}.par{}.{:05d}.png".format(
            config.PLOT_PREFIX_MASS_RADIUS, pid, num))
        fname.parent.mkdir(exist_ok=True)
        if fname.exists() and not overwrite:
            print('[plot_mass_radius] file already exists. Skipping...')
            return
        plots.mass_radius(s, pid, num, ax=ax)
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
    ds = s.load_hdf5(num, quantities=['dens',], load_method='xarray')
    pds = s.load_par(num)
    fig = plots.plot_sinkhistory(s, ds, pds)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_core_structure(s, pid, overwrite=False):
    rmax = s.cores[pid].tidal_radius.max()
    for num in s.cores[pid].index:
        fname = Path(s.savdir, 'figures', "core_structure.par{}.{:05d}.png".format(pid, num))
        if fname.exists() and not overwrite:
            print('[plot_core_structure] file already exists. Skipping...')
            return
        msg = '[plot_core_structure] processing model {} pid {} num {}'
        msg = msg.format(s.basename, pid, num)
        print(msg)
        fig = plots.core_structure(s, pid, num, rmax=rmax)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        plt.close(fig)


def plot_diagnostics(s, pid, overwrite=False):
    """Creates diagnostics plots for a given model

    Save projections in {basedir}/figures for all snapshots.

    Parameters
    ----------
    s : LoadSim
        LoadSim instance
    pid : int
        Particle ID
    overwrite : bool, optional
        Flag to overwrite
    """
    fname = Path(s.savdir, 'figures',
                 'diagnostics_normalized.par{}.png'.format(pid))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_diagnostics] file already exists. Skipping...')
        return

    msg = '[plot_diagnostics] model {} pid {}'
    print(msg.format(s.basename, pid))

    fig = plots.plot_diagnostics(s, pid, normalize_time=True)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)

    fname = Path(s.savdir, 'figures', 'diagnostics.par{}.png'.format(pid))
    if fname.exists() and not overwrite:
        return
    fig = plots.plot_diagnostics(s, pid, normalize_time=False)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_radial_profile_at_tcrit(s, nrows=5, ncols=6, overwrite=False):
    fname = Path(s.savdir, 'figures', 'radial_profile_at_tcrit.png')
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_radial_profile_at_tcrit] file already exists. Skipping...')
        return

    msg = '[plot_radial_profile_at_tcrit] Processing model {}'
    print(msg.format(s.basename))

    if len(s.good_cores()) > nrows*ncols:
        raise ValueError("Number of good cores {} exceeds the number of panels.".format(len(s.good_cores())))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), sharex=True,
                            gridspec_kw={'hspace':0.05, 'wspace':0.12})
    for pid, ax in zip(s.good_cores(), axs.flat):
        plots.radial_profile_at_tcrit(s, pid, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.text(0.6, 0.86, f"pid {pid}", transform=ax.transAxes)
        cores = s.cores[pid]
        nc = cores.attrs['numcrit']
        core = cores.loc[nc]
        ax.text(0.6, 0.73, "{:.2f} tff".format(core.tnorm1),
                transform=ax.transAxes)
    for ax in axs[:, 0]:
        ax.set_ylabel(r'$\rho/\rho_0$')
    for ax in axs[-1, :]:
        ax.set_xlabel(r'$r/R_\mathrm{tidal}$')
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

        lid = s.cores[pid].loc[num].leaf_id
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


def calculate_go15_core_mass(s, overwrite=False):
    """Calculate core mass using the definition of GO15

    Core mass is defined as the enclosed mass within the largest closed contour
    at t_coll
    """
    fname = Path(s.savdir) / 'mcore_go15.p'
    if fname.exists():
        if not overwrite:
            return
        else:
            fname.unlink()
    mcore = {}
    for pid in s.pids:
        cores = s.cores[pid]
        ncoll = cores.attrs['numcoll']
        ds = s.load_hdf5(ncoll, quantities=['dens'])
        gd = s.load_dendro(ncoll)
        lid = cores.loc[ncoll].leaf_id
        if np.isnan(lid):
            mcore[pid] = np.nan
        else:
            rho = gd.filter_data(ds.dens, lid, drop=True)
            mcore[pid] = (rho*s.dV).sum()
    with open(fname, 'wb') as f:
        pickle.dump(mcore, f)


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
