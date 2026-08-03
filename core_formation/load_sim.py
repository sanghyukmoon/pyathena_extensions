import os.path as osp
import warnings
import pandas as pd
import xarray as xr
# Bottleneck does not use stable sum.
# See xarray #1346, #7344 and bottleneck #193, #462 and more.
# Let's disable third party softwares to go conservative.
# Accuracy is more important than performance.
xr.set_options(use_bottleneck=False, use_numbagg=False)
import numpy as np
from pathlib import Path
import pickle
from scipy.interpolate import interp1d
from scipy import signal
from astropy import units as au
from astropy import constants as ac
from pyathena.load_sim import LoadSim as LoadSimBase
from pyathena.util.units import Units
from pyathena.io.timing_reader import TimingReader

from . import models, tools, config, hst, slc_prj, myio


class LoadSim(LoadSimBase, hst.Hst, slc_prj.SliceProj, tools.LognormalPDF,
                           TimingReader):
    """LoadSim class for analyzing core collapse simulations.

    Attributes
    ----------
    rho0 : float
        Mean density of the cloud in the code unit.
    cs : float
        Sound speed in the code unit.
    gconst : float
        Gravitational constant in the code unit.
    tff : float
        Free fall time in the code unit.
    tcr : float
        Half-box flow crossing time in the code unit.
    Mach : float
        Mach number.
    sonic_length : float
        Sonic length in the code unit.
    basedir : str
        Base directory
    problem_id : str
        Prefix of the Athena++ problem
    dx : float
        Uniform cell spacing in x direction.
    dy : float
        Uniform cell spacing in y direction.
    dz : float
        Uniform cell spacing in z direction.
    tcoll_cores : pandas DataFrame
        t_coll core information container.
    cores : dict of pandas DataFrame
        All preimages of t_coll cores.
    """

    def __init__(self, basedir_or_Mach=None, method='virial_rcrit', savdir=None,
                 verbose=False, override_all=False, override_cores=False,
                 override_rprofs=False, override_derived_cores=False,
                 load_derived_cores=True):
        """The constructor for LoadSim class for core formation simulations.

        Parameters
        ----------
        basedir_or_Mach : str or float
            Path to the directory where all data is stored;
            Alternatively, Mach number
        method : str
            Which definition of t_crit to use.
        savdir : str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        verbose : bool or str or int
            Print verbose messages using logger. If True/False, set logger
            level to 'DEBUG'/'WARNING'. If string, it should be one of the
            string representation of python logging package:
            ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Numerical values from 0 ('NOTSET') to 50 ('CRITICAL') are also
            accepted.
        """

        # Set unit system
        # [L] = L_{J,0}, [M] = M_{J,0}, [V] = c_s
        self.rho0 = 1.0
        self.cs = 1.0
        self.gconst = np.pi
        self.tff0 = tools.tfreefall(self.rho0, self.gconst)

        if override_all==True:
            override_cores = True
            override_rprofs = True
            override_derived_cores = True

        if isinstance(basedir_or_Mach, (Path, str)):
            basedir = basedir_or_Mach
            super().__init__(basedir, savdir=savdir, load_method='xarray',
                             units=Units('code'), verbose=verbose)

            # Override physical units assuming dense sub-patch of a GMC
            nH0 = 200*au.cm**-3  # Mean Hydrogen number density
            T = 10*au.K  # Temperature
            mH = 1.008*au.u  # Mass of a hydrogen atom
            mu = 14/6  # Average molecular weight per particle
            muH = 1.4  # Average molecular weight per hydrogen
            cs = np.sqrt(ac.k_B*T / (mu*mH))
            rho0 = muH*nH0*mH
            LJ0 = np.sqrt(np.pi*cs**2/(ac.G*rho0)).to('pc')
            MJ0 = (rho0*LJ0**3).to('Msun')
            tJ0 = (LJ0/cs).to('Myr')
            units_dict = {'unit_system': 'cloud',
                          'mass_cgs': MJ0.cgs.value,
                          'length_cgs': LJ0.cgs.value,
                          'time_cgs': tJ0.cgs.value,
                          'mean_mass_per_hydrogen': (muH*mH).cgs.value}
            self.u = Units('custom', units_dict=units_dict)
            self.u.number_density = (self.u.density/(self.u.muH*self.u.mH)).to('cm-3')
            self.u.column_density = (self.u.number_density*self.u.length).to('cm-2')

            self.Mach = self.par['problem']['Mach']
            if self.Mach in {5, 10}:
                if self.Mach == 5:
                    self.color = 'r'
                else:
                    self.color = 'b'
            if self.basename.replace(".", "") in models.hydro_old:
                # Old hydro models does not have 'configure' block
                # Simply set mhd = False
                self.mhd = False
            else:
                if self.par['configure']['Magnetic_fields'] == 'ON':
                    self.mhd = True
                else:
                    self.mhd = False

            self.dt_output = {}
            for k, v in self.par.items():
                if k.startswith('output'):
                    self.dt_output[v['file_type']] = v['dt']


            tools.LognormalPDF.__init__(self, self.Mach)
            TimingReader.__init__(self, self.basedir, self.problem_id)

            # Set nums dictionary (when hdf5 is stored in elsewhere for storage reasons)
            if not hasattr(self, 'nums'):
                if hasattr(self, 'nums_parbin'):
                    self.nums = self.nums_parbin['par0']
                elif hasattr(self, 'nums_partab'):
                    self.nums = self.nums_partab['par0']

            # Set domain
            Lbox = set(self.domain['Lx'])
            self.dx, self.dy, self.dz = self.domain['dx']
            self.dV = self.dx*self.dy*self.dz
            if len(Lbox) == 1:
                self.Lbox = Lbox.pop()
            else:
                raise ValueError("Box must be cubic")

            self.tcr = 0.5*self.Lbox/self.Mach
            self.sonic_length = tools.get_sonic(self.Mach, self.Lbox)

            # Find the collapse time and corresponding snapshot numbers
            self.tcoll_cores = self._load_tcoll_cores(
                savdir=Path(self.savdir, config.CORE_DIR),
                force_override=override_cores
            )
            try:
                fname = Path(self.savdir, 'GRID', 'minima.p')
                with open(fname, 'rb') as handle:
                    self.minima = pickle.load(handle)
            except FileNotFoundError:
                pass

            if len(self.tcoll_cores) > 0:
                try:
                    # Load cores
                    savdir = Path(self.savdir, config.CORE_DIR)
                    self.cores = self._load_cores(
                        savdir=savdir,
                        force_override=override_cores
                    )
                except FileNotFoundError:
                    self.logger.warning("Cannot find core files to load.")
                    pass

            if hasattr(self, 'cores'):
                try:
                    # Load radial profiles
                    savdir = Path(self.savdir, config.RPROF_DIR)
                    self.rprofs = self._load_radial_profiles(
                        savdir = savdir,
                        force_override = override_rprofs
                    )
                except FileNotFoundError:
                    self.logger.warning("Cannot find radial profile files to load. "
                                        "Have you run concat_radial_profiles() to "
                                        "concatenate individual radial profiles "
                                        "into one file?")
                    pass
            # Load derived core informations using various alternative critical times
            if load_derived_cores:
                self.cores_dict = {}
                for mtd in ['empirical', 'virial_rcrit']:
                    savdir = Path(self.savdir, config.CORE_DIR)
                    try:
                        self.cores_dict[mtd] = self.update_core_props(
                            method = mtd,
                            prefix = f'cores_tcrit_{mtd}',
                            savdir = savdir,
                            force_override = override_derived_cores
                        )
                    except (AttributeError, KeyError):
                        self.logger.warning(
                            f"Failed to update core props for method {mtd}, model {self.basename}"
                        )
                try:
                    self.select_cores(method)
                except KeyError:
                    self.logger.warning(f"Failed to select core with method {method} for model {self.basename}")
        elif isinstance(basedir_or_Mach, (float, int)):
            self.Mach = basedir_or_Mach
            tools.LognormalPDF.__init__(self, self.Mach)
        elif basedir_or_Mach is None:
            pass
        else:
            raise ValueError("Unknown parameter type for basedir_or_Mach")

    def load_hdf5(self, num, sparse=False, **kwargs):
        """Load hdf5 file

        Parameters
        ----------
        num : int
            Snapshot number.
        sparse : bool
            If True, load only the header information.
        """
        if sparse:
            outid = self._hdf5_outid_def
            outvar = self._hdf5_outvar_def
            fname = Path(
                self.basedir, "sparse", f"{self.problem_id}.{num:05d}.athdf"
            )
            if not fname.exists():
                raise FileNotFoundError('sparse hdf5 file does not exist.')
            if 'chunks' in kwargs:
                chunks = (kwargs['chunks']['x'],
                          kwargs['chunks']['y'],
                          kwargs['chunks']['z'])
            else:
                raise ValueError("chunks must be specified for sparse hdf5")
            return myio.read_sparse_hdf5(fname, chunks)
        else:
            return LoadSimBase.load_hdf5(self, num, **kwargs)

    def load_par(self, num, **kwargs):
        """Load partab or parbin"""
        if 'parbin' in self.files:
            return self.load_parbin(num, **kwargs)
        elif 'partab' in self.files:
            return self.load_partab(num, **kwargs)
        else:
            raise FileNotFoundError("partab or parbin not found")

    def load_dendro(self, num, pruned=True):
        """Load pickled dendrogram object

        Parameters
        ----------
        num : int
            Snapshot number.
        pruned : bool
            If true, load the pruned dendrogram
        """
        if pruned:
            fname = Path(self.savdir, 'GRID',
                         'dendrogram.pruned.{:05d}.p'.format(num))
        else:
            fname = Path(self.savdir, 'GRID',
                         'dendrogram.{:05d}.p'.format(num))

        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    def num_to_time(self, num):
        ds = self.load_hdf5(num, header_only=True)
        return ds['Time']

    def select_cores(self, method):
        self.cores = self.cores_dict[method].copy()

    def good_cores(self, nres=8):
        """List of resolved cores"""
        good_cores = []
        for pid, cores in self.cores.items():
            if cores.attrs['track_failed']:
                # Exclude cores that failed to be tracked before collapse.
                continue
            if tools.test_resolved_core(self, cores, nres):
                # Exclude cores that are not resolved at the critical time.
                good_cores.append(pid)
        return good_cores

    @LoadSimBase.Decorators.check_pickle
    def update_core_props(self, method,
                          prefix=None, savdir=None, force_override=False):
        """Update core properties

        Calculate lagrangian core properties using the radial profiles
        Add normalized times

        Parameters
        ----------

        Returns
        -------
        pandas.DataFrame
            Updated core dataframe.
        """
        core_dict = {}
        for pid in self.pids:
            cores = self.cores[pid].copy()
            if cores.attrs['track_failed']:
                core_dict[pid] = cores
                continue

            rprofs = self.rprofs[pid]

            min_dst, mw_dst, min_dst_to_core = [], [], []
            for num in cores.index:
                pds = self.load_par(num)
                core = cores.loc[num]
                dst, mass = [], []
                if len(pds) == 0:
                    min_dst.append(np.nan)
                    mw_dst.append(np.nan)
                else:
                    for _, par in pds.iterrows():
                        dst.append(self.distance_between(core.leaf_id,
                                                         self.cartesian_to_flatindex(par.x1, par.x2, par.x3))[()])
                        mass.append(par.mass)
                    min_dst.append(min(dst))
                    mw_dst.append(np.average(dst, weights=mass))
                for cid in self.pids:
                    other_cores = self.cores[cid]
                    if num not in other_cores.index:
                        continue
                    other_core = other_cores.loc[num]
                    if other_core.leaf_id == core.leaf_id:
                        continue
                    dst.append(self.distance_between(core.leaf_id, other_core.leaf_id))
                if len(dst) == 0:
                    min_dst_to_core.append(np.inf)
                else:
                    min_dst_to_core.append(min(dst))
            cores['min_dst_to_star'] = min_dst
            cores['mw_dst_to_star'] = mw_dst
            cores['min_dst_to_pscore'] = min_dst_to_core

#            # Critical radius based on menc/mmax, restricted to
#            # 0 <= r <= min_dst_to_star for each snapshot.
#            ratio = rprofs.menc / rprofs.mmax
#            dst_to_star = cores.min_dst_to_star.replace(np.nan, np.inf)
#            dst_to_star = xr.DataArray(dst_to_star.to_numpy(), coords=dict(t=rprofs.t), dims='t')
#            within_cut = (rprofs.r >= 0) & (rprofs.r <= dst_to_star)
#            virial_rcrit = ratio.where(within_cut).idxmax('r')
#            cores['virial_rcrit'] = pd.Series(virial_rcrit.data,
#                                              index=rprofs.num.data)

            r = rprofs.r.to_numpy()
            virial_rcrit = []
            for num in cores.index:
                rprf = rprofs.sel(num=num)
                x = (rprf.menc/rprf.mmax).to_numpy()
                x[0] = 0  # Avoid NaN
                # order=4 means that a point is considered a local maximum
                # if it is greater than its 4 neighbors on each side.
                peaks, = signal.argrelextrema(x, np.greater, order=4)
                if len(peaks) > 0:
                    virial_rcrit.append(r[peaks][0])
                else:
                    virial_rcrit.append(np.nan)
            cores['virial_rcrit'] = pd.Series(virial_rcrit, index=cores.index)

            # Find critical time
            ncrit, rcrit = tools.critical_time_old(self, cores.copy(),
                                                   rprofs.copy(), method=method)
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
                                     f"Model {self.basename}, par {pid}, ncrit = {ncrit}"
                                     f" crit_method {method}")
                if rcore > rprf.r.max()[()]:
                    msg = (
                        f"Core radius exceeds the maximum rprof radius for "
                        f"model {self.basename}, par {pid}. ncrit = {ncrit}, "
                        f"rcore = {rcore:.2f}; rprf_max = {rprf.r.max().data[()]:.2f}"
                    )
                    self.logger.warning(msg)
                    continue
                if np.isfinite(rcore):
                    mcore = rprf.menc.interp(r=rcore).data[()]
                    mean_density = mcore / (4*np.pi*rcore**3/3)
                    tff_crit = tools.tfreefall(mean_density, self.gconst)
                else:
                    mcore = np.nan
                    mean_density = np.nan
                    tff_crit = np.nan
                cores.attrs['tcrit'] = core.time
                cores.attrs['rcore'] = rcore
                cores.attrs['mcore'] = mcore
                cores.attrs['mean_density'] = mean_density
                cores.attrs['tff_crit'] = tff_crit

            # Load Lagrangian props
            fname = Path(savdir, f'lprops_tcrit_{method}.par{pid}.p')
            if fname.exists():
                lprops = pd.read_pickle(fname).sort_index()
                if set(lprops.columns).issubset(cores.columns):
                    cores = cores.drop(lprops.columns, axis=1)

                # Save attributes before performing join, which will drop them.
                attrs = cores.attrs.copy()
                attrs.update(lprops.attrs)
                cores = cores.join(lprops)
                # Reattach attributes
                cores.attrs = attrs

                # Net force
                Fnet = cores.Fthm + cores.Ftrb + cores.Fcen + cores.Fani - cores.Fgrv
                if self.mhd:
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
                phst = self.load_parhst(pid)
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


            # Try finding observed properties and attach them
            try:
                prestellar_cores = cores.loc[:cores.attrs['numcoll']]
                oprops = []
                for num, core in prestellar_cores.iterrows():
                    fname = Path(savdir, 'observables.par{}.{:05d}.p'.format(pid, num))
                    if fname.exists():
                        oprops.append(pd.read_pickle(fname))
                if len(oprops) > 0:
                    oprops = pd.DataFrame(oprops).set_index('num').sort_index()

                    # Save attributes before performing join, which will drop them.
                    attrs = cores.attrs.copy()
                    attrs.update(oprops.attrs)
                    cores = cores.join(oprops)
                    # Reattach attributes
                    cores.attrs = attrs
            except:
                pass

            # Sort attributes
            cores.attrs = {k: cores.attrs[k] for k in sorted(cores.attrs)}

            core_dict[pid] = cores

        return core_dict

    def flatindex_to_cartesian(self, flatindex, return_index=False):
        """Cartesian coordinates corresponding to flattened index

        Parameters
        ----------
        flatindex : int
            Flattened index assuming C-ordering (i.e., k, j, i)

        Returns
        -------
        x, y, z : float
        """
        k, j, i = np.unravel_index(flatindex, self.domain['Nx'].T, order='C')
        if return_index:
            return i, j, k
        else:
            x, y, z = (self.domain['le'] + np.array([i+0.5, j+0.5, k+0.5])*self.domain['dx'])
            return x, y, z

    def cartesian_to_flatindex(self, x, y, z):
        """Flattened index corresponding to Cartesian coordinates

        If x, y, z do not correspond to cell center, find closest cell center.

        Parameters
        ----------
        x, y, z : float

        Returns
        -------
        flatindex : int
            Flattened index assuming C-ordering (i.e., k, j, i)
        """
        i, j, k = ((np.array([x, y, z]) - self.domain['le'])
                   // self.domain['dx']).astype(int)
        flatidx = np.ravel_multi_index(
            (k, j, i), self.domain['Nx'].T, mode='raise', order='C'
        )
        return flatidx

    def distance_between(self, idx1, idx2):
        """Calculates periodic distance between two flattened indices

        Parameters
        ----------
        idx1, idx2 : int
            Flattened indices
        """
        pos1 = self.flatindex_to_cartesian(idx1)
        pos2 = self.flatindex_to_cartesian(idx2)
        return tools.periodic_distance(pos1, pos2, self.Lbox)

    def trajectory_start_num(self, cores, *, f_mul):
        """Find the earliest snapshot with a continuous tracked minimum.
        """
        cores = cores.sort_index(ascending=False)
        core0 = cores.iloc[0]
        core1 = cores.iloc[1]
        for num, core2 in cores.iloc[2:].iterrows():
            dst0 = self.distance_between(core0.leaf_id, core1.leaf_id)
            pos0 = np.array(self.flatindex_to_cartesian(core0.leaf_id))
            pos1 = np.array(self.flatindex_to_cartesian(core1.leaf_id))
            pos2 = np.array(self.flatindex_to_cartesian(core2.leaf_id))
            pos_extrapolated = pos0 + 2*(pos1 - pos0)
            dst = tools.periodic_distance(pos2, pos_extrapolated, self.Lbox)
            if dst > f_mul*max(dst0, self.dx):
                num_start = num+1
                return num_start
            core0 = core1
            core1 = core2

    def core_trajectory(self, cores, return_nums=False, num_start=None):
        """Return the backward core trajectory as a continuous path.

        The trajectory is traversed from the collapse time to the past. When
        the path crosses a periodic boundary, coordinates are unwrapped so the
        returned path remains continuous.

        Parameters
        ----------
        cores : pandas.DataFrame
            DataFrame containing the core information for a given pid.
        return_nums : bool, optional
            If True, also return the snapshot numbers associated with the
            tracked trajectory.
        num_start : int, optional
            Earliest snapshot to include in the trajectory.

        Returns
        -------
        xv, yv, zv : np.ndarray
            Unwrapped trajectory coordinates from future to past.
        nums : np.ndarray, optional
            Snapshot numbers from past to future. Returned only when
            ``return_nums`` is True.
        """
        if num_start is not None:
            cores = cores.loc[num_start:]
        if cores.empty:
            raise ValueError(f'No core snapshots at or after num_start={num_start}')

        cores = cores.sort_index(ascending=False)
        widths = np.asarray(self.domain['re']) - np.asarray(self.domain['le'])

        pos0 = np.asarray(self.flatindex_to_cartesian(cores.iloc[0].leaf_id),
                          dtype=float)
        pos_prev_wrapped = pos0.copy()
        pos_prev_unwrapped = pos0.copy()

        nums = [cores.index[0]]
        trajectory = [pos0.copy()]
        for num, core in cores.iloc[1:].iterrows():
            pos_wrapped = np.asarray(self.flatindex_to_cartesian(core.leaf_id),
                                     dtype=float)
            displacement = np.array([
                tools.periodic_operator(delta, -0.5*width, 0.5*width)
                for delta, width in zip(pos_wrapped - pos_prev_wrapped, widths)
            ])
            nums.append(num)
            trajectory.append(pos_prev_unwrapped + displacement)
            pos_prev_unwrapped = trajectory[-1]
            pos_prev_wrapped = pos_wrapped
        trajectory = np.asarray(trajectory)
        if return_nums:
            nums = np.asarray(nums)
            return nums[::-1], trajectory[::-1, 0], trajectory[::-1, 1], trajectory[::-1, 2]
        return trajectory[::-1, 0], trajectory[::-1, 1], trajectory[::-1, 2]

    def apply_periodic_bc(self, x, y, z):
        """Apply periodic boundary conditions"""
        x = tools.sawtooth(x, self.domain['le'][0], self.domain['re'][0],
                           self.domain['le'][0], self.domain['re'][0])
        y = tools.sawtooth(y, self.domain['le'][1], self.domain['re'][1],
                           self.domain['le'][1], self.domain['re'][1])
        z = tools.sawtooth(z, self.domain['le'][2], self.domain['re'][2],
                           self.domain['le'][2], self.domain['re'][2])
        return x, y, z

    @LoadSimBase.Decorators.check_pickle
    def _load_tcoll_cores(self, prefix='tcoll_cores', savdir=None, force_override=False):
        """Read .csv output and find their collapse time and snapshot number.

        Additionally store their mass, position, velocity at the time of
        collapse.
        """
        x1, x2, x3, v1, v2, v3 = {}, {}, {}, {}, {}, {}
        time, num = {}, {}
        for pid in self.pids:
            phst = self.load_parhst(pid).iloc[0]
            x1[pid] = phst.x1
            x2[pid] = phst.x2
            x3[pid] = phst.x3
            v1[pid] = phst.v1
            v2[pid] = phst.v2
            v3[pid] = phst.v3
            time[pid] = phst.time - phst.age
            num[pid] = np.floor(time[pid] / self.dt_output['hdf5']).astype('int')
        tcoll_cores = pd.DataFrame(
            dict(x1=x1, x2=x2, x3=x3,
                 v1=v1, v2=v2, v3=v3,
                 time=time, num=num),
            dtype=object
        )
        tcoll_cores.index.name = 'pid'
        return tcoll_cores

    @LoadSimBase.Decorators.check_pickle
    def _load_cores(self, prefix='cores', savdir=None, force_override=False):
        cores_dict = {}
        pids_not_found = []

        # Try reading the go15 mass
        try:
            fname = Path(savdir, 'mcore_go15.p')
            with open(fname, 'rb') as f:
                mcore_go15 = pickle.load(f)
            mcore_go15_found = True
        except FileNotFoundError:
            mcore_go15_found = False


        for pid in self.pids:
            fname = Path(savdir, f'cores.par{pid}.p')
            cores = pd.read_pickle(fname).sort_index()
            num_start = self.trajectory_start_num(cores, f_mul=3.0)

            num_start_min = int(
                (self.tcoll_cores.loc[1].time - 1.5 / self.u.Myr)
                / self.dt_output['hdf5']
            )
            # Prevent excessive back-tracking to early times
            num_start = max(num_start, num_start_min)

            cores = cores.loc[num_start:]

            # Read critical TES info and concatenate to self.cores
            # Try reading critical TES pickles
            tes_crit = []
            for num in cores.index:
                try:
                    fname = Path(savdir, f'critical_tes.par{pid}.{num:05d}.p')
                    tes_crit.append(pd.read_pickle(fname))
                except FileNotFoundError:
                    pids_not_found.append(pid)
                    break
            if len(tes_crit) > 0:
                tes_crit = pd.DataFrame(tes_crit).set_index('num').sort_index()

                # Save attributes before performing join, which will drop them.
                attrs = cores.attrs.copy()
                attrs.update(tes_crit.attrs)
                cores = cores.join(tes_crit)

                # Reattach attributes
                cores.attrs = attrs
            if mcore_go15_found:
                cores.attrs['mcore_go15'] = mcore_go15[pid]

            # Find collapse time
            cores.attrs['tcoll'] = self.tcoll_cores.loc[pid].time

            # Add attributes
            if len(cores) == 1 and cores.index[0] == cores.attrs['numcoll']:
                cores.attrs['track_failed'] = True
            else:
                cores.attrs['track_failed'] = False

            # Sort attributes
            cores.attrs = {k: cores.attrs[k] for k in sorted(cores.attrs)}

            cores_dict[pid] = cores

        if len(pids_not_found) > 0:
            msg = f"{self.basename}: Some critical TES files are missing for pid {pids_not_found}"
            self.logger.warning(msg)
        return cores_dict

    def concat_radial_profiles(self):
        """Concatenate per-snapshot 3D radial profiles into one pickle file."""
        savdir = Path(self.savdir, config.RPROF_DIR)
        savdir.mkdir(parents=True, exist_ok=True)
        fname_concat = savdir / 'radial_profile.concatenated.p'

        rprofs_dict = {}
        pids_not_found = []
        pids_not_found_prj = []
        for pid in self.pids:
            cores = self.cores[pid]
            rprofs_pid, nums = [], []
            min_nr = None
            for num in cores.index:
                try:
                    leaf_id = cores.loc[num].leaf_id
                    fname = savdir / f'radial_profile.{leaf_id}.{num:05d}.nc'
                    rprf = xr.load_dataset(fname)
                    if min_nr is None:
                        min_nr = rprf.sizes['r']
                    else:
                        min_nr = min(min_nr, rprf.sizes['r'])
                    rprofs_pid.append(rprf)
                    nums.append(num)
                except FileNotFoundError:
                    print(f"Missing radial profile for pid {pid}, num {num}.")
                    pids_not_found.append(pid)
                    break
            if len(rprofs_pid) > 0:
                rprf = xr.concat(rprofs_pid, 't')
                rprf = rprf.assign_coords(num=('t', nums))
                rprf = rprf.isel(r=slice(0, min_nr))

                prj_rprofs, nums = [], []
                min_nr = None
                for num in cores.index:
                    try:
                        fname = savdir / f'prj_radial_profile.par{pid}.{num:05d}.nc'
                        prj_rprf = xr.load_dataset(fname)
                        if min_nr is None:
                            min_nr = prj_rprf.sizes['R']
                        else:
                            min_nr = min(min_nr, prj_rprf.sizes['R'])
                        prj_rprofs.append(prj_rprf)
                        nums.append(num)
                    except FileNotFoundError:
                        pids_not_found_prj.append(pid)
                        break
                if len(prj_rprofs) > 0:
                    prj_rprofs = xr.concat(prj_rprofs, 't')
                    prj_rprofs = prj_rprofs.assign_coords(dict(num=('t', nums)))
                    prj_rprofs = prj_rprofs.isel(R=slice(0, min_nr))
                    rprf = rprf.merge(prj_rprofs, compat="no_conflicts")

                rprf = rprf.set_xindex('num')
                rprofs_dict[pid] = rprf

        if len(pids_not_found) > 0:
            msg = f"Some radial profiles are missing for pid {pids_not_found}."
            self.logger.warning(msg)
        if len(pids_not_found_prj) > 0:
            msg = f"Some projected radial profiles are missing for pid {pids_not_found_prj}."
            self.logger.warning(msg)

        if len(rprofs_dict) == 0:
            raise FileNotFoundError('No radial profile files found to concatenate.')

        with open(fname_concat, 'wb') as handle:
            pickle.dump(rprofs_dict, handle)
        return rprofs_dict

    @LoadSimBase.Decorators.check_pickle
    def _load_radial_profiles(self, prefix='radial_profile', savdir=None, force_override=False):
        """
        Raises
        ------
        FileNotFoundError
            If individual radial profiles are not found
        KeyError
            If `cores` has not been initialized (due to missing files, etc.)
        """
        fname_concat = savdir / 'radial_profile.concatenated.p'

        if not fname_concat.exists():
            raw_rprofs_dict = self.concat_radial_profiles()
        else:
            with open(fname_concat, 'rb') as handle:
                raw_rprofs_dict = pickle.load(handle)

        rprofs_dict = {}
        for pid, rprofs in raw_rprofs_dict.items():
            rprofs = rprofs.copy()
            for axis in [1, 2, 3, 'x', 'y', 'z']:
                rprofs[f'dvel{axis}_sq_mw'] = (rprofs[f'vel{axis}_sq_mw']
                                             - rprofs[f'vel{axis}_mw']**2)

            dr = rprofs.r.data[1] - rprofs.r.data[0]
            rc = rprofs.r.data
            rf = np.insert(0.5*(rc[1:] + rc[:-1]), 0, 0)
            rf = np.append(rf, rf[-1]+dr)
            # vshell_full is the volume between the inner and outer bin edges
            vshell_full = 4*np.pi/3*(rf[1:]**3 - rf[:-1]**3)
            # vshell_half is the volume between the outer bin edge and the
            # bin center.
            vshell_half = 4*np.pi/3*(rf[1:]**3 - rc**3)
            # Note that for the first bin, vshell_full = vshell_half.

            rprofs['vshell_full'] = xr.DataArray(
                vshell_full,
                dims='r',
                coords=dict(r=rprofs.r)
            )
            rprofs['vshell_half'] = xr.DataArray(
                vshell_half,
                dims='r',
                coords=dict(r=rprofs.r)
            )

            rprofs['menc'] = rprof_cumsum_r(rprofs, rprofs.rho)

            # Virial terms
            rdotg = rprofs.xgx_mw + rprofs.ygy_mw + rprofs.zgz_mw

            rprofs['Omega_G'] = -rprof_cumsum_r(rprofs, rprofs.rho*rdotg)

            # See the Radial Profile from Cartesian Grid Data slide
            # or July 24 research note.
            hdr = 0.5*(rprofs.r[1] - rprofs.r[0]).data[()]
            rl = rprofs.r - hdr
            ru = rprofs.r + hdr
            r_inv_avg = (3/2) * (ru + rl) / (ru**2 + ru*rl + rl**2)
            r2_avg = (3/5) * (
                ru**4 + ru**3*rl + ru**2*rl**2 + ru*rl**3 + rl**4
            ) / (ru**2 + ru*rl + rl**2)
            r5_avg = (3/8) * (
                ru**7 + ru**6*rl + ru**5*rl**2 + ru**4*rl**3
                + ru**3*rl**4 + ru**2*rl**5 + ru*rl**6 + rl**7
            ) / (ru**2 + ru*rl + rl**2)
            vshell_inner_half = rprofs.vshell_full - rprofs.vshell_half
            menc_inner_edge = rprofs.menc - rprofs.rho*vshell_inner_half
            menc_inner_edge -= 4*np.pi*rprofs.rho*rl**3/3
            rdotg_sph = xr.where(
                rprofs.r > 0,
                -self.gconst*(
                    menc_inner_edge*r_inv_avg
                    + 4*np.pi*rprofs.rho/3*r2_avg
                ),
                0
            )
            rprofs['Omega_G_sph'] = -rprof_cumsum_r(rprofs, rprofs.rho*rdotg_sph)
            rprofs['Omega_G0'] = xr.where(
                rprofs.r > 0,
                self.gconst*(
                    menc_inner_edge**2*r_inv_avg
                    + 8*np.pi*rprofs.rho/3*menc_inner_edge*r2_avg
                    + (4*np.pi*rprofs.rho/3)**2*r5_avg
                ),
                0
            )

            rprofs['Omega_K_thm'] = rprof_cumsum_r(rprofs, 3*self.cs**2*rprofs.rho)

            vsq = (rprofs.vel1_sq_mw + rprofs.vel2_sq_mw + rprofs.vel3_sq_mw)
            rprofs['Omega_K_kin'] = rprof_cumsum_r(rprofs, rprofs.rho*vsq)

            rprofs['Omega_K'] = rprofs['Omega_K_thm'] + rprofs['Omega_K_kin']
            rprofs['Omega_S_thm'] = 4*np.pi*rprofs.r**3*self.cs**2*rprofs.rho
            rprofs['Omega_S_kin'] = 4*np.pi*rprofs.r**3*rprofs.rho*rprofs.vel1_sq_mw
            rprofs['Omega_S'] = rprofs['Omega_S_thm'] + rprofs['Omega_S_kin']
            rprofs['alpha_vir'] = rprofs['Omega_K'] / rprofs['Omega_G']
            if self.mhd:
                magnetic_energy_density = (rprofs.b1_sq + rprofs.b2_sq + rprofs.b3_sq)/2
                rprofs['Omega_M'] = rprof_cumsum_r(rprofs, magnetic_energy_density)
                t_rr = rprofs.b1_sq - magnetic_energy_density
                rprofs['Omega_S_mag'] = -4*np.pi*rprofs.r**3*t_rr
                rprofs['Omega_S'] += rprofs['Omega_S_mag']
                rprofs['gamma_M'] = rprofs['Omega_M'] / (2*rprofs['Omega_K'])
            else:
                rprofs['Omega_M'] = xr.zeros_like(rprofs.rho)
                rprofs['Omega_S_mag'] = xr.zeros_like(rprofs.rho)
                rprofs['gamma_M'] = xr.zeros_like(rprofs.rho)
            rprofs['gamma_S'] = rprofs['Omega_S'] / rprofs['Omega_G']
            rprofs['Alpha'] = rprofs.Omega_K + rprofs.Omega_M - rprofs.Omega_G - rprofs.Omega_S
            rprofs['ptot'] = rprofs.rho*(self.cs**2 + rprofs.vel1_sq_mw)
            rprofs['peq'] = (rprofs.Omega_K + rprofs.Omega_M - rprofs.Omega_S_mag - rprofs.Omega_G) / (4*np.pi*rprofs.r**3)

            # Maximum pressure from McCrea analysis
            rgrav = self.gconst*rprofs.menc/self.cs**2
            pgrav = self.cs**8/(4*np.pi*self.gconst**3*rprofs.menc**2)
            sigma_1d_sq = rprofs.Omega_K_kin/(3*rprofs.menc)
            a_grv = rprofs.Omega_G/rprofs.Omega_G0
            a_grv_eff = a_grv.copy()
            if self.mhd:
                brms = np.sqrt(2*rprofs.Omega_M/(4*np.pi*rprofs.r**3/3))
                bflux = np.pi*rprofs.r**2*brms
                b_mag = (rprofs.Omega_M - rprofs.Omega_S_mag) / (bflux**2/rprofs.r)
                mmag2 = b_mag/a_grv/self.gconst*bflux**2
                a_grv_eff *= (1 - mmag2/rprofs.menc**2)
            else:
                mmag2 = xr.zeros_like(a_grv)
            rprofs['mcrit_mag'] = np.sqrt(mmag2.where(mmag2 >= 0))

            param_dict = {
                'mmax': {
                    'mmag2': mmag2,
                    'sigma_tot2': self.cs**2 + sigma_1d_sq,
                },
                'mmax_thm': {
                    'mmag2': xr.zeros_like(mmag2),
                    'sigma_tot2': self.cs**2,
                },
                'mmax_trb': {
                    'mmag2': xr.zeros_like(mmag2),
                    'sigma_tot2': sigma_1d_sq,
                },
                'mmax_mag': {
                    'mmag2': mmag2,
                    'sigma_tot2': 0,
                },
                'mmax_thm_trb': {
                    'mmag2': xr.zeros_like(mmag2),
                    'sigma_tot2': self.cs**2 + sigma_1d_sq
                },
                'mmax_thm_mag': {
                    'mmag2': mmag2,
                    'sigma_tot2': self.cs**2
                },
                'mmax_trb_mag': {
                    'mmag2': mmag2,
                    'sigma_tot2': sigma_1d_sq
                },
            }
            for key in param_dict.copy().keys():
                param_dict[key]['a'] = a_grv
                param_dict[f'{key}_fixed_a'] = param_dict[key].copy()
                param_dict[f'{key}_fixed_a']['a'] = xr.ones_like(a_grv)
            for key, params in param_dict.items():
                c_J = np.sqrt(3**7 / (4 * np.pi * 2**8 * params['a'].where(params['a'] > 0)**3))
                # Cubic coefficients for x^3 + ax^2 + bx + c = 0
                a = -3*params['mmag2'] - c_J**2 * params['sigma_tot2']**4 / (self.gconst**3 * rprofs.ptot)
                b = 3*params['mmag2']**2
                c = -params['mmag2']**3
                x = cubic_root(a, b, c)
                rprofs[key] = np.sqrt(x.where(x >= 0))

            rhoavg = rprofs.menc / (4*np.pi*rprofs.r**3/3)
            mgrav = self.cs**3/self.gconst**1.5/np.sqrt(rhoavg)
            mbe = 1.86*mgrav
            rprofs['mBE'] = mbe
            rprofs['mTES'] = mbe*(1 + sigma_1d_sq/2)
            mass_to_flux_crit = 1 / (2*np.pi*np.sqrt(self.gconst))
            if self.mhd:
                rprofs['mPhi'] = rprofs.phi_B*np.sqrt(4*np.pi)*mass_to_flux_crit
            else:
                rprofs['mPhi'] = xr.zeros_like(rprofs.rho)

            rprofs = rprofs.merge(tools.radial_acceleration(self, rprofs), compat="no_conflicts")
            if 'num' not in rprofs.indexes:
                rprofs = rprofs.set_xindex('num')

            rprofs_dict[pid] = rprofs.transpose('t', 'r', ...)

        return rprofs_dict

    def load_power_spectrum(self, force_override=False):
        """
        Raises
        ------
        FileNotFoundError
            If individual radial profiles are not found
        KeyError
            If `cores` has not been initialized (due to missing files, etc.)
        """
        savdir = Path(self.savdir, config.FOURIER_DIR)
        pspec = []
        for num in self.nums:
            fname = Path(savdir, f'power_spectrum.{num:05d}.p')
            ps = xr.open_dataset(fname)
            pspec.append(ps)
        pspec = xr.concat(pspec, 't')
        pspec = pspec.assign_coords(dict(num=('t', self.nums)))
        pspec = pspec.set_xindex('num')
        return pspec


class LoadSimAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None, verbose=True):

        # Default models
        if models is None:
            models = dict()
        self.models = []
        self.basedirs = dict()

        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                if verbose:
                    msg = "[LoadSimAll]: "\
                          "Model {0:s} doesn\'t exist: {1:s}".format(mdl, basedir)
                    print(msg)
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir

    def set_model(self, model, **kwargs):
        return LoadSim(self.basedirs[model], **kwargs)

    def itercore(self, models=None, nres=8, fmul_prune=3, **kwargs):
        if models is None:
            models = self.models
        for mdl in models:
            s = self.set_model(mdl, **kwargs)
            if nres == 0:
                for pid in s.pids:
                    cores = s.cores[pid]
                    rprofs = s.rprofs[pid]
                    yield s, pid, cores, rprofs
            else:
                for pid in s.good_cores(nres):
                    cores = s.cores[pid]
                    rprofs = s.rprofs[pid]
                    yield s, pid, cores, rprofs

    def itercritcore(self, models=None, nres=8, **kwargs):
        if models is None:
            models = self.models
        for s, pid, cores, rprofs in self.itercore(models=models, nres=nres,
                                                   **kwargs):
            num = cores.attrs['numcrit']
            core = cores.loc[num]
            rprf = rprofs.sel(num=num)
            yield s, pid, core, rprf


def safe_cbrt(val):
    """Sign-safe cube root for xarray DataArrays."""
    return np.sign(val) * np.abs(val)**(1/3)


def cubic_root(a, b, c):
    """Calculate the real root of a cubic equation x^3 + ax^2 + bx + c = 0"""
    p = (3*b - a**2) / 3
    q = (2*a**3 - 9*a*b + 27*c) / 27
    disc = (q/2)**2 + (p/3)**3

    sqrt_disc = np.sqrt(disc.where(disc >= 0))
    x_cardano = safe_cbrt(-q/2 + sqrt_disc) + safe_cbrt(-q/2 - sqrt_disc) - a/3

    arg = -q/2/np.sqrt(-(p.where(disc < 0)/3)**3)
    arg_violation = xr.where(disc < 0, np.abs(arg) > 1 + 1e-10, False)
    if arg_violation.any():
        max_violation = float((np.abs(arg) - 1).where(arg_violation).max())
        warnings.warn(
            f"arccos argument out of [-1, 1] by up to {max_violation:.2e} in "
            f"{int(arg_violation.sum())} cells. Possible numerical issue near disc=0."
        )
    phi = np.arccos(arg.where(disc < 0))
    r = 2*np.sqrt(-p.where(disc < 0)/3)
    x0 = r*np.cos(phi/3) - a/3
    x1 = r*np.cos((phi + 2*np.pi)/3) - a/3
    x2 = r*np.cos((phi + 4*np.pi)/3) - a/3
    x_three = xr.concat([x0, x1, x2], 'root')
    x_cardano_disc_neg = x_three.max(dim='root')
    x = xr.where(disc >= 0, x_cardano, x_cardano_disc_neg)
    return x

def rprof_cumsum_r(rprofs, rprf_var):
#    res = (4*np.pi*rprofs.r**2*rprf_var).cumulative_integrate('r')
    res = (rprf_var*rprofs.vshell_full).cumsum('r')
    # correct for outer half of the shell
    res -= rprf_var*rprofs.vshell_half
    return res
