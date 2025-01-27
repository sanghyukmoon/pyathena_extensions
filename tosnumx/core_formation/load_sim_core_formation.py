import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import pickle
from scipy.interpolate import interp1d
from astropy import units as au
from astropy import constants as ac
from pyathena.load_sim import LoadSim
from pyathena.util.units import Units
from pyathena.io.timing_reader import TimingReader

from . import tools, hst, slc_prj


class LoadSimCoreFormation(LoadSim, hst.Hst, slc_prj.SliceProj, tools.LognormalPDF,
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

    def __init__(self, basedir_or_Mach=None, method='empirical', savdir=None,
                 load_method='pyathena', verbose=False, force_override=False):
        """The constructor for LoadSimCoreFormation class

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
        load_method : str
            Load hdf5 using 'pyathena' or 'yt'. Default value is 'pyathena'.
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

        if isinstance(basedir_or_Mach, (Path, str)):
            basedir = basedir_or_Mach
            super().__init__(basedir, savdir=savdir, load_method=load_method,
                             units=Units('code'), verbose=verbose)
            self.Mach = self.par['problem']['Mach']

            tools.LognormalPDF.__init__(self, self.Mach)
            TimingReader.__init__(self, self.basedir, self.problem_id)

#            # Set nums dictionary (when hdf5 is stored in elsewhere for storage reasons)
#            if self.nums is None:
#                self.nums = self.nums_partab['par0']

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
            self.tcoll_cores = self._load_tcoll_cores(force_override=force_override)

            try:
                # Load cores
                savdir = Path(self.savdir, 'cores')
                self.cores = self._load_cores(savdir=savdir, force_override=force_override)
                # Remove bud nodes TODO(SMOON) Is this really necessary?
                if len(self.cores) > 0:
                    pids = self.pids.copy()
                    for pid in pids:
                        if not self.cores[pid].attrs['tcoll_resolved']:
                            self.pids.remove(pid)
            except FileNotFoundError:
                self.logger.warning("Failed to load core information")
                pass

            try:
                # Load radial profiles
                savdir = Path(self.savdir, 'radial_profile')
                self.rprofs = self._load_radial_profiles(savdir=savdir, force_override=force_override)
            except (AttributeError, FileNotFoundError, KeyError):
                self.logger.warning("Failed to load radial profiles")
                pass

            # Load derived core informations using various alternative critical times
            self.cores_dict = {}
            for mtd in ['empirical', 'predicted']:  # retire experimental pred_xis
                try:
                    # Calculate derived core properties using the predicted critical time
                    savdir = Path(self.savdir, 'cores')
                    self.cores_dict[mtd] = self.update_core_props(method=mtd, prefix=f'cores_tcrit_{mtd}',
                                                                  savdir=savdir, force_override=force_override)
                except (AttributeError, KeyError):
                    self.logger.warning(f"Failed to update core properties for model {self.basename}, method {mtd}")

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

        # Override Unit system assuming dense sub-patch of a GMC
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

    def select_cores(self, method):
        method_list = {'empirical', 'predicted', 'pred_xis'}
        if method not in method_list:
            raise Exception("Method must be one of {}".format(sorted(method_list)))
        self.cores = self.cores_dict[method].copy()

    def good_cores(self, nres=8):
        """List of resolved and isolated cores"""
        good_cores = []
        for pid, cores in self.cores.items():
            if cores.attrs['isolated'] and tools.test_resolved_core(self, cores, nres):
                good_cores.append(pid)
        return good_cores

    @LoadSim.Decorators.check_pickle
    def update_core_props(self, method,
                          prefix=None, savdir=None, force_override=False):
        """Update core properties

        Calculate lagrangian core properties using the radial profiles
        Set isolated flag
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
            rprofs = self.rprofs[pid]

            # Find critical time
            ncrit = tools.critical_time(self, pid, method)
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
                rcore = core.critical_radius
                if rcore > rprf.r.max()[()]:
                    msg = (
                        f"Core radius exceeds the maximum rprof radius for "
                        f"model {self.basename}, par {pid}. ncrit = {ncrit}, "
                        f"rcore = {rcore:.2f}; rprf_max = {rprf.r.max().data[()]:.2f}"
                    )
                    self.logger.warning(msg)
                    continue
                mcore = rprf.menc.interp(r=rcore).data[()]
                mean_density = mcore / (4*np.pi*rcore**3/3)
                tff_crit = tools.tfreefall(mean_density, self.gconst)
                cores.attrs['tcrit'] = core.time
                cores.attrs['rcore'] = rcore
                cores.attrs['mcore'] = mcore
                cores.attrs['mean_density'] = mean_density
                cores.attrs['tff_crit'] = tff_crit

            # Test isolatedness
            if tools.test_isolated_core(self, cores):
                cores.attrs['isolated'] = True
            else:
                cores.attrs['isolated'] = False

            # Load Lagrangian props
            fname = Path(self.savdir, 'cores', f'lprops_tcrit_{method}.par{pid}.p')
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
                    fname = Path(self.savdir, 'cores',
                                 'observables.par{}.{:05d}.p'
                                 .format(pid, num))
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

    @LoadSim.Decorators.check_pickle
    def _load_tcoll_cores(self, prefix='tcoll_cores', savdir=None, force_override=False):
        """Read .csv output and find their collapse time and snapshot number.

        Additionally store their mass, position, velocity at the time of
        collapse.
        """
        # find collapse time and the snapshot numbers at the time of collapse
        dt_output = {}
        for k, v in self.par.items():
            if k.startswith('output'):
                dt_output[v['file_type']] = v['dt']

        x1, x2, x3, v1, v2, v3 = {}, {}, {}, {}, {}, {}
        time, num = {}, {}
        for pid in self.pids:
            phst = self.load_parhst(pid)
            phst0 = phst.iloc[0]
            x1[pid] = phst0.x1
            x2[pid] = phst0.x2
            x3[pid] = phst0.x3
            v1[pid] = phst0.v1
            v2[pid] = phst0.v2
            v3[pid] = phst0.v3
            time[pid] = phst0.time
            num[pid] = np.floor(phst0.time / dt_output['hdf5']
                                ).astype('int')
        tcoll_cores = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3,
                                             v1=v1, v2=v2, v3=v3,
                                             time=time, num=num),
                                        dtype=object)
        tcoll_cores.index.name = 'pid'
        return tcoll_cores

    @LoadSim.Decorators.check_pickle
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

            prestellar_cores = cores.loc[:cores.attrs['numcoll']]

            if cores.attrs['tcoll_resolved']:
                # Read critical TES info and concatenate to self.cores

                # Try reading critical TES pickles
                tes_crit = []
                for num in cores.index:
                    try:
                        fname = Path(self.savdir, 'critical_tes',
                                     f'critical_tes.par{pid}.{num:05d}.p')
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

            # Sort attributes
            cores.attrs = {k: cores.attrs[k] for k in sorted(cores.attrs)}

            cores_dict[pid] = cores



        if len(pids_not_found) > 0:
            msg = f"Some critical TES files are missing for pid {pids_not_found}"
            self.logger.warning(msg)
        return cores_dict

    @LoadSim.Decorators.check_pickle
    def _load_radial_profiles(self, prefix='radial_profile', savdir=None, force_override=False):
        """
        Raises
        ------
        FileNotFoundError
            If individual radial profiles are not found
        KeyError
            If `cores` has not been initialized (due to missing files, etc.)
        """
        rprofs_dict = {}
        pids_not_found = []
        pids_not_found_prj = []
        for pid in self.pids:
            cores = self.cores[pid]
            rprofs, nums = [], []
            min_nr = None
            for num in cores.index:
                try:
                    fname = Path(savdir,
                                 f'radial_profile.par{pid}.{num:05d}.nc')
                    rprf = xr.open_dataset(fname)
                    if min_nr is None:
                        min_nr = rprf.sizes['r']
                    else:
                        min_nr = min(min_nr, rprf.sizes['r'])
                    rprofs.append(rprf)
                    nums.append(num)
                except FileNotFoundError:
                    pids_not_found.append(pid)
                    break
            if len(rprofs) > 0:
                rprofs = xr.concat(rprofs, 't')
                rprofs = rprofs.assign_coords(dict(num=('t', nums)))
                # Slice data to common range in r.
                rprofs = rprofs.isel(r=slice(0, min_nr))
                for axis in [1, 2, 3, 'x', 'y', 'z']:
                    rprofs[f'dvel{axis}_sq_mw'] = (rprofs[f'vel{axis}_sq_mw']
                                                 - rprofs[f'vel{axis}_mw']**2)
                rprofs['menc'] = (4*np.pi*rprofs.r**2*rprofs.rho
                                  ).cumulative_integrate('r')
                rprofs = rprofs.merge(tools.calculate_accelerations(rprofs))
                rprofs = rprofs.set_xindex('num')

            # Read projected radial profiles
            prj_rprofs, nums = [], []
            min_nr = None
            for num in cores.index:
                try:
                    fname = Path(savdir,
                                 f'prj_radial_profile.par{pid}.{num:05d}.nc')
                    rprf = xr.open_dataset(fname)
                    if min_nr is None:
                        min_nr = rprf.sizes['R']
                    else:
                        min_nr = min(min_nr, rprf.sizes['R'])
                    prj_rprofs.append(rprf)
                    nums.append(num)
                except FileNotFoundError:
                    pids_not_found_prj.append(pid)
                    break
            if len(prj_rprofs) > 0:
                prj_rprofs = xr.concat(prj_rprofs, 't')
                prj_rprofs = prj_rprofs.assign_coords(dict(num=('t', nums)))
                # Slice data to common range in R.
                prj_rprofs = prj_rprofs.isel(R=slice(0, min_nr))
                if 'num' in rprofs.indexes:
                    rprofs = rprofs.drop_indexes('num')
                rprofs = rprofs.merge(prj_rprofs)
                rprofs = rprofs.set_xindex('num')

            rprofs_dict[pid] = rprofs
        if len(pids_not_found) > 0:
            msg = f"Some radial profiles are missing for pid {pids_not_found}."
            self.logger.warning(msg)
        if len(pids_not_found_prj) > 0:
            msg = f"Some projected radial profiles are missing for pid {pids_not_found_prj}."
            self.logger.warning(msg)

        return rprofs_dict


class LoadSimCoreFormationAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()
        self.models = []
        self.basedirs = dict()
        self.simdict = dict()

        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                msg = "[LoadSimCoreFormationAll]: "\
                      "Model {0:s} doesn\'t exist: {1:s}".format(mdl, basedir)
                print(msg)
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir

    def set_model(self, model, method='empirical', savdir=None,
                  load_method='pyathena', verbose=False,
                  reset=False, force_override=False):
        self.model = model
        if reset or force_override:
            self.sim = LoadSimCoreFormation(self.basedirs[model],
                                            method=method,
                                            savdir=savdir,
                                            load_method=load_method,
                                            verbose=verbose,
                                            force_override=force_override)
            self.simdict[model] = self.sim
        else:
            try:
                self.sim = self.simdict[model]
            except KeyError:
                self.sim = LoadSimCoreFormation(self.basedirs[model],
                                                method=method,
                                                savdir=savdir,
                                                load_method=load_method,
                                                verbose=verbose,
                                                force_override=force_override)
                self.simdict[model] = self.sim

        return self.sim
