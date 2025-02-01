import numpy as np
import xarray as xr
import pandas as pd
from scipy.special import erfcinv, erfc
from scipy.stats import linregress
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pathlib import Path
from pyathena.util import transform
from tesphere import utils, tes

from . import config

class LognormalPDF:
    """Lognormal probability distribution function"""

    def __init__(self, Mach, b=0.4, weight='mass'):
        """Constructor of the LognormalPDF class

        Parameter
        ---------
        Mach : float
            Sonic Mach number
        b : float, optional
            Parameter in the density dispersion-Mach number relation.
            Default to 0.4, corresponding to natural mode mixture.
            See Fig. 8 of Federrath et al. 2010.
        weight : string, optional
            Weighting of the PDF. Default to mass-weighting.
        """
        self.mu = 0.5*np.log(1 + b**2*Mach**2)
        self.var = 2*self.mu
        self.sigma = np.sqrt(self.var)
        if weight == 'mass':
            pass
        elif weight == 'volume':
            self.mu *= -1
        else:
            ValueError("weight must be either mass or volume")

    def fx(self, x):
        """The mass fraction between x and x+dx

        Parameter
        ---------
        x : float
            Logarithmic density contrast, ln(rho/rho_0).
        """
        f = (1 / np.sqrt(2*np.pi*self.var))*np.exp(-(x - self.mu)**2
                                                   / (2*self.var))
        return f

    def probability_between(self, dl, du):
        return self.mfrac_above(dl) - self.mfrac_above(du)

    def mfrac_above(self, rhothr):
        """Return the mass fraction above density rhothr"""
        xthr = np.log(rhothr)
        tthr = (xthr - self.mu) / np.sqrt(2*self.var)
        return 0.5*erfc(tthr)

    def get_contrast(self, frac):
        """Calculates density contrast for given mass coverage

        Returns rho/rho_0 below which frac (0 to 1) of the total mass
        is contained.

        Parameter
        ---------
        frac : float
            Mass fraction.
        """
        x = self.mu + np.sqrt(2)*self.sigma*erfcinv(2 - 2*frac)
        return np.exp(x)


def find_tcoll_core(s, pid):
    """Find the GRID-dendro ID of the t_coll core of particle pid

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    pid : int
        Particle id.

    Returns
    -------
    lid : int or None
        ID of the leaf corresponding to t_coll core. If unresolved, return None.
    """
    # load dendrogram at t = t_coll
    num = s.tcoll_cores.loc[pid].num
    gd = s.load_dendro(num)

    # find closeast leaf node to this particle
    pos_particle = s.tcoll_cores.loc[pid][['x1', 'x2', 'x3']]
    pos_particle = pos_particle.to_numpy()
    dst = [get_periodic_distance(get_coords_node(s, lid), pos_particle, s.Lbox)
           for lid in gd.leaves]
    lid = gd.leaves[np.argmin(dst)]

    # return the grid-dendro ID of the t_coll core
    return lid


# TODO Can we predict the new sink position using the mean velocity inside the core?
# But that would require loading the hdf5 snapshot, making the core tracking more expensive.
# TODO Stopping condition due to leaf distance is just arbitrary, because in principle if the
# leaf disappears by a merger, it would keep tracking. We need more physically motivated stopping
# condition
def track_cores(s, pid):
    """Perform reverse core tracking

    Parameters
    ----------
    s : LoadSim
    pid : int

    Returns
    -------
    cores : pandas.DataFrame

    See also
    --------
    track_protostellar_cores : Forward core tracking after t_coll into
                               the protostellar stage.
    """
    # start from t = t_coll and track backward
    nums = np.arange(s.tcoll_cores.loc[pid].num, config.GRID_NUM_START-1, -1)
    num = nums[0]
    msg = f'[track_cores] processing model {s.basename} pid {pid} num {num}'
    print(msg)

    lid = find_tcoll_core(s, pid)
    ds = s.load_hdf5(num, header_only=True)

    if lid is None:
        msg = (
            f'[track_cores] t_coll core for pid {pid} is unresolved. '
            ' do not perform core tracking for this core.'
        )
        print(msg)
        nums_track = [num,]
        time = [ds['Time'],]
        leaf_id = [np.nan,]
        leaf_radius = [np.nan,]
        tidal_radius = [np.nan,]
        tcoll_resolved = False
    else:
        tcoll_resolved = True
        gd = s.load_dendro(num)

        # Calculate effective radius of this leaf
        rleaf = reff_sph(gd.len(lid)*s.dV)

        # Do the tidal correction to neglect attached substructures.

        # Calculate tidal radius
        rtidal = calculate_tidal_radius(s, gd, lid, lid)

        nums_track = [num,]
        time = [ds['Time'],]
        leaf_id = [lid,]
        leaf_radius = [rleaf,]
        tidal_radius = [rtidal,]

        for num in nums[1:]:
            msg = '[track_cores] processing model {} pid {} num {}'
            print(msg.format(s.basename, pid, num))
            gd = s.load_dendro(num)
            ds = s.load_hdf5(num, header_only=True)
            pds = s.load_par(num)

            # find closeast leaf to the previous preimage
            dst = [get_node_distance(s, leaf, leaf_id[-1]) for leaf in gd.leaves]
            lid = gd.leaves[np.argmin(dst)]
            rleaf = reff_sph(gd.len(lid)*s.dV)

            # If there is sink particle in the leaf, stop tracking.
            idx = np.floor((pds[['x1', 'x2', 'x3']] - s.domain['le']) / s.dx).astype('int')
            idx = idx[['x3', 'x2', 'x1']]
            idx = idx.values
            idx = idx[:, 0]*s.domain['Nx'][1]*s.domain['Nx'][0] + idx[:,1]*s.domain['Nx'][0] + idx[:, 2]
            flag = 0
            for idx_ in idx:
                if idx_ in gd.get_all_descendant_cells(lid):
                    flag += 1
            if flag > 0:
                break

            # Calculate tidal radius
            rtidal = calculate_tidal_radius(s, gd, lid, lid)

            # If the center moved more than the tidal radius, stop tracking.
            if get_node_distance(s, lid, leaf_id[-1]) > tidal_radius[-1]:
                break

            nums_track.append(num)
            time.append(ds['Time'])
            leaf_id.append(lid)
            leaf_radius.append(rleaf)
            tidal_radius.append(rtidal)

    # SMOON: Using dtype=object is to prevent automatic upcasting from int to float
    # when indexing a single row. Maybe there is a better approach.
    cores = pd.DataFrame(dict(time=time,
                              leaf_id=leaf_id,
                              leaf_radius=leaf_radius,
                              tidal_radius=tidal_radius),
                         index=nums_track, dtype=object).sort_index()

    # Set attributes
    cores.attrs['pid'] = pid
    cores.attrs['numcoll'] = cores.index[-1]
    cores.attrs['tcoll_resolved'] = tcoll_resolved

    return cores


def track_protostellar_cores(s, pid):
    """Perform forward core tracking

    Parameters
    ----------
    s : LoadSim
    pid : int

    Returns
    -------
    cores : pandas.DataFrame

    See also
    --------
    track_cores : Reverse core tracking from t_coll back into
                  the prestellar stage.
    """
    # Load prestellar core list
    # Do not load from self.cores, which might already contain the derived core properties.
    # We do not want to write derived properties into cores.par{}.p.
    fname = Path(s.savdir, 'cores', 'cores.par{}.p'.format(pid))
    cores = pd.read_pickle(fname).sort_index()
    ncoll = cores.attrs['numcoll']

    # Select prestellar part
    cores = cores.loc[:ncoll]

    # nums after t_coll
    nums = [num for num in s.nums if num > ncoll]

    nums_track = []
    time = []
    leaf_id = []
    leaf_radius = []
    tidal_radius  = []

    for num in nums:
        msg = '[track_protostellar_cores] processing model {} pid {} num {}'
        print(msg.format(s.basename, pid, num))
        gd = s.load_dendro(num)
        ds = s.load_hdf5(num, header_only=True)
        pds = s.load_par(num)

        if pid not in pds.index:
            # This sink particle has merged to other sink. Stop tracking
            break

        # Find closet leaf to the sink particle
        sink_pos = pds.loc[pid][['x1', 'x2', 'x3']].to_numpy()
        dst = [get_periodic_distance(get_coords_node(s, lid), sink_pos, s.Lbox)
               for lid in gd.leaves]
        lid = gd.leaves[np.argmin(dst)]
        rleaf = reff_sph(gd.len(lid)*s.dV)

        # Calculate tidal radius
        rtidal = calculate_tidal_radius(s, gd, lid, lid)

        nums_track.append(num)
        time.append(ds['Time'])
        leaf_id.append(lid)
        leaf_radius.append(rleaf)
        tidal_radius.append(rtidal)

    tmp = pd.DataFrame(dict(time=time,
                            leaf_id=leaf_id,
                            leaf_radius=leaf_radius,
                            tidal_radius=tidal_radius),
                       index=nums_track, dtype=object).sort_index()
    tmp.attrs = cores.attrs

    cores = pd.concat([cores, tmp])

    return cores


def calculate_tidal_radius(s, gd, node, leaf=None):
    """Calculate tidal radius of this node

    Tidal radius is defined as the distance to the closest node, excluding
    itself and its descendants.

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    gd : grid_dendro.Dendrogram
        Dendrogram object.
    node : int
        ID of the grid-dendro node.

    Returns
    -------
    rtidal : float
        Tidal radius.
    """
    if node == gd.trunk:
        # If this node is a trunk, tidal radius is the half the box size,
        # assuming periodic boundary condition.
        return 0.5*s.Lbox
    if leaf is None:
        leaf = gd.find_minimum(node)
    nodes = set(gd.nodes.keys()) - set(gd.descendants[node]) - {node}
    dst = [get_node_distance(s, nd, leaf) for nd in nodes]
    rtidal = np.min(dst)
    return rtidal


def calculate_critical_tes(s, rprf, core):
    """Calculates critical tes given the radial profile.

    Given the radial profile, find the critical tes at the same central
    density. return the ambient density, radius, power law index, and the sonic
    scale.

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    rprf : xarray.Dataset
        Object containing radial profiles.
    core : pandas.Series
        Object containing core informations

    Returns
    -------
    res : dict
        center_density, edge_density, critical_radius, pindex, sonic_radius
    """
    # Set scale length and mass based on the center and edge densities
    rhoc = rprf.rho.isel(r=0).data[()]
    r0 = s.cs/np.sqrt(4*np.pi*s.gconst*rhoc)
    m0 = s.cs**3/np.sqrt(4*np.pi*s.gconst**3*rhoc)
    mtidal = (4*np.pi*rprf.r**2*rprf.rho).sel(r=slice(0, core.tidal_radius)
                                              ).integrate('r').data[()]
    mean_tidal_density = mtidal / (4*np.pi*core.tidal_radius**3/3)

    rmin_fit = 0.5*s.dx
    rmax_fit = max(core.tidal_radius, 16.5*s.dx)
    # Select data for sonic radius fit
    rds = rprf.r.sel(r=slice(rmin_fit, rmax_fit)).data
    vr = np.sqrt(rprf.dvel1_sq_mw.sel(r=slice(rmin_fit, rmax_fit)).data)

    res = linregress(np.log(rds), np.log(vr/s.cs))
    pindex = min(res.slope, 0.9999)  # Apply ceiling to pindex
    intercept = res.intercept

    if pindex <= 0:
        rs = dcrit = rcrit = mcrit = np.nan
    else:
        # sonic radius
        rs = np.exp(-intercept/pindex)

        # Find critical TES at the central density
        xi_s = rs / r0
        try:
            ts = tes.TES(pindex=pindex, rsonic=xi_s)
            dcrit = np.exp(ts.ucrit)
            rcrit = ts.rcrit*r0
            mcrit = ts.mcrit*m0
        except UserWarning:
            dcrit = rcrit = mcrit = np.nan

    res = dict(tidal_mass=mtidal, center_density=rhoc,
               mean_tidal_density=mean_tidal_density, sonic_radius=rs, pindex=pindex,
               critical_contrast=dcrit, critical_radius=rcrit,
               critical_mass=mcrit)
    return res


def calculate_radial_profile(s, ds, origin, rmax, lvec=None):
    """Calculates radial profiles of various properties at selected position

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    ds : xarray.Dataset
        Object containing simulation data.
    origin : tuple-like
        Coordinate origin (x0, y0, z0).
    rmax : float
        Maximum radius of radial bins.
    lvec : array, optional
        Angular momentum vector to align the polar axis.

    Returns
    -------
    rprof : xarray.Dataset
        Angle-averaged radial profiles.

    Notes
    -----
    vel1, vel2, vel3 : Mass-weighted mean velocities (v_r, v_theta, v_phi).
    vel1_sq_mw, vel2_sq_mw, vel3_sq_mw : Mass-weighted variance of the velocities.
    gacc1_mw : Mass-weighted mean gravitational acceleration.
    phi_mw : Mass-weighted mean gravitational potential.
    """
    # Sometimes, tidal radius is so small that the angular momentum vector
    # Cannot be computed. In this case, fall back to default behavior.
    # (to_spherical will assume z axis as the polar axis).
    if lvec is not None and (np.array(lvec)**2).sum() == 0:
        lvec = None

    # Slice data
    nbin = int(np.ceil(rmax/s.dx))
    ledge = 0.5*s.dx
    redge = (nbin + 0.5)*s.dx
    ds = ds.sel(x=slice(origin[0] - redge, origin[0] + redge),
                y=slice(origin[1] - redge, origin[1] + redge),
                z=slice(origin[2] - redge, origin[2] + redge))

    # Convert density and velocities to spherical coord.
    gacc = {}
    for dim, axis in zip(['x', 'y', 'z'], [1, 2, 3]):
        # Recenter velocity and calculate gravitational acceleration
        vel_ = ds[f'mom{axis}']/ds.dens
        ds[f'vel{dim}'] = vel_ - vel_.sel(x=origin[0], y=origin[1], z=origin[2])
        gacc[dim] = -ds.phi.differentiate(dim)

    _, (ds['vel1'], ds['vel2'], ds['vel3'])\
        = transform.to_spherical((ds.velx, ds.vely, ds.velz), origin, lvec)
    _, (ds['gacc1'], _, _)\
        = transform.to_spherical(gacc.values(), origin, lvec)
    ds = ds.drop_vars(['mom1', 'mom2', 'mom3'])
    ds = ds.rename_vars(dict(dens='rho'))

    # Perform radial binnings
    rprofs = {}

    # Volume-weighted averages
    for k in ['rho']:
        rprf_c = ds[k].sel(x=origin[0], y=origin[1], z=origin[2]).drop_vars(['x', 'y', 'z'])
        rprf = transform.fast_groupby_bins(ds[k], 'r', ledge, redge, nbin)
        rprofs[k] = xr.concat([rprf_c, rprf], 'r')

    # Mass-weighted averages
    for k in ['gacc1', 'velx', 'vely', 'velz', 'vel1', 'vel2', 'vel3', 'phi']:
        rprf_c = ds[k].sel(x=origin[0], y=origin[1], z=origin[2]).drop_vars(['x', 'y', 'z'])
        rprf = transform.fast_groupby_bins(ds.rho*ds[k], 'r', ledge, redge, nbin) / rprofs['rho']
        rprofs[k+'_mw'] = xr.concat([rprf_c, rprf], 'r')

    # Mass-weighted average of the quantity squared
    for k in ['velx', 'vely', 'velz', 'vel1', 'vel2', 'vel3']:
        rprf_c = ds[k].sel(x=origin[0], y=origin[1], z=origin[2]).drop_vars(['x', 'y', 'z'])**2
        rprf = transform.fast_groupby_bins(ds.rho*ds[k]**2, 'r', ledge, redge, nbin) / rprofs['rho']
        rprofs[k+'_sq_mw'] = xr.concat([rprf_c, rprf], 'r')

    rprofs = xr.Dataset(rprofs)

    # Drop theta and phi coordinates
    for k in ['th', 'ph']:
        if k in rprofs:
            rprofs = rprofs.drop_vars(k)

    return rprofs


def calculate_prj_radial_profile(s, num, origin):
    """Calculate projected radial profile of column density and velocities

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    core : pandas.Series
        Object containing core informations

    Returns
    -------
    rprf : dict
        Dictionary containing projected radial profiles along x,y,z directions
    """

    # Read the central position of the core and recenter the snapshot
    xc, yc, zc = origin

    prj = s.read_prj(num)

    xycoordnames = dict(z=['x', 'y'],
                        x=['y', 'z'],
                        y=['z', 'x'])

    xycenters = dict(z=[xc, yc],
                     x=[yc, zc],
                     y=[zc, xc])

    # Calculate surface density radial profiles
    rprofs = {}
    ledge = 0.5*s.dx
    nbin = s.domain['Nx'][0]//2 - 1
    redge = (nbin + 0.5)*s.dx
    for i, ax in enumerate(['x', 'y', 'z']):
        x1, x2 = xycoordnames[ax]
        x1c, x2c = xycenters[ax]


        # Volume-weighted averages
        for qty in [k for k in prj[ax].keys() if k.startswith('Sigma_gas')]:
            ds = prj[ax][qty].copy(deep=True)
            ds, new_center = recenter_dataset(ds, {x1:x1c, x2:x2c})
            ds.coords['R'] = np.sqrt((ds.coords[x1]- new_center[x1])**2
                                     + (ds.coords[x2] - new_center[x2])**2)
            rprf_c = xr.DataArray(ds.sel({x1:new_center[x1], x2:new_center[x2]}).data[()],
                                  dims='R', coords={'R':[0,]})
            rprf = transform.fast_groupby_bins(ds, 'R', ledge, redge, nbin)
            rprf = xr.concat([rprf_c, rprf], dim='R')
            rprofs[f'{ax}_{qty}'] = rprf

        # Mass-weighted averages
        for qty in [k for k in prj[ax].keys() if k.startswith('vel_nc')]:
            ds = prj[ax][qty].copy(deep=True)
            ds, new_center = recenter_dataset(ds, {x1:x1c, x2:x2c})
            ds.coords['R'] = np.sqrt((ds.coords[x1]- new_center[x1])**2
                                     + (ds.coords[x2] - new_center[x2])**2)
            nth = qty.split('vel_nc')[1]  # read threshold density string
            w = prj[ax][f'Sigma_gas_nc{nth}'].copy(deep=True)
            w, _ = recenter_dataset(w, {x1:x1c, x2:x2c})
            w.coords['R'] = np.sqrt((w.coords[x1]- new_center[x1])**2
                                    + (w.coords[x2] - new_center[x2])**2)
            rprf_c = xr.DataArray(ds.sel({x1:new_center[x1], x2:new_center[x2]}).data[()],
                                  dims='R', coords={'R':[0,]})
            num = transform.fast_groupby_bins(w*ds, 'R', ledge, redge, nbin)
            den = transform.fast_groupby_bins(w, 'R', ledge, redge, nbin)
            rprf = num/den
            rprf = xr.concat([rprf_c, rprf], dim='R')
            rprofs[f'{ax}_{qty}'] = rprf

        # RMS averages
        for qty in [k for k in prj[ax].keys() if k.startswith('veldisp_nc')]:
            ds = prj[ax][qty].copy(deep=True)
            ds, new_center = recenter_dataset(ds, {x1:x1c, x2:x2c})
            ds.coords['R'] = np.sqrt((ds.coords[x1]- new_center[x1])**2
                                     + (ds.coords[x2] - new_center[x2])**2)
            nth = qty.split('veldisp_nc')[1]  # read threshold density string
            w = prj[ax][f'Sigma_gas_nc{nth}'].copy(deep=True)
            w, _ = recenter_dataset(w, {x1:x1c, x2:x2c})
            w.coords['R'] = np.sqrt((w.coords[x1]- new_center[x1])**2
                                    + (w.coords[x2] - new_center[x2])**2)
            rprf_c = xr.DataArray(ds.sel({x1:new_center[x1], x2:new_center[x2]}).data[()],
                                  dims='R', coords={'R':[0,]})
            num = transform.fast_groupby_bins(w*ds**2, 'R', ledge, redge, nbin)
            den = transform.fast_groupby_bins(w, 'R', ledge, redge, nbin)
            rprf = np.sqrt(num/den)
            rprf = xr.concat([rprf_c, rprf], dim='R')
            rprofs[f'{ax}_{qty}'] = rprf

    rprofs = xr.Dataset(rprofs)

    return rprofs


def calculate_lagrangian_props(s, cores, rprofs):
    """Calculate Lagrangian properties of cores

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    cores : pandas.DataFrame
        Object containing core informations.
    rprofs : xarray.Dataset
        Object containing radial profiles.

    Returns
    -------
    lprops : pandas.DataFrame
        Object containing Lagrangian properties of cores.
    """
    # Slice cores that have corresponding radial profiles
    common_indices = sorted(set(cores.index) & set(rprofs.num.data))
    cores = cores.loc[common_indices]
    ncrit = cores.attrs['numcrit']
    ncoll = cores.attrs['numcoll']
    rcore = cores.attrs['rcore']
    mcore = cores.attrs['mcore']

    if np.isnan(ncrit):
        radius = menc_crit = rhoe = rhoavg = np.nan
        vinfall = vcom = sigma_mw = sigma_1d = sigma_1d_trb = sigma_1d_blk = np.nan
        Fthm = Ftrb = Fcen = Fani = Fgrv = np.nan
    else:
        radius, menc_crit, rhoe, rhoavg = [], [], [], []
        vinfall, vcom, sigma_mw, sigma_1d, sigma_1d_trb, sigma_1d_blk = [], [], [], [], [], []
        Fthm, Ftrb, Fcen, Fani, Fgrv = [], [], [], [], []
        for num, core in cores.iterrows():
            rprof = rprofs.sel(num=num)

            # Find radius which encloses mcore.
            if rprof.menc.isel(r=-1) < mcore:
                # In this case, no radius up to maximum tidal radius encloses
                # mcore. This means we are safe to set rcore = Rtidal.
                r_M = np.inf
            else:
                r_M = brentq(lambda x: rprof.menc.interp(r=x) - mcore,
                             rprof.r.isel(r=0), rprof.r.isel(r=-1))
            radius.append(r_M)

            # enclosed mass within the critical radius
            if np.isnan(core.critical_radius):
                menc_crit.append(np.nan)
            else:
                menc_crit.append(rprof.menc.interp(r=core.critical_radius).data[()])


            # Mass-weighted infall speed
            rprf = rprof.sel(r=slice(0, r_M))
            vin = rprf.vel1_mw.weighted(rprf.r**2*rprf.rho).mean().data[()]
            vinfall.append(vin)

            # Mass-weighted velocity dispersion
            rprf = rprof.sel(r=slice(0, r_M))

            sigmw = np.sqrt(rprf.dvel1_sq_mw.weighted(rprf.r**2*rprf.rho).mean().data[()])
            sigma_mw.append(sigmw)

            vx_com = rprf.velx_mw.weighted(rprf.r**2*rprf.rho).mean()
            vy_com = rprf.vely_mw.weighted(rprf.r**2*rprf.rho).mean()
            vz_com = rprf.velz_mw.weighted(rprf.r**2*rprf.rho).mean()
            vcom.append(np.sqrt(vx_com**2 + vy_com**2 + vz_com**2).data[()])


            # Mass-weighted 1D velocity dispersion from 3D average
            sig1d = np.sqrt((rprf.velx_sq_mw.weighted(rprf.r**2*rprf.rho).mean()
                           + rprf.vely_sq_mw.weighted(rprf.r**2*rprf.rho).mean()
                           + rprf.velz_sq_mw.weighted(rprf.r**2*rprf.rho).mean()
                           - vx_com**2 - vy_com**2 - vz_com**2).data[()]/3)
            sigma_1d.append(sig1d)

            # turbulent component of 1D velocity dispersion
            sig1d = np.sqrt((rprf.dvel1_sq_mw.weighted(rprf.r**2*rprf.rho).mean()
                           + rprf.dvel2_sq_mw.weighted(rprf.r**2*rprf.rho).mean()
                           + rprf.dvel3_sq_mw.weighted(rprf.r**2*rprf.rho).mean()).data[()]/3)
            sigma_1d_trb.append(sig1d)

            # bulk component of 1D velocity dispersion
            sig1d = np.sqrt(((rprf.vel1_mw**2).weighted(rprf.r**2*rprf.rho).mean()
                           + (rprf.vel2_mw**2).weighted(rprf.r**2*rprf.rho).mean()
                           + (rprf.vel3_mw**2).weighted(rprf.r**2*rprf.rho).mean()).data[()]/3)
            sigma_1d_blk.append(sig1d)

            # select r = r_M
            rprf = rprof.interp(r=r_M)
            rhoe.append(rprf.rho.data[()])
            rhoavg.append(mcore / (4*np.pi*r_M**3/3))
            Fthm.append(rprf.Fthm.data[()])
            Ftrb.append(rprf.Ftrb.data[()])
            Fcen.append(rprf.Fcen.data[()])
            Fani.append(rprf.Fani.data[()])
            Fgrv.append(rprf.Fgrv.data[()])
    lprops = pd.DataFrame(data = dict(radius=radius, menc_crit=menc_crit, edge_density=rhoe, mean_density=rhoavg,
                                      vinfall=vinfall, vcom=vcom, sigma_mw=sigma_mw, sigma_1d=sigma_1d, sigma_1d_trb=sigma_1d_trb, sigma_1d_blk=sigma_1d_blk,
                                      Fthm=Fthm, Ftrb=Ftrb, Fcen=Fcen, Fani=Fani, Fgrv=Fgrv),
                          index = cores.index)

    # Attach some attributes
    # Velocity dispersion at t_crit
    if np.isnan(ncrit):
        vcom = np.nan
        sigma_r = np.nan
        sigma_1d = np.nan
        sigma_1d_trb = np.nan
    else:
        vcom = lprops.loc[ncrit].vcom
        sigma_r = lprops.loc[ncrit].sigma_mw
        sigma_1d = lprops.loc[ncrit].sigma_1d
        sigma_1d_trb = lprops.loc[ncrit].sigma_1d_trb
    lprops.attrs['vcom'] = vcom
    lprops.attrs['sigma_r'] = sigma_r
    lprops.attrs['sigma_1d'] = sigma_1d
    lprops.attrs['sigma_1d_trb'] = sigma_1d_trb

    # Free-fall time at t_coll
    lprops.attrs['tff_coll'] = tfreefall(lprops.loc[ncoll].mean_density, s.gconst)

    return lprops


def calculate_cumulative_energies(s, rprf, core):
    """Calculate cumulative energies based on radial profiles

    Use the mass-weighted mean gravitational potential at the tidal radius
    as the reference point. Mass-weighted mean is appropriate if we want
    the d(egrv)/dr = 0 as R -> Rtidal.

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    rprf : xarray.Dataset
        Object containing radial profiles.
    core : pandas.Series
        Object containing core informations.

    Returns
    -------
    rprf : xarray.Dataset
        Object containing radial profiles, augmented by energy fields
    """
    # TODO(SMOON) change the argument core to rmax and
    # substitute tidal_radius below to rmax.
    # Also, return the bound radius.
    # from scipy.interpolate import interp1d
    # etot_f = interp1d(rprf.r, rprf.etot)
    # rcore = brentq(etot_f, rprf.r[1], core.tidal_radius)

    # Thermal energy
    gm1 = (5/3 - 1)
    ethm = (4*np.pi*rprf.r**2*s.cs**2*rprf.rho/gm1).cumulative_integrate('r')

    # Kinetic energy
    vsq = rprf.vel1_sq_mw + rprf.vel2_sq_mw + rprf.vel3_sq_mw
    vcomsq = rprf.vel1_mw**2 + rprf.vel2_mw**2 + rprf.vel3_mw**2
    ekin = ((4*np.pi*rprf.r**2*0.5*rprf.rho*vsq).cumulative_integrate('r')
            - vcomsq*(4*np.pi*rprf.r**2*0.5*rprf.rho).cumulative_integrate('r'))

    # Gravitational energy
    phi0 = rprf.phi_mw.interp(r=core.tidal_radius)
    egrv = ((4*np.pi*rprf.r**2*rprf.rho*rprf.phi_mw).cumulative_integrate('r')
            - phi0*(4*np.pi*rprf.r**2*rprf.rho).cumulative_integrate('r'))

    rprf['ethm'] = ethm
    rprf['ekin'] = ekin
    rprf['egrv'] = egrv
    rprf['etot'] = ethm + ekin + egrv

    return rprf


def calculate_infall_rate(rprofs, cores):
    time, vr, mdot = [], [], []
    for num, rtidal in cores.tidal_radius.items():
        rprf = rprofs.sel(num=num).interp(r=rtidal)
        time.append(rprf.t.data[()])
        vr.append(-rprf.vel1_mw.data[()])
        mdot.append((-4*np.pi*rprf.r**2*rprf.rho*rprf.vel1_mw).data[()])
    if 'num' in rprofs.indexes:
        rprofs = rprofs.drop_indexes('num')
    rprofs['infall_speed'] = xr.DataArray(vr, coords=dict(t=time))
    rprofs['infall_rate'] = xr.DataArray(mdot, coords=dict(t=time))
    if 'num' not in rprofs.indexes:
        rprofs = rprofs.set_xindex('num')
    return rprofs


def calculate_accelerations(s, rprf):
    """Calculate RHS of the Lagrangian EOM (force per unit mass)

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    rprf : xarray.Dataset
        Radial profiles

    Returns
    -------
    acc : xarray.Dataset
        Accelerations appearing in Lagrangian EOM

    """
    if 'num' in rprf.indexes:
        rprf = rprf.drop_indexes('num')
    pthm = rprf.rho*s.cs**2
    ptrb = rprf.rho*rprf.dvel1_sq_mw
    acc = dict(adv=rprf.vel1_mw*rprf.vel1_mw.differentiate('r'),
               thm=-pthm.differentiate('r') / rprf.rho,
               trb=-ptrb.differentiate('r') / rprf.rho,
               cen=((rprf.vel2_mw**2 + rprf.vel3_mw**2) / rprf.r).where(rprf.r > 0, other=0),
               grv=rprf.gacc1_mw,
               ani=((rprf.dvel2_sq_mw + rprf.dvel3_sq_mw - 2*rprf.dvel1_sq_mw)
                    / rprf.r).where(rprf.r > 0, other=0))
    if s.mhd:
        pmag = 0.5*(rprf.b2**2 + rprf.b3**2 - rprf.b1**2)
        acc['mag'] = -pmag.differentiate('r') / rprf.rho
    else:
        acc['mag'] = rprf.rho*0

    acc = xr.Dataset(acc)
    acc['dvdt_lagrange'] = acc.thm + acc.trb + acc.mag + acc.grv + acc.cen + acc.ani
    acc['dvdt_euler'] = acc.dvdt_lagrange - acc.adv

    dm = 4*np.pi*rprf.r**2*rprf.rho
    acc['Fadv'] = (dm*acc.adv).cumulative_integrate('r')
    acc['Fthm'] = (dm*acc.thm).cumulative_integrate('r')
    acc['Ftrb'] = (dm*acc.trb).cumulative_integrate('r')
    acc['Fmag'] = (dm*acc.mag).cumulative_integrate('r')
    acc['Fcen'] = (dm*acc.cen).cumulative_integrate('r')
    acc['Fgrv'] = (-dm*acc.grv).cumulative_integrate('r')
    acc['Fani'] = (dm*acc.ani).cumulative_integrate('r')

    return acc


def calculate_observables(s, core, rprf):
    """Calculate observable properties of a core"""
    nthr_list = [10, 30, 100]
    num = core.name
    obsprops = dict()
    obsprops['num'] = num
    prj = s.read_prj(num)
    xc, yc, zc = get_coords_node(s, core.leaf_id)
    xycoordnames = dict(z=['x', 'y'],
                        x=['y', 'z'],
                        y=['z', 'x'])
    xycenters = dict(z=[xc, yc],
                     x=[yc, zc],
                     y=[zc, xc])

    # Read 3d data cube
    dens_3d = s.load_hdf5(num, quantities=['dens']).dens
    dens_3d, new_center_3d = recenter_dataset(dens_3d, dict(x=xc, y=yc, z=zc))
    for i, ax in enumerate(['x', 'y', 'z']):
        x1, x2 = xycoordnames[ax]
        x1c, x2c = xycenters[ax]
        dens_3d.coords[f'{ax}_rpos'] = np.sqrt((dens_3d.coords[x1] -
                                                new_center_3d[x1])**2
                                               + (dens_3d.coords[x2] -
                                                  new_center_3d[x2])**2)
        dens_3d.coords[f'{ax}_rlos'] = np.abs(dens_3d.coords[ax] - new_center_3d[ax])

    # Simplest background subtraction -- average column in the whole box
    dcol_bgr0 = s.rho0*s.Lbox

    # Observable properties from radial profiles,
    # without any density thresholding.
    # This is analogous to dust map.
    for i, ax in enumerate(['x', 'y', 'z']):
        dcol = rprf[f'{ax}_Sigma_gas']
        # Central column density
        dcol_c = dcol.sel(R=0).data[()] - dcol_bgr0
        try:
            # Calculate FWHM quantities
            rfwhm = obs_core_radius(dcol, 'fwhm', dcol_bgr=dcol_bgr0)
            mfwhm = ((dcol - dcol_bgr0)*2*np.pi*dcol.R
                     ).sel(R=slice(0, rfwhm)).integrate('R').data[()]
            dcol_fwhm = dcol.interp(R=rfwhm)
            mfwhm_bgrsub = ((dcol - dcol_fwhm)*2*np.pi*dcol.R
                            ).sel(R=slice(0, rfwhm)).integrate('R').data[()]
            dfwhm = mfwhm / (4*np.pi*rfwhm**3/3)
        except ValueError:
            rfwhm = mfwhm = mfwhm_bgrsub = dfwhm = np.nan
        obsprops[f'{ax}_radius'] = rfwhm
        obsprops[f'{ax}_mass'] = mfwhm
        obsprops[f'{ax}_mass_bgrsub'] = mfwhm_bgrsub
        obsprops[f'{ax}_mean_density'] = dfwhm
        obsprops[f'{ax}_center_column_density'] = dcol_c


    # Observable properties using density thresholding.
    # Analogous to molecular line observations.
    for nthr in nthr_list:
        d3dthr = dens_3d.where(dens_3d > nthr, other=0)
        for ax in ['x', 'y', 'z']:
            x1, x2 = xycoordnames[ax]
            x1c, x2c = xycenters[ax]

            # POS FWHM radius
            dcol_prf = rprf[f'{ax}_Sigma_gas_nc{nthr}']
            try:
                rfwhm = obs_core_radius(dcol_prf, method='fwhm')
            except:
                rfwhm = np.nan

            # POS radius using background thresholding
            dcol_map = prj[ax][f'Sigma_gas_nc{nthr}'].copy(deep=True)
            dv_map = prj[ax][f'veldisp_nc{nthr}'].copy(deep=True)
            dcol_map, _ = recenter_dataset(dcol_map, {x1: x1c, x2: x2c})
            dv_map, new_center = recenter_dataset(dv_map, {x1: x1c, x2: x2c})
            rpos = np.sqrt((dv_map.coords[x1] - new_center[x1])**2
                           + (dv_map.coords[x2] - new_center[x2])**2)

            # Set a threshold column density for a given "tracer"
            dcol_bgr = dcol_bgr0*s.mfrac_above(nthr/s.rho0)

            # POS radius at which any pixel falls below dcol_bgr
            rmax = rpos.where(dcol_map < dcol_bgr).min().data[()]

            # POS radius using filling factor thresholding
            afrac_thres = 0.5
            flag = xr.where(dcol_map > dcol_bgr, 1, 0)
            flag.coords['R'] = dens_3d.coords[f'{ax}_rpos']
            ledge = 0.5*s.dx
            nbin = s.domain['Nx'][0]//2 - 1
            redge = (nbin + 0.5)*s.dx
            afrac = transform.fast_groupby_bins(flag, 'R', ledge, redge, nbin)
            xb = afrac.R.data[afrac < afrac_thres][0]
            xa = xb - s.dx
            try:
                rmax2 = brentq(lambda x: interp1d(afrac.R.data, afrac.data)(x) - afrac_thres, xa, xb)
            except ValueError:
                rmax2 = np.nan

            # Loop over different plane-of-sky radius definitions
            for rcore_pos, method in zip([obsprops[f'{ax}_radius'], rfwhm, rmax, rmax2], ['fwhm_dust', 'fwhm', 'f0', 'f50']):
                obsprops[f'{ax}_pos_radius_{method}_nc{nthr}'] = rcore_pos
                if np.isfinite(rcore_pos):
                    obsprops[f'{ax}_velocity_dispersion_{method}_nc{nthr}'] =\
                        np.sqrt((dv_map.where(rpos < rcore_pos)**2
                                 ).weighted(dcol_map).mean()).data[()]

                    # True line-of-sight distance
                    rlos_crd = dens_3d.coords[f'{ax}_rlos']
                    rpos_crd = dens_3d.coords[f'{ax}_rpos']
                    # Optimized numpy operation using broadcast; almost order of faster than
                    # built-in xarray weighted average which is commented out.
                    # Slower version looks like:
#                    rlos_true = rlos_crd.where(rpos_crd < rcore_pos
#                                               ).weighted(d3dthr).mean().data[()]
                    # Faster version:
                    arr, msk, wgt = xr.broadcast(rlos_crd, rpos_crd < rcore_pos, d3dthr)
                    arr = arr.transpose('z', 'y', 'x').data
                    msk = msk.transpose('z', 'y', 'x').data
                    wgt = wgt.transpose('z', 'y', 'x').data
                    try:
                        # True line-of-sight distance defined by density-weighted average
                        # of |z - z0| over a cylinder R < R_pos
                        # Note that we are not using rms average.
                        rlos_true = np.average(arr[msk], weights=wgt[msk])
                    except ZeroDivisionError:
                        rlos_true = np.nan
                    obsprops[f'{ax}_los_radius_{method}_nc{nthr}'] = rlos_true
                    obsprops[f'{ax}_mean_column_density_{method}_nc{nthr}']\
                            = dcol_prf.sel(R=slice(0, rcore_pos)).weighted(dcol_prf.R).mean().data[()]
                else:
                    obsprops[f'{ax}_velocity_dispersion_{method}_nc{nthr}'] = np.nan
                    obsprops[f'{ax}_los_radius_{method}_nc{nthr}'] = np.nan
                    obsprops[f'{ax}_mean_column_density_{method}_nc{nthr}'] = np.nan
    return obsprops


def column_density(rcyl, frho, rmax):
    """Calculate column density

    Parameters
    ----------
    rcyl : float
        Cylindrical radius at which the column density is computed
    frho : function
        The function rho(r) that returns the volume density at a given
        spherical radius.
    rmax : float
        The maximum radius to integrate out.

    Returns
    -------
    dcol : float
        Column density.
    """
    def func(z, rcyl):
        r = np.sqrt(rcyl**2 + z**2)
        return frho(r)
    if isinstance(rcyl, np.ndarray):
        dcol = []
        for R in rcyl:
            zmax = np.sqrt(rmax**2 - R**2)
            res, _ = quad(func, 0, zmax, args=(R,), epsrel=1e-2, limit=200)
            dcol.append(2*res)
        dcol = np.array(dcol)
    else:
        zmax = np.sqrt(rmax**2 - rcyl**2)
        res, _ = quad(func, 0, zmax, args=(rcyl,), epsrel=1e-2, limit=200)
        dcol = 2*res
    return dcol


def critical_time(s, pid, method='empirical'):
    cores = s.cores[pid].copy()
    if len(cores) == 0:
        return np.nan
    cores = cores.loc[:cores.attrs['numcoll']]
    rprofs = s.rprofs[pid]

    ncrit = None

    if method == 'empirical':
        # Earliest time after which the net force integrated within r_crit
        # remains negative until the end of the collapse.
        # To find this "empirical critical time", we start from t_coll
        # and march backward in time.
        for num, core in cores.sort_index(ascending=False).iterrows():
            # Exclude t_coll snapshot at which the turbulence has amplified
            # to produce nagative linewidth-size slope.
            if num in cores.index[-2:]:
                if np.isnan(core.critical_radius):
                    n2coll = cores.attrs['numcoll'] - num
                    msg = (f"Critical radius at t_coll - {n2coll}"
                           f" is NaN for par {pid}, method {method}."
                           " This may have been caused by negative pindex."
                           " Continuing...")
                    s.logger.warning(msg)
                    continue
            rprf = rprofs.sel(num=num)

            # Net force at the critical radius is negative after the
            # critical time, throughout the collapse.
            if np.isfinite(core.critical_radius):
                fnet = (rprf.Fthm + rprf.Ftrb + rprf.Fcen + rprf.Fani
                        - rprf.Fgrv)
                fnet = fnet.interp(r=core.critical_radius).data[()]
            else:
                fnet = np.nan
            # Whatever fnet is, if it is not negative, we should break.
            # That is, when rcrit = NaN or inf, we should break.
            # However, NaN can be artificial, we can probably impose
            # the upper limit on p.
            if not fnet < 0:
                ncrit = num + 1
                break
    elif method in ['predicted', 'pred_xis']:
        for num, core in cores.sort_index(ascending=True).iterrows():
            if method == 'predicted':
                # Predicted critical time using R_tidal_avg and Menc
                rprf = rprofs.sel(num=num)
                if np.isfinite(core.critical_radius):
                    menc = rprf.menc.interp(r=core.critical_radius).data[()]
                else:
                    menc = np.nan
                rtidal_avg = 0.5*(core.leaf_radius + core.tidal_radius)
                cond1 = rtidal_avg >= core.critical_radius
                cond2 = menc >= core.critical_mass
                if cond1 and cond2:
                    ncrit = num
                    break
            elif method == 'pred_xis':
                # Predicted critical time using xi_s
                r0 = s.cs / np.sqrt(4*np.pi*s.gconst*core.center_density)
                xi_s = core.sonic_radius / r0
                cond1 = xi_s > 8.99  # r_crit = r_s at xi_s = 8.99 for p=0.5
                cond2 = core.pindex > 0 and core.pindex < 1
                if cond1 and cond2:
                    ncrit = num
                    break

    if ncrit is None or ncrit == cores.index[-1] + 1:
        # If the critical condition is satisfied for all time, or is not
        # satisfied at t_coll, set ncrit to NaN.
        # TODO: we may not want to discard those that the critical condition
        # is satisfied for all times.
        ncrit = np.nan
    return ncrit


def get_coords_minimum(dat):
    """returns coordinates at the minimum of dat

    Args:
        dat : xarray.DataArray instance (usually potential)
    Returns:
        x0, y0, z0
    """
    center = dat.argmin(...)
    x0, y0, z0 = [dat.isel(center).coords[dim].data[()]
                  for dim in ['x', 'y', 'z']]
    return x0, y0, z0


def get_coords_node(s, nd):
    """Get coordinates of the generating point of this node

    Parameters
    ----------
    s : LoadSim
        Simulation metadata.
    nd : int
        GRID-dendro node ID.

    Returns
    -------
    coordinates: tuple representing physical coordinates (x, y, z)
    """
    k, j, i = np.unravel_index(nd, s.domain['Nx'].T, order='C')
    coordinates = (s.domain['le']
                   + np.array([i+0.5, j+0.5, k+0.5])*s.domain['dx'])
    return coordinates


def get_periodic_distance(pos1, pos2, Lbox, return_axis_distance=False):
    hLbox = 0.5*Lbox
    axis_distance = []
    for x1, x2 in zip(pos1, pos2):
        dst = np.abs(x1-x2)
        dst = Lbox - dst if dst > hLbox else dst
        axis_distance.append(dst)
    axis_distance = np.array(axis_distance)
    dst = np.sqrt((axis_distance**2).sum())
    if return_axis_distance:
        return axis_distance
    else:
        return dst


def get_node_distance(s, nd1, nd2):
    """Calculate periodic distance between two nodes

    Parameters
    ----------
    s : LoadSim
        Simulation metadata.
    nd1 : int
        GRID-dendro node ID
    nd2 : int
        GRID-dendro node ID
    """
    pos1 = get_coords_node(s, nd1)
    pos2 = get_coords_node(s, nd2)
    # TODO generalize this
    dst = get_periodic_distance(pos1, pos2, s.Lbox)
    return dst


def get_sonic(Mach_outer, l_outer, p=0.5):
    """returns sonic scale assuming linewidth-size relation v ~ R^p
    """
    if Mach_outer == 0:
        return np.inf
    lambda_s = l_outer*Mach_outer**(-1/p)
    return lambda_s


def recenter_dataset(ds, center):
    """Recenter whole dataset or dataarray.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset to be recentered.
    center : dict
        {x:xc, y:yc} or {x:xc, y:yc, z:zc}, etc.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Recentered dataset.
    tuple
        Position of the new center. This must be the grid coordinates
        closest, but not exactly the same, to (0, 0, 0).
    """
    shift, new_center = {}, {}
    for dim, pos in center.items():
        hNx = ds.sizes[dim] // 2
        coords = ds.coords[dim].data
        dx = coords[1] - coords[0]
        shift[dim] = hNx - np.where(np.isclose(coords, pos, atol=0.1*dx))[0][0]
        new_center[dim] = ds.coords[dim].isel({dim: hNx}).data[()]

    return ds.roll(shift), new_center


def get_rhocrit_KM05(lmb_sonic):
    """Equation (17) of Krumholz & McKee (2005)

    Args:
        lmb_sonic: sonic length devided by Jeans length at mean density.
    Returns:
        rho_crit: critical density devided by mean density.
    """
    phi_x = 1.12
    rho_crit = (phi_x/lmb_sonic)**2
    return rho_crit


def roundup(a, decimal):
    return np.ceil(a*10**decimal) / 10**decimal


def rounddown(a, decimal):
    return np.floor(a*10**decimal) / 10**decimal


def test_resolved_core(s, cores, nres):
    """Test if the given core is sufficiently resolved.

    Returns True if the critical radius at t_crit is greater than
    nres*dx.

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata
    pid : int
        Particle ID.
    nres : int
        Minimum number of cells to be considered resolved.

    Returns
    -------
    bool
        True if a core is resolved, false otherwise.
    """
    ncrit = cores.attrs['numcrit']
    if np.isnan(ncrit):
        return False
    ncells = cores.loc[ncrit].critical_radius / s.dx
    if ncells >= nres:
        return True
    else:
        return False


def test_isolated_core(s, cores):
    """Test if the given core is isolated.

    Criterion for an isolated core is that the core must not contain
    any particle at the time of collapse.

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    pid : int
        Particle ID.

    Returns
    -------
    bool
        True if a core is isolated, false otherwise.
    """
    ncrit = cores.attrs['numcrit']
    if np.isnan(ncrit):
        return False
    pds = s.load_par(ncrit)
    pstar = pds[['x1', 'x2', 'x3']]
    core = cores.loc[ncrit]

    nd = core.leaf_id
    pcore = get_coords_node(s, nd)

    dst = np.sqrt(((pstar - pcore)**2).sum(axis=1))

    return (dst > core.tidal_radius).all()


def lpdensity(r, cs, gconst):
    """Larson-Penston density profile

    Parameter
    ---------
    r : float
        Radius.
    cs : float
        Isothermal sound speed.
    gconst : float
        Gravitational constant.

    Returns
    -------
    float
        Asymptotic Larson-Penston density
    """

    return 8.86*cs**2/(4*np.pi*gconst*r**2)


def lpradius(m, cs, gconst):
    """Equivalent Larson-Penston radius containing mass m

    Parameter
    ---------
    m : float
        Mass.
    cs : float
        Isothermal sound speed.
    gconst : float
        Gravitational constant.

    Returns
    -------
    float
        Equivalent radius
    """
    return gconst*m/8.86/cs**2


def tfreefall(dens, gconst):
    """Free fall time at a given density.

    Parameter
    ---------
    dens : float
        Density.
    gconst : float
        Gravitational constant.

    Returns
    -------
    float
        Gravitational free-fall time
    """
    return np.sqrt(3*np.pi/(32*gconst*dens))


def reff_sph(vol):
    """Effective radius of a volume

    Reff = (3*vol/(4 pi))**(1/3)

    Parameter
    ---------
    vol : float
        Volume

    Returns
    -------
    float
        Effective spherical radius
    """
    fac = 0.6203504908994000865973817
    return fac*vol**(1/3)


def obs_core_radius(rprf_dcol, method='fwhm', dcol_bgr=0):
    """Observational core radius

    The radius at which the column density drops by 10% of the
    central value.

    Parameters
    ----------
    rprf_dcol : xarray.DataArray
        The radial column density profile.

    Returns
    -------
    robs : float
    """
    match method:
        case 'fwhm':
            rprf_dcol = rprf_dcol - dcol_bgr
            dcol_c = rprf_dcol.isel(R=0).data[()]
            idx = (rprf_dcol.data < 0.5*dcol_c).nonzero()[0]
            if len(idx) < 1:
                raise ValueError(f"Core radius with method {method} cannot be found")
            else:
                idx = idx[0]
            rmax = rprf_dcol.R.isel(R=idx).data[()]
            robs = utils.fwhm(interp1d(rprf_dcol.R.data[()], rprf_dcol.data),
                              rmax, which='column')
        case 'background':
            idx = (rprf_dcol.data < dcol_bgr).nonzero()[0]
            if len(idx) < 1:
                raise ValueError(f"Core radius with method {method} cannot be found")
            else:
                idx = idx[0]
            xa = rprf_dcol.R.isel(R=idx-1).data[()]
            xb = rprf_dcol.R.isel(R=idx).data[()]
            dcol_itp = interp1d(rprf_dcol.R.data, rprf_dcol.data)
            robs = brentq(lambda x: dcol_itp(x) - dcol_bgr, xa, xb)
    return robs


def get_evol_norm(vmin=-3, vmid=0, vmax=1):
    """Get a normalization for color coding evolutionary time

    Blue (vmin)  ->  white (vmid)  ->  red (vmax).
    """
    # Color scale
    from matplotlib import colors
    alpha = np.log(0.5) / np.log((vmid - vmin)/(vmax - vmin))

    def _forward(x):
        t = (x - vmin) / (vmax - vmin)
        return t**alpha

    def _inverse(x):
        return vmin + (vmax - vmin)*x**(1/alpha)
    norm = colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)
    return norm


def get_evol_cbar(mappable, ax=None, cax=None, ticks=[-3, -1, 0, 0.5, 1],
                  label=r'$\dfrac{t - t_\mathrm{crit}}{\Delta t_\mathrm{coll}}$'):
    """Get an appropriate color bar for get_evol_norm"""
    import matplotlib.pyplot as plt
    cbar = plt.colorbar(mappable, ax=ax, cax=cax, label=label)
    cbar.solids.set(alpha=1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    cbar.ax.minorticks_off()
    return cbar


def sawtooth(x, xmin, xmax, ymin, ymax):
    """Sawtooth curve

    Linear from [xmin, ymin] to [xmax, ymax] and then periodic elsewhere
    """
    t = ((x - xmax) + (x - xmin)) / (xmax - xmin)
    p = 2
    u = 2*(t / p - np.floor(0.5 + t / p))
    y = 0.5*(ymax - ymin)*u + 0.5*(ymax + ymin)
    return y
