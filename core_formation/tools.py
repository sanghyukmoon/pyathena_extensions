import numpy as np
import xarray as xr
# Bottleneck does not use stable sum.
# See xarray #1346, #7344 and bottleneck #193, #462 and more.
# Let's disable third party softwares to go conservative.
# Accuracy is more important than performance.
xr.set_options(use_bottleneck=False, use_numbagg=False)
import pandas as pd
import dask
import dask.array as da
from scipy.special import erfcinv, erfc
from scipy.optimize import brentq, curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.ndimage import label, uniform_filter
from astropy import units as au
from pathlib import Path
from pyathena.util import transform
from tesphere import utils, tes

from . import load_sim, config


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
    lid : int
        Flat index of the center of the t_coll core.
    """
    # Snapshot number right before the collapse
    num = s.tcoll_cores.loc[pid].num

    # Load the flat indices at minima
    idx_at_minima = s.minima[num]

    # find closeast minimum to this particle
    pos_particle = s.tcoll_cores.loc[pid][['x1', 'x2', 'x3']]
    pos_particle = pos_particle.to_numpy()
    dst = [periodic_distance(s.flatindex_to_cartesian(lid),
                             pos_particle, s.Lbox)
           for lid in idx_at_minima]
    lid = idx_at_minima[np.argmin(dst)]

    # return the flat index of the center of the t_coll core.
    return lid

def track_cores(s, pid):
    """Perform reverse core tracking

    Parameters
    ----------
    s : LoadSim
    pid : int
    ncells_min : int, optional
        Minimum number of cells in a leaf. Default to 27.

    Returns
    -------
    cores : pandas.DataFrame

    See also
    --------
    track_protostellar_cores : Forward core tracking after t_coll into
                               the protostellar stage.
    """
    # start from t = t_coll and track backward
    numcoll = s.tcoll_cores.loc[pid].num
    nums = np.arange(numcoll, config.GRID_NUM_START-1, -1)
    num = nums[0]
    msg = f'[track_cores] processing model {s.basename} pid {pid} num {num}'
    print(msg)

    lid = find_tcoll_core(s, pid)

    # Test if any star particle is contained inside t_coll core.
    # TODO: This requires dendrogram construction; We should change algorithm
    # for usage in AthenaK

    assert s.par['output2']['file_type'] == 'hdf5' and s.par['output2']['variable'] == 'cons'
    dt_hdf5 = s.par['output2']['dt']

    nums_track = [num,]
    time = [s.num_to_time(num),]
    leaf_id = [lid,]
    for num in nums[1:]:
        print(f'[track_cores] processing model {s.basename} pid {pid} num {num}')
        minima = s.minima[num]
        lid_old = lid

        # find closeast leaf to the previous preimage
        dst = [s.distance_between(lid, lid_old) for lid in minima]
        lid = minima[np.argmin(dst)]

        nums_track.append(num)
        time.append(s.num_to_time(num))
        leaf_id.append(lid)
    # SMOON: Using dtype=object is to prevent automatic upcasting from int to float
    # when indexing a single row. Maybe there is a better approach.
    cores = pd.DataFrame(dict(time=time, leaf_id=leaf_id),
                         index=nums_track, dtype=object).sort_index()

    # Set attributes
    cores.attrs['pid'] = pid
    cores.attrs['numcoll'] = numcoll

    return cores


def tidal_radius():
    # TODO implement tidal radius calculation based on radial profiles of
    # sign of gravitational acceleration
    pass

def local_dendrogram(arr, center_pos, domain_left_edge, domain_cell_size,
                     hw=0.5, prune=True, ncells_min=27):
    """Construct a local dendrogram

    Parameters
    ----------
    arr : xarray.DataArray
        Input array to construct dendrogram. Usually, gravitational potential.
    center_pos : tuple
        Center position of the local dendrogram (x, y, z).
    domain_left_edge : tuple
        Left edge of the global domain. (xmin, ymin, zmin)
    domain_cell_size : tuple
        Cell size of the global domain. (dx, dy, dz)
    hw : float, optional
        Half width of the local domain. Default to 0.5.
    ncells_min : int, optional
        Minimum number of cells in a leaf. Default to 27.

    Returns
    -------
    gd : grid_dendro.Dendrogram
    """
    from grid_dendro import dendrogram
    x0, y0, z0 = center_pos
    xl, yl, zl = domain_left_edge
    dx, dy, dz = domain_cell_size
    arr, center, shift = recenter_dataset(arr, dict(x=x0, y=y0, z=z0))
    shape = arr.shape
    arr = arr.sel(dict(x=slice(-hw, hw), y=slice(-hw, hw), z=slice(-hw, hw)))

    il = ((arr.x[0].data - xl) // dx).astype(np.int32)
    jl = ((arr.y[0].data - yl) // dy).astype(np.int32)
    kl = ((arr.z[0].data - zl) // dz).astype(np.int32)
    start_indices = np.array([kl, jl, il]) - np.array([shift['z'], shift['y'], shift['x']])

    if isinstance(arr.data, da.Array):
        arr = arr.data.compute()
    else:
        arr = arr.data
    gd = dendrogram.Dendrogram(arr, boundary_flag='outflow')
    gd.construct()
    if prune:
        gd.prune(ncells_min)
    gd.reindex(start_indices, shape, direction='backward')
    return gd


def critical_tes_property(s, rprf, core):
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

    rmin_fit = 3.5*s.dx
    rmax_fit = 16.5*s.dx
    # Select data for sonic radius fit
    rds = rprf.r.sel(r=slice(rmin_fit, rmax_fit)).data
    vr = np.sqrt(
            rprf.dvel1_sq_mw.sel(r=slice(rmin_fit, rmax_fit)).data
            + rprf.dvel2_sq_mw.sel(r=slice(rmin_fit, rmax_fit)).data
            + rprf.dvel3_sq_mw.sel(r=slice(rmin_fit, rmax_fit)).data
            ) / np.sqrt(3)
    res, _ = curve_fit(lambda r, a, b: a*np.log(r) + b, rds, np.log(vr/s.cs))
    pindex, intercept = res[0], res[1]
    pmax = 0.999 # Apply ceiling to avoid unphysical pindex
    if pindex >= pmax:
        res, _ = curve_fit(lambda r, b: pmax*np.log(r) + b, rds, np.log(vr/s.cs))
        pindex, intercept = pmax, res[0]

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

    res = dict(center_density=rhoc,
               sonic_radius=rs, pindex=pindex,
               critical_contrast=dcrit, critical_radius=rcrit,
               critical_mass=mcrit)
    return res


def radial_profile(s, ds, origin, rmax=None, nsub=4, compute_flux=False):
    """Calculates radial profiles of various properties at selected position

    This function returns lazy Dataset if the inputs are dask array.

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    ds : xarray.Dataset
        Object containing simulation data.
    origin : tuple or list
        Coordinate origin (x0, y0, z0).
    rmax : float, optional
        Maximum radius of radial bins. If None, use s.Lbox/2
    nsub : int, optional
        Number of subcells per cell for subcell correction. Default to 4.

    Returns
    -------
    rprof : xarray.Dataset
        Angle-averaged radial profiles.

    Notes
    -----
    vel1, vel2, vel3 : Mass-weighted mean velocities (v_r, v_theta, v_phi).
    vel1_sq_mw, vel2_sq_mw, vel3_sq_mw : Mass-weighted variance of the velocities.
    gacc1_mw : Mass-weighted mean gravitational acceleration.
    frac_neg_gacc1 : Fraction of cells in each radial bin with inward
        radial gravitational acceleration (g_r < 0).
    vshell : Effective shell volume under the current discrete radial binning.
    phi_mw : Mass-weighted mean gravitational potential.
    mdot_x, mdot_y, mdot_z : Contributions from the Cartesian velocity
        components to the signed mass flux through the spherical surface at
        radius r, with positive values corresponding to inward flux.
    phi_B : Magnetic flux through a circular aperture of radius r whose normal
        follows the enclosed mean magnetic field direction.
    mdot_b, mdot_bperp1, mdot_bperp2 : Contributions from the velocity
        components along the orthonormal basis defined by the enclosed mean
        magnetic field direction and the two perpendicular directions.
    """
    # Define helper functions
    def _plane_basis(normal):
        normal = np.array(normal, dtype=float)
        norm = np.sqrt((normal**2).sum())
        if not np.isfinite(norm) or norm == 0:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal /= norm
        ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(normal, ref)
        e1 /= np.sqrt((e1**2).sum())
        e2 = np.cross(normal, e1)
        return e1, e2, normal

    def _magnetic_flux(radius, normal):
        e1, e2, normal = _plane_basis(normal)
        q = np.arange(-radius, radius + hdx, s.dx)
        u = xr.DataArray(q, dims='u', coords=dict(u=q))
        v = xr.DataArray(q, dims='v', coords=dict(v=q))
        xq = origin[0] + u*e1[0] + v*e2[0]
        yq = origin[1] + u*e1[1] + v*e2[1]
        zq = origin[2] + u*e1[2] + v*e2[2]
        R = np.sqrt(u**2 + v**2)
        bnorm = ds.bx*normal[0] + ds.by*normal[1] + ds.bz*normal[2]
        plane = bnorm.interp(x=xq, y=yq, z=zq, method='linear')
        return (plane.where(R <= radius).sum(('u', 'v')) * s.dx**2).rename('phi_B')

    def _spherical_at_subcells(vec, basis):
        """Project Cartesian vectors onto spherical bases at subcells."""
        vx, vy, vz = vec
        return tuple(
            vx*ex + vy*ey + vz*ez
            for ex, ey, ez in basis
        )

    def _radial_binning(qty, mass_weighted=False, qty_sub=None):
        """Inner wrapper function for radial binning

        Central sphere and the first few nonzero radial bins are corrected with
        subcell method, while the rest of the bins are calculated with simple
        binning. The number of subcell bins is determined by subcell_region_radius, which is
        set to 4 by default.

        Parameters
        ----------
        qty : xarray.DataArray
            Quantity to be binned.
        mass_weighted : bool, optional
            Whether to calculate mass-weighted average. Default to False.
        qty_sub : xarray.DataArray, optional
            Quantity evaluated at subcell positions with dimensions
            ``(subcell, cell)``. If omitted, the parent-cell value is used
            for every subcell belonging to that parent.

        Returns
        -------
        rprf : xarray.DataArray
            Radial profile of the quantity.
        vshell : xarray.DataArray
            Effective shell volume under the current discrete radial binning.

        Notes
        -----
        mass_weighted=True assumes that rprofs['rho'] is already calculated.
        """
        # Perform normal radial binning for entire domain
        dat = ds.rho*qty if mass_weighted else qty
        rprf, bin_cnt = transform.groupby_bins(dat, 'r', nbin, (hdx, redge),
                                               return_count=True)
        if mass_weighted:
            rprf = rprf / rprofs['rho']

        # Overwrite the central sphere and first few nonzero bins with
        # subcell corrected values.
        vshell = (bin_cnt*s.dV).rename('vshell')

        if qty_sub is None:
            qty_patch = (
                qty.sel(**subcell_region)
                .reset_coords(drop=True)
                .transpose('z', 'y', 'x')
                .stack(cell=('z', 'y', 'x'))
            )
            weight = rho_patch if mass_weighted else 1
            numer = (
                weight*qty_patch*subcell_frac*s.dV
            ).sum('cell')
        else:
            weight = rho_patch if mass_weighted else 1
            numer = (
                weight*qty_sub*flag_subcell_in_shell*dV_sub
            ).sum(('subcell', 'cell'))
        denom = shell_mass if mass_weighted else shell_volume
        rprf_patch = numer/denom

        # Stitch the subcell-corrected values with the rest of the radial profile
        rprf = xr.concat([rprf_patch, rprf.isel(r=slice(subcell_region_radius, None))], dim='r')
        vshell = xr.concat([shell_volume, vshell.isel(r=slice(subcell_region_radius, None))], dim='r')
        return rprf, vshell

    if rmax is None:
        rmax = s.Lbox/2

    # =========================================================================
    # ====== Step 1. Define variables and select the region of interest =======
    # =========================================================================

    # Select the region within the maximum radius
    nbin = int(np.ceil(rmax/s.dx))
    assert nbin > 0, f"nbin must be positive, got {nbin}"
    hdx = 0.5*s.dx
    redge = (nbin + 0.5)*s.dx
    ds = ds.sel(x=slice(origin[0] - redge, origin[0] + redge),
                y=slice(origin[1] - redge, origin[1] + redge),
                z=slice(origin[2] - redge, origin[2] + redge))
    ds = ds.rename_vars(dict(dens='rho'))
    if s.mhd:
        ds = ds.rename_vars(dict(Bcc1='bx', Bcc2='by', Bcc3='bz'))
    # Subtract the velocity at the potential minimum
    vel_origin = {}
    for dim, axis in zip(['x', 'y', 'z'], [1, 2, 3]):
        vel_ = ds[f'mom{axis}']/ds.rho
        vel_origin[dim] = vel_.sel(x=origin[0], y=origin[1], z=origin[2])
        ds[f'vel{dim}'] = vel_ - vel_origin[dim]
    ds = ds.drop_vars(['mom1', 'mom2', 'mom3'])
    # Angular momenta
    ds['Ldens_x'] = ds.rho*((ds.y - origin[1])*ds.velz - (ds.z - origin[2])*ds.vely)
    ds['Ldens_y'] = ds.rho*((ds.z - origin[2])*ds.velx - (ds.x - origin[0])*ds.velz)
    ds['Ldens_z'] = ds.rho*((ds.x - origin[0])*ds.vely - (ds.y - origin[1])*ds.velx)
    # Gravitational accelerations
    for dim in ['x', 'y', 'z']:
        ds[f'gacc{dim}'] = -ds.phi.differentiate(dim)

    # Transform vector fields to spherical coordinates
    _, (ds['vel1'], ds['vel2'], ds['vel3'])\
        = transform.to_spherical((ds.velx, ds.vely, ds.velz), origin)
    _, (ds['gacc1'], ds['gacc2'], ds['gacc3'])\
        = transform.to_spherical((ds.gaccx, ds.gaccy, ds.gaccz), origin)
    if s.mhd:
        _, (ds['b1'], ds['b2'], ds['b3'])\
            = transform.to_spherical((ds.bx, ds.by, ds.bz), origin)

    # flag cells where g_r is negative
    # TODO  need to set g_r = 0 at r = 0 before doing radial_binning.
    ds['flag_neg_gacc1'] = xr.where(ds.gacc1 < 0, 1.0, 0.0)

    # =========================================================================
    # ====== Step 2. Calculate radial profiles ================================
    # =========================================================================

    subcell_region_radius = min(4, nbin)
    if not isinstance(nsub, (int, np.integer)) or nsub <= 0 or nsub % 2:
        raise ValueError('nsub must be a positive even integer')

    if subcell_region_radius > 0:
        # 1. Select the subcell region and construct geometry
        # --------------------------------------------------------
        redge_sub = (subcell_region_radius + 0.5)*s.dx
        subcell_region = dict(x=slice(origin[0] - redge_sub, origin[0] + redge_sub),
                         y=slice(origin[1] - redge_sub, origin[1] + redge_sub),
                         z=slice(origin[2] - redge_sub, origin[2] + redge_sub))
        ds_patch = ds.sel(**subcell_region)

        # Calculate the cell-centered x,y,z coordinates of the subcells
        # Subcell location is specified by the parent cell coordinates (x,y,z)
        # and the offset from the parent cell, (subx, suby, subz).
        dx_sub = s.dx/nsub
        offsets = xr.DataArray(-0.5*s.dx + (np.arange(nsub) + 0.5)*dx_sub, dims='sub')
        xsub = (ds_patch.x - origin[0]) + offsets.rename(sub='subx')
        ysub = (ds_patch.y - origin[1]) + offsets.rename(sub='suby')
        zsub = (ds_patch.z - origin[2]) + offsets.rename(sub='subz')

        # Keep the unstacked geometry for calculating the spherical basis at
        # each subcell position.
        Rsub = np.sqrt(xsub**2 + ysub**2)
        rsub = np.sqrt(Rsub**2 + zsub**2)

        # Spherical basis vectors evaluated at subcell positions. Because
        # nsub is even, no subcell center lies at r=0 or R=0.
        sin_th = Rsub/rsub
        cos_th = zsub/rsub
        sin_ph = ysub/Rsub
        cos_ph = xsub/Rsub
        zero_sub = xr.zeros_like(rsub)
        spherical_basis_sub = (
            (sin_th*cos_ph, sin_th*sin_ph, cos_th),
            (cos_th*cos_ph, cos_th*sin_ph, -sin_th),
            (-sin_ph + zero_sub, cos_ph + zero_sub, zero_sub),
        )

        # Transform selected vectors to spherical components
        ds_sub = {}
        ds_sub['vel1'], ds_sub['vel2'], ds_sub['vel3'] = _spherical_at_subcells(
            (ds_patch.velx, ds_patch.vely, ds_patch.velz),
            spherical_basis_sub
        )
        ds_sub['gacc1'], _, _ = _spherical_at_subcells(
            (ds_patch.gaccx, ds_patch.gaccy, ds_patch.gaccz),
            spherical_basis_sub
        )
        ds_sub['flag_neg_gacc1'] = xr.where(ds_sub['gacc1'] < 0, 1.0, 0.0)
        if s.mhd:
            b_sub = _spherical_at_subcells(
                (ds_patch.bx, ds_patch.by, ds_patch.bz),
                spherical_basis_sub
            )
            ds_sub.update(
                b1=b_sub[0],
                b2=b_sub[1],
                b3=b_sub[2],
            )
        # Drop inherited cell-center r, th, and ph coordinates before using
        # r as the radial-shell dimension.
        ds_sub = xr.Dataset(ds_sub).reset_coords(drop=True)
        ds_sub = ds_sub.stack(
            cell=('z', 'y', 'x'),
            subcell=('subz', 'suby', 'subx')
        ).transpose('subcell', 'cell')
        rsub = rsub.stack(
            cell=('z', 'y', 'x'),
            subcell=('subz', 'suby', 'subx')
        ).transpose('subcell', 'cell')
        # 2. Calculate the fraction of each parent cell assigned to subcells
        # ------------------------------------------------------------------
        ibin = np.floor((rsub + hdx)/s.dx).astype(int)

        # Fraction of each parent cell assigned to the central sphere and the
        # first few nonzero radial shells.
        flag_subcell_in_shell = xr.concat(
            [
                xr.where(ibin == i, 1.0, 0.0)
                for i in range(subcell_region_radius + 1)
            ],
            dim='r',
        ).assign_coords(r=np.arange(subcell_region_radius + 1)*s.dx)
        subcell_frac = flag_subcell_in_shell.mean('subcell')
        dV_sub = s.dV/nsub**3

        shell_volume = (subcell_frac*s.dV).sum('cell').rename('vshell')
        rho_patch_grid = ds_patch.rho.reset_coords(drop=True)
        rho_patch = (
            rho_patch_grid
            .transpose('z', 'y', 'x')
            .stack(cell=('z', 'y', 'x'))
        )
        shell_mass = (rho_patch*subcell_frac*s.dV).sum('cell').rename('mshell')

    # 1. Scalars
    # ----------
    rprofs = {}
    rprofs['rho'], rprofs['vshell'] = _radial_binning(ds.rho)
    rprofs['frac_neg_gacc1'], _ = _radial_binning(
        ds.flag_neg_gacc1,
        qty_sub=ds_sub.flag_neg_gacc1,
    )
    rprofs['phi_mw'], _ = _radial_binning(ds.phi, mass_weighted=True)

    # 2. Vectors
    # ----------
    rprofs['gacc1'], _ = _radial_binning(
        ds.gacc1,
        qty_sub=ds_sub.gacc1,
    )
    rprofs['gacc1_mw'], _ = _radial_binning(
        ds.gacc1,
        mass_weighted=True,
        qty_sub=ds_sub.gacc1,
    )

    # Cartesian components
    # -------------------- remain constant within each parent cell.
    for k in ['velx', 'vely', 'velz']:
        rprofs[k+'_mw'], _ = _radial_binning(
            ds[k],
            mass_weighted=True,
        )
        rprofs[k+'_sq_mw'], _ = _radial_binning(
            ds[k]**2,
            mass_weighted=True,
        )
    # integrand for virial energy term
    rprofs['xgx_mw'], _ = _radial_binning((ds.x - origin[0])*ds.gaccx, mass_weighted=True)
    rprofs['ygy_mw'], _ = _radial_binning((ds.y - origin[1])*ds.gaccy, mass_weighted=True)
    rprofs['zgz_mw'], _ = _radial_binning((ds.z - origin[2])*ds.gaccz, mass_weighted=True)
    # angular momentum
    for k in ['Ldens_x', 'Ldens_y', 'Ldens_z']:
        rprofs[k], _ = _radial_binning(ds[k])
    if s.mhd:
        for k in ['bx', 'by', 'bz']:
            rprofs[k], _ = _radial_binning(ds[k])
            rprofs[k+'_sq'], _ = _radial_binning(ds[k]**2)

    # Spherical components
    # -------------------- use the basis evaluated at each subcell.
    for k in ['vel1', 'vel2', 'vel3']:
        rprofs[k+'_mw'], _ = _radial_binning(
            ds[k],
            mass_weighted=True,
            qty_sub=ds_sub[k],
        )
        rprofs[k+'_sq_mw'], _ = _radial_binning(
            ds[k]**2,
            mass_weighted=True,
            qty_sub=ds_sub[k]**2,
        )
    if s.mhd:
        for k in ['b1', 'b2', 'b3']:
            rprofs[k], _ = _radial_binning(
                ds[k],
                qty_sub=ds_sub[k],
            )
            rprofs[k+'_sq'], _ = _radial_binning(
                ds[k]**2,
                qty_sub=ds_sub[k]**2,
            )
    rprofs = xr.Dataset(rprofs)

    if s.mhd:
        venc = rprofs.vshell.cumsum('r')
        rprofs['mean_bx'] = (rprofs.bx * rprofs.vshell).cumsum('r') / venc
        rprofs['mean_by'] = (rprofs.by * rprofs.vshell).cumsum('r') / venc
        rprofs['mean_bz'] = (rprofs.bz * rprofs.vshell).cumsum('r') / venc
        bmean_norm = np.sqrt(rprofs.mean_bx**2 + rprofs.mean_by**2 + rprofs.mean_bz**2)
        rprofs['bhat_x'] = rprofs.mean_bx / bmean_norm
        rprofs['bhat_y'] = rprofs.mean_by / bmean_norm
        rprofs['bhat_z'] = rprofs.mean_bz / bmean_norm
        # Define B-normal plane at each radius and calculate
        # 1. Two perpendicular unit vectors.
        # 2. Magnetic flux
        bhat_x, bhat_y, bhat_z = dask.compute(
            rprofs.bhat_x, rprofs.bhat_y, rprofs.bhat_z
        )
        bhat_x = bhat_x.data
        bhat_y = bhat_y.data
        bhat_z = bhat_z.data
        bperp1_x, bperp1_y, bperp1_z = [], [], []
        bperp2_x, bperp2_y, bperp2_z = [], [], []
        if compute_flux:
            phi_B = []
        for radius, nx, ny, nz in zip(rprofs.r.values, bhat_x, bhat_y, bhat_z):
            bperp1, bperp2, bhat = _plane_basis((nx, ny, nz))
            bperp1_x.append(bperp1[0])
            bperp1_y.append(bperp1[1])
            bperp1_z.append(bperp1[2])
            bperp2_x.append(bperp2[0])
            bperp2_y.append(bperp2[1])
            bperp2_z.append(bperp2[2])
            if compute_flux:
                phi_B.append(_magnetic_flux(radius, bhat))
        rprofs['bperp1_x'] = xr.DataArray(bperp1_x, dims='r', coords=dict(r=rprofs.r))
        rprofs['bperp1_y'] = xr.DataArray(bperp1_y, dims='r', coords=dict(r=rprofs.r))
        rprofs['bperp1_z'] = xr.DataArray(bperp1_z, dims='r', coords=dict(r=rprofs.r))
        rprofs['bperp2_x'] = xr.DataArray(bperp2_x, dims='r', coords=dict(r=rprofs.r))
        rprofs['bperp2_y'] = xr.DataArray(bperp2_y, dims='r', coords=dict(r=rprofs.r))
        rprofs['bperp2_z'] = xr.DataArray(bperp2_z, dims='r', coords=dict(r=rprofs.r))
        if compute_flux:
            rprofs['phi_B'] = xr.concat(phi_B, dim='r').assign_coords(r=rprofs.r)

    # 3. Tensors
    # ----------
    rhat = {
        'x': xr.where(ds.r > 0, (ds.x - origin[0])/ds.r, 0.0),
        'y': xr.where(ds.r > 0, (ds.y - origin[1])/ds.r, 0.0),
        'z': xr.where(ds.r > 0, (ds.z - origin[2])/ds.r, 0.0)
    }
    mdot_scale = -4*np.pi*rprofs.r**2
    mass_flux = {} # < rho v_i rhat_j > for i,j in {x,y,z}
    for i in 'xyz':
        for j in 'xyz':
            mass_flux[(i, j)], _ = _radial_binning(ds.rho*ds[f'vel{i}']*rhat[j])
        rprofs[f'mdot_{i}'] = mdot_scale*mass_flux[(i, i)]

    # For Cartesian directions, mdot is simply the diagonal components.
    # For MHD, we also want to calculate mdot along the field direction.
    # See May 5, 2026 notes for mathmatical expressions
    if s.mhd:
        for basis, prefix in [
            (dict(x=rprofs.bhat_x, y=rprofs.bhat_y, z=rprofs.bhat_z), 'mdot_b'),
            (dict(x=rprofs.bperp1_x, y=rprofs.bperp1_y, z=rprofs.bperp1_z), 'mdot_bperp1'),
            (dict(x=rprofs.bperp2_x, y=rprofs.bperp2_y, z=rprofs.bperp2_z), 'mdot_bperp2'),
        ]:
            mdot = 0
            for i in ['x', 'y', 'z']:
                for j in ['x', 'y', 'z']:
                    mdot = mdot + basis[i]*basis[j]*mass_flux[(i, j)]
            rprofs[prefix] = mdot_scale*mdot

    # Register velocity at origin
    rprofs['velx_origin'] = vel_origin['x']
    rprofs['vely_origin'] = vel_origin['y']
    rprofs['velz_origin'] = vel_origin['z']

    # Drop theta and phi coordinates
    for k in ['th', 'ph']:
        if k in rprofs:
            rprofs = rprofs.drop_vars(k)

    return rprofs


def radial_profile_projected(s, num, origin):
    """Calculate projected radial profile of column density and velocities

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    num : int
        Snapshot number.
    origin : tuple, list, or numpy.ndarray
        Coordinate origin (x0, y0, z0).


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
    hdx = 0.5*s.dx
    nbin = s.domain['Nx'][0]//2 - 1
    redge = (nbin + 0.5)*s.dx

    def _rprf_incl_center(qty, ax, mass_weighted=False, rms=False):
        """Inner wrapper function for radial binning
        TODO
        ----
        make this function consistent with radial_binning
        used in volumetric radial_profile function.

        mass_weighted=True assumes that rprofs['rho'] is already calculated.
        """
        x1, x2 = xycoordnames[ax]
        x1c, x2c = xycenters[ax]
        ds = prj[ax][qty].copy(deep=True)
        ds, new_center, _ = recenter_dataset(ds, {x1:x1c, x2:x2c})
        ds.coords['R'] = np.sqrt((ds.coords[x1] - new_center[x1])**2
                                 + (ds.coords[x2] - new_center[x2])**2)
        rprf_c = ds.sel({x1:new_center[x1], x2:new_center[x2]}).drop_vars([x1, x2])
        if mass_weighted:
            method, nth = qty.split('mtd')[1].split('_nc')
            w = prj[ax][f'Sigma_gas_mtd{method}_nc{nth}'].copy(deep=True)
            w, _, _ = recenter_dataset(w, {x1:x1c, x2:x2c})
            w.coords['R'] = np.sqrt((w.coords[x1] - new_center[x1])**2
                                    + (w.coords[x2] - new_center[x2])**2)
            if rms:
                rprf = np.sqrt(transform.groupby_bins(ds**2*w, 'R', nbin, (hdx, redge), skipna=True)
                               / transform.groupby_bins(w, 'R', nbin, (hdx, redge), skipna=True))
            else:
                rprf = (transform.groupby_bins(ds*w, 'R', nbin, (hdx, redge), skipna=True)
                        / transform.groupby_bins(w, 'R', nbin, (hdx, redge), skipna=True))
        else:
            if rms:
                rprf = np.sqrt(transform.groupby_bins(ds**2, 'R', nbin, (hdx, redge), skipna=True))
            else:
                rprf = transform.groupby_bins(ds, 'R', nbin, (hdx, redge), skipna=True)
        rprf = xr.concat([rprf_c, rprf], dim='R')
        return rprf

    for i, ax in enumerate(['x', 'y', 'z']):
        # Volume-weighted averages
        for qty in [k for k in prj[ax].keys() if k.startswith('Sigma_gas')]:
            rprofs[f'{ax}_{qty}'] = _rprf_incl_center(qty, ax)

        # Mass-weighted averages
        for qty in [k for k in prj[ax].keys() if k.startswith('vel_mtd')]:
            rprofs[f'{ax}_{qty}_mw'] = _rprf_incl_center(qty, ax, mass_weighted=True)

        for qty in [k for k in prj[ax].keys() if k.startswith('veldisp_mtd')]:
            rprofs[f'{ax}_{qty}'] = _rprf_incl_center(qty, ax)
            rprofs[f'{ax}_{qty}_mw'] = _rprf_incl_center(qty, ax, mass_weighted=True)

        # RMS averages
        for qty in [k for k in prj[ax].keys() if k.startswith('veldisp_mtd')]:
            rprofs[f'{ax}_{qty}_rms'] = _rprf_incl_center(qty, ax, rms=True)
            rprofs[f'{ax}_{qty}_rms_mw'] = _rprf_incl_center(qty, ax, mass_weighted=True, rms=True)

    rprofs = xr.Dataset(rprofs)

    return rprofs


def lagrangian_property(s, cores, rprofs):
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

    if np.isnan(ncrit) or np.isnan(rcore):
        raise ValueError('ncrit and rcore must be defined to calculate Lagrangian properties')
    else:
        rprofs = rprofs.sel(num=common_indices)

        idx_hi = (rprofs.menc >= mcore).argmax('r')
        if np.any(idx_hi == 0):
            raise ValueError('mcore is smaller than the smallest enclosed mass in the radial profile')
        idx_lo = idx_hi - 1

        m_lo = rprofs.menc.isel(r=idx_lo)
        m_hi = rprofs.menc.isel(r=idx_hi)
        r_lo = rprofs.r.isel(r=idx_lo)
        r_hi = rprofs.r.isel(r=idx_hi)
        r_M = r_lo + (r_hi - r_lo) / (m_hi - m_lo) * (mcore - m_lo)


        critical_radius = xr.DataArray(cores.critical_radius.to_numpy(),
                                       dims='t',
                                       coords=dict(t=rprofs.t))
        menc_crit = rprofs.menc.interp(r=critical_radius)
        menc_crit = menc_crit.where(np.isfinite(critical_radius))

        within_r_M = rprofs.r <= r_M
        w = rprofs.r**2*rprofs.rho
        denom = w.where(within_r_M).sum('r')
        def mw_mean(da):
            num = (da.where(within_r_M)*w).sum('r')
            return num / denom

        # Mass-weighted infall speed
        vinfall = mw_mean(rprofs.vel1_mw)

        # Mass-weighted velocity dispersion
        sigma_mw = np.sqrt(mw_mean(rprofs.dvel1_sq_mw))

        vx_com = mw_mean(rprofs.velx_mw)
        vy_com = mw_mean(rprofs.vely_mw)
        vz_com = mw_mean(rprofs.velz_mw)
        vcom = np.sqrt(vx_com**2 + vy_com**2 + vz_com**2)

        sigma_1d = np.sqrt((mw_mean(rprofs.velx_sq_mw)
                            + mw_mean(rprofs.vely_sq_mw)
                            + mw_mean(rprofs.velz_sq_mw)
                            - vx_com**2 - vy_com**2 - vz_com**2) / 3)

        sigma_1d_trb = np.sqrt((mw_mean(rprofs.dvel1_sq_mw)
                                + mw_mean(rprofs.dvel2_sq_mw)
                                + mw_mean(rprofs.dvel3_sq_mw)) / 3)

        sigma_1d_blk = np.sqrt((mw_mean(rprofs.vel1_mw**2)
                                + mw_mean(rprofs.vel2_mw**2)
                                + mw_mean(rprofs.vel3_mw**2)) / 3)

        rprf = rprofs.interp(r=r_M)
        rhoavg = mcore / (4*np.pi*r_M**3/3)

        lprops = pd.DataFrame(data = dict(radius=r_M.to_numpy(),
                                          menc_crit=menc_crit.to_numpy(),
                                          edge_density=rprf.rho.to_numpy(),
                                          mean_density=rhoavg.to_numpy(),
                                          vinfall=vinfall.to_numpy(),
                                          vcom=vcom.to_numpy(),
                                          sigma_mw=sigma_mw.to_numpy(),
                                          sigma_1d=sigma_1d.to_numpy(),
                                          sigma_1d_trb=sigma_1d_trb.to_numpy(),
                                          sigma_1d_blk=sigma_1d_blk.to_numpy(),
                                          Fthm=rprf.Fthm.to_numpy(),
                                          Ftrb=rprf.Ftrb.to_numpy(),
                                          Fcen=rprf.Fcen.to_numpy(),
                                          Fani=rprf.Fani.to_numpy(),
                                          Fgrv=rprf.Fgrv.to_numpy()),
                              index = cores.index)
        if s.mhd:
            lprops['mcrit_mag'] = rprf.mcrit_mag.to_numpy()
            lprops['Fmag'] = rprf.Fmag.to_numpy()

    # Attach some attributes
    # Velocity dispersion at t_crit
    lprp = lprops.loc[ncrit]
    lprops.attrs['vcom'] = lprp.vcom
    lprops.attrs['sigma_r'] = lprp.sigma_mw
    lprops.attrs['sigma_1d'] = lprp.sigma_1d
    lprops.attrs['sigma_1d_trb'] = lprp.sigma_1d_trb

    # Free-fall time at t_coll
    lprops.attrs['tff_coll'] = tfreefall(lprops.loc[ncoll].mean_density, s.gconst)

    return lprops


def cumulative_energy(s, rprf, core):
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


def infall_rate(rprofs, cores):
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


def radial_acceleration(s, rprf):
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
        t_rr = 0.5*(rprf.b1_sq - rprf.b2_sq - rprf.b3_sq)
        acc['mag'] = (
            t_rr.differentiate('r')
            + ((2*rprf.b1_sq - rprf.b2_sq - rprf.b3_sq)
               / rprf.r).where(rprf.r > 0, other=0)
        ) / rprf.rho
    else:
        acc['mag'] = rprf.rho*0

    acc = xr.Dataset(acc)
    acc['dvdt_lagrange'] = (acc.thm + acc.trb + acc.mag + acc.grv
                            + acc.cen + acc.ani)
    acc['dvdt_euler'] = acc.dvdt_lagrange - acc.adv

    acc['Fadv'] = load_sim.rprof_cumsum_r(rprf, rprf.rho*acc.adv)
    acc['Fthm'] = load_sim.rprof_cumsum_r(rprf, rprf.rho*acc.thm)
    acc['Ftrb'] = load_sim.rprof_cumsum_r(rprf, rprf.rho*acc.trb)
    acc['Fmag'] = load_sim.rprof_cumsum_r(rprf, rprf.rho*acc.mag)
    acc['Fcen'] = load_sim.rprof_cumsum_r(rprf, rprf.rho*acc.cen)
    acc['Fgrv'] = -load_sim.rprof_cumsum_r(rprf, rprf.rho*acc.grv)
    acc['Fani'] = load_sim.rprof_cumsum_r(rprf, rprf.rho*acc.ani)


    # Net forces
    acc['fnet'] = (acc.thm + acc.trb + acc.cen + acc.ani
                   + acc.mag + acc.grv) / (-acc.grv)
    acc['Fnet'] = (acc.Fthm + acc.Ftrb + acc.Fcen + acc.Fani
                   + acc.Fmag - acc.Fgrv) / acc.Fgrv
    return acc


def observable(s, core, rprf):
    """Calculate observable properties of a core"""
    nthr_list = [10, 30, 100]
    num = core.name
    obsprops = dict()
    obsprops['num'] = num
    prj = s.read_prj(num)
    xc, yc, zc = s.flatindex_to_cartesian(core.leaf_id)
    xycoordnames = dict(z=['x', 'y'],
                        x=['y', 'z'],
                        y=['z', 'x'])
    xycenters = dict(z=[xc, yc],
                     x=[yc, zc],
                     y=[zc, xc])

    # Read 3d data cube
    dens_3d = s.load_hdf5(num, quantities=['dens']).dens
    dens_3d, new_center_3d, _ = recenter_dataset(dens_3d, dict(x=xc, y=yc, z=zc))
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

    def threshold(dens, ncrit, method):
        if method=='tophat':
            return dens.where((dens >= ncrit)&(dens < 10*ncrit), other=0)
        elif method=='step':
            return dens.where(dens >= ncrit, other=0)
        else:
            raise ValueError(f'Unknown method {method}')

    # Observable properties using density thresholding.
    # Analogous to molecular line observations.
    for threshold_method in ['tophat', 'step']:
        for nthr in nthr_list:
            d3dthr = threshold(dens_3d, nthr, threshold_method)
            for ax in ['x', 'y', 'z']:
                pos_radius = dict(fwhm_dust=obsprops[f'{ax}_radius'])
    
                x1, x2 = xycoordnames[ax]
                x1c, x2c = xycenters[ax]
    
                # POS FWHM radius
                dcol_prf = rprf[f'{ax}_Sigma_gas_mtd{threshold_method}_nc{nthr}']
                try:
                    pos_radius['fwhm'] = obs_core_radius(dcol_prf, method='fwhm')
                except:
                    pos_radius['fwhm'] = np.nan

                for rpos_pc in [0.02, 0.05, 0.1]:
                    pos_radius[f'fixed{rpos_pc}'] = (rpos_pc*au.pc / s.u.length).cgs.value

                # Set up 2D maps
                dcol_map = prj[ax][f'Sigma_gas_mtd{threshold_method}_nc{nthr}'].copy(deep=True)
                dv_map = prj[ax][f'veldisp_mtd{threshold_method}_nc{nthr}'].copy(deep=True)
                dcol_map, _, _ = recenter_dataset(dcol_map, {x1: x1c, x2: x2c})
                dv_map, new_center, _ = recenter_dataset(dv_map, {x1: x1c, x2: x2c})
                rpos = np.sqrt((dv_map.coords[x1] - new_center[x1])**2
                               + (dv_map.coords[x2] - new_center[x2])**2)

                # POS radius at which any pixel falls below dcol_bgr
                dcol_c = dcol_prf.isel(R=0).data[()]
                bgr_level = dict(mean=dcol_map.mean(),
                                 c01=dcol_c*0.1,
                                 c02=dcol_c*0.2)
                for k, v in bgr_level.items():
                    try:
                        pos_radius[f'bgr_{k}'] = rpos.where(dcol_map < v).min().data[()]
                    except:
                        pos_radius[f'bgr_{k}'] = np.nan

    
                # Loop over different plane-of-sky radius definitions
                for method, rcore_pos in pos_radius.items():
                    obsprops[f'{ax}_pos_radius_{method}_mtd{threshold_method}_nc{nthr}'] = rcore_pos
                    if np.isfinite(rcore_pos):
                        obsprops[f'{ax}_velocity_dispersion_{method}_mtd{threshold_method}_nc{nthr}']\
                                = np.sqrt((dv_map.where(rpos < rcore_pos)**2
                                          ).weighted(dcol_map).mean()).data[()]
                        obsprops[f'{ax}_mean_column_density_{method}_mtd{threshold_method}_nc{nthr}']\
                                = dcol_prf.sel(R=slice(0, rcore_pos)
                                               ).weighted(dcol_prf.R).mean().data[()]
    
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
                            # Note that we are not using rms average (why not?).
                            rlos_true = np.average(arr[msk], weights=wgt[msk])
                        except ZeroDivisionError:
                            rlos_true = np.nan
                        obsprops[f'{ax}_los_radius_{method}_mtd{threshold_method}_nc{nthr}'] = rlos_true
                    else:
                        obsprops[f'{ax}_velocity_dispersion_{method}_mtd{threshold_method}_nc{nthr}'] = np.nan
                        obsprops[f'{ax}_los_radius_{method}_mtd{threshold_method}_nc{nthr}'] = np.nan
                        obsprops[f'{ax}_mean_column_density_{method}_mtd{threshold_method}_nc{nthr}'] = np.nan
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


def critical_time_old(s, cores, rprofs, *, method):
    """
    Return
    ------
    ncrit : float
        Critical time in terms of snapshot number.
    rcrit : float
        Critical radius at ncrit.
    """
    if len(cores) == 0:
        return np.nan
    cores = cores.loc[:cores.attrs['numcoll']]
    pid = cores.attrs['pid']

    ncrit = None
    rcrit = None

    if method == 'empirical':
        # Earliest time after which the net force integrated within r_crit
        # remains negative until the end of the collapse.
        # To find this "empirical critical time", we start from t_coll
        # and march backward in time.
        num_buffer = 2
        if np.all(cores.iloc[-(num_buffer+1):].pindex < 0):
            s.logger.warning(
                "pindex in the last three snapshots is negative."
               f" for pid = {pid}."
                " cannot calculate critical radius and thus the critical time"
            )
            ncrit = np.nan
            rcrit = np.nan
        else:
            for num, core in cores.sort_index(ascending=False).iterrows():
                # Exclude t_coll snapshot at which the turbulence has amplified
                # to produce nagative linewidth-size slope.
                if num in cores.index[-num_buffer:]:
                    if np.isnan(core.critical_radius):
                        n2coll = cores.attrs['numcoll'] - num
                        msg = (f"Critical radius at t_coll - {n2coll}"
                               f" is NaN for par {pid}, method {method}."
                               " This may have been caused by negative pindex."
                               " Continuing...")
                        s.logger.warning(msg)
                        continue
                    if np.isinf(core.critical_radius):
                        n2coll = cores.attrs['numcoll'] - num
                        msg = (f"Critical radius at t_coll - {n2coll}"
                               f" is inf for par {pid}, method {method}."
                               " This may have been caused by very small rsonic"
                               " Continuing...")
                        s.logger.warning(msg)
                        continue
                rprf = rprofs.sel(num=num)
                # Net force at the critical radius is negative after the
                # critical time, throughout the collapse.
                if np.isfinite(core.critical_radius):
                    rprf = rprf.interp(r=core.critical_radius)
                    fnet = (rprf.Fthm + rprf.Ftrb + rprf.Fcen + rprf.Fani
                            - rprf.Fgrv)
                    if s.mhd:
                        fnet += rprf.Fmag
                    fnet = fnet.data[()]
                else:
                    fnet = np.nan
                # Whatever fnet is, if it is not negative, we should break.
                # That is, when rcrit = NaN or inf, we should break.
                # However, NaN can be artificial, we can probably impose
                # the upper limit on p.
                if not fnet < 0:
                    ncrit = num + 1
                    if ncrit == cores.index[-1] + 1:
                        ncrit = np.nan
                        rcrit = np.nan
                    else:
                        rcrit = cores.loc[ncrit].critical_radius
                        if not np.isfinite(rcrit):
                            msg = (f"Critical radius at ncrit = {ncrit} is not "
                                   f"finite for par {pid}: "
                                   f"method={method}, rcrit={cores.loc[ncrit].critical_radius}.")
                            s.logger.warning(msg)
                            ncrit = np.nan
                    break
        if ncrit == cores.attrs['numcoll'] and np.isnan(cores.loc[ncrit].critical_radius):
            # If ncrit is ncoll at which critical radius was nan, set ncrit to NaN.
            ncrit = np.nan
    elif method == 'virial_rcrit':
        for num, core in cores.sort_index(ascending=False).iterrows():
            rprf = rprofs.sel(num=num)
            # Net force at the critical radius is negative after the
            # critical time, throughout the collapse.
            if np.isnan(core.virial_rcrit):
                raise Exception(f"{s.basename}: virial_rcrit is NaN at num = {num} for pid = {pid}. Cannot calculate net force at r_crit.")
            rprf = rprf.interp(r=core.virial_rcrit)
            # Whatever fnet is, if it is not negative, we should break.
            # That is, when rcrit = NaN or inf, we should break.
            # However, NaN can be artificial, we can probably impose
            # the upper limit on p.
            if core.virial_rcrit <= 3*s.dx:
                fnet_std = 0
            else:
                fnet_std = rprofs.fnet.sel(num=num, r=slice(3*s.dx, core.virial_rcrit)).std().data[()]
            if rprf.Fnet > 0 or fnet_std > 0.3:
                ncrit = num + 1
                if ncrit == cores.index[-1] + 1:
                    s.logger.warning(f"{s.basename}: Net force is positive at t_coll! pid = {pid}")
                    rcrit = np.nan
                else:
                    rcrit = cores.loc[ncrit].virial_rcrit
                break
    elif method == 'quadrant':
        raise ValueError("The 'quadrant' method is no longer supported. Please use 'empirical' method instead.")

    if ncrit is None or ncrit == cores.index[-1] + 1:
        # If the critical condition is satisfied for all time, or is not
        # satisfied at t_coll, set ncrit to NaN.
        # TODO: we may not want to discard those that the critical condition
        # is satisfied for all times.
        ncrit = np.nan
        rcrit = np.nan
    return ncrit, rcrit

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

def periodic_distance1d(x1, x2, w):
    """Returns periodic distance between two coordinates.

    Parameters
    ----------
    x1 : array_like
        (array of) First coordinate
    x2 : array_like
        (array of) Second coordinate
    w : scalar
        The period.
    """
    assert len(x1) == len(x2)
    hw = 0.5*w
    pdst = np.abs(periodic_operator(x1 - x2, -hw, hw))
    return pdst

def periodic_distance(x1, x2, w, return_axis_distance=False):
    """Returns periodic distance between two coordinates.

    Parameters
    ----------
    x1 : array_like
        Position of the first point.
    x2 : array_like
        Position of the second point.
    w : scalar or array_like
        The array of the period.
    return_axis_distance : bool, optional
        If True, return the distance along each axis.
    """
    assert len(x1) == len(x2)
    ndim = len(x1)
    if np.isscalar(w):
        w = np.ones(ndim)*w
    else:
        assert len(w) == ndim
        w = np.array(w)

    axis_distance = []
    for x1_, x2_, w_ in zip(x1, x2, w):
        x1v = np.atleast_1d(x1_)
        x2v = np.atleast_1d(x2_)
        axis_distance.append(periodic_distance1d(x1v, x2v, w_))
    axis_distance = np.array(axis_distance)
    pdst = np.sqrt((axis_distance**2).sum(axis=0))
    if return_axis_distance:
        return axis_distance
    else:
        return pdst.squeeze()

def periodic_operator(x, a, b):
    """The periodic operator.

    Parameters
    ----------
    x : array_like
        Input array.
    a : float
        Lower bound.
    b : float
        Upper bound.

    Returns
    -------
    array_like
        The periodic operator applied to x.

    Reference
    ---------
    https://tommohr.dev/pbc/
    """
    w = b - a
    n = np.floor((x - a) / w)
    return x - n*w

def get_sonic(Mach_outer, l_outer, p=0.5):
    """returns sonic scale assuming linewidth-size relation v ~ R^p
    """
    if Mach_outer == 0:
        return np.inf
    lambda_s = l_outer*Mach_outer**(-1/p)
    return lambda_s


def recenter_dataset(ds, center, by_index=False):
    """Recenter whole dataset or dataarray.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset to be recentered.
    center : dict
        {x:xc, y:yc} or {x:xc, y:yc, z:zc}, etc.

    Returns
    -------
    ds_recentered : xarray.Dataset or xarray.DataArray
        Recentered dataset.
    new_center : tuple
        Position of the new center. This must be the grid coordinates
        closest, but not exactly the same, to (0, 0, 0).
    shift : dict
        Shifts in each dimension.
    """
    shift, new_center = {}, {}
    for dim, pos in center.items():
        hNx = ds.sizes[dim] // 2
        coords = ds.coords[dim].data
        dx = coords[1] - coords[0]
        if by_index:
            shift[dim] = hNx - pos
        else:
            shift[dim] = hNx - np.where(np.isclose(coords, pos, atol=0.1*dx))[0][0]
        new_center[dim] = ds.coords[dim].isel({dim: hNx}).data[()]
    ds_recentered = ds.roll(shift)

    return ds_recentered, new_center, shift


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
    rcore = cores.attrs['rcore']
    if np.isnan(rcore):
        return False
    ncells = rcore / s.dx
    if ncells >= nres:
        return True
    else:
        return False


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
                  label=r'$\dfrac{t - t_\mathrm{crit}}{\Delta t_\mathrm{coll}}$',
                  location='right'):
    """Get an appropriate color bar for get_evol_norm"""
    import matplotlib.pyplot as plt
    cbar = plt.colorbar(mappable, ax=ax, cax=cax, label=label, location=location)
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


def dask_init(ncores=96, memory='740 GiB', nprocs=32, scale=1, wtime='00:30:00'):
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    cluster = SLURMCluster(cores=ncores, memory=memory, processes=nprocs,
                           interface="ib0", walltime=wtime)
    cluster.scale(scale)
    client = Client(cluster)
    return client


def find_closest_leaf(s, gd, flatidx):
    """Find the closest leaf to the given position

    Parameters
    ----------
    s : LoadSim
        LoadSim instance.
    gd : grid_dendro.Dendrogram
        Dendrogram object.
    flatidx : int
        Flat index of the cell.

    Returns
    -------
    lid : int
        Flat index of the closest leaf node.
    """
    dst = [s.distance_between(lid, flatidx) for lid in gd.leaves]
    lid = gd.leaves[np.argmin(dst)]
    return lid
