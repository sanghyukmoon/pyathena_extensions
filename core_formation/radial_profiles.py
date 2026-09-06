"""Derived radial-profile quantities and their numerical helpers."""

import warnings

import numpy as np
import xarray as xr


def derive_radial_profiles(s, rprofs):
    """Return one core's profiles with derived fields recomputed.

    Parameters
    ----------
    s : LoadSim
        Simulation providing only cs, gconst, and mhd.
    rprofs : xarray.Dataset
        Assembled profiles with time and radius dimensions and required raw
        fields. Existing derived fields are recomputed for the same MHD mode.

    Returns
    -------
    xarray.Dataset
        A copy with derived fields and dimension order (t, r, ...).
        Neither the input dataset nor the simulation is modified.
    """
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
    rprofs['Lx_enc'] = rprof_cumsum_r(rprofs, rprofs.Ldens_x)
    rprofs['Ly_enc'] = rprof_cumsum_r(rprofs, rprofs.Ldens_y)
    rprofs['Lz_enc'] = rprof_cumsum_r(rprofs, rprofs.Ldens_z)
    Lnorm = np.sqrt(rprofs.Lx_enc**2 + rprofs.Ly_enc**2 + rprofs.Lz_enc**2)
    rprofs['lhat_x'] = rprofs.Lx_enc / Lnorm
    rprofs['lhat_y'] = rprofs.Ly_enc / Lnorm
    rprofs['lhat_z'] = rprofs.Lz_enc / Lnorm
    if s.mhd:
        rprofs['costh_BL'] = (
            rprofs.bhat_x*rprofs.lhat_x
            + rprofs.bhat_y*rprofs.lhat_y
            + rprofs.bhat_z*rprofs.lhat_z
        )

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
        -s.gconst*(
            menc_inner_edge*r_inv_avg
            + 4*np.pi*rprofs.rho/3*r2_avg
        ),
        0
    )
    rprofs['Omega_G_sph'] = -rprof_cumsum_r(rprofs, rprofs.rho*rdotg_sph)
    rprofs['Omega_G0'] = xr.where(
        rprofs.r > 0,
        s.gconst*(
            menc_inner_edge**2*r_inv_avg
            + 8*np.pi*rprofs.rho/3*menc_inner_edge*r2_avg
            + (4*np.pi*rprofs.rho/3)**2*r5_avg
        ),
        0
    )

    rprofs['Omega_K_thm'] = rprof_cumsum_r(rprofs, 3*s.cs**2*rprofs.rho)

    vsq = (rprofs.vel1_sq_mw + rprofs.vel2_sq_mw + rprofs.vel3_sq_mw)
    rprofs['Omega_K_kin'] = rprof_cumsum_r(rprofs, rprofs.rho*vsq)

    rprofs['Omega_K'] = rprofs['Omega_K_thm'] + rprofs['Omega_K_kin']
    rprofs['Omega_S_thm'] = 4*np.pi*rprofs.r**3*s.cs**2*rprofs.rho
    rprofs['Omega_S_kin'] = 4*np.pi*rprofs.r**3*rprofs.rho*rprofs.vel1_sq_mw
    rprofs['Omega_S'] = rprofs['Omega_S_thm'] + rprofs['Omega_S_kin']
    rprofs['alpha_vir'] = rprofs['Omega_K'] / rprofs['Omega_G']
    if s.mhd:
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
    rprofs['ptot'] = rprofs.rho*(s.cs**2 + rprofs.vel1_sq_mw)
    rprofs['peq'] = (rprofs.Omega_K + rprofs.Omega_M - rprofs.Omega_S_mag - rprofs.Omega_G) / (4*np.pi*rprofs.r**3)

    # Maximum pressure from McCrea analysis
    rgrav = s.gconst*rprofs.menc/s.cs**2
    pgrav = s.cs**8/(4*np.pi*s.gconst**3*rprofs.menc**2)
    sigma_1d_sq = rprofs.Omega_K_kin/(3*rprofs.menc)
    agrv = rprofs.Omega_G/((3/5)*rprofs.Omega_G0)
    rprofs['a_grv'] = agrv
    if s.mhd:
        flux = np.sqrt(4*np.pi)*rprofs.phi_B
        bmag = (rprofs.Omega_M - rprofs.Omega_S_mag) / (flux**2/(6*np.pi**2*rprofs.r))
        rprofs['b_mag'] = bmag
        cphi2 = 5*bmag / (18*np.pi**2*agrv)
        rprofs['c_phi'] = np.sqrt(cphi2.where(cphi2 >= 0))
    else:
        flux = xr.zeros_like(rprofs.rho)
        cphi2 = xr.zeros_like(agrv)

    param_dict = {
        'all': {
            'sigma_tot2': s.cs**2 + sigma_1d_sq,
            'c_phi2': cphi2,
        },
        'thm': {
            'sigma_tot2': s.cs**2,
            'c_phi2': xr.zeros_like(cphi2),
        },
        'trb': {
            'sigma_tot2': sigma_1d_sq,
            'c_phi2': xr.zeros_like(cphi2),
        },
        'mag': {
            'sigma_tot2': 0,
            'c_phi2': cphi2,
        },
        'thm_trb': {
            'sigma_tot2': s.cs**2 + sigma_1d_sq,
            'c_phi2': xr.zeros_like(cphi2),
        },
        'thm_mag': {
            'sigma_tot2': s.cs**2,
            'c_phi2': cphi2,
        },
        'trb_mag': {
            'sigma_tot2': sigma_1d_sq,
            'c_phi2': cphi2,
        },
    }
    for key in param_dict.copy().keys():
        param_dict[key]['a_grv'] = agrv
        param_dict[f'{key}0'] = param_dict[key].copy()
        param_dict[f'{key}0']['a_grv'] = 1.17*xr.ones_like(agrv)
        param_dict[f'{key}0']['c_phi2'] = (0.17**2)*xr.ones_like(cphi2)
    for key, params in param_dict.items():
        agrv = params['a_grv']
        c_J = 3**4 * 5**3 / (2**10 * np.pi * agrv.where(agrv > 0)**3)
        mmag2 = params['c_phi2']/s.gconst*flux**2 if s.mhd else xr.zeros_like(agrv)
        sigma2 = params['sigma_tot2']
        rprofs[f"pmax_{key}"] = (
            c_J * sigma2**4 / (s.gconst**3*rprofs.menc**2*(1 - mmag2/rprofs.menc**2)**3)
        ).where(rprofs.menc**2 > mmag2, other=np.nan)
        # Cubic coefficients for x^3 + ax^2 + bx + c = 0
        a = -3*mmag2 - c_J * sigma2**4 / (s.gconst**3 * rprofs.ptot)
        b = 3*mmag2**2
        c = -mmag2**3
        x = cubic_root(a, b, c)
        rprofs[f"mmax_{key}"] = np.sqrt(x.where(x >= 0))

        # fixed sonic radius
        sigma_trb2 = sigma2 - s.cs**2
        xi = 32*agrv/45*sigma_trb2*s.gconst*rprofs.menc/(s.cs**4*rprofs.r)
        eta = 0.5 + 0.5*(1 + xi)**1.5 + 3*xi/4 + 3*xi**2/16
        rprofs[f"pmax_{key}_fs"] = (
            c_J * s.cs**8*eta / (s.gconst**3*rprofs.menc**2*(1 - mmag2/rprofs.menc**2)**3)
        ).where(rprofs.menc**2 > mmag2, other=np.nan)
        a = -3*mmag2 - c_J*s.cs**8*eta / (s.gconst**3 * rprofs.ptot)
        b = 3*mmag2**2
        c = -mmag2**3
        x = cubic_root(a, b, c)
        rprofs[f"mmax_{key}_fs"] = np.sqrt(x.where(x >= 0))


    rhoavg = rprofs.menc / (4*np.pi*rprofs.r**3/3)
    mgrav = s.cs**3/s.gconst**1.5/np.sqrt(rhoavg)
    mbe = 1.86*mgrav
    rprofs['mBE'] = mbe
    rprofs['mTES'] = mbe*(1 + sigma_1d_sq/(2*s.cs**2))
    rprofs['mPhi'] = 0.17/np.sqrt(s.gconst)*flux

    rprofs['adv'] = (
        rprofs.vel1_mw*rprofs.vel1_mw.differentiate('r')
    )
    pthm = rprofs.rho*s.cs**2
    ptrb = rprofs.rho*rprofs.dvel1_sq_mw
    rprofs['thm'] = -pthm.differentiate('r') / rprofs.rho
    rprofs['trb'] = -ptrb.differentiate('r') / rprofs.rho
    rprofs['cen'] = (
        (rprofs.vel2_mw**2 + rprofs.vel3_mw**2) / rprofs.r
    ).where(rprofs.r > 0, other=0)
    rprofs['grv'] = rprofs.gacc1_mw
    rprofs['ani'] = (
        (rprofs.dvel2_sq_mw + rprofs.dvel3_sq_mw
         - 2*rprofs.dvel1_sq_mw) / rprofs.r
    ).where(rprofs.r > 0, other=0)

    if s.mhd:
        rprofs['mag'] = (
            t_rr.differentiate('r')
            + ((2*rprofs.b1_sq - rprofs.b2_sq - rprofs.b3_sq)
               / rprofs.r).where(rprofs.r > 0, other=0)
        ) / rprofs.rho
    else:
        rprofs['mag'] = rprofs.rho*0

    rprofs['dvdt_lagrange'] = (
        rprofs.thm + rprofs.trb + rprofs.mag + rprofs.grv
        + rprofs.cen + rprofs.ani
    )
    rprofs['dvdt_euler'] = rprofs.dvdt_lagrange - rprofs.adv

    rprofs['Fadv'] = rprof_cumsum_r(rprofs, rprofs.rho*rprofs.adv)
    rprofs['Fthm'] = rprof_cumsum_r(rprofs, rprofs.rho*rprofs.thm)
    rprofs['Ftrb'] = rprof_cumsum_r(rprofs, rprofs.rho*rprofs.trb)
    rprofs['Fmag'] = rprof_cumsum_r(rprofs, rprofs.rho*rprofs.mag)
    rprofs['Fcen'] = rprof_cumsum_r(rprofs, rprofs.rho*rprofs.cen)
    rprofs['Fgrv'] = -rprof_cumsum_r(rprofs, rprofs.rho*rprofs.grv)
    rprofs['Fani'] = rprof_cumsum_r(rprofs, rprofs.rho*rprofs.ani)

    rprofs['fnet'] = (
        (rprofs.thm + rprofs.trb + rprofs.cen + rprofs.ani
         + rprofs.mag + rprofs.grv) / (-rprofs.grv)
    ).where(rprofs.r > 0, other=0)

    rprofs['Fnet'] = (
        (rprofs.Fthm + rprofs.Ftrb + rprofs.Fcen + rprofs.Fani
        + rprofs.Fmag - rprofs.Fgrv) / rprofs.Fgrv
    ).where(rprofs.r > 0, other=0)

    return rprofs.transpose('t', 'r', ...)


def rprof_cumsum_r(rprofs, rprf_var):
#    res = (4*np.pi*rprofs.r**2*rprf_var).cumulative_integrate('r')
    res = (rprf_var*rprofs.vshell_full).cumsum('r')
    # correct for outer half of the shell
    res -= rprf_var*rprofs.vshell_half
    return res


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
