"""Collection of plotting scripts

Recommended function signature:
    def plot_something(s, ds, ax=None)
"""

# python modules
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.markers import MarkerStyle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patheffects as pe
import numpy as np
import xarray as xr
# Bottleneck does not use stable sum.
# See xarray #1346, #7344 and bottleneck #193, #462 and more.
# Let's disable third party softwares to go conservative.
# Accuracy is more important than performance.
xr.set_options(use_bottleneck=False, use_numbagg=False)
from tesphere import tes
from grid_dendro import energy

from . import tools, models, load_sim


class SpaceTimePlotter():
    """Helper class to create space-time plots with a consistent layout and formatting.
    """
    def plot_spacetime(self, s, pid, *, size='compact', mtd_crit='virial'):
        s.select_cores(mtd_crit)
        cores = s.cores[pid]
        rprofs = s.rprofs[pid].transpose('t', 'r', ...)

        layout = self.create_layout(size)
        axs = layout['axs_dict']

        # min-max for mass ratio plots
        vmin, vmax = 0, 2

        # Net force (cumulative)
        self.imshow(
            rprofs.Fnet,
            axs['Fnet'],
            vmin=-1,
            vmax=1,
            cmap='RdBu_r',
            label=r'$F_\mathrm{net}/F_\mathrm{grv}$'
        )
        # Net force
        self.imshow(
            rprofs.fnet,
            axs['fnet'],
            vmin=-1,
            vmax=1,
            cmap='RdBu_r',
            label=r'$f_\mathrm{net}/f_\mathrm{grv}$'
        )
        # Mass-to-critical mass ratio
        self.imshow(
            rprofs.menc / rprofs.mmax,
            axs['mass_ratio'],
            vmin=vmin,
            vmax=vmax,
            cmap='cmr.watermelon',
        )
        # Critical ratio
        self.imshow(
            rprofs.fnet,
            axs['mass_ratio_over_fnet'],
            vmin=-1,
            vmax=1,
            cmap='RdBu_r',
            label=r'$f_\mathrm{net}/f_\mathrm{grv}$'
        )
        if size in ['medium', 'full']:
            self.imshow(
                rprofs.menc / rprofs.mmax_fixed_a,
                axs['mass_ratio_fixed_a'],
                vmin=vmin,
                vmax=vmax,
                cmap='cmr.watermelon',
                label=r'$M_\mathrm{enc}/M_{\mathrm{max},a=1}$'
            )

            self.imshow(
                rprofs.menc / rprofs.mmax_thm,
                axs['mass_ratio_thm'],
                vmin=vmin,
                vmax=vmax,
                cmap='cmr.watermelon',
                label=r'$M_\mathrm{enc}/M_{\mathrm{max,thm}}$'
            )

            self.imshow(
                rprofs.menc / rprofs.mmax_thm_trb,
                axs['mass_ratio_thm_trb'],
                vmin=vmin,
                vmax=vmax,
                cmap='cmr.watermelon',
                label=r'$M_\mathrm{enc}/M_{\mathrm{max,thm+trb}}$'
            )

            eta_grav = rprofs.Omega_G / rprofs.Omega_G_sph
            self.imshow(
                eta_grav,
                axs['eta_grav'],
                vmin=0,
                vmax=1,
                cmap='cmr.savanna',
                label=r'$\eta_\mathrm{grav}$'
            )
            rtidal = xr.where(eta_grav < 0.8, eta_grav.r, np.nan).min('r')
            axs['eta_grav'].plot(rtidal, cores.time, color='tab:green', ls='-')

            self.imshow(rprofs.vel1_mw/s.cs, axs['vin'],
                        vmin=-1, vmax=1, cmap='RdBu_r',
                        label=r'$\left<v_r\right>_\rho/c_s$')
        if size == 'full':
            self.imshow(
                rprofs.menc / rprofs.mmax_thm_fixed_a,
                axs['mass_ratio_thm_fixed_a'],
                vmin=vmin,
                vmax=vmax,
                cmap='cmr.watermelon',
                label=r'$M_\mathrm{enc}/M_{\mathrm{max,thm},a=1}$'
            )

            self.imshow(
                rprofs.menc / rprofs.mmax_thm_trb_fixed_a,
                axs['mass_ratio_thm_trb_fixed_a'],
                vmin=vmin,
                vmax=vmax,
                cmap='cmr.watermelon',
                label=r'$M_\mathrm{enc}/M_{\mathrm{max,thm+trb},a=1}$'
            )

            self.imshow(
                rprofs.Fnet,
                axs['mass_ratio_over_Fnet'],
                vmin=-1,
                vmax=1,
                cmap='RdBu_r',
                label=r'$F_\mathrm{net}/F_\mathrm{grv}$'
            )

            if s.mhd:
                # Magnetic Surface to volume
                self.imshow(
                    (rprofs.Omega_M - rprofs.Omega_S_mag) / rprofs.Omega_M,
                    axs['net_magnetic_energy_normalized'],
                    vmin=-1,
                    vmax=1,
                    cmap='cmr.watermelon',
                    label=r'$(\mathcal{B} - \mathcal{B}_0)/\mathcal{B}$'
                )

                # Gravity to magnetic
                self.imshow(
                    np.sqrt(
                        rprofs.Omega_G
                        / (rprofs.Omega_M - rprofs.Omega_S_mag)
                    ),
                    axs['grav_to_mag_energy'],
                    vmin=0,
                    vmax=2,
                    cmap='cmr.watermelon',
                    label=r'$\mu_\Phi \equiv (\mathcal{W}/\mathcal{B})^{1/2}$'
                          r'$ = M/M_\Phi$'
                )

                # Mass-to-flux ratio
                brms = np.sqrt(2*rprofs.Omega_M/(4*np.pi*rprofs.r**3/3))
                bflux = np.pi*rprofs.r**2*brms
                mucrit = 1/(2*np.pi*np.sqrt(s.gconst))
                self.imshow(
                    (rprofs.menc / (bflux * np.sqrt(4*np.pi))) / mucrit,
                    axs['mass_to_flux'],
                    vmin=0,
                    vmax=2,
                    cmap='cmr.watermelon',
                    label=r'$\mu_{\Phi,\mathrm{obs}}$'
                          r'$ \equiv 2\pi \sqrt{G}(M/\Phi)$'
                )

                # B field strength
                self.imshow(
                    brms*s.u.muG,
                    axs['Brms'],
                    norm=LogNorm(8e0, 2e2),
                    cmap='viridis',
                    label=r'$B_\mathrm{rms}\,[\mu\mathrm{G}]$'
                )
            else:
                axs['net_magnetic_energy_normalized'].remove()
                axs['grav_to_mag_energy'].remove()
                axs['mass_to_flux'].remove()
                axs['Brms'].remove()


        # Annotate const-mass lines
        for ax in [axs['mass_ratio_over_fnet'], axs['mass_ratio_over_Fnet']]:
            if len(rprofs.t) > 1:
                cs = rprofs.menc.plot.contour(
                    ax=ax,
                    levels=[0.5e-2, 1e-2, 2e-2, 4e-2, 8e-2, 16e-2, 32e-2, 64e-2],
                    add_labels=False,
                    colors='cyan',
                    linewidths=1
                )
                cs.set_path_effects([
                    pe.Stroke(linewidth=2, foreground='k'),
                    pe.Normal(),
                ])
                cs = (rprofs.menc/rprofs.mmax).plot.contour(
                    ax=ax,
                    levels=[0.7, 0.8, 0.9, 1.0],
                    add_labels=False,
                    colors='k',
                    linewidths=1.5
                )
                cs.set_path_effects([
                    pe.Stroke(linewidth=3, foreground='white'),
                    pe.Normal(),
                ])
                ax.clabel(cs, fontsize=13)


        # Evolution
        lw=1
        alpha=0.3

        if s.basedir in models.mach10.values():
            mdls_parent = models.mach10
        elif s.basedir in models.M10B2N1024.values():
            mdls_parent = models.M10B2N1024
        sa = load_sim.LoadSimAll(mdls_parent, verbose=False)
        for mdl in mdls_parent:
            _s = sa.set_model(mdl)
            _s.select_cores(mtd_crit)
            for _pid in _s.good_cores(8):
                cores = _s.cores[_pid]
                axs['evol_Fnet'].plot(cores.tnorm2, cores.Fnet, 'k-', lw=lw, alpha=alpha)
                axs['evol_vin'].plot(cores.tnorm2, cores.vinfall/s.cs, 'k-', lw=lw, alpha=alpha)
                axs['evol_sigma'].plot(cores.tnorm2, cores.sigma_1d/s.cs, 'k-', lw=lw, alpha=alpha)
                axs['evol_rho'].plot(cores.tnorm2, cores.center_density/s.rho0, 'k-', lw=lw, alpha=alpha)
        cores = s.cores[pid]
        axs['evol_Fnet'].plot(cores.tnorm2, cores.Fnet, c='tab:cyan', lw=lw*2)
        axs['evol_vin'].plot(cores.tnorm2, cores.vinfall/s.cs, c='tab:cyan', lw=lw*2)
        axs['evol_sigma'].plot(cores.tnorm2, cores.sigma_1d/s.cs, c='tab:cyan', lw=lw*2)
        axs['evol_rho'].plot(cores.tnorm2, cores.center_density/s.rho0, c='tab:cyan', lw=lw*2)

        for ax in layout['axs_array'].flat:
            if ax == axs['mass_ratio_over_fnet'] or ax == axs['mass_ratio_over_Fnet']:
                continue
            ax.plot(cores.radius, cores.time, ls='-', c='tab:cyan', label=r'$r_M$', lw=3)
            try:
                numcrit, rcrit = tools.critical_time_old(s, pid, method='virial')
                tcrit = cores.loc[numcrit].time
                ax.plot(rcrit, tcrit, 'o', c='tab:cyan')
            except:
                pass
            ax.plot(cores.critical_radius, cores.time, ls=':', c='tab:red', label=r'$R_\mathrm{MO}$')
            ax.plot(cores.virial_rcrit, cores.time, ls='--', c='tab:red', label=r'$r_\mathrm{crit}$')
            ax.plot(cores.leaf_radius, cores.time, ls='-.', c='g', label=r'$r_\mathrm{tidal,avg}$')
            ax.plot(cores.tidal_radius, cores.time, ls='--', c='g', label=r'$r_\mathrm{tidal,max}$')
            ax.plot(cores.min_dst_to_star, cores.time, ls='--', color='gold', label=r'$D_*$')
            ax.plot(cores.mw_dst_to_star, cores.time, color='gold', label=r'$D_{*,\mathrm{mw}}$')
            try:
                numcrit, rcrit = tools.critical_time_old(s, pid, method='empirical')
                tcrit = cores.loc[numcrit].time
                ax.plot(rcrit, tcrit, 'o', c='r')
            except:
                pass

        for ax in layout['axs_array'][:,0]:
            ax.set_xscale('log')
            ax.set_xlim(4e-3, 2e0)
            if len(cores) > 1:
                hdt = 0.5*(cores.time.iloc[1] - cores.time.iloc[0])
                ax.set_ylim(cores.time.iloc[0]-hdt, cores.time.iloc[-1]+hdt)
            else:
                pass
            ax.set_ylabel(r'$t/t_{J,0}$')
        for ax in layout['axs_array'][-1,:]:
            ax.set_xlabel(r'$r/L_{J,0}$')

        # Forces
        axs['evol_Fnet'].axhline(0, lw=1, ls='--', c='k')
        axs['evol_Fnet'].set_ylim(-1, 1)
        axs['evol_Fnet'].set_yticks(np.arange(-1, 1.1, 0.5))
        axs['evol_Fnet'].set_ylabel(r'$\mathcal{F}_\mathrm{imb}\equiv F_\mathrm{net}/|F_\mathrm{grv}|$')

        # Infall velocities
        axs['evol_vin'].set_ylim(-2, 0)
        axs['evol_vin'].set_ylabel(r'$\overline{v}_{r,\mathrm{core}}/c_s$')
        axs['evol_vin'].set_yticks(np.arange(-2, 0.1, 0.5))

        # Velocity dispersion
        axs['evol_sigma'].set_ylim(0, 2)
        axs['evol_sigma'].set_ylabel(r'$\sigma_\mathrm{1D}(r_M)/c_s$')
        axs['evol_sigma'].set_yticks(np.arange(0, 2.1, 0.5))

        # Contrasts
        axs['evol_rho'].set_yscale('log')
        axs['evol_rho'].set_ylim(1e1, 1e5)
        axs['evol_rho'].set_ylabel(r'$\rho_c/\rho_0$')

        layout['axs_evol_array'][-1].set_xticks(np.arange(-1, 1.1, 0.5));
        layout['axs_evol_array'][-1].set_xlim(-1, 1)
        layout['axs_evol_array'][-1].set_xlabel(r'$(t - t_\mathrm{crit})/\Delta t_\mathrm{coll}$')

        layout['axs_array'][0,-1].legend(loc='upper left', bbox_to_anchor=(1.35, 1), frameon=True, fontsize=15)

        suptitle = f'Model {s.basename}, pid {pid}'
        suptitle += f'        M_core = {cores.attrs['mcore']:.3f} M_J0 = {cores.attrs['mcore']*s.u.mass.to('Msun'):.3f}'
        suptitle += f'        R_core = {cores.attrs['rcore']:.3f} L_J0 = {cores.attrs['rcore'] / s.dx:.1f} dx = {cores.attrs['rcore']*s.u.length.to('pc'):.3f}'
        layout['fig'].suptitle(suptitle, y=0.9, fontsize=20)

        return layout['fig']

    def format_log(self, cb, base=2.0):
        cb.locator = ticker.LogLocator(base=base)
        cb.formatter = ticker.LogFormatterSciNotation(base=base)
        cb.minorlocator = ticker.NullLocator()
        try:
            cb.update_ticks()
        except:
            pass

    def imshow(self, da, ax, *, label=None, log_base=None, **kwargs):
        m = da.plot.imshow(ax=ax, cbar_kwargs=dict(label=label) if label else None, add_labels=False, **kwargs)
        if log_base is not None:
            format_log(self, m.colorbar, base=log_base)
        return m

    def create_layout(self, size):
        if size == 'compact':
            nrows = 2
            ncols = 2
        elif size == 'medium':
            nrows = 3
            ncols = 3
        elif size == 'full':
            nrows = 4
            ncols = 4
        else:
            raise ValueError("size must be either compact or full")

        fig = plt.figure(figsize=(8.53*(ncols+1), 6*nrows))

        # Outer layout: left block (large panels) + right block (small stacked panels)
        outer = GridSpec(
                1, 2, figure=fig,
                width_ratios=[4*ncols/2, 1.2],
                wspace=0.24
                )

        # Left: 2x2 large axes, with tight vertical spacing for shared x in each column
        left = GridSpecFromSubplotSpec(
                nrows, ncols, subplot_spec=outer[0],
                hspace=0.11, wspace=0.3
                )
        axs_spacetime = np.empty([nrows, ncols], dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                if i == 0 and j == 0:
                    axs_spacetime[i, j] = fig.add_subplot(left[i, j])
                else:
                    axs_spacetime[i, j] = fig.add_subplot(
                            left[i, j],
                            sharex=axs_spacetime[0, 0],
                            sharey=axs_spacetime[0, 0]
                            )

        # Right: 4 stacked small axes, sharing x
        right = GridSpecFromSubplotSpec(
                4, 1, subplot_spec=outer[1],
                hspace=0.1
                )
        axs_evolution = np.empty(4, dtype=object)
        axs_evolution[0] = fig.add_subplot(right[0, 0])
        axs_evolution[1] = fig.add_subplot(right[1, 0], sharex=axs_evolution[0])
        axs_evolution[2] = fig.add_subplot(right[2, 0], sharex=axs_evolution[0])
        axs_evolution[3] = fig.add_subplot(right[3, 0], sharex=axs_evolution[0])

        # Hide lower x tick labels where axes share x
        for ax in axs_evolution[:3]:
            ax.tick_params(labelbottom=False)

        if size == 'compact':
            axs = [
                    ['Fnet'      , 'fnet'                ],
                    ['mass_ratio', 'mass_ratio_over_fnet'],
                    ]
        if size == 'medium':
            axs = [
                    ['Fnet'                     , 'fnet'              , 'vin'               ],
                    ['mass_ratio_thm'           , 'mass_ratio_thm_trb', 'mass_ratio'        ],
                    ['mass_ratio_over_fnet'     , 'eta_grav'          , 'mass_ratio_fixed_a'],
                    ]
        if size == 'full':
            axs = [
                    ['Fnet'                  , 'fnet'                          , 'vin'               , 'Brms'                ],
                    ['mass_ratio_thm'        , 'mass_ratio_thm_trb'            , 'mass_ratio'        , 'mass_ratio_over_fnet'],
                    ['mass_ratio_thm_fixed_a', 'mass_ratio_thm_trb_fixed_a'    , 'mass_ratio_fixed_a', 'mass_ratio_over_Fnet'],
                    ['eta_grav'              , 'net_magnetic_energy_normalized', 'grav_to_mag_energy', 'mass_to_flux'        ],
                    ]
        axs = dict(zip(np.array(axs).flatten(), axs_spacetime.flatten()))
        axs['evol_Fnet'] = axs_evolution[0]
        axs['evol_vin'] = axs_evolution[1]
        axs['evol_sigma'] = axs_evolution[2]
        axs['evol_rho'] = axs_evolution[3]
        layout = {
                'fig': fig,
                'axs_array': axs_spacetime,
                'axs_evol_array': axs_evolution,
                'axs_dict': axs
                }
        return layout

def plot_projection(s, ds, field='dens', axis='z', op='sum',
                    vmin=1e-1, vmax=2e2, cmap='pink_r', alpha=1,
                    ax=None, cax=None, noplot=False,
                    add_colorbar=True, transpose=False):
    """Plot projection of the selected variable along the given axis.

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    ds : xarray.Dataset
        Object containing fluid variables.
    field : str, optional
        Variable to plot.
    axis : str, optional
        Axis to project.
    vmin : float, optional
        Minimum color range.
    vmax : float, optional
        Maximum color range.
    cmap : str, optional
        Color map.
    alpha : float, optional
        Transparency.
    ax : matplotlib.axes, optional
        Axes to draw contours.
    cax : matplotlib.axes, optional
        Axes to draw color bar.
    add_colorbar : bool, optional
        If true, add color bar.
    transpose : bool, optional
        If true, transpose x and y axis.
    """
    # some domain informations
    xmin, ymin, zmin = s.domain['le']
    xmax, ymax, zmax = s.domain['re']
    Lx, Ly, Lz = s.domain['Lx']

    if isinstance(ds, xr.Dataset):
        # Reset the domain information, for the case when
        # ds is a part of the whole domain.
        xmin = ds.x[0] - 0.5*s.dx
        ymin = ds.y[0] - 0.5*s.dy
        zmin = ds.z[0] - 0.5*s.dz
        xmax = ds.x[-1] + 0.5*s.dx
        ymax = ds.y[-1] + 0.5*s.dy
        zmax = ds.z[-1] + 0.5*s.dz
        Lx = xmax - xmin
        Ly = ymax - ymin
        Lz = zmax - zmin

    wh = dict(zip(('x', 'y', 'z'), ((Ly, Lz), (Lz, Lx), (Lx, Ly))))
    extent = dict(zip(('x', 'y', 'z'), ((ymin, ymax, zmin, zmax),
                                        (zmin, zmax, xmin, xmax),
                                        (xmin, xmax, ymin, ymax))))
    permutations = dict(z=('y', 'x'), y=('x', 'z'), x=('z', 'y'))
    field_dict_pyathena = dict(dens='dens', mask='mask')

    fld = field_dict_pyathena[field]
    if op == 'sum':
        prj = ds[fld].integrate(axis).transpose(*permutations[axis])
    elif op == 'max':
        prj = ds[fld].max(axis).transpose(*permutations[axis])
    if noplot:
        return prj
    else:
        prj = prj.to_numpy()

    if ax is None:
        ax = plt.gca()
    if transpose:
        prj = prj.T
        extent = {k: v[2:] + v[0:2] for k, v in extent.items()}
    img = ax.imshow(prj, norm=LogNorm(vmin, vmax), origin='lower',
                    extent=extent[axis], cmap=cmap, alpha=alpha)
    if add_colorbar:
        plt.colorbar(cax=cax)
    return img


def plot_energies(s, ds, rprf, core, gd, node, ax=None):
    if ax is not None:
        plt.sca(ax)

    rprf = tools.cumulative_energy(s, rprf, core)
    plt.plot(rprf.r, rprf.ethm, ls='-', c='tab:blue', label='thermal')
    plt.plot(rprf.r, rprf.ekin, ls='-', c='tab:orange', label='kinetic')
    plt.plot(rprf.r, rprf.egrv, ls='-', c='tab:green', label='gravitational')
    plt.plot(rprf.r, rprf.etot, ls='-', c='tab:red', label='total')

    data = dict(rho=ds.dens.to_numpy(),
                vel1=(ds.mom1/ds.dens).to_numpy(),
                vel2=(ds.mom2/ds.dens).to_numpy(),
                vel3=(ds.mom3/ds.dens).to_numpy(),
                prs=s.cs**2*ds.dens.to_numpy(),
                phi=ds.phi.to_numpy(),
                dvol=s.dV)
    reff, engs = energy.calculate_cumulative_energies(gd, data, node)
    plt.plot(reff, engs['ethm'], ls='--', c='tab:blue')
    plt.plot(reff, engs['ekin'], ls='--', c='tab:orange')
    plt.plot(reff, engs['egrv'], ls='--', c='tab:green')
    plt.plot(reff, engs['etot'], ls='--', c='tab:red')
    plt.axhline(0, linestyle=':', color='tab:gray')
    plt.legend(loc='lower left')


def plot_grid_dendro_contours(s, gd, nodes, coords, axis='z', color='k',
                              lw=0.5, ax=None, transpose=False, recenter=None,
                              select=None):
    """Draw contours at the boundary of GRID-dendro objects

    Parameters
    ----------
    s : LoadSim
        Object containing simulation metadata.
    gd : grid_dendro.dendrogram.Dendrogram
        GRID-dendro dendrogram instance.
    nodes : int or array of ints
        ID of selected GRID-dendro nodes
    coords : xarray.core.coordinates.DatasetCoordinates
        xarray coordinate instance.
    axis : str, optional
        Axis to project.
    color : str, optional
        Contour line color.
    lw : float, optional
        Contour line width.
    ax : matplotlib.axes, optional
        Axes to draw contours.
    transpose : bool, optional
        If true, transpose x and y axis.
    recenter : tuple, optional
        New (x, y, z) coordinates of the center.
    select : dict, optional
        Selected region to slice data. If recenter is True, the selected
        region is understood in the recentered coordinates.
    """
    # some domain informations
    xmin, xmax = coords['x'].min(), coords['x'].max()
    ymin, ymax = coords['y'].min(), coords['y'].max()
    zmin, zmax = coords['z'].min(), coords['z'].max()
    sizes = coords.sizes
    extent = dict(zip(('x', 'y', 'z'), ((ymin, ymax, zmin, zmax),
                                        (zmin, zmax, xmin, xmax),
                                        (xmin, xmax, ymin, ymax))))
    permutations = dict(z=('y', 'x'), y=('x', 'z'), x=('z', 'y'))

    if isinstance(nodes, (int, np.int32, np.int64)):
        nodes = [nodes,]
    for nd in nodes:
        mask = xr.DataArray(np.ones([sizes['z'], sizes['y'], sizes['x']],
                                    dtype=bool), coords=coords)
        mask = gd.filter_data(mask, nd, fill_value=0)
        if recenter is not None:
            mask, _, _ = tools.recenter_dataset(mask, recenter)
        if select is not None:
            mask = mask.sel(select)

        mask = mask.max(dim=axis)

        if mask.max() == 0 or mask.min() == 1:
            # If a core is outside the selected region, or the
            # selected region is entirely contained in a core,
            # no contour can be drawn.
            continue

        mask = mask.transpose(*permutations[axis])

        if ax is not None:
            plt.sca(ax)
        if transpose:
            mask = mask.T
            extent = {k: v[2:] + v[0:2] for k, v in extent.items()}

        mask.plot.contour(levels=[0.5], linewidths=lw, colors=color,
                          add_labels=False)
    plt.xlim(extent[axis][0], extent[axis][1])
    plt.ylim(extent[axis][2], extent[axis][3])


def plot_cum_forces(s, rprf, core, ax=None, lw=1):
    """Plot cumulative force per unit mass
    """
    if ax is not None:
        plt.sca(ax)

    plt.plot(rprf.r, rprf.Fthm/rprf.Fgrv, lw=lw, c='tab:blue', label=r'$F_\mathrm{thm}$')
    plt.plot(rprf.r, rprf.Ftrb/rprf.Fgrv, lw=lw, c='tab:orange', label=r'$F_\mathrm{trb}$')
    plt.plot(rprf.r, rprf.Fcen/rprf.Fgrv, lw=lw, c='tab:green', label=r'$F_\mathrm{cen}$')
    plt.plot(rprf.r, rprf.Fani/rprf.Fgrv, lw=lw, c='tab:purple', label=r'$F_\mathrm{ani}$')
    fnet = rprf.Fthm + rprf.Ftrb + rprf.Fcen + rprf.Fani - rprf.Fgrv
    if s.mhd:
        plt.plot(rprf.r, rprf.Fmag/rprf.Fgrv, lw=lw, c='tab:red', label=r'$F_\mathrm{mag}$')
        fnet += rprf.Fmag
    plt.plot(rprf.r, fnet/rprf.Fgrv, 'k-', lw=1.5*lw, label=r'$F_\mathrm{net}$')


    plt.axhline(0, c='k', lw=1, ls='--')
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$F/F_\mathrm{grv,0}$')
    plt.ylim(-1, 2)
    plt.legend(ncol=3, loc='lower left')


def plot_forces(s, rprf, ax=None, xlim=(0, 0.2), ylim=(-2, 3)):
    if ax is not None:
        plt.sca(ax)

    (rprf.thm/(-rprf.grv)).plot(lw=1, color='tab:blue', label=r'$f_\mathrm{thm}$')
    (rprf.trb/(-rprf.grv)).plot(lw=1, color='tab:orange', label=r'$f_\mathrm{trb}$')
    (rprf.cen/(-rprf.grv)).plot(lw=1, color='tab:green', label=r'$f_\mathrm{cen}$')
    (rprf.ani/(-rprf.grv)).plot(lw=1, color='tab:purple', label=r'$f_\mathrm{ani}$')
    fnet = (rprf.thm + rprf.trb + rprf.cen + rprf.ani + rprf.grv)
    if s.mhd:
        (rprf.mag/(-rprf.grv)).plot(lw=1, color='tab:red', label=r'$f_\mathrm{mag}$')
        fnet += rprf.mag

    plt.plot(rprf.r, fnet/(-rprf.grv), lw=1.5, color='k', label=r'$f_\mathrm{net}$')


    plt.axhline(0, linestyle=':')
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$f/|f_\mathrm{grv}|$')
    plt.xlim(xlim)
    plt.ylim(ylim)


def plot_diagnostics(s, pid, normalize_time=True):
    """Create four-row plot showing history of core properties

    Parameters
    ----------
    s : LoadSim
        Simulation metadata
    pid : int
        Unique particle ID.
    normalize_time : bool, optional
        Flag to use normalized time (t-tcoll)/tff
    """
    fig, axs = plt.subplots(4, 1, figsize=(7, 15), sharex='col',
                            gridspec_kw=dict(hspace=0.1))

    # Load cores
    cores = s.cores[pid].sort_index()
    if normalize_time:
        time = cores.tnorm2
    else:
        time = cores.time

    # Plot net force
    plt.sca(axs[0])
    fnet = (cores.Fthm + cores.Ftrb + cores.Fcen + cores.Fani - cores.Fgrv)/cores.Fgrv
    plt.plot(time, fnet, c='k')
    plt.ylim(-1, 1)
    plt.ylabel(r'$F_\mathrm{net}/ F_\mathrm{grv}$')
    if pid in s.good_cores():
        plt.title('{}, core {}'.format(s.basename, pid))
    else:
        plt.title('{}, core {}'.format(s.basename, pid)+r'$^*$')

    # Plot individual force components.
    plt.twinx()
    plt.plot(time, cores.Fthm, lw=1, c='cyan', label=r'$F_\mathrm{thm}$')
    plt.plot(time, cores.Ftrb, lw=1, c='gray', label=r'$F_\mathrm{trb}$')
    plt.plot(time, cores.Fcen, lw=1, c='olive', label=r'$F_\mathrm{cen}$')
    plt.plot(time, cores.Fgrv, lw=1, c='pink', label=r'$F_\mathrm{grv}$')
    plt.plot(time, cores.Fani, lw=1, c='brown', ls=':', label=r'$F_\mathrm{ani}$')
    plt.yscale('log')
    plt.ylim(1e-1, 1e2)
    plt.legend(loc='center left', bbox_to_anchor=(1.08, 0.5))

    # Plot radii
    plt.sca(axs[1])
    plt.plot(time, cores.tidal_radius, c='tab:blue',
             label=r'$R_\mathrm{tidal}$')
    plt.plot(time, cores.leaf_radius, c='tab:blue', ls='-', lw=1)
    plt.plot(time, cores.sonic_radius, c='tab:green',
             label=r'$R_\mathrm{sonic}$')
    plt.plot(time, cores.critical_radius, c='tab:red',
             label=r'$R_\mathrm{crit}$')
    plt.plot(time, cores.radius, c='k',
             label=r'$r_\mathrm{M}$')
    plt.yscale('log')
    plt.ylim(1e-2, 1e0)
    plt.ylabel(r'$R/L_{J,0}$')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Plot mass
    plt.sca(axs[2])
    plt.plot(time, cores.tidal_mass, c='tab:blue',
             label=r'$M_\mathrm{tidal}$')
    plt.plot(time, cores.menc_crit, c='tab:blue', ls='-.',
             label=r'$M_\mathrm{enc}$')
    plt.plot(time, cores.critical_mass, c='tab:red',
             label=r'$M_\mathrm{crit}$')
    plt.yscale('log')
    plt.ylim(1e-3, 1e1)
    plt.ylabel(r'$M/M_{J,0}$')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Plot densities
    plt.sca(axs[3])
    plt.plot(time, cores.center_density, c='tab:blue', ls='-',
             label=r'$\rho_c$')
    plt.plot(time, cores.edge_density, c='tab:blue', ls='--',
             label=r'$\rho_e$')
    rhoavg = cores.attrs['mcore'] / (4*np.pi*cores.radius**3/3)
    plt.plot(time, rhoavg, c='tab:blue', ls=':',
             label=r'$\overline{\rho}$')
    plt.yscale('log')
    plt.ylabel(r'$\rho/\rho_0$')
    plt.legend(loc='upper left', bbox_to_anchor=(1.12, 1))
    plt.ylim(1e0, 1e5)

    if normalize_time:
        plt.xlim(-2, 1)
        plt.xlabel(r'$(t - t_\mathrm{crit})/(t_\mathrm{coll} - t_\mathrm{crit})$')
    else:
        for ax in axs:
            ax.axvline(cores.attrs['tcrit'], color='k', ls=':')
        plt.xlabel(r'$t/t_{J,0}$')
    for ax in axs:
        ax.grid()
    return fig


def plot_core_evolution(s, pid, num, rmax=None):
    if rmax is None:
        if np.isfinite(s.cores[pid].attrs['rcore']):
            rmax = 2*s.cores[pid].attrs['rcore']
        else:
            rmax = s.cores[pid].tidal_radius.max()
    hw = 1.1*rmax

    # Load data
    ds = s.load_hdf5(num, quantities=['dens'], load_method='xarray')
#    gd = s.load_dendro(num)
    core = s.cores[pid].loc[num]
    core_tcrit_pred = s.cores_dict['predicted'][pid].loc[num]
    core_tcrit_emp = s.cores_dict['empirical'][pid].loc[num]
    rprf = s.rprofs[pid].sel(num=num)

    # Average tidal radius
    rtidal = 0.5*(core.tidal_radius + core.leaf_radius)

    # Find the location of the core
    xc, yc, zc = s.flatindex_to_cartesian(core.leaf_id)

    # Load sink particles
    pds = s.load_par(num)
    pds = pds[((pds.x1 > xc - hw) & (pds.x1 < xc + hw)
             & (pds.x2 > yc - hw) & (pds.x2 < yc + hw)
             & (pds.x3 > zc - hw) & (pds.x3 < zc + hw))]
    pds = pds.copy()
    pds.loc[:, ('x1', 'x2', 'x3')] -= np.array([xc, yc, zc])
    for pos, idx in zip(['x1', 'x2', 'x3'], [0, 1, 2]):
        pds.loc[:, pos] = tools.sawtooth(pds[pos],
                                         s.domain['le'][idx], s.domain['re'][idx],
                                         s.domain['le'][idx], s.domain['re'][idx])

    # Create figure
    fig = plt.figure(figsize=(35, 21))
    gs = gridspec.GridSpec(3, 5, wspace=0.23, hspace=0.15,
                           width_ratios=[1.2, 1.2, 1.2, 1.2, 1.2])

    xlim = dict(z=(xc-hw, xc+hw),
                x=(yc-hw, yc+hw),
                y=(zc-hw, zc+hw))
    ylim = dict(z=(yc-hw, yc+hw),
                x=(zc-hw, zc+hw),
                y=(xc-hw, xc+hw))
    xlabel = dict(z=r'$x/L_{J,0}$', x=r'$y/L_{J,0}$', y=r'$z/L_{J,0}$')
    ylabel = dict(z=r'$y/L_{J,0}$', x=r'$z/L_{J,0}$', y=r'$x/L_{J,0}$')

    xycoords = dict(z=['x1', 'x2'],
                    x=['x2', 'x3'],
                    y=['x3', 'x1'])

    axs = dict(proj=[fig.add_subplot(gs[i, 0]) for i in [0, 1, 2]],
               zoom=[fig.add_subplot(gs[i, 1]) for i in [0, 1, 2]],
               rho=[fig.add_subplot(gs[0, i]) for i in [2, 3]],
               force=[fig.add_subplot(gs[1, i]) for i in [2, 3]],
               veldisp=fig.add_subplot(gs[2, 2]),
               vel=fig.add_subplot(gs[2, 3]),
#               energy=fig.add_subplot(gs[0, 4]),
               acc=fig.add_subplot(gs[:, 4]))

    # Zoom-in dataset
    sel = dict(x=slice(-hw, hw), y=slice(-hw, hw), z=slice(-hw, hw))
    d, _, _ = tools.recenter_dataset(ds, dict(x=xc, y=yc, z=zc))
    d = d.sel(sel)

    for i, prj_axis in enumerate(['z', 'x', 'y']):
        # 1. Projections
        plt.sca(axs['proj'][i])
        plot_projection(s, ds, axis=prj_axis, add_colorbar=False)
        rec = plt.Rectangle((xlim[prj_axis][0], ylim[prj_axis][0]),
                            2*hw, 2*hw, fill=False, ec='r')
        plt.gca().add_artist(rec)
        plt.xlabel(xlabel[prj_axis])
        plt.ylabel(ylabel[prj_axis])

        # 2. Zoom-in projections
        plt.sca(axs['zoom'][i])
        plot_projection(s, d, axis=prj_axis, add_colorbar=False)
#        if core.leaf_id not in gd.leaves:
#            # Due to path-dependent nature of dendrogram pruning, some ids
#            # may not match in global dendrogram and local dendrogram.
#            # For now, let's simply pass this edge case.
#            pass
#        else:
#            if core.leaf_id != gd.trunk:
#                plot_grid_dendro_contours(s, gd, gd.sibling(core.leaf_id), ds.coords, axis=prj_axis,
#                                          recenter=dict(x=xc, y=yc, z=zc), select=sel, color='k')
#            plot_grid_dendro_contours(s, gd, core.leaf_id, ds.coords, axis=prj_axis,
#                                      recenter=dict(x=xc, y=yc, z=zc), select=sel, color='g')
        if rtidal <= np.sqrt(2)*hw:
            c0 = plt.Circle((0, 0), rtidal, fill=False, color='g', lw=1, ls='--')
            plt.gca().add_artist(c0)
        if np.isfinite(core.critical_radius) and core.critical_radius <= np.sqrt(2)*hw:
            c0 = plt.Circle((0, 0), core.critical_radius, fill=False, color='r', lw=1, ls='-')
            plt.gca().add_artist(c0)
        if np.isfinite(core.radius) and core.radius <= np.sqrt(2)*hw:
            c0 = plt.Circle((0, 0), core.radius, fill=False, color='b', lw=1, ls='-.')
            plt.gca().add_artist(c0)

        # Overplot star particles
        plt.scatter(pds[xycoords[prj_axis][0]], pds[xycoords[prj_axis][1]],
                    marker=MarkerStyle('*', fillstyle='full'), color='y')

        plt.xlim(-hw, hw)
        plt.ylim(-hw, hw)
        plt.xlabel(xlabel[prj_axis])
        plt.ylabel(ylabel[prj_axis])


    # 4. Radial profiles
    # Density
    r = np.linspace(s.dx/2, 2*rmax)
    rhoLP = tools.lpdensity(r, s.cs, s.gconst)
    for ax in axs['rho']:
        plt.sca(ax)
        plt.plot(rprf.r, rprf.rho, 'k-+')
        plt.plot(r, rhoLP, 'k--')

    # overplot critical tes
    r0 = s.cs / np.sqrt(4*np.pi*s.gconst*core.center_density)
    xi_max = rprf.r.isel(r=-1).data[()]/r0
    if np.isfinite(core.sonic_radius):
        try:
            ts = tes.TES(pindex=core.pindex, rsonic=core.sonic_radius/r0)
            xi = np.logspace(np.log10(ts._rfloor), np.log10(xi_max))
            for ax in axs['rho']:
                ax.plot(xi*r0, core.center_density*ts.density(xi), 'r--', lw=1.5)
        except UserWarning:
            pass

    # overplot critical BE
    ts = tes.TES()
    xi = np.logspace(np.log10(ts._rfloor), np.log10(xi_max))
    for ax in axs['rho']:
        ax.plot(xi*r0, core.center_density*ts.density(xi), 'r:', lw=1)
        ax.set_xlabel(r'$r/L_{J,0}$')
        ax.set_ylabel(r'$\rho/\rho_0$')
        ax.set_ylim(1e0, tools.lpdensity(s.dx/2, s.cs, s.gconst))

    plt.sca(axs['rho'][0])
    plt.plot(s.dx/2, rprf.rho.isel(r=0), marker=MarkerStyle(4, fillstyle='full'), ms=20)
    plt.xlim(s.dx/2, 2*rmax)
    plt.xscale('log')
    plt.yscale('log')

    plt.sca(axs['rho'][1])
    plt.xlim(0, rmax)
    plt.yscale('log')

    # Forces
    for ax in axs['force']:
        plot_cum_forces(s, rprf, core, ax)
    plt.sca(axs['force'][0])
    plt.xlim(s.dx/2, 2*rmax)
    plt.xscale('log')
    plt.legend(loc='upper right')
    plt.sca(axs['force'][1])
    plt.xlim(0, rmax)
    plt.legend([], [])

    # Velocities
    plt.sca(axs['vel'])
    plt.plot(rprf.r, rprf.vel1_mw, marker='+', label=r'$v_r$')
    plt.plot(rprf.r, rprf.vel2_mw, marker='+', label=r'$v_\theta$')
    plt.plot(rprf.r, rprf.vel3_mw, marker='+', label=r'$v_\phi$')
    plt.axhline(0, ls=':')
    plt.xlim(0, rmax)
    plt.ylim(-2.5, 1.5)
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$\left<v\right>_\rho/c_s$')
    plt.legend(loc='upper right')

    # Velocity dispersions
    plt.sca(axs['veldisp'])
    plt.loglog(rprf.r, np.sqrt(rprf.dvel1_sq_mw), marker='+', label=r'$v_r$')
    plt.loglog(rprf.r, np.sqrt(rprf.dvel2_sq_mw), marker='+',
               label=r'$v_\theta$')
    plt.loglog(rprf.r, np.sqrt(rprf.dvel3_sq_mw), marker='+',
               label=r'$v_\phi$')
    rcl = tools.reff_sph(s.Lbox**3)
    plt.plot(rprf.r, s.Mach*(rprf.r/rcl)**0.5, 'k--')
    plt.plot(rprf.r, (rprf.r/(s.sonic_length/2))**1, 'k--')

    # overplot linear fit
    if not np.isnan(core.sonic_radius) and not np.isinf(core.sonic_radius):
        plt.plot(rprf.r, (rprf.r/core.sonic_radius)**(core.pindex), 'r--',
                 lw=1)

    plt.xlim(s.dx/2, 2*rmax)
    plt.ylim(1e-1, 1e1)
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$\left<v^2\right>^{1/2}_\rho/c_s$')
    plt.legend(loc='lower right')

    # 5. Energies
    # TODO On-the-fly calculation is too expensive.
    # Until precalculating the energies, disable the plot
#    axs['energy'].remove()
#    plt.sca(axs['energy'])
#    plot_energies(s, ds, rprf, core, gd, core.leaf_id)
#    if emin is not None and emax is not None:
#        plt.ylim(emin, emax)
#    plt.xlim(0, rmax)

    # 6. Accelerations
    plt.sca(axs['acc'])
    plot_forces(s, rprf)
    plt.title('')
    plt.xlim(0, rmax)
    plt.legend(ncol=3, fontsize=17, loc='upper right')

    # Annotations
    plt.sca(axs['rho'][0])
    plt.text(0.6, 0.9, r'$t={:.3f}$'.format(ds.Time)+r'$\,t_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')

    # Annotate normalized time; if either core_tcrit_pred or core_tcrit_emp is unresolved, this will raise
    # AttributeError.
    try:
        plt.text(0.6, 0.8, r'$\tau_\mathrm{evol,pred}=$'+r'${:.2f}$'.format(core_tcrit_pred.tnorm2),
                 transform=plt.gca().transAxes, backgroundcolor='w')
    except AttributeError:
        pass
    try:
        plt.text(0.6, 0.7, r'$\tau_\mathrm{evol}=$'+r'${:.2f}$'.format(core_tcrit_emp.tnorm2),
                 transform=plt.gca().transAxes, backgroundcolor='w')
    except AttributeError:
        pass

    plt.text(0.6, 0.6, f'n={num}',
             transform=plt.gca().transAxes, backgroundcolor='w')

    plt.text(0.05, 0.05, r'$r_M={:.2f}$'.format(core.radius)+r'$\,L_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    plt.text(0.05, 0.15, r'$r_\mathrm{crit}=$'+r'${:.2f}$'.format(core.critical_radius)+r'$\,L_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    plt.text(0.05, 0.25, r'$r_\mathrm{tidal}=$'+r'${:.2f}$'.format(rtidal)+r'$\,L_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    plt.text(0.05, 0.35, r'$r_s=$'+r'${:.2f}$'.format(core.sonic_radius)+r'$\,L_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')

    for ax in (axs['rho'][0], axs['rho'][1], axs['force'][0], axs['force'][1],
               axs['vel'], axs['veldisp'], axs['acc']):
        plt.sca(ax)
        ln1 = plt.axvline(rtidal, c='g', lw=1, ls='--')
        ln2 = plt.axvline(core.critical_radius, ls='-', c='r')
        ln3 = plt.axvline(core.sonic_radius, ls=':', c='tab:gray')
        ln4 = plt.axvline(core.radius, ls='-.', c='b')

    plt.sca(axs['rho'][1])
    lgd = plt.legend([ln1, ln2, ln3, ln4], [r'$r_\mathrm{tidal}$',
                                            r'$r_\mathrm{crit}$',
                                            r'$r_s$',
                                            r'$r_M$'],
                     loc='upper right')
    plt.gca().add_artist(lgd)

    return fig


def mass_radius(s, pid, num, rmax=None, ax=None):
    if rmax is None:
        rmax = s.cores[pid].tidal_radius.max()
    core = s.cores[pid].loc[num]
    rprf = s.rprofs[pid].sel(num=num)

    lw = 1.5

    menc = (4*np.pi*rprf.r**2*rprf.rho).cumulative_integrate('r')
    plt.plot(rprf.r, menc, 'k-+', lw=lw)

    tse = tes.TESe(p=core.pindex, xi_s=core.sonic_radius*np.sqrt(core.edge_density))
    uc, rc, mc = tse.get_crit()
    ymax = s.cores[pid].tidal_mass.max()
    nsample = 100
    rds, mass = np.zeros(nsample), np.zeros(nsample)
    for i, u0 in enumerate(np.linspace(0, 4*uc, nsample)):
        rds[i] = tse.get_radius(u0)
        mass[i] = tse.get_mass(u0)
    for i in [1,2,4]:
        rhoe = core.edge_density*i
        plt.plot(rds/np.sqrt(rhoe), mass/np.sqrt(rhoe), 'k-', lw=lw/2)
    plt.plot(rds, rds*mc/rc, 'k-', lw=lw)
    plt.fill_between(rds, rds*mc/rc, y2=ymax, facecolor='lightgray')

    # Critical equilibrium profile
    rhoe = core.edge_density/1.25
    r = np.linspace(0.01, rc)
    u, _ = tse.solve(r, uc)
    rho = xr.DataArray(np.exp(u), coords=dict(r=r))
    menc = (4*np.pi*r**2*rho).cumulative_integrate('r')
    plt.plot(r/np.sqrt(rhoe), menc/np.sqrt(rhoe), c='k', lw=lw, ls='-.')

    plt.xlim(0, rmax)
    plt.ylim(0, ymax)
    plt.xlabel(r'$R/L_{J,0}$')
    plt.ylabel(r'$M/M_{J,0}$')
    plt.axvline(core.tidal_radius, lw=1, ls='-', c='tab:gray')
    plt.axvline(core.critical_radius_e, lw=1, ls='-.', c='tab:gray')
    plt.axhline(core.critical_mass_e, lw=1, ls='-.', c='tab:gray')
    plt.axvline(core.critical_radius, lw=1, ls='--', c='tab:gray')


def core_structure(s, pid, num, rmax=None):
    core = s.cores[pid].loc[num]
    rprf = s.rprofs[pid].sel(num=num)
    if rmax is None:
        rmax = core.tidal_radius

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10),
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for ax in axs[0]:
        plt.sca(ax)
        plt.plot(rprf.r, rprf.rho, 'k-+')
        rhoLP = tools.lpdensity(rprf.r, s.cs, s.gconst)
        plt.plot(rprf.r, rhoLP, 'k--', lw=1)

        # overplot critical tes
        r0 = s.cs / np.sqrt(4*np.pi*s.gconst*core.center_density)
        xi_min = rprf.r.isel(r=1).data[()]/r0
        xi_max = rprf.r.isel(r=-1).data[()]/r0
        xi = np.logspace(np.log10(xi_min), np.log10(xi_max))
        if not np.isnan(core.sonic_radius) and not np.isinf(core.sonic_radius):
            ts = tes.TES(pindex=core.pindex, rsonic=core.sonic_radius/r0)
            plt.plot(xi*r0, ts.density(xi), 'r--', lw=1.5)

        # overplot critical BE
        ts = tes.TES()
        plt.plot(xi*r0, ts.density(xi), 'r:', lw=1)

        plt.axhline(core.edge_density, ls='-.', c='tab:gray')
        plt.yscale('log')
        plt.ylim(1e0, tools.lpdensity(s.dx/2, s.cs, s.gconst))

    for ax in axs[1]:
        plot_cum_forces(s, rprf, core, ax)

    axs[0,0].set_ylabel(r'$\rho/\rho_0$')
    for ax in axs[:,0]:
        ax.set_xscale('log')
    for ax in axs[:,1]:
        ax.set_ylabel('')
    for ax in axs[:,0]:
        ax.set_xlim(rprf.r[0]/2, rmax*2)
    for ax in axs[:,1]:
        ax.set_xlim(0, rmax)
    for ax in axs[1]:
        ax.set_xlabel(r'$r/L_{J,0}$')

    plt.sca(axs[0,0])
    plt.text(0.5, 0.9, r'$t={:.3f}$'.format(core.time)+r'$\,t_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    plt.text(0.5, 0.8, r'$M={:.2f}$'.format(core.tidal_mass)+r'$\,M_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    plt.text(0.5, 0.7, r'$R={:.2f}$'.format(core.tidal_radius)+r'$\,L_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    plt.text(0.05, 0.05, r'$t-t_\mathrm{crit}=$'+r'${:.2f}$'.format(core.tnorm2)
             + r'$\,\Delta t_\mathrm{coll}$', transform=plt.gca().transAxes,
             backgroundcolor='w')

    for ax in axs.flat:
        plt.sca(ax)
        ln1 = plt.axvline(core.tidal_radius, c='tab:gray', lw=1)
        ln2 = plt.axvline(core.critical_radius, ls='--', c='tab:gray')
        ln3 = plt.axvline(core.sonic_radius, ls=':', c='tab:gray')
    plt.sca(axs[0,1])
    lgd = plt.legend([ln1, ln2, ln3], [r'$R_\mathrm{tidal}$',
                                       r'$R_\mathrm{crit,c}$',
                                       r'$R_\mathrm{sonic}$'],
                     loc='upper right')
    return fig


def radial_profile_at_tcrit(s, pid, ax=None, lw=1.5):
    cores = s.cores[pid]
    num = cores.attrs['numcrit']
    core = cores.loc[num]
    rprf = s.rprofs[pid].sel(num=num)

    if ax is not None:
        plt.sca(ax)

    rcrit = core.critical_radius

    plt.plot(rprf.r/rcrit, rprf.rho/core.center_density, ls='-', marker='+', color='k', lw=lw)

    # Overplot critical TES
    r0 = s.cs / np.sqrt(4*np.pi*s.gconst*core.center_density)
    try:
        ts = tes.TES(pindex=core.pindex, rsonic=core.sonic_radius/r0)
        xi = np.logspace(np.log10(ts._rfloor), np.log10(rcrit/r0))
        plt.plot(xi*r0/rcrit, ts.density(xi), c='tab:red', lw=lw)
    except UserWarning:
        s.logger.warning(f"model {s.basename} pid {pid} has out-of-range pindex at tcrit")

    # Overplot critical BE
    ts = tes.TES()
    xi = np.logspace(np.log10(ts._rfloor), np.log10(rcrit/r0))
    plt.plot(xi*r0/rcrit, ts.density(xi), c='tab:blue', lw=lw)
    plt.plot(ts.rcrit*r0/rcrit, ts.density(ts.rcrit), c='tab:blue',
             marker=MarkerStyle('o', fillstyle='full'))

    plt.xlim(0, 1)
    plt.ylim(5e-3, 1e0)
    plt.xlabel(r'$r/R_\mathrm{crit}$')
    plt.ylabel(r'$\rho/\rho_c$')
    plt.yscale('log')

# DEPRECATED


def plot_sinkhistory(s, ds, pds):
    # find end time
    ds_end = s.load_hdf5(s.nums[-1], header_only=True)
    tend = ds_end['Time']

    # create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, wspace=0, hspace=0)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    # plot projections
    for ax, axis in zip((ax0, ax1, ax2), ('z', 'x', 'y')):
        plot_projection(s, ds, ax=ax, axis=axis, add_colorbar=False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax0.plot(pds.x1, pds.x2, '*', color='b', ms=8, mew=0.5, alpha=0.7)
    ax1.plot(pds.x2, pds.x3, '*', color='b', ms=8, mew=0.5, alpha=0.7)
    ax2.plot(pds.x3, pds.x1, '*', color='b', ms=8, mew=0.5, alpha=0.7)

    # plot particle history
    plt.sca(ax3)
    for pid in s.pids:
        phst = s.load_parhst(pid)
        time = phst.time
        mass = phst.mass
        tslc = time < ds.Time
        plt.plot(time[tslc], mass[tslc])
    plt.axvline(ds.Time, linestyle=':', color='k', linewidth=0.5)
    if len(s.tcoll_cores) > 0:
        plt.xlim(s.tcoll_cores.time.iloc[0], tend)
    plt.ylim(1e-2, 1e1)
    plt.yscale('log')
    plt.xlabel(r'$t/t_\mathrm{J,0}$')
    plt.ylabel(r'$M_*/M_\mathrm{J,0}$')
    return fig


def plot_Pspec(s, ds, ax=None, ax_twin=None):
    """Requires load_method='xarray'"""
    # prepare frequencies
    from scipy.fft import fftn, fftfreq, fftshift
    from pyathena.util.transform import groupby_bins
    Lx, Ly, Lz = s.domain['Lx']
    Nx, Ny, Nz = s.domain['Nx']
    dx, dy, dz = s.domain['Lx']/(s.domain['Nx'])
    kx = fftshift(2*np.pi*fftfreq(Nx, dx))
    ky = fftshift(2*np.pi*fftfreq(Ny, dy))
    kz = fftshift(2*np.pi*fftfreq(Nz, dz))
    # Do FFTs
    for axis in [1, 2, 3]:
        vel = ds['mom'+str(axis)]/ds.dens
        ds['vel'+str(axis)] = vel
        vhat = fftn(vel.data, vel.shape)*dx*dy*dz
        vhat = fftshift(vhat)
        ds['vhat'+str(axis)] = xr.DataArray(
            vhat,
            coords=dict(kz=('z', kz), ky=('y', ky), kx=('x', kx)),
            dims=('z', 'y', 'x'))
    # Calculate 3D power spectrum
    Pk = np.abs(ds.vhat1)**2 + np.abs(ds.vhat2)**2 + np.abs(ds.vhat3)**2
    # Set radial wavenumbers
    Pk.coords['k'] = np.sqrt(Pk.kz**2 + Pk.ky**2 + Pk.kx**2)
    kmin = np.sqrt((2*np.pi/Lx)**2 + (2*np.pi/Ly)**2 + (2*np.pi/Lz)**2)
    kmax = np.sqrt((2*np.pi/(2*dx))**2 + (2*np.pi/(2*dy))**2
                   + (2*np.pi/(2*dz))**2)
    lmin, lmax = 2*np.pi/kmax, 2*np.pi/kmin
    # Perform spherical average
    Pk = groupby_bins(Pk, 'k', np.linspace(kmin, kmax, min(Nx, Ny, Nz)//2))

    # Plot results
    if ax is not None:
        plt.sca(ax)

    Pk.plot(marker='+')
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(Pk.k, Pk[0]*(Pk.k/Pk.k.min())**(-4), 'k--', lw=1, label=r'$n=-4$')
    plt.plot(Pk.k, Pk[0]*(Pk.k/Pk.k.min())**(-(11/3)), 'k-.', lw=1, label=r'$n=-11/3$')
    plt.xlim(kmin, kmax)
    plt.xlabel(r'$k/L_J^{-1}$')
    plt.ylim(1e-7, 1e3)
    plt.ylabel(r'$P(k)=|\mathbf{v}(k)|^2$')
    plt.legend(loc='upper right')
    if ax_twin is not None:
        plt.sca(ax_twin)
    else:
        plt.twiny()
    plt.xscale('log')
    xticks = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
    plt.gca().set_xticks(1/xticks)
    plt.gca().set_xticklabels(xticks)
    plt.xlim(1/lmax, 1/lmin)
    plt.axvline(s.dx, ls='--', c='k', lw=1)
    plt.xlabel(r'$l/L_J$')


def plot_central_density_evolution(s, ax=None):
    rho_crit_KM05 = tools.get_rhocrit_KM05(s.sonic_length)
    if ax is not None:
        plt.sca(ax)

    for pid in s.pids:
        rprf = s.rprofs[pid]
        rprf.rho.isel(r=0).plot(label=f'par{pid}')
        plt.yscale('log')
        plt.ylim(1e1, tools.lpdensity(0.5*s.dx, s.cs, s.gconst))
    plt.axhline(rho_crit_KM05, linestyle='--')
    plt.text(s.rprofs[1].t.min(), rho_crit_KM05, r"$\rho_\mathrm{crit, KM05}$",
             fontsize=18, ha='left', va='bottom')
    plt.text(s.rprofs[1].t.min(), 14.1*rho_crit_KM05,
             r"$14.1\rho_\mathrm{crit, KM05}$", fontsize=18,
             ha='left', va='bottom')
    plt.axhline(14.1*rho_crit_KM05, linestyle='--', lw=1)
    plt.legend(loc=(1.01, 0))
    plt.ylabel(r'$\rho_c/\rho_0$')
    plt.xlabel(r'$t/t_J$')
    plt.title(s.basename)


def plot_PDF(s, ds, ax=None):
    """Requires load_method='xarray'"""
    def gaussian(x, mu, sig):
        return np.exp(-(x - mu)**2 / (2*sig**2))/np.sqrt(2*np.pi*sig**2)

    rho = np.sort(ds.dens.data.flatten()).astype('float64')
    x = np.log(rho)
    mu_v = x.mean()
    mu_m = np.average(x, weights=rho)
    sig_v = x.std()
    sig_m = np.sqrt(np.average((x-mu_m)**2, weights=rho))

    if ax is not None:
        plt.sca(ax)

    # volume weighting
    fm, edges = np.histogram(np.log(rho), bins=100, density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    plt.stairs(fm, np.exp(edges), color='r')
    plt.plot(np.exp(centers), gaussian(centers, mu_v, sig_v), 'r:', lw=1)

    # mass weighting
    fm, edges = np.histogram(np.log(rho), weights=rho, bins=100, density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    plt.stairs(fm, np.exp(edges), color='b')
    plt.plot(np.exp(centers), gaussian(centers, mu_m, sig_m), 'b:', lw=1)

    # annotations
    plt.text(0.03, 0.9, r'$\mu = {:.2f}$'.format(mu_v),
             transform=plt.gca().transAxes, fontsize=15, c='r')
    plt.text(0.03, 0.8, r'$\sigma = {:.2f}$'.format(sig_v),
             transform=plt.gca().transAxes, fontsize=15, c='r')
    plt.text(0.03, 0.7, r'$\sigma^2/2 = {:.2f}$'.format(sig_v**2/2),
             transform=plt.gca().transAxes, fontsize=15, c='r')
    plt.text(0.03, 0.6, r'$e^\mu = {:.2f}$'.format(np.exp(mu_v)),
             transform=plt.gca().transAxes, fontsize=15, c='r')
    plt.text(0.72, 0.9, r'$\mu = {:.2f}$'.format(mu_m),
             transform=plt.gca().transAxes, fontsize=15, c='b')
    plt.text(0.72, 0.8, r'$\sigma = {:.2f}$'.format(sig_m),
             transform=plt.gca().transAxes, fontsize=15, c='b')
    plt.text(0.72, 0.7, r'$\sigma^2/2 = {:.2f}$'.format(sig_m**2/2),
             transform=plt.gca().transAxes, fontsize=15, c='b')
    plt.text(0.72, 0.6, r'$e^\mu = {:.2f}$'.format(np.exp(mu_m)),
             transform=plt.gca().transAxes, fontsize=15, c='b')
    plt.xscale('log')
    plt.xlabel(r'$\rho/\rho_0$')
    plt.ylabel('volume or mass fraction')
    plt.title(r'$t = {:.2f}$'.format(ds.Time))
    plt.xlim(1e-4, 1e4)
    plt.ylim(0, 0.3)
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])
