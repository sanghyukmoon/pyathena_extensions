from pathlib import Path
import numpy as np
import argparse
import subprocess
from multiprocessing import Pool

from core_formation import config, tasks, models, load_sim

if __name__ == "__main__":
    sa = load_sim.LoadSimAll(models.models)

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs='+', type=str,
                        help="List of models to process")
    parser.add_argument("--pids", nargs='+', type=int,
                        help="List of particle ids to process")
    parser.add_argument("--np", type=int, default=1,
                        help="Number of processors")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite everything")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Loop through nums in revserse direction")
    parser.add_argument("--combine-partab", action="store_true",
                        help="Join partab files")
    parser.add_argument("--combine-partab-full", action="store_true",
                        help="Join partab files including last output")
    parser.add_argument("--run-grid", action="store_true",
                        help="Run GRID-dendro")
    parser.add_argument("--prune", action="store_true",
                        help="Prune dendrogram")
    parser.add_argument("--track-cores", action="store_true",
                        help="Perform reverse core tracking (prestellar phase)")
    parser.add_argument("--radial-profile", action="store_true",
                        help="Calculate radial profiles of each cores")
    parser.add_argument("--critical-tes", action="store_true",
                        help="Calculate critical TES of each cores")
    parser.add_argument("--lagrangian-props", action="store_true",
                        help="Calculate Lagrangian properties of cores")
    parser.add_argument("--projections", action="store_true",
                        help="Calculate projections")
    parser.add_argument("--prj-radial-profile", action="store_true",
                        help="Calculate radial profiles of each cores")
    parser.add_argument("--observables", action="store_true",
                        help="Calculate observable properties of cores")
    parser.add_argument("--linewidth-size", action="store_true",
                        help="Calculate linewidth-size relation")
    parser.add_argument("--make-movie", action="store_true",
                        help="Create movies")
    parser.add_argument("--plot-core-evolution", action="store_true",
                        help="Create core evolution plots")
    parser.add_argument("--plot-sink-history", action="store_true",
                        help="Create sink history plots")
    parser.add_argument("--plot-pdfs", action="store_true",
                        help="Create density pdf and velocity power spectrum")
    parser.add_argument("--plot-diagnostics", action="store_true",
                        help="Create diagnostics plot for each core")
    parser.add_argument("--pid-start", type=int)
    parser.add_argument("--pid-end", type=int)

    args = parser.parse_args()

    # Select models
    for mdl in args.models:
        s = sa.set_model(mdl, force_override=True)

        if args.pid_start is not None and args.pid_end is not None:
            pids = np.arange(args.pid_start, args.pid_end+1)
        else:
            pids = s.pids
        if args.pids:
            pids = args.pids
        pids = sorted(list(set(s.pids) & set(pids)))

        # Combine output files.
        if args.combine_partab:
            print(f"Combine partab files for model {mdl}")
            tasks.combine_partab(s, remove=True, include_last=False)

        if args.combine_partab_full:
            print(f"Combine all partab files for model {mdl}")
            tasks.combine_partab(s, remove=True, include_last=True)

        # Run GRID-dendro.
        if args.run_grid:
            def wrapper(num):
                tasks.run_grid(s, num, overwrite=args.overwrite)
            print(f"Run GRID-dendro for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.nums[config.GRID_NUM_START:], 1)

        # Run GRID-dendro.
        if args.prune:
            def wrapper(num):
                tasks.prune(s, num, overwrite=args.overwrite)
            print(f"Run GRID-dendro for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.nums[config.GRID_NUM_START:], 1)

        # Find t_coll cores and save their GRID-dendro node ID's.
        if args.track_cores:
            def wrapper(pid):
                tasks.core_tracking(s, pid, overwrite=args.overwrite)
            print(f"Perform core tracking for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, pids)

        # Calculate radial profiles of t_coll cores and pickle them.
        if args.radial_profile:
            msg = ("calculate and save radial profiles for "
                   f"model {mdl}")
            print(msg)
            def wrapper(num):
                tasks.radial_profile(s, num, pids, overwrite=args.overwrite,
                                     full_radius=True, days_overwrite=0)
            nums = s.nums[::-1] if args.reverse else s.nums
            with Pool(args.np) as p:
                p.map(wrapper, nums)

        # Find critical tes
        if args.critical_tes:
            print(f"find critical tes for cores for model {mdl}")
            for pid in pids:
                cores = s.cores[pid]
                def wrapper(num):
                    tasks.critical_tes(s, pid, num, overwrite=args.overwrite)
                with Pool(args.np) as p:
                    p.map(wrapper, cores.index)

        # Calculate Lagrangian properties
        if args.lagrangian_props:
            s = sa.set_model(mdl, force_override=True)
            def wrapper(pid):
                method_list = ['empirical', 'predicted']
                for method in method_list:
                    s.select_cores(method)
                    if pid in s.cores:
                        tasks.lagrangian_props(s, pid, method=method, overwrite=args.overwrite)
            print(f"Calculate Lagrangian properties for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, pids)

        # Calculate radial profiles of t_coll cores and pickle them.
        if args.projections:
            s = sa.set_model(mdl, force_override=True)
            msg = ("calculate and save projections for " f"model {mdl}")
            print(msg)
            def wrapper(num):
                tasks.projections(s, num, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper, s.nums)

        # Calculate radial profiles of t_coll cores and pickle them.
        if args.prj_radial_profile:
            s = sa.set_model(mdl, force_override=True)
            msg = ("calculate and save projected radial profiles for "
                   f"model {mdl}")
            print(msg)
            def wrapper(num):
                tasks.prj_radial_profile(s, num, pids, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper, s.nums)

        # Find observables
        if args.observables:
            s = sa.set_model(mdl, force_override=True)
            print(f"Calculate observable core properties for model {mdl}")
            for pid in pids:
                cores = s.cores[pid]
                cores = cores.loc[:cores.attrs['numcoll']]
                def wrapper(num):
                    tasks.observables(s, pid, num, overwrite=args.overwrite)
                with Pool(args.np) as p:
                    p.map(wrapper, cores.index)


        # Resample AMR data into uniform grid
#        print(f"resample AMR to uniform for model {mdl}")
#        tasks.resample_hdf5(s)

        # Calculate radial profiles of t_coll cores and pickle them.
        if args.linewidth_size:
            for num in [74]:
                ds = s.load_hdf5(num, quantities=['dens', 'mom1', 'mom2', 'mom3'])
                ds['vel1'] = ds.mom1/ds.dens
                ds['vel2'] = ds.mom2/ds.dens
                ds['vel3'] = ds.mom3/ds.dens
                def wrapper(seed):
                    tasks.calculate_linewidth_size(s, num, seed=seed, overwrite=args.overwrite, ds=ds)
                with Pool(args.np) as p:
                    p.map(wrapper, np.arange(1000))

                def wrapper2(pid):
                    tasks.calculate_linewidth_size(s, num, pid=pid, overwrite=args.overwrite, ds=ds)
                with Pool(args.np) as p:
                    p.map(wrapper2, s.good_cores())

            def wrapper3(pid):
                ncrit = s.cores[pid].attrs['numcrit']
                tasks.calculate_linewidth_size(s, ncrit, pid=pid, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper3, s.good_cores())

        # make plots
        if args.plot_core_evolution:
            s = sa.set_model(mdl, force_override=True)
            print(f"draw core evolution plots for model {mdl}")
            for pid in pids:
                for method in ['empirical', 'predicted']:
                    s.select_cores(method)
                    cores = s.cores[pid]
                    def wrapper(num):
                        tasks.plot_core_evolution(s, pid, num, method=method,
                                                  overwrite=args.overwrite)
                    with Pool(args.np) as p:
                        p.map(wrapper, cores.index)

        if args.plot_sink_history:
            s = sa.set_model(mdl, force_override=True)
            def wrapper(num):
                tasks.plot_sink_history(s, num, overwrite=args.overwrite)
            print(f"draw sink history plots for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.nums)

        if args.plot_pdfs:
            def wrapper(num):
                tasks.plot_pdfs(s, num, overwrite=args.overwrite)
            print(f"draw PDF-power spectrum plots for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.nums)

        if args.plot_diagnostics:
            print(f"draw diagnostics plots for model {mdl}")
            for pid in s.good_cores():
                tasks.plot_diagnostics(s, pid, overwrite=args.overwrite)

        # make movie
        if args.make_movie:
            print(f"create movies for model {mdl}")
            srcdir = Path(s.savdir, "figures")
            plot_prefix = [
#                    config.PLOT_PREFIX_PDF_PSPEC,
                    config.PLOT_PREFIX_SINK_HISTORY,
                          ]
            for prefix in plot_prefix:
                subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d",
                                srcdir])
            prefix = config.PLOT_PREFIX_CORE_EVOLUTION
            for pid in pids:
                for method in ['empirical', 'predicted']:
                    s.select_cores(method)
                    prf = f"{prefix}.par{pid}.tcrit_{method}"
                    subprocess.run(["make_movie", "-p", prf, "-s", srcdir,
                                    "-d", srcdir])
