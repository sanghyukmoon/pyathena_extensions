from pathlib import Path
import numpy as np
import argparse
import subprocess
import multiprocessing as mp

from core_formation import config, tasks, models, load_sim, tools

# Python 3.14 defaults to forkserver on POSIX. This runner relies on
# inherited state and local wrapper functions, so keep the historical fork
# behavior explicitly.
Pool = mp.get_context("fork").Pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs='+', type=str,
                        help="List of models to process")
    parser.add_argument("--np", type=int, default=1,
                        help="Number of processors")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite everything")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Loop through nums in revserse direction")
    parser.add_argument("--track-cores", action="store_true",
                        help="Perform reverse core tracking (prestellar phase)")
    parser.add_argument("--critical-tes", action="store_true",
                        help="Calculate critical TES of each cores")
    parser.add_argument("--collapse-history", action="store_true",
                        help="Calculate intrinsic collapse histories")
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

    args = parser.parse_args()
    sa = load_sim.LoadSimAll(models.models)

    # Select models
    for mdl in args.models:
        # Find t_coll cores and save their GRID-dendro node ID's.
        if args.track_cores:
            s = sa.set_model(mdl, skip_collapse_history=True)
            def wrapper(pid):
                tasks.track_cores(s, [pid,], overwrite=args.overwrite)
            print(f"Perform core tracking for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.pids)

        # Find critical tes
        if args.critical_tes:
            s = sa.set_model(mdl, skip_collapse_history=True)
            print(f"find critical tes for cores for model {mdl}")
            def wrapper(pid):
                tasks.critical_tes(s, pid, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper, s.cores)

        # Calculate each requested core's collapse history.
        if args.collapse_history:
            s = sa.set_model(mdl, skip_collapse_history=True)
            for onset_def in tools.COLLAPSE_ONSET_DEFINITIONS:
                def wrapper(pid):
                    tasks.collapse_history(s, s.cores[pid], onset_def,
                                           overwrite=args.overwrite)
                with Pool(args.np) as p:
                    p.map(wrapper, s.cores)

        if args.projections:
            s = sa.set_model(mdl)
            msg = ("calculate and save projections for " f"model {mdl}")
            print(msg)
            def wrapper(num):
                tasks.projections(s, num, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper, s.nums)

        if args.prj_radial_profile:
            s = sa.set_model(mdl)
            msg = ("calculate and save projected radial profiles for "
                   f"model {mdl}")
            print(msg)
            def wrapper(num):
                tasks.prj_radial_profile(s, num, s.cores, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper, s.nums)

        # Find observables
        if args.observables:
            s = sa.set_model(mdl, override_all=True)
            print(f"Calculate observable core properties for model {mdl}")
            for pid, cores in s.cores.items():
                cores = cores.loc[:cores.attrs['numcoll']]
                def wrapper(num):
                    tasks.observables(s, pid, num, overwrite=args.overwrite)
                with Pool(args.np) as p:
                    p.map(wrapper, [num for num in cores.index if num in s.nums_with_proj])

        # Calculate radial profiles of t_coll cores and pickle them.
        if args.linewidth_size:
            s = sa.set_model(mdl, override_all=True)
            all_cores = s.select_cores()
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
                    p.map(wrapper2, s.good_cores(all_cores))

            def wrapper3(pid):
                ncrit = all_cores[pid].attrs['numcrit']
                tasks.calculate_linewidth_size(s, ncrit, pid=pid, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper3, s.good_cores(all_cores))

        # make plots
        if args.plot_core_evolution:
            s = sa.set_model(mdl, override_all=True)
            print(f"draw core evolution plots for model {mdl}")
            for cores in s.select_cores().values():
                def wrapper(num):
                    tasks.plot_core_evolution(s, cores, num, overwrite=args.overwrite)
                with Pool(args.np) as p:
                    p.map(wrapper, [num for num in cores.index if num in s.nums_with_proj])

        if args.plot_sink_history:
            s = sa.set_model(mdl, override_all=True)
            def wrapper(num):
                tasks.plot_sink_history(s, num, overwrite=args.overwrite)
            print(f"draw sink history plots for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.nums_with_proj)

        # make movie
        if args.make_movie:
            s = sa.set_model(mdl)
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
            for pid in s.cores:
                for onset_def in [tools.CollapseOnsetDefinition()]:
                    prf = f"{prefix}.par{pid}.tcrit_{onset_def.filename_token}"
                    subprocess.run(["make_movie", "-p", prf, "-s", srcdir,
                                    "-d", srcdir])
