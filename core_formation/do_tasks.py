from dask.distributed import Client
from dask_jobqueue.slurm import SLURMRunner
import argparse

from core_formation import load_sim, models, tasks


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs='+', type=str,
                        help="List of models to process")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite everything")
    parser.add_argument("--radial-profile", action="store_true",
                        help="Calculate radial profiles of each cores")
    parser.add_argument("--save-minima", action="store_true",
                        help="Find minima and save to pickle")
    args = parser.parse_args()

    sa = load_sim.LoadSimAll(models.models)

    with SLURMRunner(scheduler_options={"interface": "ib0"}, worker_options={"interface": "ib0"}) as runner:
    
        # The runner object contains the scheduler address info and can be used to construct a client.
        with Client(runner) as client:
    
            # Wait for all the workers to be ready before continuing.
            client.wait_for_workers(runner.n_workers)
            print(f"Number of workers = {runner.n_workers}")

            for mdl in args.models:
                s = sa.set_model(mdl, force_override=True)

                if args.radial_profile:
                    msg = ("calculate and save radial profiles for "
                           f"model {mdl}")
                    print(msg)
                    for num in s.nums:
                        tasks.radial_profile(s, num, s.pids, overwrite=args.overwrite,
                                             full_radius=True, days_overwrite=0)

                if args.find_minima:
                    msg = ("Find minimas and save to pickle for "
                           f"model {mdl}")
                    print(msg)
                    tasks.save_minima(s, overwrite=args.overwrite)
