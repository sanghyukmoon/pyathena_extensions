# Repository Guidelines

## Project Structure & Module Organization

This repository contains scientific analysis extensions built on `pyathena`. The `core_formation/` package uses a flat module layout. `load_sim.py` defines the main `LoadSim` interface; `tools.py` contains numerical helpers; `tasks.py` implements batch analyses; and `plots.py`, `slc_prj.py`, `hst.py`, and `stats.py` provide specialized analysis and visualization routines. Model definitions and output-directory names belong in `models.py` and `config.py`. Cluster entry points include `do_tasks.py` and `profile_radial_profile.py`. `do_dasks.py` is deprecated.

There is currently no dedicated test directory. SLURM scripts, scheduler JSON, `.out`/`.err` logs, NetCDF files, and pickle products are generated artifacts, not source code.

## Environment & Development Commands

Create and activate the Conda environment from the repository root:

```bash
mamba env create -f env.yml
mamba activate pyathena
```

Run lightweight validation before committing:

```bash
python -m py_compile core_formation/*.py
python -c "from core_formation import load_sim, tasks, tools"
```

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8-style names: `snake_case` for functions and variables, `CamelCase` for classes, and uppercase names for configuration constants. Follow existing NumPy/xarray idioms and retain physical-variable terminology already used by datasets. Prefer lazy Dask operations for domain-sized arrays, but make graph growth and materialization explicit. Numerical accuracy takes precedence over speed; preserve the package convention disabling xarray's Bottleneck and Numbagg reductions where relevant. No formatter or linter is currently enforced, so avoid unrelated reformatting.

## Testing Guidelines

For numerical changes, add focused synthetic-array checks when practical and compare dimensions, coordinates, units, and representative values. If adding pytest coverage, place files under `tests/` as `test_<module>.py`. Do not require Athena simulation data or a SLURM cluster for unit tests. For Dask changes, test both lazy graph construction and computed results on small arrays.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as `Move angular momentum calc...` and `Apply correct subcell avg...`. Keep commits focused. Pull requests should describe the scientific or behavioral effect, identify affected models/tasks, report validation commands, and note memory or graph-size implications. Include before/after figures only for plotting or scientifically visible changes; never commit credentials, private paths, large simulation products, or scheduler logs.

## On-the-fly analysis project coordination

For core_formation work in this project, read /home/sm69/athena_radial_profile/AGENTS.md, /home/sm69/athena_radial_profile/PLANS.md, and /home/sm69/athena_radial_profile/plans/onthefly-analysis-master.md before planning or implementation.

The single shared output-time specification is /home/sm69/athena_radial_profile/OUTPUT_CONTRACT.md. Read it when changing timelines, loaders, numbering, or mixed-data consumers. Do not duplicate it here or assume target behavior is already implemented. Contract revisions require reviewing Athena++, base readers, application consumers, tests, and old-output compatibility together.

Every future project planning session must update the master plan and its dedicated smaller plan; every implementation milestone must update both. Use the master's terminology: t_coll is sink-particle formation, t_crit is the onset of runaway collapse, r_crit/M_crit are virial-based quantities, and r_TES/M_TES are TES predictions. Scientific formulas and existing identifiers are not to be changed merely for terminology. Unresolved projection physics requires user review.

Preserve unrelated user edits. Use the pyathena Python environment and place all new simulation/analysis artifacts only under /scratch/gpfs/sm69/onthefly-rprof-test. Keep focused commits with Co-authored-by: Codex <codex@openai.com>. Update explanatory PDFs after major milestones.
