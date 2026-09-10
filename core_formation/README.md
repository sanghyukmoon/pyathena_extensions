# Comparing collapse-onset definitions

`LoadSim` keeps tracked trajectories and TES properties in `s.cores`. Derived
trajectories for all 17 definitions are cached in `s.cores_dict[onset_def][pid]`.
Selection returns a shallow dictionary copy whose DataFrames are shared with
`s.cores_dict`. Adding or removing dictionary entries affects only the returned
dictionary; editing a DataFrame also changes the cached table.

```python
from core_formation import load_sim, plots

s = load_sim.LoadSim(basedir, savdir=savdir, legacy=False)
all_cores = s.select_cores()  # virial mass, fixed form factors, net force, std
pressure_cores = s.select_cores(rcrit_from="virial_pressure", criterion="overpressure")
tes_cores = s.select_cores(rcrit_from="tes")

cores = all_cores[pid]
pressure_core = pressure_cores[pid]
print(cores.attrs["tcrit"], pressure_core.attrs["tcrit"])
plots.plot_core_evolution(s, cores, num)
plots.plot_core_evolution(s, pressure_core, num)

resolved_pids = s.good_cores(all_cores, nres=8)
```


## Evolution figures from stored projections

`plot_core_evolution(s, cores, num)` and `plot_sinkhistory(s, num)` require
on-the-fly `.proj.h5` files and a simulation loaded with `legacy=False`.
Choose `num` from `s.nums_with_projection`; for core evolution it must also
belong to the selected trajectory. Neither figure loads volumetric HDF5.

Both figures display viewing axes in z, y, x order, with horizontal/vertical
coordinates (x, y), (x, z), and (y, z), respectively. Core-evolution zoom panels
crop the full-LOS maps in the image plane. Velocity arrows are density-weighted
means relative to the central-cell velocity saved in the same radial profile;
magnetic streamlines use density-weighted means. Particle and minimum markers
in zoom panels represent the local 3D cube, including periodic neighbors.

`plot_projection(s, projection, field='Sigma_gas', axis='z', ...)` now renders
2D maps. `b_stream` uses the transverse `rhoB1/2/3` integrals divided by
`Sigma_gas`; `v_quiver` requires prepared transverse `vel1/2/3` maps in the
caller's chosen velocity frame. There is no volume-input or legacy fallback.

The `plot_core_evolution` and `plot_sink_history` batch tasks use discovered
projection epochs and initialize in non-legacy mode. Filenames and overwrite
behavior are unchanged; explicitly overwrite existing images to regenerate
them. The sink-history label shows sinks formed by the snapshot time over the
total historical count, including sinks that later merged. Beside it, SFE is
the current snapshot's total sink mass divided by the initial uniform gas mass
(`rho0 * Lbox**3`), displayed as a percentage. Curves retain the
strict snapshot-time cutoff and the `tend + 0.01` right-hand limit so the final
marker remains visible.

## Selection details

The selection keywords are `rcrit_from` (`"tes"`, `"virial_mass"`, or
`"virial_pressure"`), `fixed_form_factor`, `criterion` (`"net_force"` or
`"overpressure"`), and `require_small_std`. Virial defaults are fixed form
factors and the standard-deviation criterion; use `False` to disable either.
TES uses its historical net-force condition without those two options. See
`CollapseOnsetDefinition` for the supported combinations. `rtes` is the TES
prediction; `rcrit` is the selected radius trajectory and equals `rtes` for TES.

One `cores` table supplies its particle ID and definition through `attrs`.
`attrs["onset_definition"]` is a JSON string that survives NetCDF storage.
Single-core plots and tasks take `(s, cores, ...)`, with no particle ID or
selection keywords. Definition-dependent task filenames include the definition
token. Population plotting takes `(s, all_cores, ...)`.

Profiles and potential minima remain on the simulation:

```python
rprf = s.rprofs[pid].sel(num=num)
center_id = pressure_core.at[num, "leaf_id"]
position = s.flatindex_to_cartesian(center_id)
all_minima = s.minima[num]
for core in pressure_core.itertuples():
    position = s.flatindex_to_cartesian(core.leaf_id)
```

Use `.at` for an individual identifier and `itertuples()` for iteration to
preserve integer dtype. The full minima population is independent of selection.

Batch entry points keep keyword selection:

```python
sa = load_sim.LoadSimAll(model_paths)
for s, pid, cores, rprofs in sa.itercore(
        rcrit_from="virial_pressure", criterion="overpressure", legacy=False):
    print(s.basename, pid, cores.attrs["tcrit"])

fig, axs = plots.plot_lookback_profiles(
    sa, lookback_times=[0.0, 0.1], quantities=[{"name": "rho"}],
    rcrit_from="virial_mass", require_small_std=False)
```

`itercore` selects once per simulation. `nres=0` yields every selected core;
positive values apply the resolution filter. `itercritcore` retains its existing
`(s, pid, core_series, rprf)` return. Space-time plots select their background
population using the target table's definition.

## Preparation and loading

The preparation order is tracking, radial profiles, critical TES, then collapse
history. TES products remain independent of the onset definition. Legacy
individual and concatenated profiles are stable inputs.

```python
from core_formation import tasks, tools

s = load_sim.LoadSim(basedir, savdir=savdir, skip_collapse_history=True)
for onset_def in tools.COLLAPSE_ONSET_DEFINITIONS:
    for pid, cores in s.cores.items():
        tasks.collapse_history(s, cores, onset_def, overwrite=True)

s = load_sim.LoadSim(basedir, savdir=savdir)
all_cores = s.select_cores()
```

The multiprocessing runner exposes `--collapse-history` in place of
`--lagrangian-props`; the Dask runner uses the `collapse_history` task name.
`--critical-tes` remains separate. History preparation includes every requested
core, without resolution filtering. An unresolved onset is saved with NaN onset
properties and no Lagrangian calculation.

Each intrinsic history is stored as
`cores/collapse_history_<onset_token>.par<pid>.nc`. These files replace the old
separate derived-core and Lagrangian products. Run the new task to generate them;
old files are not translated or deleted. Normal analysis requires all expected
cores for every configured onset definition. Preparation uses
`skip_collapse_history=True` and does not require these files.

`_load_collapse_history` is an internal reader of intrinsic histories. Its
required `filebase` and `savdir` arguments are supplied by initialization, which
checks that every expected core is present. Initialization
separately joins available observational products for consumers; observations
never enter the intrinsic history NetCDF or its aggregate pickle. Updating
observations does not require recomputing collapse histories.

The existing pyathena `check_pickle` decorator accelerates loading with disposable
aggregate dictionaries:

| Data | Pickle |
|---|---|
| Base trajectories and TES | `cores/cores.p` |
| Complete radial profiles | `radial_profile/radial_profile.p` |
| Intrinsic histories | `cores/collapse_history_<onset_token>.p` |
| Observations | `cores/observables.p` |

`override_cores`, `override_rprofs`, `override_collapse_history`, and
`override_observables` rebuild their respective aggregates. `override_all` refreshes
all enabled loading stages; it does not enable skipped histories. Constructor
`override_collapse_history=True` (or reader `force_override=True`) reloads NetCDF,
not scientific calculations. Use `tasks.collapse_history(..., overwrite=True)`
for scientific recomputation.

History, tracking/TES, and observational tasks invalidate their directly affected
aggregate pickles. Reload for analysis after preparation. Upstream scientific
changes require explicit downstream task reruns. No timestamp scans, compatibility
repair, or automatic downstream recalculation are performed. Delete incompatible
pickle caches or explicitly override them; their durable source products remain.

For legacy profiles, `override_rprofs=True` recomputes derived fields from the
existing concatenation and replaces only the complete-profile pickle. Concatenation
is created if absent or rebuilt by explicitly calling `concat_radial_profiles()`.
For non-legacy profiles, the aggregate is rebuilt directly from on-the-fly outputs;
per-core complete-profile NetCDF intermediates are no longer used. Continue using
separate savdirs when comparing the two source modes.

The definition constructs the chosen ratio with `onset_def.virial_ratio(profile)`
and evaluates a single snapshot with `onset_def.is_collapsing(profile, rcrit)`.
The latter interpolates internally and derives radial spacing from `profile.r`
for the optional standard-deviation condition. `critical_time(s, cores)` reads
the onset definition from the table metadata and handles the backward search.
The historical TES search retains its separate condition evaluation and terminal
exceptions.

`tools.critical_time(s, cores)` and
`tools.lagrangian_property(s, cores)` obtain profiles from
`s.rprofs[cores.attrs["pid"]]`. Numerical routines that do not receive the
simulation and core table continue to accept profiles explicitly.
