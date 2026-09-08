# Comparing collapse-onset definitions

`LoadSim` keeps tracked trajectories and TES properties in `s.cores`. Derived
trajectories for all 17 definitions are cached in `s.cores_dict[definition][pid]`.
Selection returns independent DataFrame copies and leaves both collections
unchanged.

```python
from core_formation import load_sim, plots

s = load_sim.LoadSim(basedir, savdir=savdir, legacy=False)
all_cores = s.select_cores()  # virial mass, fixed form factors, net force, std
pressure_cores = s.select_cores(rcrit="virial_pressure", criterion="overpressure")
tes_cores = s.select_cores(rcrit="tes")

cores = all_cores[pid]
pressure_core = pressure_cores[pid]
print(cores.attrs["tcrit"], pressure_core.attrs["tcrit"])
plots.plot_core_evolution(s, cores, num)
plots.plot_core_evolution(s, pressure_core, num)

resolved_pids = s.good_cores(all_cores, nres=8)
```

The selection keywords are `rcrit` (`"tes"`, `"virial_mass"`, or
`"virial_pressure"`), `fixed_form_factor`, `criterion` (`"net_force"` or
`"overpressure"`), and `require_small_std`. Virial defaults are fixed form
factors and the standard-deviation criterion; use `False` to disable either.
TES uses its historical net-force condition without those two options. See
`CollapseOnsetDefinition` for the supported combinations. `rtes` is the TES
prediction; `rcrit` is the selected radius trajectory and equals `rtes` for TES.

One `cores` table supplies its particle ID and definition through `attrs`.
`attrs["collapse_definition"]` is a JSON string that survives NetCDF storage.
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
        rcrit="virial_pressure", criterion="overpressure", legacy=False):
    print(s.basename, pid, cores.attrs["tcrit"])

fig, axs = plots.plot_lookback_profiles(
    sa, lookback_times=[0.0, 0.1], quantities=[{"name": "rho"}],
    rcrit="virial_mass", require_small_std=False)
```

`itercore` selects once per simulation. `nres=0` yields every selected core;
positive values apply the resolution filter. `itercritcore` retains its existing
`(s, pid, core_series, rprf)` return. Space-time plots select their background
population using the target table's definition.

Existing derived caches receive definition metadata in memory on reading;
reading does not rewrite the files. After generating Lagrangian or observational
products, explicitly refresh derived caches with `override_derived_cores=True`
on a new `LoadSim`. The selector does not generate these optional products.
Plots that use dendrogram properties still require those properties.
