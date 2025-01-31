# About

* This repository contains my personal python packages for simulation data analysis.
* All the packages are built on top of Jeong-Gyu Kim's [pyathena](https://github.com/jeonggyukim/pyathena) platform, which provides 1) an I/O interface to *Athena* and *Athena++* output files, 2) a high-level `LoadSim` container class, and 3) useful utility functions.


# Rationale

* There's no single analysis tool that serves all purpose.
* One has to invent a particular, as opposed to general, analysis methods to address a specific scientific question.


# Recommendations
The packages contained in this repository may serve as an example for those who wish to write their own **pyathena extensions**. The following guidelines may be useful:

* Write your own local `load_sim.py` module that contains `LoadSim` class derived from `pyathena.LoadSim`. To avoid namespace conflict:
```python
from pyathena.load_sim import LoadSim as LoadSimBase
class LoadSim(LoadSimBase, ...
```
* If a class or function that you wrote is general enough, consider moving it to upstream [pyathena](https://github.com/jeonggyukim/pyathena).
