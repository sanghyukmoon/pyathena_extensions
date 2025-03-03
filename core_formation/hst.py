import numpy as np
import pandas as pd
from pyathena.io.read_hst import read_hst
from pyathena.load_sim import LoadSim as LoadSimBase

class Hst:

    @LoadSimBase.Decorators.check_pickle_hst
    def read_hst(self, savdir=None, force_override=False):
        """Function to read hst and convert quantities to convenient units
        """

        hst = read_hst(self.files['hst'], force_override=force_override)

        # Time in unit of free-fall time
        hst['time_in_tff'] = hst['time']/self.tff0
        # Timestep
        hst['dt_in_tff'] = hst['dt']/self.tff0

        # Kinetic and gravitational energies
        hst['KE'] = (hst['1KE'] + hst['2KE'] + hst['3KE'])

        # Mass weighted velocity dispersions
        for name, ax in zip(('x', 'y', 'z'), ('1', '2', '3')):
            KE = hst['{}KE'.format(ax)]
            hst['v{}'.format(name)] = np.sqrt(2*KE/hst['mass'])

        # 3D Mach number
        hst['Mach'] = np.sqrt(hst['vx']**2 + hst['vy']**2 + hst['vz']**2) / self.cs

        hst.set_index('time', inplace=True)
        self.hst = hst
        return hst
