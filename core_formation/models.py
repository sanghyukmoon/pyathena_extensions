M5N512 = {f"M5J2P{iseed}N512": f"/scratch/gpfs/sm69/cores/hydro/M5.J2.P{iseed}.N512" for iseed in range(0, 40)}
M5N1024 = {f"M5J2P{iseed}N1024": f"/scratch/gpfs/sm69/cores/hydro/M5.J2.P{iseed}.N1024" for iseed in range(0, 20)}
m2 = {f"M10J4P{iseed}N1024": f"/scratch/gpfs/sm69/cores/hydro/M10.J4.P{iseed}.N1024" for iseed in range(0, 2)}
m3 = {f"M10J4P{iseed}N1024": f"/projects2/EOSTRIKE/sanghyuk/cores/M10.J4.P{iseed}.N1024" for iseed in range(2, 7)}
M10N1024 = {**m2, **m3}

# backward compatibility
mach5 = M5N512
mach10 = M10N1024
hydro = {**mach5, **mach10} # we may update hydro later, while keeping the old one

hydro_old = hydro.copy()
hydro_old['M15J2P1N512'] = "/projects2/EOSTRIKE/sanghyuk/cores/M15.J2.P1.N512"
hydro_old['M3J4P1N1024'] = "/tigress/sm69/cores/hydro/M3.J4.P1.N1024"

mhd = {"M10J4B4P1N1024": "/tigress/sm69/cores/mhd/M10.J4.B4.P1.N1024"}

# All models
models = {**M5N512, **M5N1024, **M10N1024, **mhd}

models['test'] = '/scratch/gpfs/sm69/cores/mhd/test'
