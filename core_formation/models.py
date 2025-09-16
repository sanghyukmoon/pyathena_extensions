M5N512 = {f"M5J2P{iseed}N512": f"/projects2/EOSTRIKE/sanghyuk/cores/M5.J2.P{iseed}.N512" for iseed in range(0, 40)}
M10N1024 = {f"M10J4P{iseed}N1024": f"/projects2/EOSTRIKE/sanghyuk/cores/M10.J4.P{iseed}.N1024" for iseed in range(0, 7)}

# backward compatibility
mach5 = M5N512
mach10 = M10N1024
hydro = {**mach5, **mach10} # we may update hydro later, while keeping the old one

hydro_old = hydro.copy()

mhd = {"M10J4B4P1N1024": "/tigress/sm69/cores/mhd/M10.J4.B4.P1.N1024",
       "M10J4B2P1N1024": "/scratch/gpfs/EOST/sanghyuk/cores/mhd/M10.J4.B2.P1.N1024",
       "M5J2B2P1N512": "/scratch/gpfs/EOST/sanghyuk/cores/mhd/M5.J2.B2.P1.N512"}

M5N512_ext = {f"M5J2P{iseed}N512": f"/scratch/gpfs/sm69/cores/hydro/M5.J2.P{iseed}.N512" for iseed in range(40, 57)}
M5N1024 = {f"M5J2P{iseed}N1024": f"/scratch/gpfs/sm69/cores/hydro/M5.J2.P{iseed}.N1024" for iseed in range(0, 139)}

# All models
models = {**M5N512, **M5N1024, **M10N1024, **mhd, **M5N512_ext}
