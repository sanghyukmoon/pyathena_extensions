Lseries = {f"L{i}": f"/tigerdata/EOSTRIKE/TIGRESS-classic/TIGRESS-GC/L{i}_512" for i in [0, 1, 2, 3]}
Sseries = {f"S{i}": f"/tigerdata/EOSTRIKE/TIGRESS-classic/TIGRESS-GC/S{i}_256" for i in [0, 1, 2, 3]}
Pseries = {f"P{i}": f"/tigress/sm69/TIGRESS-GC/L1_512_P{i}" for i in [15, 50, 100]}
Bseries = {f"B{i}": f"/tigress/sm69/TIGRESS-GC/M1B{i}_512" for i in [10, 30]}
Bseries['B100'] = "/tigress/sm69/public_html/M1B100_512"
Bseries['Binf'] = "/tigress/sm69/public_html/M1Binf_512"
models = {**Lseries, **Sseries, **Pseries, **Bseries}
