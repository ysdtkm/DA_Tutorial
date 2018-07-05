#!/usr/bin/env python3

PRIME = 10 ** 6 + 3
SEED = PRIME * 2

T_RUN = 10 ** 4
DT = 0.01
ACYC_STEP = int(2.5 / DT)
assert abs(DT * ACYC_STEP - 2.5) < 1.0e-6
FLAG_R = "Cheng"
OBS_NET = "full_spectral"

EDIM = 37
RHO = 1.0

BCOV_FROM = "Cheng"
TDVAR_METHOD = "oi"
SIGMA_B = 0.0002

SKIP_HEAVY_PLOT = True
