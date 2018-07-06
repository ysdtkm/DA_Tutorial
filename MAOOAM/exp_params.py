#!/usr/bin/env python3

PRIME = 10 ** 6 + 3
SEED = PRIME * 2
X0_INIT = "Takuma"
RANDOM_SAMPLE_INIT = True

T_RUN = 10 ** 3
DT = 0.5
ACYC_STEP = int(2.5 / DT)
assert abs(DT * ACYC_STEP - 2.5) < 1.0e-6
FLAG_R = "Cheng"
ERROR_FREE_OBS = False
OBS_NET = "full_spectral"

EDIM = 37
RHO = 1.0

BCOV_FROM = "Cheng"
TDVAR_METHOD = "oi"
SIGMA_B = 0.005
DIAGONALIZE_B = False

SKIP_HEAVY_PLOT = False
