#!/usr/bin/env python3

PRIME = 10 ** 6 + 3
SEED = PRIME * 2
X0_INIT = "Cheng"
RANDOM_SAMPLE_INIT = True
INIT_SIGMAS = 0.1

T_RUN = 10 ** 4
DT = 0.1
ACYC_STEP = int(round(2.5 / DT))
ERROR_FREE_OBS = False
OBS_NET = "ocn_grid"
FLAG_R = "hbht"

EDIM = 37
RHO = 1.0
RELAX = 0.6

BCOV_FROM = "t0829ocn"
TDVAR_METHOD = "inv"
SIGMA_B = 0.01
DIAGONALIZE_B = False

ALPHA = 0.5

SKIP_HEAVY_PLOT = True
