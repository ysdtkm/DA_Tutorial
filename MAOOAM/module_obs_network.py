import sys
import unittest
from functools import lru_cache
import numpy as np
import pickle
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from exp_params import SEED, OBS_NET

NDIM = 36

def get_grid_val(waves, x, y, is_atm, elem):
    assert waves.__class__ == np.ndarray
    assert waves.shape == (NDIM,)
    assert y.__class__ in [float, np.float32, np.float64]
    assert x.__class__ in [float, np.float32, np.float64]
    assert elem in ["psi", "tmp", "u", "v"]

    f0 = 1.032e-4
    n = 1.5
    na = 10
    no = 8
    R = 287.0
    L = 5000000.0 / np.pi

    gridval = 0.0
    if is_atm:
        types = ["A", "K", "L", "A", "K", "L", "K", "L", "K", "L"]
        hs = [0, 1, 1, 0, 1, 1, 2, 2, 2, 2]
        ps = [1, 1, 1, 2, 2, 2, 1, 1, 2, 2]
        for j in range(na):
            j_all = j + na if elem == "tmp" else j
            if elem == "u":
                if types[j] == "A":
                    gridval += waves[j_all] * np.sqrt(2.0) * ps[j] * np.sin(ps[j] * y)
                elif types[j] == "K":
                    gridval += waves[j_all] * 2.0 * np.cos(hs[j] * n * x) \
                               * ps[j] * (-1) * np.cos(ps[j] * y)
                else:
                    gridval += waves[j_all] * 2.0 * np.sin(hs[j] * n * x) \
                               * ps[j] * (-1) * np.cos(ps[j] * y)
            elif elem == "v":
                if types[j] == "A":
                    gridval += 0.0
                elif types[j] == "K":
                    gridval += waves[j_all] * 2.0 * hs[j] * n * (-1) \
                               * np.sin(hs[j] * n * x) * np.sin(ps[j] * y)
                else:
                    gridval += waves[j_all] * 2.0 * hs[j] * n \
                               * np.cos(hs[j] * n * x) * np.sin(ps[j] * y)
            else:
                if types[j] == "A":
                    gridval += waves[j_all] * np.sqrt(2.0) * np.cos(ps[j] * y)
                elif types[j] == "K":
                    gridval += waves[j_all] * 2.0 * np.cos(hs[j] * n * x) * np.sin(ps[j] * y)
                else:
                    gridval += waves[j_all] * 2.0 * np.sin(hs[j] * n * x) * np.sin(ps[j] * y)
    else:
        hos = [1, 1, 1, 1, 2, 2, 2, 2]
        pos = [1, 2, 3, 4, 1, 2, 3, 4]
        for j in range(no):
            j_all = j + (na * 2 + no) if elem == "tmp" else j + na * 2
            if elem == "u":
                gridval += waves[j_all] * 2.0 * np.sin(0.5 * hos[j] * n * x) \
                           / (-1) * pos[j] * np.cos(pos[j] * y)
            elif elem == "v":
                gridval += waves[j_all] * 2.0 * 0.5 * hos[j] * n \
                           * np.cos(0.5 * hos[j] * n * x) * np.sin(pos[j] * y)
            else:
                gridval += waves[j_all] * 2.0 * np.sin(0.5 * hos[j] * n * x) * np.sin(pos[j] * y)
    if elem == "tmp":
        gridval *= (f0 ** 2 * L ** 2) / R
        if is_atm:
            gridval *= 2
    elif elem == "psi":
        gridval *= L ** 2 * f0
    else:
        gridval *= L * f0
    return gridval

def __get_obs_grid_atmos():
    n = 1.5
    xmax = 2.0 * np.pi / n
    ymax = np.pi
    nxobs = 5
    nyobs = 2
    x1d = np.linspace(0, xmax, nxobs, endpoint=False)
    y1d = np.linspace(0, ymax, nyobs, endpoint=False) + ymax / nyobs * 0.25
    x2d, y2d = np.meshgrid(x1d, y1d)
    return x2d, y2d

def __get_obs_grid_ocean():
    n = 1.5
    xmax = 2.0 * np.pi / n
    ymax = np.pi
    nxobs = 2
    nyobs = 4
    x1d = np.linspace(0, xmax, nxobs, endpoint=False) + xmax / nxobs * 0.5
    y1d = np.linspace(0, ymax, nyobs, endpoint=False) + ymax / nyobs * 0.25
    x2d, y2d = np.meshgrid(x1d, y1d)
    return x2d, y2d

@lru_cache(maxsize=1)
def __model_state_example():
    xini = np.array([
        4.695340259215241E-002,
        2.795833230987369E-002,
        -2.471191763590483E-002,
        -7.877635082773315E-003,
        -4.448292568544942E-003,
        -2.756238610924190E-002,
        -4.224400051368891E-003,
        5.914241112882518E-003,
        -1.779437742222920E-004,
        5.224450720394076E-003,
        4.697982667229096E-002,
        5.149282577209392E-003,
        -1.949084549066326E-002,
        4.224006062949761E-004,
        -1.247786759371923E-002,
        -9.825952138046594E-003,
        -2.610941795170075E-005,
        2.239286581216401E-003,
        -7.891896725509534E-004,
        7.470171905055880E-004,
        -9.315932162526787E-007,
        3.650179005106874E-005,
        1.064122403269511E-006,
        3.937836448211443E-008,
        -2.208288760403859E-007,
        -3.753762121228048E-006,
        -7.105126469908465E-006,
        1.518110190916469E-008,
        -5.773178576933025E-004,
        0.187369278208256,
        1.369868543156558E-003,
        7.023608700166264E-002,
        -4.539810680860224E-004,
        -1.882650440363933E-003,
        -3.900412687995408E-005,
        -1.753655087903711E-007])
    return xini

def model_state_example():
    return __model_state_example()

def get_h_full_coverage():
    nobs = 36
    h_mat = np.empty((nobs, NDIM))
    xgrid_atm, ygrid_atm = __get_obs_grid_atmos()
    xgrid_ocn, ygrid_ocn = __get_obs_grid_ocean()
    nobs_atm = 10
    nobs_ocn = 8
    assert nobs_atm * 2 + nobs_ocn * 2 == nobs
    for i in range(NDIM):
        state_unit = np.zeros(NDIM)
        state_unit[i] = 1.0
        for j in range(nobs):
            if j < nobs_atm:
                k = j
                is_atm = True
                elem = "u"
                xgrid = xgrid_atm
                ygrid = ygrid_atm
            elif j < nobs_atm * 2:
                k = j - nobs_atm
                is_atm = True
                elem = "tmp"
                xgrid = xgrid_atm
                ygrid = ygrid_atm
            elif j < nobs_atm * 2 + nobs_ocn:
                k = j - nobs_atm * 2
                is_atm = False
                elem = "u"
                xgrid = xgrid_ocn
                ygrid = ygrid_ocn
            elif j < nobs_atm * 2 + nobs_ocn * 2:
                k = j - (nobs_atm * 2 + nobs_ocn)
                is_atm = False
                elem = "tmp"
                xgrid = xgrid_ocn
                ygrid = ygrid_ocn
            else:
                raise Exception("__get_h_full_coverage overflow")
            h_mat[j, i] = get_grid_val(state_unit, xgrid.flatten()[k], ygrid.flatten()[k], is_atm, elem)
    return h_mat

def get_h():
    natm = 20
    if OBS_NET == "full_spectral":
        h = np.identity(NDIM)
    elif OBS_NET == "atm_spectral":
        h = np.identity(NDIM)
        h = h[:natm, :]
    elif OBS_NET == "ocn_spectral":
        h = np.identity(NDIM)
        h = h[natm:, :]
    elif OBS_NET == "full_grid":
        h = get_h_full_coverage()
    elif OBS_NET == "atm_grid":
        h = get_h_full_coverage()
        h = h[:natm, :]
    elif OBS_NET == "ocn_grid":
        h = get_h_full_coverage()
        h = h[natm:, :]
    elif OBS_NET == "Kriti_ocean":
        ht = read_fort_txt_2d("binary_const/20180707_ht_kriti_ocean.txt")
        h = ht.T
        assert h.shape == (NDIM - natm, NDIM)
    else:
        raise ValueError
    return h

def mask_h_mat(h, mask):
    nobs = h.shape[0]
    assert nobs == 36
    assert mask in ["atm", "ocn", None]
    if mask == "atm":
        return h[20:, :]
    elif mask == "ocn":
        return h[:20, :]
    return h

def get_h_comparison():
    nobs = 8
    h_mat = np.empty((nobs, NDIM))
    x = 2.2988646295052755
    y = 2.246833659321132
    for i in range(NDIM):
        state_unit = np.zeros(NDIM)
        state_unit[i] = 1.0
        elems = ["u", "v", "psi", "tmp", "u", "v", "psi", "tmp"]
        is_atms = [True] * 4 + [False] * 4
        for j in range(nobs):
            h_mat[j, i] = get_grid_val(state_unit, x, y, is_atms[j], elems[j])
    return h_mat

def plot_mat(mat):
    # plt.imshow(mat, cmap="RdBu_r")
    plt.imshow(mat, norm=matplotlib.colors.SymLogNorm(linthresh=0.1),
               cmap="RdBu_r")
    plt.colorbar()
    plt.savefig("tmp.png")
    plt.close()

def read_fort_txt_2d(filename):
    with open(filename, "r") as f:
        lines = list(f.readlines())
    nl = len(lines)
    nc = len(lines[0].split())
    mat = np.empty((nl, nc))
    for i in range(nl):
        l = lines[i].replace("D", "E").split()
        assert len(l) == nc
        lf = [float(x) for x in l]
        mat[i, :] = np.array(lf)
    return mat

if __name__ == "__main__":
    pass

