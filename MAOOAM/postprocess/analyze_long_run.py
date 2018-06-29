#!/usr/bin/env python3

from functools import lru_cache
import sys
import numpy as np
from tqdm import trange

NX = 36
NL = 10 ** 8  # 10 ** 8
MAX_VAL = 1.0
SRC = "/lustre/kritib/AOSC658/MAOOAM-DAS/kriti_fortfiles/long_run/fort.200"

@lru_cache(maxsize=1)
def get_mean_and_squared_mean(filepath):
    su = np.zeros(NX)
    ss = np.zeros(NX)
    with open(filepath, "r") as f:
        for i in trange(NL, desc="get_mean", ascii=True, disable=(not sys.stdout.isatty())):
            li = f.readline()
            na = np.fromstring(li.replace("D", "E"), dtype=float, sep=" ")
            assert na.shape == (NX,)
            su += na
            ss += na ** 2
    me = su / NL
    ms = ss / NL
    assert np.max(np.abs(me)) < MAX_VAL
    assert np.max(ms) < MAX_VAL ** 2
    return me, ms

def get_clim_cov(filepath):
    me, ms = get_mean_and_squared_mean(filepath)
    cov = np.zeros((NX, NX))
    with open(filepath, "r") as f:
        for i in trange(NL, desc="get_cov ", ascii=True, disable=(not sys.stdout.isatty())):
            li = f.readline()
            na = np.fromstring(li.replace("D", "E"), dtype=float, sep=" ")
            assert na.shape == (NX,)
            anom = na - me
            cov += anom_to_cov(anom)
    cov /= NL
    assert np.max(cov) < MAX_VAL ** 2
    return cov

def test_get_clim_cov():
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    cov = get_clim_cov(SRC)
    me, ms = get_mean_and_squared_mean(SRC)
    assert np.allclose(np.diag(cov), ms - me ** 2)

    ma = np.max(np.abs(cov))
    norm = matplotlib.colors.SymLogNorm(linthresh=ma * 0.1 ** 6)
    cm = plt.imshow(cov, norm=norm, cmap="RdBu_r")
    plt.colorbar(cm)
    plt.savefig("tmp.pdf", bbox_inches="tight")
    plt.close()

def save_mean_stdv_clim_cov():
    me, ms = get_mean_and_squared_mean(SRC)
    cov = get_clim_cov(SRC)
    std = (ms - me ** 2) ** 0.5
    assert np.allclose(np.diag(cov) ** 0.5, std)
    np.save("mean.npy", me)
    np.save("stdv.npy", std)
    np.save("cov.npy", cov)
    print("mean:\n", me)
    print("stdv:\n", std)

def anom_to_cov(anom):
    cov = anom[:, None] @ anom[None, :]
    assert cov.shape == (NX, NX)
    return cov

if __name__ == "__main__":
    save_mean_stdv_clim_cov()


