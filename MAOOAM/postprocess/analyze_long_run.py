#!/usr/bin/env python3

from functools import lru_cache
import sys
import unittest
import numpy as np
from tqdm import trange

NX = 36
NL = 10 ** 4  # 10 ** 8
MAX_VAL = 1.0
SRC = "/lustre/kritib/AOSC658/MAOOAM-DAS/kriti_fortfiles/long_run/fort.200"

def get_mean_and_cov(filepath):
    # a51p7
    su = np.zeros(NX)
    dot = np.zeros((NX, NX))
    with open(filepath, "r") as f:
        for i in trange(NL, desc="get", ascii=True, disable=(not sys.stdout.isatty())):
            li = f.readline()
            na = np.fromstring(li.replace("D", "E"), dtype=float, sep=" ")
            assert na.shape == (NX,)
            su += na
            dot += na[:, None] @ na[None, :]
    me = su / NL
    cov = dot / NL - me[:, None] @ me[None, :]
    return me, cov

def save_mean_stdv_clim_cov():
    me, cov = get_mean_and_cov(SRC)
    np.save("mean.npy", me)
    np.save("cov.npy", cov)
    print("mean:\n", me)
    print("stdv:\n", np.diag(cov) ** 0.5)

class TestAnalyzeLongRun(unittest.TestCase):
    def test_get_mean_and_cov(self):
        import matplotlib
        matplotlib.use("pdf")
        import matplotlib.pyplot as plt
        me, cov = get_mean_and_cov(SRC)
        ma = np.max(np.abs(cov))
        norm = matplotlib.colors.SymLogNorm(linthresh=ma * 0.1 ** 6)
        cm = plt.imshow(cov, norm=norm, cmap="RdBu_r")
        plt.colorbar(cm)
        plt.savefig("tmp.pdf", bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    save_mean_stdv_clim_cov()
