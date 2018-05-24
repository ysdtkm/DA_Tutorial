#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess as sp
import itertools, functools

def main():
    params1 = [0.001 * 10 ** (x / 20.0) for x in range(-9, 11)]
    params2 = [1]
    basedir = "/lustre/tyoshida/shrt/exec"
    exps = ["t0658", "t0659", "t0660", "t0661"]
    ress = {}
    for exp in exps:
        ress[exp] = get_rmses(f"{basedir}/{exp}", params1, params2, "sigma_b_%.06f", "none_%d")

    names = {"t0658": "tau = 1000", "t0659": "tau = 100", "t0660": "tau = 10", "t0661": "tau = 1"}
    print(ress["t0658"].shape)

def get_rmses(wdir_base, params1, params2, p1_fmt, p2_fmt):
    job = functools.partial(get_rmse, wdir_base, p1_fmt, p2_fmt)
    res = [[job(p1, p2) for p2 in params2] for p1 in params1]
    return np.array(res)

def get_rmse(wdir_base, p1_fmt, p2_fmt, p1, p2):
    s1 = (p1_fmt % p1).replace(".", "_")
    s2 = (p2_fmt % p2).replace(".", "_")
    dname = "%s/%s/%s" % (wdir_base, s1, s2)
    try:
        res = np.load(f"{dname}/rmse_hybrid.npy")
        print("%s available" % dname)
    except:
        res = np.empty(5)
        res[:] = np.nan
        print("%s failed" % dname)
    return res

if __name__ == "__main__":
    main()

