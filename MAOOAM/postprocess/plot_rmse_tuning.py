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
    exps = ["t0658", "t0659", "t0660", "t0661"]
    names = {"t0658": "tau = 1000", "t0659": "tau = 100", "t0660": "tau = 10", "t0661": "tau = 1"}
    ress = {}
    for exp in exps:
        ress[exp] = exec_parallel(f"../{exp}", params1, params2, "sigma_b_%.06f", "none_%d")
    plot_reduced_rmse(params1, params2, ress, names)

def plot_reduced_rmse(params1, params2, ress, taus):
    n1, n2 = len(params1), len(params2)
    nr = 5
    names = ["atmos_psi", "atmos_temp", "ocean_psi", "ocean_temp", "all"]
    res_npy = np.empty((len(ress), nr, n1, n2))
    for ie, exp in enumerate(ress):
        for i1 in range(n1):
            for i2 in range(n2):
                res_npy[ie, :, i1, i2] = ress[exp][i1][i2]
    for ir in range(nr):
        plt.rcParams["font.size"] = 16
        plt.tight_layout()
        for ie, exp in enumerate(ress):
            plt.loglog(params1, res_npy[ie, ir, :, 0], label=taus[exp])
        plt.xlabel("sqrt(a) = sqrt(spectral radius of B)")
        plt.ylim([0.0005, 0.5])
        plt.ylabel("RMS error")
        plt.legend()
        plt.savefig("out/rmse_onedim_%s.pdf" % names[ir], bbox_inches="tight")
        plt.close()

def inverse_itertools_2d_product(params1, params2, map_result):
    # return 2D list, un-flattening the map_result
    n1 = len(params1)
    n2 = len(params2)
    res = [[map_result[i1 * n2 + i2] for i2 in range(n2)] for i1 in range(n1)]
    return res

def exec_parallel(wdir_base, params1, params2, p1_fmt, p2_fmt):
    params_prod = itertools.product(params1, params2)
    job = functools.partial(exec_single_job, wdir_base, p1_fmt, p2_fmt)
    res = list(map(job, params_prod))
    return inverse_itertools_2d_product(params1, params2, res)

def exec_single_job(wdir_base, p1_fmt, p2_fmt, param):
    p1, p2 = param
    s1 = (p1_fmt % p1).replace(".", "_")
    s2 = (p2_fmt % p2).replace(".", "_")
    dname = "%s/%s/%s" % (wdir_base, s1, s2)
    try:
        res = np.load(f"{dname}/rmse_hybrid.npy")
        print("%s done" % dname)
    except:
        res = np.empty(5)
        res[:] = np.nan
        print("%s failed" % dname)
    return res

if __name__ == "__main__":
    main()

