#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import subprocess as sp
import itertools, functools

def main():
    sp.run("rm -rf out && mkdir -p out", shell=True, check=True)
    params1 = [0.001 * 10 ** (x / 10.0) for x in range(-4, 6)]
    params2 = list(range(1, 11))
    basedir = "/lustre/tyoshida/shrt/exec"
    exps = ["t0674", "t0675", "t0676", "t0677"]
    ress = {}
    for exp in exps:
        ress[exp] = get_rmses(f"{basedir}/{exp}", params1, params2, "sigma_b_%.06f", "none_%d")
        plot_rmse(ress[exp], exp, params1, params2, "sigma_b_%.06f")

def plot_rmse(res, exp, params1, params2, fmt1):
    cmps = ["atmos_psi", "atmos_temp", "ocean_psi", "ocean_temp", "all"]
    assert res.shape == (len(params1), len(params2), len(cmps))
    names = {"t0674":"tau = 10", "t0675":"tau = 1", "t0676":"tau = 100", "t0677":"tau = 1000"}

    for ic, c in enumerate(cmps):
        # plt.tight_layout()
        plt.rcParams["font.size"] = 16
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.16, top=0.9, left=0.16, right=0.95)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(0.0003, 0.004)
        plt.ylim(0.0008, 0.4)
        for ip2, p2 in enumerate(params2):
            plt.scatter(params1, res[:, ip2, ic], color="k", marker="x", alpha=0.3)
        plt.plot(params1, np.mean(res[:, :, ic], axis=1), alpha=0.8)
        ax.set_xticks([0.0003, 0.001, 0.003])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.title(names[exp])
        plt.xlabel("sqrt(a) = sqrt(spectral radius of B)")
        plt.ylabel("RMS error (nondimensional)")
        plt.savefig(f"out/{c}_{exp}.pdf", bb_inches="tight")
        plt.close("all")

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
        # print("%s available" % dname)
    except:
        res = np.empty(5)
        res[:] = np.nan
        print("%s failed" % dname)
    return res

if __name__ == "__main__":
    main()

