#!/usr/bin/env python3

import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from util_parallel import Change, shell, exec_parallel

def main_hybrid():
    wdir_base = sys.argv[1]
    params1 = [0.0, 0.1, 0.2, 0.5]
    params2 = list(range(2, 38))
    changes1 = [Change("analysis_init.py", 70, "alpha", "alpha = %f")]
    changes2 = [Change("analysis_init.py", 83, "das.edim", "das.edim = %d")]
    shell("mkdir -p %s/out" % wdir_base)
    res = exec_parallel(wdir_base, "template", params1, params2, "alpha_%.02f", "ens_%02d",
                        changes1, changes2, "make", "out.pdf")
    plot_reduced_rmse(params1, params2, res)

def main_3dvar():
    wdir_base = sys.argv[1]
    params1 = list(np.geomspace(0.001, 0.1, 20))
    params2 = [1]
    changes1 = [Change("analysis_init.py", 101, "sigma_b", "sigma_b = %f")]
    changes2 = []
    shell("mkdir -p %s/out" % wdir_base)
    res = exec_parallel(wdir_base, "template", params1, params2, "sigma_b_%.06f", "none_%d",
                        changes1, changes2, "make", "out.pdf")
    plot_reduced_rmse(params1, params2, res)

def plot_reduced_rmse(params1, params2, res):
    if params1 is params2 is res is None:
        with open("plot_reduced_rmse.pkl", "rb") as f:
            params1, params2, res = pickle.load(f)
    else:
        with open("plot_reduced_rmse.pkl", "wb") as f:
            pickle.dump([params1, params2, res], f)
    n1, n2 = len(params1), len(params2)
    nr = len(res[0][0])
    res_npy = np.empty((nr, n1, n2))
    names = ["atmos_psi", "atmos_temp", "ocean_psi", "ocean_temp", "all"]
    for i1 in range(n1):
        for i2 in range(n2):
            res_npy[:, i1, i2] = res[i1][i2]
    for ir in range(nr):
        fig, ax = plt.subplots()
        cm = plt.imshow(res_npy[ir, :, :], cmap="Reds")
        cm.set_clim(0, 0.2)
        plt.colorbar(cm)
        plt.xlabel("none")
        ax.set_xticks(range(n2))
        ax.set_xticklabels(params2, rotation=90)
        plt.ylabel("sigma_b")
        ax.set_yticks(range(n1))
        ax.set_yticklabels(params1)
        plt.savefig("out/rmse_%s.pdf" % names[ir])
        plt.close()

        if n2 >= 2:
            continue
        plt.loglog(params1, res_npy[ir, :, 0])
        plt.xlabel("sigma_b")
        # plt.ylim([0, 0.01])
        plt.ylabel("RMS error")
        plt.savefig("out/rmse_onedim_%s.pdf" % names[ir])
        plt.close()

if __name__ == "__main__":
    main_3dvar()

