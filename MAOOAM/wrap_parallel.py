#!/usr/bin/env python3

import sys
from util_parallel import Change, shell, exec_parallel
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

def main():
    wdir_base = sys.argv[1]
    params1 = [0.001 * 10 ** (x / 5.0) for x in range(-1, 5)]
    params2 = [1]
    changes1 = [Change("analysis_init.py", 99, "sigma_b", "sigma_b = %f")]
    changes2 = []
    shell("mkdir -p %s/out" % wdir_base)
    res = exec_parallel(wdir_base, "template", params1, params2, "sigma_b_%.06f", "none_%d",
                        changes1, changes2, "make", "out.pdf")
    plot_reduced_rmse(params1, params2, res)

def plot_reduced_rmse(params1, params2, res):
    n1, n2 = len(params1), len(params2)
    nr = len(res[0][0])
    res_npy = np.empty((nr, n1, n2))
    names = ["atmos_psi", "atmos_temp", "ocean_psi", "ocean_temp", "all"]
    for i1 in range(n1):
        for i2 in range(n2):
            res_npy[:, i1, i2] = res[i1][i2]
    for ir in range(nr):
        fig, ax = plt.subplots()
        cm = plt.imshow(res_npy[ir, :, :])
        # cm.set_clim(0, 0.05)
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
        plt.ylabel("RMS error")
        plt.savefig("out/rmse_onedim_%s.pdf" % names[ir])
        plt.close()

if __name__ == "__main__":
    main()

