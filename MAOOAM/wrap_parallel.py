#!/usr/bin/env python3

import sys
from util_parallel import Change, shell, exec_parallel
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

def main():
    wdir_base = sys.argv[1]
    params1 = list(reversed(range(3, 38, 2)))
    params2 = [1.0 + 0.01 * i for i in range(11)]
    changes1 = [Change("analysis_init.py", 79, "das.edim", "das.edim = %d")]
    changes2 = [Change("class_da_system.py", 420, "rho", "    rho = %f")]
    shell("mkdir -p %s/out" % wdir_base)
    res = exec_parallel(wdir_base, "template", params1, params2, "ens_%02d", "infl_%.02f",
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
        plt.xlabel("inflation rho")
        ax.set_xticks(range(n2))
        ax.set_xticklabels(params2)
        plt.ylabel("ensemble size")
        ax.set_yticks(range(n1))
        ax.set_yticklabels(params1)
        plt.savefig("out/rmse_%s.pdf" % names[ir])
        plt.close()

if __name__ == "__main__":
    main()

