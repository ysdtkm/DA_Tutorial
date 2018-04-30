#!/usr/bin/env python3

import sys
from util_parallel import Change, shell, exec_parallel
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

def main():
    wdir_base = sys.argv[1]
    params1 = list(reversed(range(3, 38, 17)))
    params2 = [1.0 + 0.05 * i for i in range(3)]
    changes1 = [Change("analysis_init.py", 79, "das.edim", "das.edim = %d")]
    changes2 = [Change("class_da_system.py", 420, "rho", "    rho = %f")]
    shell("mkdir -p %s/out" % wdir_base)
    res = exec_parallel(wdir_base, "template", params1, params2, "ens_%02d", "infl_%.02f",
                  changes1, changes2, "make", "out.pdf")
    print(list(res))

def plot_reduced_rmse(params1, params2, res):
    n1, n2 = len(params1), len(params2)
    nr = len(res[0][0])
    res_npy = np.empty((nr, n1, n2))
    for i1 in range(n1):
        for i2 in range(n2):
            res_npy[:, i1, i2] = res[i1][i2]
    for ir in range(nr):
        cm = plt.imshow(res_npy[ir, :, :])
        plt.colorbar(cm)
        plt.savefig("rmse.pdf")
        plt.close()

if __name__ == "__main__":
    main()

