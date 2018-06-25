#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from module_obs_network import get_h_full_coverage

def get_r_luyu(name):
    n = 36
    r = np.zeros((n, n))
    assert name in ["Luyu", "Kriti"]
    with open(f"binary_const/clim_std_{name}_20180619.txt", "r") as f:
        for i in range(n):
            t = f.readline().strip().replace("D", "E")
            r[i, i] = float(t) ** 2
    return r

def get_h_b_ht():
    h = get_h_full_coverage()
    diag_b_10_percent = get_r_luyu("Luyu")
    hbht = h @ diag_b_10_percent @ h.T
    # ttk diag_r = np.array([1.0] * 10 + [1.0] * 10 + [0.01] * 8 + [0.1] * 8) ** 2
    return np.diag(np.diag(hbht))

def print_h_b_ht():
    cm = plt.imshow(get_h_b_ht(), norm=matplotlib.colors.SymLogNorm(linthresh=0.1 ** 8), cmap="RdBu_r")
    plt.colorbar(cm)
    plt.savefig("out.pdf")
    plt.close()
    print(np.diag(get_h_b_ht()) ** 0.5)

def compare_luyu_kriti():
    print(np.diag(get_r_luyu("Luyu")))
    print(np.diag(get_r_luyu("Kriti")))
    plt.semilogy(np.diag(get_r_luyu("Luyu")), label="Luyu")
    plt.semilogy(np.diag(get_r_luyu("Kriti")), label="Kriti")
    plt.legend()
    plt.savefig("out.pdf")
    plt.close()

if __name__ == "__main__":
    print_h_b_ht()
