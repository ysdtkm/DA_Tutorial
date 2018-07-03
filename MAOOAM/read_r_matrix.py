#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from module_obs_network import get_h

def read_diag_r(name):
    n = 36
    r = np.zeros((n, n))
    assert name in ["Luyu", "Kriti", "Cheng"]
    if name == "Cheng":
        fname = "binary_const/20180702_cheng_r.txt"
    else:
        fname = f"binary_const/clim_std_{name}_20180619.txt"
    with open(fname, "r") as f:
        for i in range(n):
            t = f.readline().strip().replace("D", "E")
            r[i, i] = float(t) ** 2
    return r

def get_b_clim_kriti():
    b = np.load("binary_const/20180629_cov_kriti_1e6tu.npy")
    assert b.shape == (36, 36)
    return b

def get_h_b_ht():
    h = get_h()
    b = get_b_clim_kriti()
    hbht = h @ b @ h.T * 0.01
    return np.diag(np.diag(hbht))

def compare_h_b_ht_diag():
    h = get_h()
    b = get_b_clim_kriti()
    hbht = h @ b @ h.T * 0.01
    hbhtd = h @ np.diag(np.diag(b)) @ h.T * 0.01
    plt.title("10% climatology variance stdv at obs location")
    plt.semilogy(np.diag(hbht) ** 0.5, label="true")
    plt.semilogy(np.diag(hbhtd) ** 0.5, label="with diagonalized Bclim")
    plt.xlabel("observation index")
    plt.ylabel("10% clim variance stdv by m/s or K")
    plt.legend()
    plt.savefig("out.pdf", bbox_inches="tight")
    plt.close()

def print_h_b_ht():
    cm = plt.imshow(get_h_b_ht(), norm=matplotlib.colors.SymLogNorm(linthresh=0.1 ** 8), cmap="RdBu_r")
    plt.colorbar(cm)
    plt.savefig("out.pdf")
    plt.close()
    print(np.diag(get_h_b_ht()) ** 0.5)

def compare_luyu_kriti():
    plt.semilogy(np.diag(read_diag_r("Luyu")), label="Luyu")
    plt.semilogy(np.diag(read_diag_r("Kriti")), label="Kriti")
    plt.semilogy(np.diag(read_diag_r("Cheng")), label="Cheng")
    plt.legend()
    plt.savefig("out.pdf")
    plt.close()

if __name__ == "__main__":
    compare_luyu_kriti()
