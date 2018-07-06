#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from module_obs_network import get_h
from exp_params import FLAG_R

def read_diag_r(name):
    # return R matrix in spectral space, which is a diag matrix (10% clim stdv) ** 2
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

def get_r(flag=FLAG_R):
    assert flag in ["hbht", "Luyu", "Kriti", "Cheng", "identity"]
    h = get_h()
    p = h.shape[0]
    if flag == "hbht":
        b = get_b_clim_kriti()
        hbht = h @ b @ h.T * 0.01
        r = np.diag(np.diag(hbht))
    elif flag in ["Luyu", "Kriti", "Cheng"]:
        r = read_diag_r(flag)
        r *= 100  # ttk
        assert r.shape == (p, p)
    elif flag == "identity":
        sigma = 0.001
        r = np.identity(p) * sigma ** 2
    else:
        raise ValueError
    return r

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
    cm = plt.imshow(get_r(), norm=matplotlib.colors.SymLogNorm(linthresh=0.1 ** 8), cmap="RdBu_r")
    plt.colorbar(cm)
    plt.savefig("out.pdf")
    plt.close()
    print(np.diag(get_r()) ** 0.5)

def compare_r():
    plt.semilogy(np.diag(get_r("hbht")), label="H B Ht")
    plt.semilogy(np.diag(get_r("Luyu")), label="Luyu")
    plt.semilogy(np.diag(get_r("Kriti")), label="Kriti old")
    plt.semilogy(np.diag(get_r("Cheng")), label="Cheng")
    plt.legend()
    plt.savefig("out.pdf")
    plt.close()

if __name__ == "__main__":
    compare_r()
