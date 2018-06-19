#!/usr/bin/env python3

import numpy as np

def get_r_luyu(name):
    n = 36
    r = np.zeros((n, n))
    assert name in ["Luyu", "Kriti"]
    with open(f"binary_const/clim_std_{name}_20180619.txt", "r") as f:
        for i in range(n):
            t = f.readline().strip().replace("D", "E")
            r[i, i] = float(t) ** 2
    return r

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.plot(np.diag(get_r_luyu("Luyu")), label="Luyu")
    plt.plot(np.diag(get_r_luyu("Kriti")), label="Kriti")
    plt.legend()
    plt.savefig("out.pdf")
    plt.close()
