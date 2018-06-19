#!/usr/bin/env python3

import numpy as np

def get_r_luyu():
    n = 36
    r = np.zeros((n, n))
    with open("binary_const/R_matrix_Luyu_20180619.txt", "r") as f:
        for i in range(n):
            t = f.readline().strip().replace("D", "E")
            r[i, i] = (float(t) * 0.1) ** 2
    return r

if __name__ == "__main__":
    print(get_r_luyu())
