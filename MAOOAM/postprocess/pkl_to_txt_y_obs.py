#!/usr/bin/env python3

import pickle

with open("y_obs.pkl", "rb") as f:
    yo = pickle.load(f)

nt, nx = yo.val.shape
assert (nt, nx) == (20000, 36)

with open("y_obs_fail.txt", "w") as f:
    for i in range(nt):
        for j in range(nx):
            f.write(str(yo.val[i, j]).replace("e", "D"))
            f.write(" ")
        f.write("\n")
