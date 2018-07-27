#!/usr/bin/env python3

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np
import module_obs_network
import IPython

def plot_r():
    # h = module_obs_network.get_h()
    h = module_obs_network.read_fort_txt_2d("binary_const/20180727_kriti_h.txt").T
    bclim = np.load("binary_const/20180629_cov_kriti_1e6tu.npy")

    luyu_r = module_obs_network.read_fort_txt_2d("binary_const/clim_std_Luyu_20180619.txt").flatten()

    kriti_r_obs_space = module_obs_network.read_fort_txt_2d("binary_const/20180727_kriti_r_in_obs_space.txt").flatten() ** 2
    kriti_hxxThT = module_obs_network.read_fort_txt_2d("binary_const/20180727_kriti_hxxtht.txt").flatten() ** 2

    r_with_diag = np.diag(h @ np.diag(luyu_r ** 2) @ h.T)
    r_no_diag = np.diag(h @ bclim * 0.01 @ h.T)
    r_one_side = (h @ luyu_r[:, None]).flatten() ** 2

    plt.semilogy(r_with_diag, label="Luyu's R (diagonal)")
    plt.semilogy(r_no_diag, label="hxxThT, mine")
    plt.semilogy(kriti_r_obs_space, label="kriti hrrhT")
    plt.semilogy(kriti_hxxThT, label="kriti_hxxThT")
    plt.semilogy(r_one_side, label="one side")
    plt.legend()
    plt.show()

    IPython.embed()

def get_kriti_r():
    h = module_obs_network.read_fort_txt_2d("binary_const/20180727_kriti_h.txt")
    ma = np.max(np.abs(h))
    norm = matplotlib.colors.SymLogNorm(linthresh=ma * 0.1 ** 16)
    cm = plt.imshow(h, norm=norm, cmap="RdBu_r")
    plt.colorbar(cm)
    plt.show()
    IPython.embed()

plot_r()
