#!/usr/bin/env python3

import sys
import subprocess as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from class_state_vector import state_vector
from class_da_system import da_system

NDIM = 36

def plot_time_colormap(dat, img_name, vmin=None, vmax=None, title="", cmap="RdBu_r"):
    assert dat.__class__ == np.ndarray
    assert len(dat.shape) == 2
    assert dat.shape[1] == NDIM
    cm = plt.imshow(dat, aspect="auto", cmap=cmap, origin="bottom")
    if (vmin is not None) and (vmax is not None):
        cm.set_clim(vmin, vmax)
    plt.colorbar(cm)
    plt.xlabel("model variable")
    plt.ylabel("time")
    plt.title(title)
    plt.savefig(img_name)
    plt.close()

def plot_mean_bcov(bcov, img_name, title):
    cm = plt.imshow(bcov, cmap="RdBu_r")
    vmax = np.max(bcov)
    cm.set_clim(-vmax, vmax)
    plt.colorbar(cm)
    plt.xlabel("model variable")
    plt.ylabel("model variable")
    plt.title(title)
    plt.savefig(img_name)
    plt.close()

def cov_to_corr(cov):
    corr = np.copy(cov)
    n = cov.shape[0]
    for i in range(n):
        for j in range(n):
            corr[i, j] /= np.sqrt(cov[i, i] * cov[j, j])
    return corr

def get_bv_dim(cov):
    eigvals = np.maximum(0, np.real(np.linalg.eigvals(cov)))
    singvals = eigvals ** 0.5
    bv_dim = np.sum(singvals) ** 2 / np.sum(singvals ** 2)
    return bv_dim

def __test_plot_time_colormap():
    nt = 100
    dat = np.random.randn(nt, NDIM)
    sp.run("mkdir -p img", shell=True, check=True)
    plot_time_colormap(dat, "img/tmp.png", None, None, "test")

def __sample_read_files():
    vlim_raw = [-0.05, 0.1]
    vlim_diff = [None, None]
    nature_file ='x_nature.pkl'
    nature = state_vector()
    nature = nature.load(nature_file)
    freerun_file = 'x_freerun.pkl'
    freerun = state_vector()
    freerun = freerun.load(freerun_file)
    sp.run("mkdir -p img", shell=True, check=True)
    plot_time_colormap(freerun.getTrajectory() - nature.getTrajectory(),
                       "img/error_free_run.png", *vlim_diff, "error free run")
    plot_time_colormap(freerun.getTrajectory(),
                       "img/freerun.png", *vlim_raw, "freerun", "viridis")
    plot_time_colormap(nature.getTrajectory(),
                       "img/nature.png", *vlim_raw, "nature", "viridis")
    for method in ["ETKF"]:
        analysis_file = 'x_analysis_{method}.pkl'.format(method=method)
        das = da_system()
        das = das.load(analysis_file)
        analysis = das.getStateVector()
        plot_time_colormap(analysis.getTrajectory() - nature.getTrajectory(),
                           "img/error_analysis_%s.png" % method, *vlim_diff,
                           "error analysis %s" % method)
        plot_time_colormap(analysis.getTrajectory(),
                           "img/analysis_%s.png" % method, *vlim_raw,
                           "analysis %s" % method, "viridis")

def read_and_plot_bcov():
    Pb_hist = np.load("Pb_hist.npy")
    assert len(Pb_hist.shape) == 3
    assert Pb_hist.shape[1] == Pb_hist.shape[2]
    mean_cov = np.mean(Pb_hist, axis=0)
    counter = Pb_hist.shape[0]
    title = "mean B cov (sample = %d, BV dim = %f)" % (counter, get_bv_dim(mean_cov))
    plot_mean_bcov(mean_cov, "img/bcov.pdf", title)

    n = Pb_hist.shape[1]
    mean_corr = np.zeros((n, n))
    for t in range(counter):
        mean_corr += cov_to_corr(Pb_hist[t, :, :])
    title = "mean B corr (sample = %d, BV dim = %f)" % (counter, get_bv_dim(mean_corr))
    plot_mean_bcov(mean_corr, "img/bcorr.pdf", title)

if __name__ == "__main__":
    __sample_read_files()
    read_and_plot_bcov()
