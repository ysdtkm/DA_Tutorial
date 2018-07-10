#!/usr/bin/env python3

import sys
import unittest
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from class_state_vector import state_vector
from class_da_system import da_system
from exp_params import SEED, SKIP_HEAVY_PLOT

NDIM = 36

def plot_time_colormap(dat, img_name, vmin=None, vmax=None, title="", cmap="RdBu_r", log=False):
    assert dat.__class__ == np.ndarray
    assert len(dat.shape) == 2
    assert dat.shape[1] == NDIM
    plt.rcParams["font.size"] = 16
    norm = matplotlib.colors.SymLogNorm(linthresh=0.001 * vmax) if log else None
    cm = plt.imshow(dat, aspect="auto", cmap=cmap, origin="bottom", norm=norm,
        extent=get_extent_bottom_origin(dat))
    if (vmin is not None) and (vmax is not None):
        cm.set_clim(vmin, vmax)
    plt.colorbar(cm)
    plt.xlabel("model variable")
    plt.ylabel("time")
    plt.title(title)
    plt.savefig(img_name, bbox_inches="tight")
    plt.close()

def plot_mean_bcov(bcov, img_name, title, log=False):
    plt.rcParams["font.size"] = 16
    vmax = 1.0
    norm = matplotlib.colors.SymLogNorm(linthresh=0.001 * vmax) if log else None
    cm = plt.imshow(bcov, cmap="RdBu_r", norm=norm, extent=get_extent_square())
    cm.set_clim(-vmax, vmax)
    plt.colorbar(cm)
    plt.xlabel("model variable")
    plt.ylabel("model variable")
    plt.title(title)
    plt.savefig(img_name, bbox_inches="tight")
    plt.close()

def plot_eig_bcov(bcov, img_name_eigval, img_name_eigvec, intvl):
    plt.rcParams["font.size"] = 16
    eigval, eigvec = np.linalg.eigh(bcov)
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]

    xs = np.arange(1, len(eigval) + 1)
    plt.plot(xs, eigval)
    plt.yscale("log")
    plt.title(r"eigenvalues of $B_0$ ($\tau$ = %d)" % intvl)
    plt.ylim(10 ** (-15), 10 ** (-3))
    plt.xlabel("eigenvector index")
    plt.ylabel("eigenvalues")
    plt.savefig(img_name_eigval, bbox_inches="tight")
    plt.close()

    cm = plt.imshow(eigvec, cmap="RdBu_r", extent=get_extent_square())
    cm.set_clim(-1, 1)
    plt.colorbar(cm)
    plt.title(r"eigenvectors of $B$ ($\tau$ = %d)" % intvl)
    plt.xlabel("eigenvector index")
    plt.ylabel("model variable")
    plt.savefig(img_name_eigvec, bbox_inches="tight")
    plt.close()

def get_extent_square():
    # left, right, bottom, top
    return [0.5, NDIM + 0.5, NDIM + 0.5, 0.5]

def get_extent_bottom_origin(data):
    sp = data.shape
    assert len(sp) == 2
    return [0.5, sp[1] + 0.5, 0.5, sp[0] + 0.5]

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

def zero_out_off_diag_blocks(cov):
    assert cov.shape == (NDIM, NDIM)
    NATM = 20
    cov2 = np.copy(cov)
    cov2[:NATM, NATM:] = 0.0
    cov2[NATM:, :NATM] = 0.0
    return cov2

def plot_all(methods):
    vlim_raw = [-0.05, 0.1]
    vlim_diff = [-0.15, 0.15]
    nature_file ='x_nature.pkl'
    nature = state_vector()
    nature = nature.load(nature_file)
    freerun_file = 'x_freerun.pkl'
    freerun = state_vector()
    freerun = freerun.load(freerun_file)
    if not SKIP_HEAVY_PLOT:
        plot_time_colormap(freerun.getTrajectory() - nature.getTrajectory(),
                           "img/error_free_run.pdf", *vlim_diff, "error free run", "RdBu_r", True)
        plot_time_colormap(freerun.getTrajectory(),
                           "img/freerun.pdf", *vlim_raw, "freerun", "viridis")
        plot_time_colormap(nature.getTrajectory(),
                           "img/nature.pdf", *vlim_raw, "nature", "viridis")
    for method in methods:
        analysis_file = 'x_analysis_{method}.pkl'.format(method=method)
        das = da_system()
        das = das.load(analysis_file)
        analysis = das.getStateVector()
        if not SKIP_HEAVY_PLOT:
            plot_time_colormap(analysis.getTrajectory() - nature.getTrajectory(),
                               "img/%s/error_analysis.pdf" % method, *vlim_diff,
                               "error analysis %s" % method, "RdBu_r", True)
            plot_time_colormap(analysis.getTrajectory(),
                               "img/%s/analysis.pdf" % method, *vlim_raw,
                               "analysis %s" % method, "viridis")

def read_and_plot_bcov():
    Pb_hist = np.load("Pb_hist.npy")
    assert len(Pb_hist.shape) == 3
    assert Pb_hist.shape[1] == Pb_hist.shape[2]

    counter = Pb_hist.shape[0]
    me, sd, fl = cov_to_mean_and_std(Pb_hist, False)
    srad = np.max(np.linalg.eigvalsh(me))
    assert srad > 0.0
    np.save("mean_b_cov.npy", me)
    plot_mean_bcov(me / srad, "img/bcov.pdf", f"B cov (sample = {counter // 2})", True)
    plot_mean_bcov(sd / srad, "img/std_cov.pdf", "Stdv of ens cov")
    plot_mean_bcov(fl, "img/flow_dependency.pdf", "Flow dependency of ens cov")
    plot_eig_bcov(me, "img/bcov_eigval.pdf", "img/bcov_eigvec.pdf", 0)

def cov_to_mean_and_std(pb_hist, to_corr=False):
    assert len(pb_hist.shape) == 3
    nt = pb_hist.shape[0]
    mat_hist = np.empty_like(pb_hist)
    for i in range(nt):
        if to_corr:
            mat_hist[i, :, :] = cov_to_corr(pb_hist[i, :, :])
        else:
            mat_hist[i, :, :] = pb_hist[i, :, :]
    mat_hist_cut = mat_hist[nt // 2:, :, :]  # discard formar half as spinup
    mean_mat = np.mean(mat_hist_cut, axis=0)
    stdv_mat = np.std(mat_hist_cut, axis=0)
    rms_mat = np.mean(mat_hist_cut ** 2, axis=0) ** 0.5
    assert np.all(rms_mat > 0.0)
    flow_dependence = stdv_mat ** 2 / rms_mat ** 2
    eps = 1.0e-8
    assert np.all(flow_dependence < 1.0 + eps)
    return mean_mat, stdv_mat, flow_dependence

def read_and_plot_mean_bcov():
    for intvl, name in [(1, "t0644"), (10, "t0645"), (100, "t0646"), (1000, "t0647")]:
        mean_cov = np.load(f"binary_const/mean_b_cov_{name}_{intvl:04d}tu.npy")
        assert len(mean_cov.shape) == 2
        assert mean_cov.shape[0] == mean_cov.shape[1] == NDIM
        title = r"normalized $B$ ($\tau$ = %d)" % intvl
        plot_mean_bcov(mean_cov / np.max(np.linalg.eigvalsh(mean_cov)), "img/bcov_%05d.pdf" % intvl, title, True)
        plot_eig_bcov(mean_cov, "img/bcov_%05d_eigval.pdf" % intvl, "img/bcov_%05d_eigvec.pdf" % intvl, intvl)

class TestPlot(unittest.TestCase):
    def test_plot_time_colormap(self):
        np.random.seed(SEED * 6)
        nt = 100
        dat = np.random.randn(nt, NDIM)
        plot_time_colormap(dat, "tmp.pdf", None, None, "test")

def main():
    np.set_printoptions(formatter={'float': '{: 10.6g}'.format}, threshold=2000, linewidth=150)
    plot_all(sys.argv[1:])
    if sys.argv[1] in ["ETKF", "hybrid"]:
        read_and_plot_bcov()

if __name__ == "__main__":
    main()
