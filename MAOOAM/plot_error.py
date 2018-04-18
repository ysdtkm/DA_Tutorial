from class_state_vector import state_vector
from class_da_system import da_system
import numpy as np
from sys import argv
import matplotlib.pyplot as plt

def main():
    nature_file ='x_nature.pkl'
    nature = state_vector()
    nature = nature.load(nature_file)
    freerun_file = 'x_freerun.pkl'
    freerun = state_vector()
    freerun = freerun.load(freerun_file)
    method = argv[1]
    analysis_file = 'x_analysis_{method}.pkl'.format(method=method)
    das = da_system()
    das = das.load(analysis_file)
    analysis = das.getStateVector()
    plot_rmse_all(nature, freerun, analysis, method, np.s_[:, :], "img/rmse_all.png")
    plot_rmse_all(nature, freerun, analysis, method, np.s_[:, 0:10], "img/rmse_atmos_psi.png")
    plot_rmse_all(nature, freerun, analysis, method, np.s_[:, 10:20], "img/rmse_atmos_temp.png")
    plot_rmse_all(nature, freerun, analysis, method, np.s_[:, 20:28], "img/rmse_ocean_psi.png")
    plot_rmse_all(nature, freerun, analysis, method, np.s_[:, 28:36], "img/rmse_ocean_temp.png")

def plot_rmse_all(nature, freerun, analysis, method, slice, img_name):
    plt.plot(nature.getTimes(),
             np.linalg.norm(freerun.getTrajectory()[slice] - nature.getTrajectory()[slice],
                            axis=1), label='Free run')
    plt.plot(nature.getTimes(),
             np.linalg.norm(analysis.getTrajectory()[slice] - nature.getTrajectory()[slice],
                            axis=1), label='Analysis ({method})'.format(method=method))
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Error', rotation='horizontal', labelpad=20)
    plt.title(img_name)
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()

main()
