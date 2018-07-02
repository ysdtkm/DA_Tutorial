import numpy as np
from class_maooam import maooam
from class_state_vector import state_vector
from class_obs_data import obs_data
from class_da_system import da_system
from module_obs_network import get_h
from module_constants import get_x_std, get_static_b
from sys import argv
from exp_params import SEED
from read_r_matrix import get_r_luyu, get_h_b_ht

#-----------------------------------------------------------------------
# Usage:
# python analysis_init.py {method}
#
# where {method} is any one of:
#  skip
#  nudging
#  OI
#  3DVar
#  ETKF
#  PF
#  Hybrid
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# Read the MAOOAM nature run
#-----------------------------------------------------------------------
np.random.seed(SEED * 3)
infile = 'x_nature.pkl'
sv = state_vector()
sv = sv.load(infile)
x_nature = sv.getTrajectory()
maxit,xdim = np.shape(x_nature)

#-----------------------------------------------------------------------
# Read the MAOOAM observations
#-----------------------------------------------------------------------
infile = 'y_obs.pkl'
obs = obs_data()
obs = obs.load(infile)

#-----------------------------------------------------------------------
# Try reducing the observed dimensions
#-----------------------------------------------------------------------
yp = list(range(0,xdim)) #[0,1,2]
# Example for Lorenz-63:
#yp = [0]    # x only
#yp = [1]    # y only
#yp = [2]    # z only
#yp = [0,1]  # x and y only
#yp = [1,2]  # y and z only
#yp = [0,2]  # z and x only
if len(yp) < xdim:
  obs.reduceDim(yp)

y_obs = obs.getVal()
y_pts = obs.getPos()
y_err = obs.getErr()
print('y_obs = ')
print(y_obs[0,:])
print('y_pts = ')
print(y_pts[0,:])

_,ydim = np.shape(y_obs)

#-----------------------------------------------------------------------
# Initialize the da system
#-----------------------------------------------------------------------
alpha = 0.1
das = da_system(alpha=alpha)
das.setStateVector(sv)
das.setObsData(obs)
das.xdim = xdim
das.ydim = ydim
das.x0 = x_nature[0,:]
das.t = sv.getTimes()
das.t0 = das.t[0]

#-----------------------------------------------------------------------
# Initialize the ensemble
#-----------------------------------------------------------------------
das.edim = 37
das.ens_bias_init = 0
das.ens_sigma_init = 0.0001
das.x0 += np.random.randn(xdim) * das.ens_sigma_init  # truth is like an ensemble member

#-----------------------------------------------------------------------
# Initialize 4D-Var parameters
#-----------------------------------------------------------------------
das.outer_loops = 3

#-----------------------------------------------------------------------
# Initialize the error covariances B and R, and the linearized 
# observation operator H
#-----------------------------------------------------------------------

I = np.identity(xdim)

# Set background error covariance
sigma_b = 0.002
B = get_static_b() * sigma_b ** 2

# Set the linear observation operator matrix as the identity by default 
H = get_h()

# Set observation error covariance
nobs = H.shape[0]
R = get_h_b_ht()

# Set constant matrix for nudging
const = 0.00003
C = np.linalg.pinv(H) * const

das.setB(B)
das.setR(R)
das.setH(H)
das.setC(C)

# Update the matrices to fit the reduced observation dimension
if len(yp) < xdim:
  das.reduceYdim(yp)

print('B = ')
print(das.getB())
print('R = ')
print(das.getR())
print('H = ')
print(das.getH())


#-----------------------------------------------------------------------
# Initialize the timesteps
#-----------------------------------------------------------------------
t_nature = sv.getTimes()
acyc_step = 5                        # (how frequently to perform an analysis)
dtau = (t_nature[acyc_step] - t_nature[0])
fcst_step = acyc_step                      # (may need to change for 4D DA methods)
fcst_dt = dtau / fcst_step
maxit,xdim = np.shape(x_nature)

# Store basic timing info in das object
das.acyc_step = acyc_step
das.dtau = dtau
das.fcst_step = fcst_step
das.fcst_dt = fcst_dt
das.dt = (t_nature[1] - t_nature[0])
das.maxit = maxit
das.xdim = xdim

#-----------------------------------------------------------------------
# Choose DA method:
#-----------------------------------------------------------------------

method = argv[1]

#-----------
# Test basic functionality
#-----------
#method='skip'

#-----------
# 3D methods
#-----------
# Nudging
#method='nudging'
# OI
#method='OI'
# 3D-Var
#method='3DVar'

#-----------
# Ensemble methods
#-----------
# EnKF
#method='ETKF'
# Particle filter
#method='PF'

#-----------
# 4D methods
#-----------
# 4D-Var
#method='4DVar'
# 4DEnVar
#method='4DEnVar'
# 4DETKF
#method='4DETKF'

#-----------
# Hybrid methods
#-----------
# Hybrid-Gain
#method='Hybrid'

das.setMethod(method)

#-----------------------------------------------------------------------
# Store DA object
#-----------------------------------------------------------------------
name = 'x_analysis_init'
outfile=name+'.pkl'
das.save(outfile)

print(das)
