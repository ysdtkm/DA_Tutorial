import numpy as np
from class_state_vector import state_vector
from class_obs_data import obs_data
from module_obs_network import get_h_full_coverage, NDIM
from exp_params import SEED
from read_r_matrix import get_r_luyu, get_h_b_ht

infile = 'x_nature.pkl'
outfile = 'y_obs.pkl'

#--------------------------------------------------------------------------------
# Define observation bias (mu) and error (sigma)
# Note: sigma will be multiplied by nature run climatological standard deviation
#--------------------------------------------------------------------------------
mu = 0
sigma = 0.001
np.random.seed(SEED * 2)

#--------------------------------------------------------------------------------
# Create observation object
#--------------------------------------------------------------------------------
obs = obs_data(name='observe_full_state', mu_init=mu, sigma_init=sigma)

#--------------------------------------------------------------------------------
# Read the nature run
#--------------------------------------------------------------------------------
sv = state_vector()
sv = sv.load(infile)
x_nature = sv.getTrajectory()

nr,nc = x_nature.shape
print('nr = ',nr)
print('nc = ',nc)

#--------------------------------------------------------------------------------
# Compute the climatological variance
#--------------------------------------------------------------------------------
x_std = np.zeros(nc)
for i in range(nc):
  x_std[i] = np.std(x_nature[:,i])

#--------------------------------------------------------------------------------
# Sample the nature run and apply noise
#--------------------------------------------------------------------------------
H = get_h_full_coverage()
nobs = H.shape[0]
R = get_h_b_ht()
assert R.shape == (nobs, nobs)
yo = np.zeros((nr, nobs))
eta = np.zeros((nr, nobs))
hx = np.zeros((nr, nobs))
for i in range(nobs):
  # eta[:, i] = np.random.normal(mu, sigma, nr)
  eta[:, i] = np.random.normal(mu, R[i, i] ** 0.5, nr)
for j in range(nr):
  # (Could apply H(x_nature) here):
  hx[j, :] = H @ x_nature[j, :]
  # Observation is simulated as H(x_nature) + noise
  yo[j, :] = hx[j, :] + eta[j, :]

pos = np.zeros_like(yo)
for i in range(nr):
  pos[i, :] = list(range(0, nobs)) #[0,1,2]

obs.setVal(yo)
obs.setErr(eta)
obs.setHx(hx)
obs.setDep(yo-hx)
obs.setPos(pos)

print(obs)

#--------------------------------------------------------------------------------
# Store the true and noisy observations
#--------------------------------------------------------------------------------
obs.save(outfile)

