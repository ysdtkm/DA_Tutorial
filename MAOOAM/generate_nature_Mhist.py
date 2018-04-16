# Tutorial: "A tour of Data Assimilation methods"
# Model: MAOOAM
# DA Methods: Nudging, 3D-Var, 4D-Var, Particle Filter, EnKF, Hybrid
import numpy as np
from class_maooam import maooam
from class_state_vector import state_vector
from class_obs_data import obs_data
from class_da_system import da_system

#-----------------------------------------------------------------------
# Exercises:
# (1) Compute the Lyapunov exponents of the nature system
# (2) Compute the forward and backward Lyapunov vectors
# (3) Compute the Covariant Lyapunov Vectors
#-----------------------------------------------------------------------


#-----------------------------------------------------------------------
# Read the MAOOAM nature run
#-----------------------------------------------------------------------
name = 'x_nature'
infile = name+'.pkl'
sv = state_vector()
sv = sv.load(infile)
x_nature = sv.getTrajectory()
t_nature = sv.getTimes()
maxit,xdim = np.shape(x_nature)
# sigma,rho,beta = sv.params

#------------------------------------------------------------------
# Initialize the MAOOAM object
#------------------------------------------------------------------
model = maooam()

#------------------------------------------------------------------
# Input a nature run and compute its corresponding Jacobian
#------------------------------------------------------------------
print('Compute approximate Tangent Linear Model (TLM) at each timestep...')
Mhist = model.compute_TLMa(x_nature,t_nature)
sv.setMhist(Mhist)

#------------------------------------------------------------------
# Store the nature run with corresponding history of the Jacobian
#------------------------------------------------------------------
outfile = name+'_Mhist.pkl'
sv.save(outfile)
