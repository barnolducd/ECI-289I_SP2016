# Brad Arnold
# ECI-289I-001, Spring 2016
# Homework 2.1 (Exploration vs. Exploitation)
# -------------------------------------------

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

# Define MatLab "Peaks" Function
def peaks(x):
  a = 3*(1-x[0])**2*np.exp(-(x[0]**2) - (x[1]+1)**2)
  b = 10*(x[0]/5 - x[0]**3 - x[1]**5)*np.exp(-x[0]**2-x[1]**2)
  c = (1/3)*np.exp(-(x[0]+1)**2 - x[1]**2)
  return a - b - c + 6.551 # add this so objective is always positive

ub = 3.0
lb = -3.0

d = 2 # Dimension - Decision Variable Space
m = 0 # Mean
s = 0.1 # Standard Deviation
prob = 0 # Probability
num_seeds = 5 # Number of Random Seeds
max_NFE = 100000
xt = np.zeros((2, max_NFE))
ft = np.zeros((num_seeds, max_NFE))

# Hill Climbing Algorithm
for seed in range(num_seeds):
  np.random.seed(seed)

  x = np.random.uniform(lb, ub, d)
  bestf = peaks(x)

  for i in range(max_NFE):
    sample_rand = np.random.uniform(0,10,1) # Random Value between 0 and 10.
    if sample_rand <= (prob*10): # If Random Value Less Than/Equal To Probability Defined
        # Sample Locally from Gaussian Distribution
        trial_x = np.random.normal(m,s,d)
    else:
        # Sample Uniformly from Full Domain
        trial_x = np.random.uniform(lb, ub, d)
    trial_f = peaks(trial_x)
    if trial_f < bestf:
      x = trial_x
      bestf = trial_f
    
    ft[seed,i] = bestf
    xt[:,i] = x

  print x
  print bestf

# Add Enumeration Line
enumy = np.arange(2.0,-3.0,-1.0)
enumx = (10**(-enumy)*(ub-lb))**d

plt.subplot(1,2,1)
xx = np.arange(lb,ub,0.01)
X1,X2 = np.meshgrid(xx, xx)
Z = peaks([X1,X2])
plt.contourf(X1,X2,Z,50,cmap=plt.cm.Blues_r)
plt.plot(xt[0,:], xt[1,:], color='k', linewidth=2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar()

plt.subplot(1,2,2)
plt.loglog(ft.T, color='steelblue', linewidth=1)
plt.loglog(enumx,10**enumy, color='indianred', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()



