# Brad Arnold
# ECI-289I-001, Spring 2016
# Homework 3.2 (Differential Evolution)
# -------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Define Rosenbrock Function
def rosenbrock(x):
  return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

ub = 3.0
lb = -3.0

d = 2 # Dimension of Decision Variable Space
num_seeds = 10

popsize = 10
CR = 0.9 # Crossover Probability
F = 0.9
max_NFE = 10000
ft = np.zeros((num_seeds, max_NFE/popsize))

# Differential Evolution (Simple Version)
for seed in range(num_seeds):
  np.random.seed(seed)

  # Random Initial Population (popsize x d matrix)
  P = np.random.uniform(lb, ub, (popsize,d))
  f = np.zeros(popsize)
  nfe = 0
  f_best, x_best = None, None

  while nfe < max_NFE:

    for i,x in enumerate(P):

      # pick two random population members
      # "x" will be the one we modify, but other variants
      # will always modify the current best solution instead
      xb,xc = P[np.random.randint(0, popsize, 2), :]
      trial_x = np.copy(x)

      # for each dimension ..
      for j in range(d):
        if np.random.rand() < CR:
          trial_x[j] = x[j] + F*(xb[j]-xc[j])

      f[i] = rosenbrock(x)
      trial_f = rosenbrock(trial_x)
      nfe += 1

      # if this is better than the parent, replace
      if trial_f < f[i]:
        P[i,:] = trial_x
        f[i] = trial_f

    # keep track of best here
    if f_best is None or f.min() < f_best:
      f_best = f.min()
      x_best = P[f.argmin(),:]

    ft[seed,nfe/popsize-1] = f_best

  # for each trial print the result (but the traces are saved in ft)
  print x_best
  print f_best

# Define Enumeration Line
enumy = np.arange(1.0,-4.0,-1.0)
enumx = (10**(-enumy)*(ub-lb))**d

plt.loglog(range(popsize,max_NFE+1,popsize), ft.T, color='steelblue', linewidth=1)
plt.loglog(enumx,10**enumy, color='indianred', linewidth=2)

plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()