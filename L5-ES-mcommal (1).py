# Brad Arnold
# ECI-289I-001, Spring 2016
# Homework 3.1 (Evolution Strategies)
# ------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Define MatLab "Peaks" Function
def peaks(x):
  a = 3*(1-x[0])**2*np.exp(-(x[0]**2) - (x[1]+1)**2)
  b = 10*(x[0]/5 - x[0]**3 - x[1]**5)*np.exp(-x[0]**2-x[1]**2)
  c = (1/3)*np.exp(-(x[0]+1)**2 - x[1]**2)
  return a - b - c + 6.551

ub = 3.0
lb = -3.0

d = 2 # Dimension of Decision Variable Space
s = 0.5 # Standard Deviation of Normal Noise
num_seeds = 10

m = 1
l = 100
max_NFE = 60000
ft = np.zeros((num_seeds, max_NFE/l))

def mutate(x, lb, ub, sigma):
  x_trial = x + np.random.normal(0, sigma, x.size)
  while np.any((x_trial > ub) | (x_trial < lb)):
    x_trial = x + np.random.normal(0, sigma, x.size)
  return x_trial

# (mu,lambda) Evolution Strategy
for seed in range(num_seeds):
  np.random.seed(seed)

  # Random Initial Population (l x d matrix)
  P = np.random.uniform(lb, ub, (l,d))
  f = np.zeros(l)
  nfe = 0
  f_best, x_best = None, None

  while nfe < max_NFE:

    # Evaluate All Solutions in Population
    for i,x in enumerate(P):
      f[i] = peaks(x)
      nfe += 1

    # Find m "Best" Parents, Truncation Selection
    ix = np.argsort(f)[:m]
    Q = P[ix, :]

    # Keep Track of "Best"
    if f_best is None or f[ix[0]] < f_best:
      f_best = f[ix[0]]
      x_best = Q[0,:]

    # Mutate: Each Parent Generates l/m Children (Integer Division)
    child = 0
    for i,x in enumerate(Q):
      for j in range(int(l/m)):
        P[child,:] = mutate(x, lb, ub, s)
        child += 1

    ft[seed,nfe/l-1] = f_best

  print x_best
  print f_best

# Define Enumeration Line
enumy = np.arange(2.0,-3.0,-1.0)
enumx = (10**(-enumy)*(ub-lb))**d

plt.loglog(range(l,max_NFE+1,l), ft.T, color='steelblue', linewidth=1)
plt.loglog(enumx,10**enumy, color='indianred', linewidth=2)

plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()






