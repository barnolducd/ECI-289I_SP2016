# Brad Arnold
# ECI-289I-001, Spring 2016
# Homework 3.1 (Evolution Strategies)
# ------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# function to optimize
def peaks(x):
  a = 3*(1-x[0])**2*np.exp(-(x[0]**2) - (x[1]+1)**2)
  b = 10*(x[0]/5 - x[0]**3 - x[1]**5)*np.exp(-x[0]**2-x[1]**2)
  c = (1/3)*np.exp(-(x[0]+1)**2 - x[1]**2)
  return a - b - c + 6.551 # add this so objective is always positive

ub = 3.0
lb = -3.0

d = 2 # dimension of decision variable space
s = 0.5 # stdev of normal noise (if this is too big, it's just random search!)
num_seeds = 10

m = 1
l = 100 # beware "lambda" is a reserved keyword
max_NFE = 60000 # should be a multiple
ft = np.zeros((num_seeds, max_NFE/l))

def mutate(x, lb, ub, sigma):
  x_trial = x + np.random.normal(0, sigma, x.size)
  while np.any((x_trial > ub) | (x_trial < lb)):
    x_trial = x + np.random.normal(0, sigma, x.size)
  return x_trial

# (mu,lambda) evolution strategy
for seed in range(num_seeds):
  np.random.seed(seed)

  # random initial population (l x d matrix)
  P = np.random.uniform(lb, ub, (l,d))
  f = np.zeros(l) # we'll evaluate them later
  nfe = 0
  f_best, x_best = None, None

  while nfe < max_NFE:

    # evaluate all solutions in the population
    for i,x in enumerate(P):
      f[i] = peaks(x)
      nfe += 1

    # find m best parents, truncation selection
    ix = np.argsort(f)[:m]
    Q = P[ix, :]

    # keep track of best here
    if f_best is None or f[ix[0]] < f_best:
      f_best = f[ix[0]]
      x_best = Q[0,:]

    # then mutate: each parent generates l/m children (integer division)
    child = 0
    for i,x in enumerate(Q):
      for j in range(int(l/m)):
        P[child,:] = mutate(x, lb, ub, s)
        child += 1

    ft[seed,nfe/l-1] = f_best

  # for each trial print the result (but the traces are saved in ft)
  print x_best
  print f_best

# add a line for "enumeration"
enumy = np.arange(2.0,-3.0,-1.0)
enumx = (10**(-enumy)*(ub-lb))**d

plt.loglog(range(l,max_NFE+1,l), ft.T, color='steelblue', linewidth=1)
plt.loglog(enumx,10**enumy, color='indianred', linewidth=2)

plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()






