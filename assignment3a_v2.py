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

ub = 3.0 # Upper Bound
lb = -3.0 # Lower Bound
d = 2 # Dimension of Decision Variable Space
num_seeds = 10 # Number of Random Seeds

m = 5 # Number of Parents Surviving Each Gen (Mu)
l = 30 # Population Size (Lambda)
max_NFE = 10000  # Maximum Number of Function Evaluations (NFE)

ft = np.zeros((num_seeds, max_NFE/l))
s_var = np.zeros((num_seeds, max_NFE/l))

def mutate(x, lb, ub, sigma):
    x_trail = x + np.random.normal(0, sigma, x.size)
    while np.any((x_trail > ub) | (x_trail < lb)):
        x_trail = x + np.random.normal(0, sigma, x.size)
    return x_trail

# (mu,lambda) Evolution Strategy
for seed in range(num_seeds):
    s = 0.5 # Standard Deviation of Normal Noise (Sigma)
    np.random.seed(seed)
    # Random Initial Population (l x d matrix)
    P = np.random.uniform(lb, ub, (l,d))
    f = np.zeros(l)
    nfe = 10
    f_best, x_best = None, None
    while nfe < max_NFE:
        # Evaluate All Solutions in Population
        for i,x in enumerate(P):
            f[i] = peaks(x)
            nfe += 1
        # Find Best Parents and Truncate Selection, Keep Track of 'Best' Values
        ix = np.argsort(f)[:m]
        Q = P[ix, :]
        if f_best is None or f[ix[0]] < f_best:
            f_best = f[ix[0]]
            x_best = Q[0,:]
        # Mutate: Each Parent Generates l/m Children (Integer Division)
        parents = np.take(f,ix) # Array of 'Best' Parent Values
        children = np.zeros(l)
        child = 0
        for i,x in enumerate(Q):
            for j in range(int(l/m)):
                P[child, :] = mutate(x, lb, ub, s)
                children[child] = peaks(P[child]) # Evaluate Children in "Peaks" Function
                child += 1
            counter = 0
            for j2 in range(m):
                for j3 in range(l/m):
                    if children[j3] <= parents[j2]: # Compare Children to Parents
                        counter +=1
                    else:
                        continue
                # Implement 1/5th Rule
                if counter > 0.2 * len(children):
                    s += 0.01
                else:
                    s -= 0.01
                if s < 0.01: # Avoid Sigma Value as Zero
                    s = 0.01
                s_var[seed,nfe/l-1] = s
        ft[seed,nfe/l-1] = f_best
    print x_best
    print f_best

# Define Enumeration Line
enumy = np.arange(2.0,-3.0,-1.0)
enumx = (10**(-enumy)*(ub-lb))**d

plt.subplot(1,2,1)
plt.loglog(range(l,max_NFE+1,l), ft.T, color='steelblue', linewidth=1)
plt.loglog(enumx,10**enumy, color='indianred', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')

plt.subplot(1,2,2)
plt.plot(range(l,max_NFE+1,l), s_var.T, color='steelblue', linewidth=1)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Sigma')
plt.show()