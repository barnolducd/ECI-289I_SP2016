# Brad Arnold
# ECI-289I-001, Spring 2016
# Homework 2.2 (Traveling Salesman)
# -------------------------------

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

# this data uses "ATT distance" defined here:
# http://debian.cse.msu.edu/CSE/pub/GA/programs/Genocop-doc/TSP/READ_ME
# (this explains the division by 10 inside the square root)
def distance(tour, xy):
  # index with list, and repeat the first city
  x = np.append(tour, tour[0])
  d = np.diff(xy[tour], axis=0) 
  return np.sqrt((d**2).sum(axis=1)/10).sum()

num_seeds = 5
max_NFE = 10000
ft = np.zeros((num_seeds, max_NFE))
T0 = 1000 # initial temperature
alpha = 0.95 # cooling parameter

# Load Brewery Data
xy = np.loadtxt('BrewCoords.txt')
nt = np.loadtxt('BrewTier.txt')
num_breweries = len(xy)

# simulated annealing
for seed in range(num_seeds):
  np.random.seed(seed)
  T = T0

  # random initial tour
  tour = np.arange(num_breweries)
  tour = np.random.permutation(tour)
  bestf = distance(tour, xy)

  for i in range(max_NFE):

    # mutate the tour using two random cities
    trial_tour = np.copy(tour) # do not operate on original list
    a = np.random.randint(num_breweries)
    b = np.random.randint(num_breweries)

    # option 1: just swap the pair (we did this in class)
    # trial_tour[a],trial_tour[b] = trial_tour[b],trial_tour[a]

    # option 2: reverse the part of the tour between these nodes
    # (the lower value needs to be "a")
    if a > b:
      a,b = b,a
    trial_tour[a:b] = trial_tour[a:b][::-1]

    trial_f = distance(trial_tour, xy)

    r = np.random.rand()
    if T > 10**-3: # protect division by zero
      P = np.min([1.0, np.exp((bestf - trial_f)/T)])
    else:
      P = 0.0
    
    if trial_f < bestf or r < P:
      tour = trial_tour
      bestf = trial_f
    
    T = T0*alpha**i
    ft[seed,i] = bestf

  print tour
  print bestf

plt.subplot(1,2,1)
tour = np.append(tour, tour[0]) # for plotting
plt.plot(xy[tour][:,0], xy[tour][:,1], marker='o')

plt.subplot(1,2,2)
plt.semilogx(ft.T, color='steelblue', linewidth=1)
plt.xlabel('Iterations')
plt.ylabel('Length of Tour')
plt.show()



