# Brad Arnold
# ECI-289I-001, Spring 2016
# Homework 1.2 (Gradient Descent)
# -------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# Define Matyas Function
def f(x):
  return 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]

# Gradient Normalized
def fp(x):
  fx0 = 0.52*x[0]-0.48*x[1]
  fx1 = 0.52*x[1]-0.48*x[0]
  mag = np.sqrt(fx0**2+fx1**2)
  return np.array([fx0/mag, fx1/mag])

alpha = 0.2 # Step Size
x = np.array([4,2]) # Initial Point
xt = []
ft = []

# Max Time Approach
for i in range(200):
  x = x - alpha*fp(x)
  xt.append(x)
  ft.append(f(x))
  print x

# Plotting Code
plt.subplot(1,2,1)
xx = np.arange(-5,5,0.01)
X1,X2 = np.meshgrid(xx, xx)
Z = 2*X1**2 + 10*X2**2
plt.contour(X1,X2,Z,50,cmap=plt.cm.Blues_r)
xt = np.array(xt)
plt.plot(xt[:,0], xt[:,1], color='k', linewidth=2)
plt.xlabel('X1')
plt.ylabel('X2')

plt.subplot(1,2,2)
plt.plot(ft, color='k', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()