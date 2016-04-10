# Brad Arnold
# ECI-289I-001, Spring 2016
# Homework 1.1 (Basic Programming Assignment)
# -------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def Hindex(L):
    L = sorted(L,reverse=True)
    H_val=0
    while H_val < len(L) and L[H_val] > H_val:
        H_val=H_val+1
    return H_val

# Part A Output
L = [2,7,1,3,11,1]
H_val = Hindex(L)
print H_val

# Part B Output
L = []
H_arr = []
for P in range(0,100,1):
    L.insert(P,P)
    H_arr.insert(P,Hindex(L))
plt.plot(L,H_arr)
plt.ylabel('h-Index (Self Citation)')
plt.xlabel('Papers Published')
plt.show()
