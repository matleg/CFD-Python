# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 16:09:44 2014
exercice - chapter 9 - section 9.3
Incompressible Couette flow, implicit Crank Nicolson technique
using numpy linalg
Anderson, John. Computational Fluid Dynamics. 1 edition.
New York: McGraw-Hill Science/Engineering/Math, 1995.
@author: ml971 hotmail.com
"""


import numpy as np
import matplotlib.pyplot as plt


###geometry
ny = 21
delta_y = 1./(ny-1)
print ("delta_y = "+str(delta_y))

###const.
E = 1.  # parameter for delta_t calculation
Re_D = 5000.
# delta_t includes E --> Couette flow steady state independent of Re_D
delta_t = E * Re_D * delta_y**2
print ("delta_t = "+str(delta_t))


###init. cond.

A = -E/2
B = 1+E

syst_eq = np.zeros((ny-2, ny-2))
for ll in range(ny-2):
    for cc in range(ny-2):
        if ll == cc:
            syst_eq[ll, cc] = B
        elif ll == cc-1:
            syst_eq[ll, cc] = A
        elif ll == cc+1:
            syst_eq[ll, cc] = A

print ("syst_eq" + str(syst_eq))

u = np.zeros((ny, 1))
u[-1, 0] = 1

buf = np.zeros_like(u)

K = np.zeros_like(u)

for t in range(250):

    print ("\n \n t = " + str(t))
    print ("u " + str(u[:, t]) + str(u.shape))

    for i in range(1, ny-1):
        K[i, t] = ((1 - E)*u[i, t] + (E/2)*(u[i+1, t] + u[i-1, t]))
    K[-2, t] = K[-2, t] - A*u[-1, t]

    u = np.hstack((u, buf))
    u[0, t+1] = 0
    u[-1, t+1] = 1
    u[1:-1, t+1] = np.linalg.solve(syst_eq, K[1:-1, t])
    print ("\n \n t+1 = " + str(t+1))
    print ("u[t+1] " + str(u[:, t+1]) + str(u.shape))
    print ("K"+str(K[:, t])+str(K.shape))

    K = np.hstack((K, buf))

y = np.linspace(0, 1, ny)
plt.figure(1)
plt.plot(u[:, 0], y, label="0dt")
plt.plot(u[:, 1], y, label="1dt")
plt.plot(u[:, 3], y, label="3dt")
plt.plot(u[:, 6], y, label="6dt")
plt.plot(u[:, 10], y, label="10dt")
plt.plot(u[:, 30], y, label="30dt")
plt.plot(u[:, 80], y, label="80dt")
plt.plot(u[:, 150], y, label="150dt")
plt.plot(u[:, 250], y, label="250dt")


plt.grid()
plt.legend()
plt.show()
