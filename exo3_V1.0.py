# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 16:09:44 2014
exercice - chapter 9 - section 9.3
Incompressible Couette flow, implicit Crank Nicolson technique
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

u = np.zeros((ny, 1))
u[-1, 0] = 1

buf = np.zeros_like(u)

a = np.zeros_like(u)
a[1:-2, 0] = A

b = np.zeros_like(u)
b[2:-1, 0] = A

d = np.zeros_like(u)
d[1:-1, 0] = B

K = np.zeros_like(u)

d_p = np.zeros_like(u)
c_p = np.zeros_like(u)

for t in range(260):

    print ("\n \n t = " + str(t))
    print ("u " + str(u[:, t]) + str(u.shape))

    for i in range(1, ny-1):
        K[i, t] = ((1 - E)*u[i, t] + (E/2)*(u[i+1, t] + u[i-1, t]))

    c_p[2, t] = (K[2, t]-K[1, t]*b[2, 0]/d[1, 0])
    d_p[2, t] = (d[2, 0]-b[2, 0]*a[1, 0]/d[1, 0])
    for i in range(3, ny-1):
        c_p[i, t] = (K[i, t]-c_p[i-1, t]*b[i, 0]/d_p[i-1, t])
        d_p[i, t] = (d[i, 0]-b[i, 0]*a[i-1, 0]/d_p[i-1, t])
    c_p[-2, t] = ((K[-2, t] - A*u[-1, t])-c_p[-3, t]*b[-2, 0]/d_p[-3, t])

    u = np.hstack((u, buf))
    u[0, t+1] = 0
    u[-1, t+1] = 1
    u[-2, t+1] = c_p[-2, t] / d_p[-2, t]
    for i in range(ny-3, 1, -1):
        u[i, t+1] = (c_p[i, t] - a[i, 0]*u[i+1, t+1])/d_p[i, t]
    u[1, t+1] = (K[1, t]-a[1, 0]*u[2, t+1])/d[1, 0]

    K = np.hstack((K, buf))
    d_p = np.hstack((d_p, buf))
    c_p = np.hstack((c_p, buf))

print ("d "+str(d[:, 0])+str(d.shape))
print ("a "+str(a[:, 0])+str(a.shape))
print ("b "+str(b[:, 0])+str(b.shape))


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
