# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:03:54 2014
exercice - chapter 7 - section 7.3
subsonic - supersonic isentropic nozzle flow - Mac-Cormack's technique
Anderson, John. Computational Fluid Dynamics. 1 edition.
New York: McGraw-Hill Science/Engineering/Math, 1995.
@author: ml971 hotmail.com
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt


dx = 0.05
x = np.arange(0, 3.1, dx)  # x from 0 to 3


timeSteps = 1500

#looked for values
A = np.zeros([len(x), 1])
rho = np.zeros([len(x), timeSteps])
rho[0, :] = 1  # boundary condition
T = np.zeros([len(x), timeSteps])
T[0, :] = 1  # boundary condition
V = np.zeros([len(x), timeSteps])
# V floating at i=0 otherwise overconstraint


#derivatives
dRhodt = np.zeros([len(x), timeSteps])
dVdt = np.zeros([len(x), timeSteps])
dTdt = np.zeros([len(x), timeSteps])


#predicted values
prho = np.zeros([len(x), timeSteps])
prho[0, :] = 1  # not calculated after, boundary condition
pV = np.zeros([len(x), timeSteps])
#pV[0,:]=0.1 #floating
pT = np.zeros([len(x), timeSteps])
pT[0, :] = 1  # not calculated after, boundary condition


#predicted derivatives
pdRhodt = np.zeros([len(x), timeSteps])
pdVdt = np.zeros([len(x), timeSteps])
pdTdt = np.zeros([len(x), timeSteps])


#average derivatives
avdRhodt = np.zeros([len(x), timeSteps])
avdVdt = np.zeros([len(x), timeSteps])
avdTdt = np.zeros([len(x), timeSteps])


#init.
def Area(xx):
    return 1 + 2.2*(xx-1.5)**2

A[:, 0] = Area(x)
##print (A)


def Rho(xx):
    return 1 - .3146*xx
rho[:, 0] = Rho(x)


def Temp(xx):
    return 1-.2314*xx
T[:, 0] = Temp(x)


def Velo(xx, t):
    return (0.1 + 1.09*xx)*t**0.5

V[:, 0] = Velo(x, T[:, 0])


temps = np.zeros([2, timeSteps+1])  # (t0=iter, t1=real)
gamma = 1.4
t = 0


while t < timeSteps-1:

    #definition and calculation of derivatives for interior points

    for i in range(1, len(x)-1):

        dRhodt[i, t] = -rho[i, t]*(V[i+1, t] - V[i, t])/dx -\
            rho[i, t]*V[i, t]*(log(A[i+1]) - log(A[i]))/dx -\
            V[i, t]*(rho[i+1, t]-rho[i, t])/dx

        dVdt[i, t] = -V[i, t]*(V[i+1, t]-V[i, t])/dx -\
            (1/gamma)*((T[i+1, t] - T[i, t])/dx +
                       (T[i, t]/rho[i, t])*((rho[i+1, t] - rho[i, t])/dx))

        dTdt[i, t] = -V[i, t]*(T[i+1, t] - T[i, t])/dx -\
            (gamma - 1)*T[i, t]*((V[i+1, t] - V[i, t])/dx +
                                 V[i, t]*(log(A[i+1]) - log(A[i]))/dx)

    #find dt min and round it

    dt = min(0.5*(dx/(T[1:29, t]**0.5 + V[1:29, t])))
    dt = 0.99*dt
    temps[0, t] = t
    temps[1, t+1] = temps[1, t] + dt

    #calculation of predicted values for interior points

    for i in range(1, len(x)-1):

        prho[i, t+1] = rho[i, t] + dRhodt[i, t]*dt
        pV[i, t+1] = V[i, t] + dVdt[i, t]*dt
        pT[i, t+1] = T[i, t] + dTdt[i, t]*dt

    pV[0, t+1] = 2*pV[1, t+1] - pV[2, t+1]
    #predicted value at i=0, floating at i=30

    #calculation of predicted value at t+dt for interior points
    for i in range(1, len(x)-1):

        pdRhodt[i, t+1] = -prho[i, t+1]*(pV[i, t+1]-pV[i-1, t+1])/dx -\
            prho[i, t+1]*pV[i, t+1]*(log(A[i]) - log(A[i-1]))/dx -\
            pV[i, t+1]*(prho[i, t+1] - prho[i-1, t+1])/dx
            # we need here pV[i-1,t+1], so pV[0,t+1] must be calculated before

        pdVdt[i, t+1] = -pV[i, t+1]*(pV[i, t+1] - pV[i-1, t+1])/dx -\
            (1/gamma)*((pT[i, t+1] - pT[i-1, t+1])/dx +
                       (pT[i, t+1]/prho[i, t+1])*((prho[i, t+1] -
                                                   prho[i-1, t+1])/dx))

        pdTdt[i, t+1] = -pV[i, t+1]*(pT[i, t+1] - pT[i-1, t+1])/dx -\
            (gamma - 1)*pT[i, t+1]*((pV[i, t+1] - pV[i-1, t+1])/dx +
                                    pV[i, t+1]*(log(A[i]) - log(A[i-1]))/dx)

    #calculation of average derivative for interior points
    for i in range(1, len(x)-1):

        avdRhodt[i, t+1] = 0.5*(dRhodt[i, t] + pdRhodt[i, t+1])

        avdVdt[i, t+1] = 0.5*(dVdt[i, t] + pdVdt[i, t+1])

        avdTdt[i, t+1] = 0.5*(dTdt[i, t] + pdTdt[i, t+1])

    #calculation of new variables for interior points
    for i in range(1, len(x)-1):
        rho[i, t+1] = rho[i, t] + avdRhodt[i, t+1]*dt
        V[i, t+1] = V[i, t] + avdVdt[i, t+1]*dt
        T[i, t+1] = T[i, t] + avdTdt[i, t+1]*dt

    #calculation of V[0], rho[30],T[30],V[30] floating
    #rho[0],T[0] fixed (reservoir at T, P fixed inlet)
    V[0, t+1] = 2*V[1, t+1] - V[2, t+1]

    rho[-1, t+1] = 2*rho[-2, t+1] - rho[-3, t+1]
    V[-1, t+1] = 2*V[-2, t+1] - V[-3, t+1]
    T[-1, t+1] = 2*T[-2, t+1] - T[-3, t+1]

    t += 1

#calculation of p (pressure) and M (mach number) speed of sound = T**0.5
p = rho*T
M = V/T**0.5

print ('rho0 ', rho[:, 0])
print ('V0 ', V[:, 0])
print ('T0 ', T[:, 0])

print ('rho1 ', rho[:, 1])
print ('V1 ', V[:, 1])
print ('T1 ', T[:, 1])

print ('rho1400 ', rho[:, 1400])
print ('V1400 ', V[:, 1400])
print ('T1400 ', T[:, 1400])

print (temps)


##plotting

f1, ax1 = plt.subplots(1)
ax1.plot(temps[0, :1000], rho[15, :1000], 'b--', label='rho/rho0')
ax1.plot(temps[0, :1000], T[15, :1000], 'r--', label='T/T0')
ax1.plot(temps[0, :1000], p[15, :1000], 'g--', label='p/p0')
ax1.plot(temps[0, :1000], M[15, :1000], 'y--', label='M/M0')
ax1.set_title('Residuals')
ax1.set_xlabel('iterations')
ax1.set_ylabel('dimensionless variable residual')
ax1.legend()


f2, (ax21, ax22) = plt.subplots(2, sharex=True)

ax21.plot(x, A[:, 0]/2, 'k-', x, -A[:, 0]/2, 'k-')
ax21.hlines(0, -0.5, x[-1] + 0.5, color='k', linestyles='dotted')

for i in range(int(x[-1]/dx)):
    ax21.vlines(x[i], A[i, 0]/2, -A[i, 0]/2, color='k', linestyles='dashed')

ax21.set_ylim([max(A[:, 0]/2 + 1), - max(A[:, 0]/2 + 1)])

ax22.plot(x, rho[:, -1], 'bo', label='rho')
ax22.plot(x, T[:, -1], 'ro', label='T')
ax22.plot(x, p[:, -1], 'go', label='p')
ax22.plot(x, M[:, -1], 'yo', label='M')
ax21.set_title('Quasi 1D nozzle flow')
ax22.set_xlabel('distance in the nozzle')
ax22.set_ylabel('dimensionless variable')
ax22.legend()

##plt.semilogy(temps[0,:1400],abs(avdVdt[15,:1400]))


##for i in [0,100,50,150,700,200]:
##    plt.plot(x,rho[:,i]*V[:,i]*A[:,0])


#plt.ion()
plt.show()
