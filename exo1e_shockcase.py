# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:05:45 2014
exercice - chapter 7 - section 7.6
a case with shock capturing
Anderson, John. Computational Fluid Dynamics. 1 edition.
New York: McGraw-Hill Science/Engineering/Math, 1995.
@author: ml971 hotmail.com
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt


dx = 0.05
x = np.arange(0, 3.05, dx)

gamma = 1.4
Cx = 0.3  # Cx = 0 -> no artificial visc, C = 0.3 -> max artificial visc

timeSteps = 2000

#wanted values, preparing tables
A = np.zeros([len(x), 1])
rho = np.zeros([len(x), timeSteps])
T = np.zeros([len(x), timeSteps])
V = np.zeros([len(x), timeSteps])

U1 = np.zeros([len(x), timeSteps])
U2 = np.zeros([len(x), timeSteps])
U3 = np.zeros([len(x), timeSteps])

pU1 = np.zeros([len(x), timeSteps])
pU2 = np.zeros([len(x), timeSteps])
pU3 = np.zeros([len(x), timeSteps])

F1 = np.zeros([len(x), timeSteps])
F2 = np.zeros([len(x), timeSteps])
F3 = np.zeros([len(x), timeSteps])
J2 = np.zeros([len(x), timeSteps])

pF1 = np.zeros([len(x), timeSteps])
pF2 = np.zeros([len(x), timeSteps])
pF3 = np.zeros([len(x), timeSteps])
pJ2 = np.zeros([len(x), timeSteps])

#Artificial viscosity
S1 = np.zeros([len(x), timeSteps])
S2 = np.zeros([len(x), timeSteps])
S3 = np.zeros([len(x), timeSteps])
pS1 = np.zeros([len(x), timeSteps])
pS2 = np.zeros([len(x), timeSteps])
pS3 = np.zeros([len(x), timeSteps])


#derivatives
dRhodt = np.zeros([len(x), timeSteps])
dVdt = np.zeros([len(x), timeSteps])
dTdt = np.zeros([len(x), timeSteps])
dU1dt = np.zeros([len(x), timeSteps])
dU2dt = np.zeros([len(x), timeSteps])
dU3dt = np.zeros([len(x), timeSteps])

pdU1dt = np.zeros([len(x), timeSteps])
pdU2dt = np.zeros([len(x), timeSteps])
pdU3dt = np.zeros([len(x), timeSteps])

avdU1dt = np.zeros([len(x), timeSteps])
avdU2dt = np.zeros([len(x), timeSteps])
avdU3dt = np.zeros([len(x), timeSteps])


#predicted values
prho = np.zeros([len(x), timeSteps])
pV = np.zeros([len(x), timeSteps])
pT = np.zeros([len(x), timeSteps])


#predicted derivatives
pdRhodt = np.zeros([len(x), timeSteps])
pdVdt = np.zeros([len(x), timeSteps])
pdTdt = np.zeros([len(x), timeSteps])


#average derivatives
avdRhodt = np.zeros([len(x), timeSteps])
avdVdt = np.zeros([len(x), timeSteps])
avdTdt = np.zeros([len(x), timeSteps])


#initialization [:, 0] (t = 0)

def Aire(xx):
    return 1+2.2*(xx-1.5)**2

A[:, 0] = Aire(x)


def Rho(xx):
    if xx <= 0.5:
        return 1
    elif xx <= 1.5:
        return 1-0.366*(xx-0.5)
    elif xx <= 2.1:
        return 0.634-0.702*(xx-1.5)
    else:
        return 0.5892-0.10228*(xx-2.1)

for i in range(len(x)):
    rho[i, 0] = Rho(x[i])


def Temp(xx):
    if xx <= 0.5:
        return 1
    elif xx <= 1.5:
        return 1-0.167*(xx-0.5)
    elif xx <= 2.1:
        return 0.833-0.4908*(xx-1.5)
    else:
        return 0.93968-0.0622*(xx-2.1)

for i in range(len(x)):
    T[i, 0] = Temp(x[i])

T[0, :] = 1  # boundary condition [0, :] (i = 0)
rho[0, :] = 1  # boundary condition
pT[0, :] = 1  # boundary condition
prho[0, :] = 1  # boundary condition


V[:, 0] = 0.59/(rho[:, 0]*A[:, 0])  # init. cond.

U1[0, :] = A[0]  # boundary condition (rho fixed initiallu, so U1 also)
pU1[0, :] = A[0]

U1[1:, 0] = rho[1:, 0]*A[1:, 0]  # init

U2[:, 0] = rho[:, 0]*A[:, 0]*V[:, 0]
U3[:, 0] = rho[:, 0]*A[:, 0]*(T[:, 0]/(gamma-1) + gamma/2*V[:, 0]**2)

temps = np.zeros([2, timeSteps+1])  # (column 0 = iter, 1 = reel)


t = 0

while t < timeSteps-1:

    #definition and calculation of Flux terms for interior points
    F1[:, t] = U2[:, t]
    for i in range(len(x)):
        F2[i, t] = ((U2[i, t]**2/U1[i, t])+((gamma-1)/gamma)*(U3[i, t]-(
            gamma/2)*(U2[i, t]**2/U1[i, t])))
        F3[i, t] = ((gamma*U2[i, t]*U3[i, t]/U1[i, t])-(
            gamma*(gamma-1)/2)*(U2[i, t]**3/U1[i, t]**2))

    for i in range(1, len(x)-1):  # production term J
        J2[i, t] = ((1/gamma)*rho[i, t]*T[i, t]*(A[i]-A[i-1])/dx)

    for i in range(1, len(x)-1):
        dU1dt[i, t] = -(F1[i+1, t]-F1[i, t])/dx
        dU2dt[i, t] = -(F2[i+1, t]-F2[i, t])/dx+J2[i, t]
        dU3dt[i, t] = -(F3[i+1, t]-F3[i, t])/dx

    # find and round dt min
    dt = min(0.5*(dx/(T[1:29, t]**0.5 + V[1:29, t])))
    dt = 0.99*dt
    temps[0, t] = t
    temps[1, t+1] = temps[1, t]+dt

    for i in range(1, len(x)-1):
        S1[i, t] = Cx*(abs(rho[i+1, t]*T[i+1, t] - 2*rho[i, t]*T[i, t] +
                           rho[i-1, t]*T[i-1, t]) /
                          (rho[i+1, t]*T[i+1, t] + 2*rho[i, t]*T[i, t] +
                           rho[i-1, t]*T[i-1, t])) * (U1[i+1, t] -
                                                      2*U1[i, t] + U1[i-1, t])
        S2[i, t] = Cx*(abs(rho[i+1, t]*T[i+1, t] - 2*rho[i, t]*T[i, t] +
                           rho[i-1, t]*T[i-1, t]) /
                          (rho[i+1, t]*T[i+1, t] + 2*rho[i, t]*T[i, t] +
                           rho[i-1, t]*T[i-1, t])) * (U2[i+1, t] -
                                                      2*U2[i, t]+U2[i-1, t])
        S3[i, t] = Cx*(abs(rho[i+1, t]*T[i+1, t] - 2*rho[i, t]*T[i, t] +
                           rho[i-1, t]*T[i-1, t]) /
                          (rho[i+1, t]*T[i+1, t] + 2*rho[i, t]*T[i, t] +
                           rho[i-1, t]*T[i-1, t])) * (U3[i+1, t] -
                                                      2*U3[i, t]+U3[i-1, t])

    #solution vectors (U) predicted for interior points at t+dt
    #and then at boundaries

    #interior
    for i in range(1, len(x)-1):

        pU1[i, t+1] = U1[i, t]+dU1dt[i, t]*dt+S1[i, t]
        pU2[i, t+1] = U2[i, t]+dU2dt[i, t]*dt+S2[i, t]
        pU3[i, t+1] = U3[i, t]+dU3dt[i, t]*dt+S3[i, t]

    #boundaries
    pU1[-1, t+1] = 2*pU1[-2, t+1]-pU1[-3, t+1]
    pU2[-1, t+1] = 2*pU2[-2, t+1]-pU2[-3, t+1]
    pV[:, t+1] = pU2[:, t+1]/pU1[:, t+1]  # floating at inlet
    pU3[-1, t+1] = 0.6784*A[-1, 0]/(gamma - 1) +\
        (gamma/2)*U2[-1, t+1]*pV[-1, t+1]  # specific exit pressure

    pU2[0, t+1] = 2*pU2[1, t+1]-pU2[2, t+1]  # floating at inlet
    pU3[0, t+1] = pU1[0, t+1]*(T[0, t+1]/(gamma-1) + gamma/2*pV[0, t+1]**2)

    for i in range(1, len(x)):
        prho[i, t+1] = pU1[i, t+1]/A[i]
        pT[i, t+1] = (gamma - 1)*((pU3[i, t+1]/pU1[i, t+1]) -
                                  (gamma/2)*(pU2[i, t+1]/pU1[i, t+1])**2)

    #predicted fluxes
    pF1[:, t+1] = pU2[:, t+1]
    for i in range(len(x)):
        pF2[i, t+1] = ((pU2[i, t+1]**2/pU1[i, t+1]) + (
            (gamma-1)/gamma)*(pU3[i, t+1]-(gamma/2)*(pU2[i, t+1]**2 /
                                                     pU1[i, t+1])))
        pF3[i, t+1] = ((gamma*pU2[i, t+1]*pU3[i, t+1]/pU1[i, t+1])-(
            gamma*(gamma-1)/2)*(pU2[i, t+1]**3/pU1[i, t+1]**2))

    #predicted derivative at t+dt for interior points
    for i in range(1, len(x)-1):
        pdU1dt[i, t+1] = -(pF1[i, t+1]-pF1[i-1, t+1])/dx
        pdU2dt[i, t+1] = -(pF2[i, t+1]-pF2[i-1, t+1])/dx
        + ((1/gamma)*prho[i, t+1]*pT[i, t+1]*(A[i]-A[i-1])/dx)
        pdU3dt[i, t+1] = - (pF3[i, t+1]-pF3[i-1, t+1])/dx

    #average derivative
    for i in range(1, len(x)-1):
        avdU1dt[i, t] = 0.5*(dU1dt[i, t]+pdU1dt[i, t+1])
        avdU2dt[i, t] = 0.5*(dU2dt[i, t]+pdU2dt[i, t+1])
        avdU3dt[i, t] = 0.5*(dU3dt[i, t]+pdU3dt[i, t+1])
##    print (avdU1dt[15, t])
##    print (avdU2dt[15, t])
##    print (avdU3dt[15, t])

    for i in range(1, len(x)-1):
            pS1[i, t+1] = Cx*(abs(prho[i+1, t+1]*pT[i+1, t+1]
                                  - 2*prho[i, t+1]*pT[i, t+1] + prho[i-1, t+1]
                                  * pT[i-1, t+1])
                              / (prho[i+1, t+1]*pT[i+1, t+1] + 2*prho[i, t+1]
                                 * pT[i, t+1] + prho[i-1, t+1]*pT[i-1, t+1])
                              )*(pU1[i+1, t+1] - 2*pU1[i, t+1]
                                 + pU1[i-1, t+1])
            pS2[i, t+1] = Cx*(abs(prho[i+1, t+1]*pT[i+1, t+1]
                                  - 2*prho[i, t+1]*pT[i, t+1]+prho[i-1, t+1]
                                  * pT[i-1, t+1])
                              / (prho[i+1, t+1]*pT[i+1, t+1] + 2*prho[i, t+1]
                                 * pT[i, t+1] + prho[i-1, t+1]*pT[i-1, t+1])
                              )*(pU2[i+1, t+1]-2*pU2[i, t+1]+pU2[i-1, t+1])
            pS3[i, t+1] = Cx*(abs(prho[i+1, t+1]*pT[i+1, t+1]
                                  - 2*prho[i, t+1]*pT[i, t+1]+prho[i-1, t+1]
                                  * pT[i-1, t+1])
                              / (prho[i+1, t+1]*pT[i+1, t+1] + 2*prho[i, t+1]
                                 * pT[i, t+1] + prho[i-1, t+1]*pT[i-1, t+1])
                              )*(pU3[i+1, t+1]-2*pU3[i, t+1]+pU3[i-1, t+1])

    #U, interior points and boundaries
    for i in range(1, len(x)-1):
        U1[i, t+1] = U1[i, t]+avdU1dt[i, t]*dt+pS1[i, t+1]
        U2[i, t+1] = U2[i, t]+avdU2dt[i, t]*dt+pS2[i, t+1]
        U3[i, t+1] = U3[i, t]+avdU3dt[i, t]*dt+pS3[i, t+1]

    U1[-1, t+1] = 2*U1[-2, t+1]-U1[-3, t+1]
    U2[-1, t+1] = 2*U2[-2, t+1]-U2[-3, t+1]

    U2[0, t+1] = 2*U2[1, t+1]-U2[2, t+1]  # floating at inlet
    V[:, t+1] = U2[:, t+1]/U1[:, t+1]  # floating at inlet
    U3[-1, t+1] = 0.6784*A[-1, 0]/(gamma - 1)
    + (gamma/2)*U2[-1, t+1]*pV[-1, t+1]  # specific exit pressure
    U3[0, t+1] = U1[0, t+1]*(T[0, t+1]/(gamma-1) + gamma/2*V[0, t+1]**2)

    for i in range(1, len(x)):
        rho[i, t+1] = U1[i, t+1]/A[i]

    T[1:, t+1] = (gamma - 1)*((U3[1:, t+1]/U1[1:, t+1])
                              - (gamma/2)*(V[1:, t+1])**2)

##    print (rho[15, t+1])
##    print (V[15, t+1])
##    print (T[15, t+1])

    t += 1

# p (pressure) and M (mach number) speed of sound = T**0.5
p = rho*T
M = V/T**0.5

print (M[:, 1400])

##plotting

f1, ax1 = plt.subplots(1)
ax1.plot(x, U2[:, 1400], label='m1')

ax1.set_title('mass')
ax1.set_xlabel('x')
ax1.set_ylabel('m')
ax1.legend()


f2, (ax21, ax22) = plt.subplots(2, sharex=True)

ax21.plot(x, A[:, 0]/2, 'k-', x, -A[:, 0]/2, 'k-')
ax21.hlines(0, -0.5, x[-1]+0.5, color='k', linestyles='dotted')

for i in range(int(x[-1]/dx)):
    ax21.vlines(x[i], A[i, 0]/2, -A[i, 0]/2, color='k', linestyles='dashed')

ax21.set_ylim([max(A[:, 0]/2+1), -max(A[:, 0]/2+1)])
ax22.plot(x, p[:, 1400], label='p')
ax22.plot(x, M[:, 1400], label='M')


ax21.set_title('Pressure and Mach')
ax22.set_xlabel('x/L')
ax22.set_ylabel('p and M (-)')

ax22.legend()

#plt.ion()
plt.show()
