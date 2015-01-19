# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 16:08:03 2014
exercice - chapter 10
Supersonic flow over flat plate
Anderson, John. Computational Fluid Dynamics. 1 edition.
New York: McGraw-Hill Science/Engineering/Math, 1995.
@author: ml971 hotmail.com

script creates a file data.sav to store values using pickle
!!!pickle python 2.7 != pickle python 3.4 !!!
"""


import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
import pickle
import os

cwd = os.getcwd()
print (cwd)


# functions
def Cal_Mu(T1):  # DYNVIS
    """calculate viscosity function of local temperature"""
    return(Mu0*(T1/T0)**(3./2.)*((T0+110)/(T1+110)))


def Cal_k(mu):  # THERMC
    """calculate k, thermal conductivity, function of local viscosity"""
    return(mu * Cp / Pr)


def Cal_dt(u1, v1, T1, Mu1, Rho1):
    """calculate the time step"""
    maxx = np.max(4./3*Mu1[1:-1, 1:-1]*(Gamma*Mu1[1:-1, 1:-1]/Pr)
    /Rho1[1:-1, 1:-1])
    dtCFL = 1/(np.abs(u1[1:-1, 1:-1])/dx + np.abs(v1[1:-1, 1:-1])/dy +
    (Gamma*R*T1[1:-1, 1:-1])**0.5*(1/dx**2+1/dy**2)**0.5 +
    2*maxx*(1/dx**2+1/dy**2))
    return (np.min(dtCFL)*K)


def Tauyy(T1, u, v, l, c, pred, corr):
    """tauxx[l, c] for predictor and corrector steps"""
    if pred:
    #d/dx -> central diff except inlet and outlet
    #d/dy -> backward
        if l == 0:  # bottom
            dvdy = (v[l+1, c]-v[l, c])/dy
        else:  # top and central
            dvdy = (v[l, c]-v[l-1, c])/dy
        if c == 0:  # inlet
            dudx = (u[l, c+1]-u[l, c])/dx
        elif c == (xmax-1):  # outlet
            dudx = (u[l, c]-u[l, c-1])/dx
        else:  # central
            dudx = (u[l, c+1]-u[l, c-1])/(2*dx)
    
    if corr:
    #d/dx -> central diff except inlet and outlet
    #d/dy -> forward
        if l == (ymax-1):  # top
            dvdy = (v[l, c]-v[l-1, c])/dy
        else:  # bottom and central
            dvdy = (v[l+1, c]-v[l, c])/dy
        if c == 0:  # inlet
            dudx = (u[l, c+1]-u[l, c])/dx
        elif c == (xmax-1):  # outlet
            dudx = (u[l, c]-u[l, c-1])/dx
        else:  # central
            dudx = (u[l, c+1]-u[l, c-1])/(2*dx)
    
    mu_t = Cal_Mu(T1[l, c])
    return ((-2./3.)*mu_t*(dudx + dvdy) + 2*mu_t*dvdy)


def Qy(T1, l, c, pred, corr):
    """qy[l, c] for predictor and corrector steps"""
    if pred:
        if l == 0:
            dTdy = (T1[l+1, c]-T1[l, c])/dy
        else:
            dTdy = (T1[l, c]-T1[l-1, c])/dy
    if corr:
        if l == (ymax-1):
            dTdy = (T1[l, c]-T1[l-1, c])/dy
        else:
            dTdy = (T1[l+1, c]-T1[l, c])/dy
        
    mu_t = Cal_Mu(T1[l, c])
    k_t = Cal_k(mu_t)
    
    return (-k_t*dTdy)


def Tauxx(T1, u, v, l, c, pred, corr):
    """tauxx[l, c] for predictor and corrector steps"""
    if pred:
    #d/dx -> backward diff except inlet
    #d/dy -> central diff except top and bottom
        if c == 0:  # inlet
            dudx = (u[l, c+1]-u[l, c])/dx
        else:  # outlet and central
            dudx = (u[l, c]-u[l, c-1])/dx
        if l == 0:  # bottom
            dvdy = (v[l+1, c]-v[l, c])/dy
        elif l == (ymax-1):  # top
            dvdy = (v[l, c]-v[l-1, c])/dy
        else:  # central
            dvdy = (v[l+1, c]-v[l-1, c])/(2*dy)
    
    if corr:
    #d/dx -> forward diff except outlet
    #d/dy -> central diff except top and bottom
        if c == (xmax-1):  # outlet
            dudx = (u[l, c]-u[l, c-1])/dx
        else:  # outlet and central
            dudx = (u[l, c+1]-u[l, c])/dx
        if l == 0:  # bottom
            dvdy = (v[l+1, c]-v[l, c])/dy
        elif l == (ymax-1):  # top
            dvdy = (v[l, c]-v[l-1, c])/dy
        else:  # central
            dvdy = (v[l+1, c]-v[l-1, c])/(2*dy)
    
    mu_t = Cal_Mu(T1[l, c])
    return ((-2./3.)*mu_t*(dudx + dvdy)+2*mu_t*dudx)
    
    
def Qx(T1, l, c, pred, corr):
    """qx[l, c] for predictor and corrector steps"""
    if pred:
        if c == 0:
            dTdx = (T1[l, c+1]-T1[l, c])/dx
        else:
            dTdx = (T1[l, c]-T1[l, c-1])/dx
    if corr:
        if c == (xmax-1):
            dTdx = (T1[l, c]-T1[l, c-1])/dx
        else:
            dTdx = (T1[l, c+1]-T1[l, c])/dx
    
    mu_t = Cal_Mu(T1[l, c])
    k_t = Cal_k(mu_t)
    
    return (-k_t*dTdx)
                    
    
def Tauxy(T1, u, v, l, c, pred, corr, EE, FF):  # line, column, pred
    """tauxy[l, c],pred/corr, E/F, l = line (0 to ymax-1)
    c = column (0 to xmax-1)""" 
    if pred and EE:  # predictor, E
    #d/dy -> central diff except top and bottom
    #d/dx -> rearward diff except inlet
        if l == 0:  # bottom
            dudy = (u[l+1, c]-u[l, c])/dy
        elif l == (ymax-1):  # top
            dudy = (u[l, c]-u[l-1, c])/dy
        else:  # central
            dudy = (u[l+1, c]-u[l-1, c])/(2*dy)
        if c == 0:  # inlet
            dvdx = (v[l, c+1]-v[l, c])/dx
        else:  # outlet & central
            dvdx = (v[l, c]-v[l, c-1])/dx
    
    if corr and EE:  # corrector, E
    #d/dy -> central diff except top and bottom
    #d/dx -> forward diff except outlet
        if l == 0:  # bottom
            dudy = (u[l+1, c]-u[l, c])/dy
        elif l == (ymax-1):  # top
            dudy = (u[l, c]-u[l-1, c])/dy
        else:  # central
            dudy = (u[l+1, c]-u[l-1, c])/(2*dy)
        if c == (xmax-1):  # outlet
            dvdx = (v[l, c]-v[l, c-1])/dx
        else:  # inlet & central
            dvdx = (v[l, c+1]-v[l, c])/dx
            
    if pred and FF:  # predictor, F
    #d/dx -> central diff except inlet and outlet
    #d/dy -> rearward except bottom
        if l == 0:  # bottom
            dudy = (u[l+1, c]-u[l, c])/dy
        else:  # top and central
            dudy = (u[l, c]-u[l-1, c])/dy
        if c == 0:  # inlet
            dvdx = (v[l, c+1]-v[l, c])/dx
        elif c == (xmax-1):  # outlet
            dvdx = (v[l, c]-v[l, c-1])/dx
        else:  # central
            dvdx = (v[l, c+1]-v[l, c-1])/(2*dx)
    
    if corr and FF:  # corrector, F
    #d/dx -> central diff except inlet and outlet
    #d/dy -> forward except top
        if l == (ymax-1):  # top
            dudy = (u[l, c]-u[l-1, c])/dy
        else:  # bottom and central
            dudy = (u[l+1, c]-u[l, c])/dy
        if c == 0:  # inlet
            dvdx = (v[l, c+1]-v[l, c])/dx
        elif c == (xmax-1):  # outlet
            dvdx = (v[l, c]-v[l, c-1])/dx
        else:  # central
            dvdx = (v[l, c+1]-v[l, c-1])/(2*dx)
            
    mu_t = Cal_Mu(T1[l, c])
    return (mu_t * (dudy + dvdx))

def Bocos(u, v, T, P):
    #boundary conditions
    u[-1, :] = 4 * M0  # top
    u[1:, 0] = 4 * M0  # all inlet except plate
    u[0, :] = 0  # along plate (bottom)
    u[1:-1, -1] = 2*u[1:-1, -2]-u[1:-1, -3]  # all outlet except plate and top

    v[:, 0] = 0  # inlet
    v[-1, :] = 0  # top
    v[0, :] = 0  # along plate (bottom)
    v[1:-1, -1] = 2*v[1:-1, -2]-v[1:-1, -3]  # all outlet except plate and top

    T[0, 1:] = Tw  # bottom plate
    T[:, 0] = T0  # inlet freestream
    T[-1, :] = T0  # top
    T[1:-1, -1] = 2*T[1:-1, -2]-T[1:-1, -3]  # all outlet except plate and top
    
    P[:, 0] = P0  # inlet
    P[-1, :] = P0  # top
    P[1:-1, -1] = 2*P[1:-1, -2]-P[1:-1, -3]  # all outlet except plate and top
    P[0, 1:] = 2*P[1, 1:]-P[2, 1:]  # all bottom except in
    
    return(u, v, T, P)


# MacCormac
def MacCor(dt, Rho, u, v, Et, e, T, P, Mu, k,
           pU1, pU2, pU3, pU5, tauxx, tauyy, tauxy,
           qx, qy):

    U1 = Rho.copy()
    U2 = (Rho * u).copy()
    U3 = (Rho * v).copy()
    U5 = Et.copy()

    #predictor step
    pred, corr = True, False
    EE, FF = True, False  # E is being calculated
    for l in range(ymax):
        for c in range(xmax):
            tauxx[l, c] = Tauxx(T, u, v, l, c, pred, corr)
            tauxy[l, c] = Tauxy(T, u, v, l, c, pred, corr, EE, FF)
            qx[l, c] = Qx(T, l, c, pred, corr)
    E1 = U2.copy()
    E2 = (Rho*u**2 + P - tauxx).copy()
    E3 = (Rho*u*v - tauxy).copy()
    E5 = ((Et+P)*u - u*tauxx - v*tauxy + qx).copy()
    
    EE, FF = False, True  # F is being calculated
    for l in range(ymax):
        for c in range(xmax):
            tauxy[l, c] = Tauxy(T, u, v, l, c, pred, corr, EE, FF)
            tauyy[l, c] = Tauyy(T, u, v, l, c, pred, corr)
            qy[l, c] = Qy(T, l, c, pred, corr)
    F1 = (U3).copy()
    F2 = (Rho*u*v - tauxy).copy()
    F3 = (Rho*v**2 + P - tauxy).copy()
    F5 = ((Et+P)*v - u*tauxy - v*tauyy + qy).copy()
    
    pU1[1:-1, 1:-1] = U1[1:-1, 1:-1] - dt*((E1[1:-1, 2:]-E1[1:-1, 1:-1])/dx+
    (F1[2:, 1:-1]-F1[1:-1, 1:-1])/dy)
    pU2[1:-1, 1:-1] = U2[1:-1, 1:-1] - dt*((E2[1:-1, 2:]-E2[1:-1, 1:-1])/dx+
    (F2[2:, 1:-1]-F2[1:-1, 1:-1])/dy)
    pU3[1:-1, 1:-1] = U3[1:-1, 1:-1] - dt*((E3[1:-1, 2:]-E3[1:-1, 1:-1])/dx+
    (F3[2:, 1:-1]-F3[1:-1, 1:-1])/dy)
    pU5[1:-1, 1:-1] = U5[1:-1, 1:-1] - dt*((E5[1:-1, 2:]-E5[1:-1, 1:-1])/dx+
    (F5[2:, 1:-1]-F5[1:-1, 1:-1])/dy)
    #values are predicted
    
    #decoding predicted values
    Rho = (pU1).copy()
    u = (pU2/pU1).copy()
    v = (pU3/pU1).copy()
    Et = (pU5).copy()
    e = (pU5/pU1 - (u**2+v**2)/2).copy()
    T = (e/Cv).copy()
    P = (Rho*R*T).copy()
    Mu = (Cal_Mu(T)).copy()
    k = (Mu*Cp/Pr).copy()
    
    #boundary conditions
    u, v, T, P = Bocos(u, v, T, P)
    
    Rho = P / (R*T)
    e = T * Cv
    Mu = Cal_Mu(T)
    k = Cal_k(Mu)
    Et = Rho*(e+(u**2+v**2)/2)
    
    pU1 = (Rho).copy()
    pU2 = (Rho * u).copy()
    pU3 = (Rho * v).copy()
    pU5 = (Et).copy()
    
    #corrector step
    pred, corr = False, True
    EE, FF = True, False  # E is being calculated
    for l in range(ymax):
        for c in range(xmax):
            tauxx[l, c] = Tauxx(T, u, v, l, c, pred, corr)
            tauxy[l, c] = Tauxy(T, u, v, l, c, pred, corr, EE, FF)
            qx[l, c] = Qx(T, l, c, pred, corr)
    E1 = (pU2).copy()
    E2 = (Rho*u**2 + P - tauxx).copy()
    E3 = (Rho*u*v - tauxy).copy()
    E5 = ((Et+P)*u - u*tauxx - v*tauxy + qx).copy()

    EE, FF = False, True  # F is being calculated
    for l in range(ymax):
        for c in range(xmax):
            tauxy[l, c] = Tauxy(T, u, v, l, c, pred, corr, EE, FF)
            tauyy[l, c] = Tauyy(T, u, v, l, c, pred, corr)
            qy[l, c] = Qy(T, l, c, pred, corr)
    F1 = (pU3).copy()
    F2 = (Rho*u*v - tauxy).copy()
    F3 = (Rho*v**2 + P - tauxy).copy()
    F5 = ((Et+P)*v - u*tauxy - v*tauyy + qy).copy()

    U1[1:-1, 1:-1] = 0.5*(U1[1:-1, 1:-1] + pU1[1:-1, 1:-1] - dt*(
    (E1[1:-1, 1:-1]-E1[1:-1, :-2])/dx+(F1[1:-1, 1:-1]-F1[:-2, 1:-1])/dy))
    U2[1:-1, 1:-1] = 0.5*(U2[1:-1, 1:-1] + pU2[1:-1, 1:-1] - dt*(
    (E2[1:-1, 1:-1]-E2[1:-1, :-2])/dx+(F2[1:-1, 1:-1]-F2[:-2, 1:-1])/dy))
    U3[1:-1, 1:-1] = 0.5*(U3[1:-1, 1:-1] + pU3[1:-1, 1:-1] - dt*(
    (E3[1:-1, 1:-1]-E3[1:-1, :-2])/dx+(F3[1:-1, 1:-1]-F3[:-2, 1:-1])/dy))
    U5[1:-1, 1:-1] = 0.5*(U5[1:-1, 1:-1] + pU5[1:-1, 1:-1] - dt*(
    (E5[1:-1, 1:-1]-E5[1:-1, :-2])/dx+(F5[1:-1, 1:-1]-F5[:-2, 1:-1])/dy))
    
    #decoding corrected values
    Rho = (U1).copy()
    u = (U2/Rho).copy()
    v = (U3/Rho).copy()
    Et = (U5).copy()
    e = (U5/Rho - (u**2+v**2)/2).copy()
    T = (e/Cv).copy()
    P = (Rho*R*T).copy()
    Mu = (Cal_Mu(T)).copy()
    k = (Mu*Cp/Pr).copy()
    
    #boundary conditions
    u, v, T, P = Bocos(u, v, T, P)
    
    Rho = P / (R*T)
    e = T * Cv
    Mu = Cal_Mu(T)
    k = Cal_k(Mu)
    Et = Rho*(e+(u**2+v**2)/2)
    
    return(Rho, u, v, Et, e, T, P, Mu, k)


def Testconv(u1,u):
    """function that test convergence comparing u1(t-1) and u(t)"""
    test_conv = u1-u
    maxim = np.amax(np.abs(test_conv))
    if maxim < converg:
        print ("solution is converged")
        return True

#init const
xmax = 70
ymax = 70
Itmax = 10000
converg = 0.00001  # converg criteria
Lhori = 0.00001  # m
M0 = 340.28  # m s-1 (Gamma*R*T0)**0.5
T0 = 288.16  # K
P0 = 101325.  # Pa

Gamma = 1.4
Pr = 0.71
Mu0 = 0.000017894  # kg m-1 s-1
R = 287.  # J kg-1 K-1
Rho0 = P0 / (R*T0)  # kg m-3
Tw = 289  # K
Cv = R/(Gamma-1)
Cp = Gamma * Cv

x = np.arange(0, xmax)
y = np.arange(0, ymax)
X, Y = np.meshgrid(x, y)

Re0 = Rho0 * 4 * M0 * Lhori / Mu0
delta = 5*Lhori/(Re0**0.5)
Lvert = 5 * delta
dx = Lhori / (xmax-1)
dy = Lvert / (ymax-1)
K = 0.3  # good untill 23 with K=0.6, bug @ 26

#init boundary variables (Rho,u,v,T,e,Et,P,Mu,k)
Rho = np.ones_like(X, dtype=np.float) * Rho0

u = np.ones_like(X, dtype=np.float) * 4 * M0
u[-1, :] = 4 * M0  # top
u[1:, 0] = 4 * M0  # all inlet except plate
u[1:, -1] = 4 * M0  # all outlet except plate
u[0, :] = 0  # along plate (bottom)

v = np.ones_like(X, dtype=np.float) * 0
v[:, 0] = 0  # inlet
v[-1, :] = 0  # top
v[0, :] = 0  # along plate (bottom)

T = np.ones_like(X, dtype=np.float) * T0
T[0, 1:] = Tw  # bottom
T[:, 0] = T0  # inlet freestream

e = np.ones_like(X, dtype=np.float) * T * Cv

Mu = np.ones_like(X, dtype=np.float) * Cal_Mu(T)

k = np.ones_like(X, dtype=np.float) * Cal_k(Mu)

P = np.ones_like(X, dtype=np.float) * P0

Et = np.ones_like(X, dtype=np.float) * Rho*(e+(u**2+v**2)/2)

its = 0  # last iteration stop stored

t_tot = 0.

try:
    with open("data.sav", 'rb') as fich:
        Rho = pickle.load(fich)
        u = pickle.load(fich)
        v = pickle.load(fich)
        T = pickle.load(fich)
        e = pickle.load(fich)
        Mu = pickle.load(fich)
        k = pickle.load(fich)
        P = pickle.load(fich)
        Et = pickle.load(fich)
        its = pickle.load(fich)
        t_tot = pickle.load(fich)
        print ("last iteration stop = " + str(its) + "\n"+
               "total time = " + str(t_tot))
except:
    print("new file: data.sav will be created at the end")
    pass


pU1 = np.ones_like(X, dtype=np.float)
pU2 = np.ones_like(X, dtype=np.float)
pU3 = np.ones_like(X, dtype=np.float)
pU5 = np.ones_like(X, dtype=np.float)
tauxx = np.ones_like(X, dtype=np.float)
tauyy = np.ones_like(X, dtype=np.float)
tauxy = np.ones_like(X, dtype=np.float)
qx = np.ones_like(X, dtype=np.float)
qy = np.ones_like(X, dtype=np.float)


##def test(v):
##    for l in range(ymax):
##        for c in range(xmax):
##            if isnan(v[l,c]):
##                print (l,c,v)
##                return (v[l,c])


#starting computations
for i in range(its, Itmax):
    tstart = time.time()
    dt = Cal_dt(u, v, T, Mu, Rho)
    
    u1 = u.copy()  # save u to test convergence

    Rho, u, v, Et, e, T, P, Mu, k = MacCor(dt, Rho, u, v, Et, e, T, P, Mu, k,
                                           pU1, pU2, pU3, pU5, tauxx, tauyy,
                                           tauxy, qx, qy)
    itt = time.time() - tstart
    t_tot += dt
    
    if Testconv(u1, u):
        # check mass balance
        inmassflow = 0.
        outmassflow = 0.
        for l in range(ymax):
            inmassflow += Rho[l, 0]*u[l, 0]*dy
            outmassflow += Rho[l, -1]*u[l, -1]*dy
        print ("inlet mass flow = " + str(inmassflow))
        print ("outlet mass flow = " + str(outmassflow))
        break
    
    if i % 100 == 0:  # save data every 100 iteration
        with open("data.sav", 'wb') as fich:
            pickle.dump(Rho, fich)
            pickle.dump(u, fich)
            pickle.dump(v, fich)
            pickle.dump(T, fich)
            pickle.dump(e, fich)
            pickle.dump(Mu, fich)
            pickle.dump(k, fich)
            pickle.dump(P, fich)
            pickle.dump(Et, fich)
            pickle.dump(i, fich)
            pickle.dump(t_tot, fich)

    print ("iteration = " + str(i) + "    iteration time = " + str(itt))


with open("data.sav", 'wb') as fich:  # save data at the end
    pickle.dump(Rho, fich)
    pickle.dump(u, fich)
    pickle.dump(v, fich)
    pickle.dump(T, fich)
    pickle.dump(e, fich)
    pickle.dump(Mu, fich)
    pickle.dump(k, fich)
    pickle.dump(P, fich)
    pickle.dump(Et, fich)
    pickle.dump(i, fich)
    pickle.dump(t_tot, fich)

#plot
plt.figure(1)
plt.title('u')
plt.contourf(X, Y, u)
plt.grid()
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")

plt.figure(2)
plt.title('P')
plt.contourf(X, Y, P)
plt.grid()
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")

plt.figure(3)
plt.title('T')
plt.contourf(X, Y, T)
plt.grid()
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")

plt.figure(4)
plt.title('v')
plt.contourf(X, Y, v)
plt.grid()
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")

plt.show()


