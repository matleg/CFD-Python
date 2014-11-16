# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 16:09:44 2014
exercice - chapter 8
2-D supersonic flow: Prandlt Meyer expansion wave
Anderson, John. Computational Fluid Dynamics. 1 edition.
New York: McGraw-Hill Science/Engineering/Math, 1995.
@author: ml971 hotmail.com
Special thanks to Ivan Padilla and Alberto Garca for sharing their matlab code
"""

import numpy as np
import math
import matplotlib.pyplot as plt

ny = 41
teta = 5.352*math.pi/180
E = 10.
gamma = 1.4
R = 8.314/0.0289645
H = 40.
Courant = 0.5
Cy = 0.6
d_eta = 1./(ny-1)  # j step in computational plane
distMax = 65


##height as a function of x[0]
def height(xx):
    if xx <= E:
        return H
    if xx > E:
        return (H+(xx-E)*math.tan(teta))


##definition of y(x): height of points
def y_calc(dist_x, x_E, h_var, nb_div):
    """distance x, x-coord of expansion corner, height from function "height"
    , y number of divisions"""
    if dist_x <= x_E:
        return np.linspace(0, h_var, nb_div)
    else:
        return np.linspace(H-h_var, H, nb_div)


##initial conditions at x = 0   [y, x]
p = np.zeros([ny, 1])   # y, x
p[:, 0] = 101000

rho = np.zeros([ny, 1])
rho[:, 0] = 1.23

T = np.zeros([ny, 1])
T[:, 0] = 286.1

u = np.zeros([ny, 1])
u[:, 0] = 678

v = np.zeros([ny, 1])
v[:, 0] = 0

M = np.zeros([ny, 1])
M[:, 0] = u[:, 0]/(gamma*R*T[:, 0])**0.5  # sqrt(gamma*(R/M)*T)

A = np.zeros([ny, 1])
B = np.zeros([ny, 1])
C = np.zeros([ny, 1])
#predictors
pu = np.zeros([ny, 1])
pv = np.zeros([ny, 1])
pp = np.zeros([ny, 1])
pT = np.zeros([ny, 1])
prho = np.zeros([ny, 1])

#buffer variable for calculations (to have a table.shape = (ny, 1) instead
#of (ny, _)
tamp = np.zeros([ny, 1])


##flux variables 1 to 4 (method to avoid: exec)
for i in ["F", "G", "dFdeps_", "dGdeps_", "pF", "pG", "pdFdeps_", "avdFdeps_"]:
    for j in range(1, 5):
        exec("{0}{1} = np.zeros([ny, 1])".format(i, j))
        #print ("{0}{1}".format(i, j) + str(eval("{0}{1}".format(i, j))))

F1[:, 0] = rho[:, 0]*u[:, 0]   # [height, iter]
F2[:, 0] = rho[:, 0]*u[:, 0]**2+p[:, 0]
F3[:, 0] = rho[:, 0]*u[:, 0]*v[:, 0]
F4[:, 0] = (gamma/(gamma-1)
            )*p[:, 0]*u[:, 0]+rho[:, 0]*u[:, 0]*((u[:, 0]**2+v[:, 0]**2)/2)

G1[:, 0] = rho[:, 0]*v[:, 0]
G2[:, 0] = rho[:, 0]*u[:, 0]*v[:, 0]
G3[:, 0] = rho[:, 0]*v[:, 0]**2+p[:, 0]
G4[:, 0] = (gamma/(gamma-1)
            )*p[:, 0]*v[:, 0]+rho[:, 0]*v[:, 0]*((u[:, 0]**2+v[:, 0]**2)/2)

machTable = np.arange(1, 4, 0.00001)


def angle(machTable):   # donne un angle en deg
    """calculation of expanstion angle for Mach between 1 and 10"""
    return ((((gamma+1)/(gamma-1))**0.5) * (
        math.atan(((gamma-1)*(machTable**2-1)/(gamma+1))**0.5)
        )/math.pi*180 - ((math.atan((machTable**2-1)**0.5)/math.pi*180)))

tab_f_cal = np.zeros_like(machTable)
for i in range(len(machTable)):
    tab_f_cal[i] = angle(machTable[i])
#print ("tab_f_cal "+str(tab_f_cal))


def find_nearest(array, value):
    """find index of Mach number corresponding to closest angle"""
    idx = (np.abs(array-value)).argmin()
    return idx


##iterations in the x-direction, Mac Cormack space marching
i = 0   # iteration number
xxx = 0.    # real x-distance
x_coords = np.array([0])


while xxx < distMax:
    lastMColumn = M[:, i]
     # print ("lastMColumn " +str(lastMColumn))
     # print(np.max(lastMColumn))
    d_eps = Courant*(height(xxx)/ny)/max(
        abs(math.tan(teta + math.asin(1./np.max(lastMColumn)))),
        abs(math.tan(teta - math.asin(1./np.max(lastMColumn)))))
     # print ("d_eps " +str(d_eps))

     # predictor
    for j in range(1, ny-1):    # y (height) [j, i] = [h, x] = [line, column]
        if xxx < E:
            detadx = 0
        if xxx > E:
            detadx = (1-j*d_eta)*math.tan(teta)/height(xxx)

        dFdeps_1[j, 0] = detadx*(F1[j, i]-F1[j+1, i]
                                 )/d_eta+(1/height(xxx)
                                          )*(G1[j, i]-G1[j+1, i])/d_eta
         # print ("dFdeps_1[j, 0] " + str(dFdeps_1[j, 0])+ " j = "+str(j))
        dFdeps_2[j, 0] = detadx*(F2[j, i]-F2[j+1, i]
                                 )/d_eta+(1/height(xxx)
                                          )*(G2[j, i]-G2[j+1, i])/d_eta
        dFdeps_3[j, 0] = detadx*(F3[j, i]-F3[j+1, i]
                                 )/d_eta+(1/height(xxx)
                                          )*(G3[j, i]-G3[j+1, i])/d_eta
         # print ("dFdeps_3[j, 0] " + str(dFdeps_3[j, 0])+ " j = "+str(j))
        dFdeps_4[j, 0] = detadx*(F4[j, i]-F4[j+1, i]
                                 )/d_eta+(1/height(xxx)
                                          )*(G4[j, i]-G4[j+1, i])/d_eta
         # always 0 and not i because no need to keep the values
         # (table with one column only)
         # d_eta = constant in computational plane

        SF1 = (Cy*abs(p[j+1, i]-2*p[j, i]+p[j-1, i])
               / (p[j+1, i]+2*p[j, i]+p[j-1, i]))*(F1[j+1, i]
                                                   - 2*F1[j, i]+F1[j-1, i])
        SF2 = (Cy*abs(p[j+1, i]-2*p[j, i]+p[j-1, i])
               / (p[j+1, i]+2*p[j, i]+p[j-1, i]))*(F2[j+1, i]
                                                   - 2*F2[j, i]+F2[j-1, i])
        SF3 = (Cy*abs(p[j+1, i]-2*p[j, i]+p[j-1, i])
               / (p[j+1, i]+2*p[j, i]+p[j-1, i]))*(F3[j+1, i]
                                                   - 2*F3[j, i]+F3[j-1, i])
         # print ("SF3 "+str(SF3))
        SF4 = (Cy*abs(p[j+1, i]-2*p[j, i]+p[j-1, i])
               / (p[j+1, i]+2*p[j, i]+p[j-1, i]))*(F4[j+1, i]
                                                   - 2*F4[j, i]+F4[j-1, i])

        pF1[j, 0] = F1[j, i]+dFdeps_1[j, 0]*d_eps + SF1
        pF2[j, 0] = F2[j, i]+dFdeps_2[j, 0]*d_eps + SF2
         # print ("pF2[j, 0] "+str(pF2[j, 0]))
        pF3[j, 0] = F3[j, i]+dFdeps_3[j, 0]*d_eps + SF3
        pF4[j, 0] = F4[j, i]+dFdeps_4[j, 0]*d_eps + SF4
         # print (pF4)

        A[j, 0] = pF3[j, 0]**2/(2*pF1[j, 0]) - pF4[j, 0]
         # print (A[j, 0])
        B[j, 0] = (gamma/(gamma-1))*pF1[j, 0]*pF2[j, 0]
         # print (B[j, 0])
        C[j, 0] = -((gamma+1)/(2*(gamma-1)))*pF1[j, 0]**3
         # print (C[j, 0])
        prho[j, 0] = (-B[j, 0]+(B[j, 0]**2-4*A[j, 0]*C[j, 0])**0.5)/(2*A[j, 0])
         # print (prho[j, 0])
        pu[j, 0] = pF1[j, 0]/prho[j, 0]
         # print (pu[j, 0])
        pv[j, 0] = pF3[j, 0]/pF1[j, 0]
         # print ("pv[j, 0] "+str(pv[j, 0]))
        pp[j, 0] = pF2[j, 0] - pF1[j, 0]*pu[j, 0]
        pT[j, 0] = pp[j, 0]/(prho[j, 0]*R)

        pG1[j, 0] = prho[j, 0]*pF3[j, 0]/pF1[j, 0]
        pG2[j, 0] = pF3[j, 0]
        pG3[j, 0] = prho[j, 0]*(pF3[j, 0]/pF1[j, 0]
                                )**2+pF2[j, 0]-pF1[j, 0]**2/prho[j, 0]
         # print ("pG3[j, 0] "+str(pG3[j, 0]))
        pG4[j, 0] = (gamma/(gamma-1)
                     )*(pF2[j, 0]-pF1[j, 0]**2/prho[j, 0]
                        )*(pF3[j, 0]/pF1[j, 0]
                           )+(prho[j, 0]/2
                              )*(pF3[j, 0]/pF1[j, 0]
                                 )*((pF1[j, 0]/prho[j, 0]
                                     )**2+(pF3[j, 0]/pF1[j, 0])**2)

    # boundary conditions for predicted values
    # bottom j = 0
    detadx = (1-0*d_eta)*math.tan(teta)/height(xxx)

    dFdeps_1[0, 0] = detadx*(F1[0, i]-F1[1, i]
                             )/d_eta+(1/height(xxx))*(G1[0, i]-G1[1, i])/d_eta
     # print ("dFdeps_1[0, 0] "+str(dFdeps_1[0, 0]))
    dFdeps_2[0, 0] = detadx*(F2[0, i]-F2[1, i]
                             )/d_eta+(1/height(xxx))*(G2[0, i]-G2[1, i])/d_eta
    dFdeps_3[0, 0] = detadx*(F3[0, i]-F3[1, i]
                             )/d_eta+(1/height(xxx))*(G3[0, i]-G3[1, i])/d_eta
     # print ("dFdeps_3[0, 0] "+str(dFdeps_3[0, 0]))
    dFdeps_4[0, 0] = detadx*(F4[0, i]-F4[1, i]
                             )/d_eta+(1/height(xxx))*(G4[0, i]-G4[1, i])/d_eta

    pF1[0, 0] = F1[0, i]+dFdeps_1[0, 0]*d_eps
     # print ("pF1[0, 0] " +str(pF1[0, 0])+ "  dFdeps_1[0, 0] "+ \
     # str(dFdeps_1[0, 0])+ " d_eps " +str(d_eps))
    pF2[0, 0] = F2[0, i]+dFdeps_2[0, 0]*d_eps
    pF3[0, 0] = F3[0, i]+dFdeps_3[0, 0]*d_eps
     # print ("pF3[0, 0] " +str(pF3[0, 0])+ "  dFdeps_3[0, 0] "+ \
     # str(dFdeps_3[0, 0])+ " d_eps " +str(d_eps))
    pF4[0, 0] = F4[0, i]+dFdeps_4[0, 0]*d_eps

    A[0, 0] = pF3[0, 0]**2/(2*pF1[0, 0]) - pF4[0, 0]
     # print (A[0, 0])
    B[0, 0] = (gamma/(gamma-1))*pF1[0, 0]*pF2[0, 0]
     # print (B[0, 0])
    C[0, 0] = -((gamma+1)/(2*(gamma-1)))*pF1[0, 0]**3
     # print (C[0, 0])
    rho_cal = (-B[0, 0]+(B[0, 0]**2-4*A[0, 0]*C[0, 0])**0.5)/(2*A[0, 0])
     # print ("rho_cal predict " +str(rho_cal))
    u_cal = pF1[0, 0]/rho_cal
     # print ("u_cal predict " +str(u_cal))
    v_cal = pF3[0, 0]/pF1[0, 0]
     # print ("v_cal predict " +str(v_cal))
    p_cal = pF2[0, 0] - pF1[0, 0]*u_cal
     # print ("p_cal predict " +str(p_cal))
    T_cal = p_cal/(rho_cal*R)
     # print ("T_cal predict " +str(T_cal))
    M_cal = (u_cal**2+v_cal**2)**0.5/(gamma*R*T_cal)**0.5
     # print ("M_cal bottom predict " +str(M_cal))

    prho[0, 0] = rho_cal
    pp[0, 0] = p_cal

    pG1[0, 0] = prho[0, 0]*pF3[0, 0]/pF1[0, 0]
    pG2[0, 0] = pF3[0, 0]
    pG3[0, 0] = prho[0, 0]*(pF3[0, 0]/pF1[0, 0]
                            )**2+pF2[0, 0]-pF1[0, 0]**2/prho[0, 0]
    pG4[0, 0] = (gamma/(gamma-1)
                 )*(pF2[0, 0]-pF1[0, 0]**2/prho[0, 0]
                    )*(pF3[0, 0]/pF1[0, 0]
                       )+(prho[0, 0]/2
                          )*(pF3[0, 0]/pF1[0, 0]
                             )*((pF1[0, 0]/prho[0, 0]
                                 )**2+(pF3[0, 0]/pF1[0, 0])**2)

    # top
    prho[ny-1, 0] = 1.23
    pu[ny-1, 0] = 678
    pv[ny-1, 0] = 0
    pp[ny-1, 0] = 101000
    pT[ny-1, 0] = 286.1

    pF1[ny-1, 0] = prho[ny-1, 0]*pu[ny-1, 0]
    pF2[ny-1, 0] = prho[ny-1, 0]*pu[ny-1, 0]**2+pp[ny-1, 0]
    pF3[ny-1, 0] = prho[ny-1, 0]*pu[ny-1, 0]*pv[ny-1, 0]
    pF4[ny-1, 0] = (gamma/(gamma-1)
                    )*(pp[ny-1, 0]*pu[ny-1, 0]
                       )+(prho[ny-1, 0]*pu[ny-1, 0]
                          )*((pu[ny-1, 0]**2+pv[ny-1, 0]**2)/2)

    pG1[ny-1, 0] = prho[ny-1, 0]*pF3[ny-1, 0]/pF1[ny-1, 0]
    pG2[ny-1, 0] = pF3[ny-1, 0]
    pG3[ny-1, 0] = prho[ny-1, 0]*(pF3[ny-1, 0]/pF1[ny-1, 0]
                                  )**2+(pF2[ny-1, 0]
                                        )-pF1[ny-1, 0]**2/prho[ny-1, 0]
    pG4[ny-1, 0] = (gamma/(gamma-1)
                    )*(pF2[ny-1, 0]-pF1[ny-1, 0]**2/prho[ny-1, 0]
                       )*(pF3[ny-1, 0]/pF1[ny-1, 0]
                          )*((pF1[ny-1, 0]/prho[ny-1, 0]
                              )**2+(pF3[ny-1, 0]/pF1[ny-1, 0])**2)
    # end of boundary conditions for predicted values

    F1 = np.hstack((F1, tamp))
    F2 = np.hstack((F2, tamp))
    F3 = np.hstack((F3, tamp))
    F4 = np.hstack((F4, tamp))

    rho = np.hstack((rho, tamp))
    u = np.hstack((u, tamp))
    v = np.hstack((v, tamp))
    p = np.hstack((p, tamp))
    T = np.hstack((T, tamp))
    M = np.hstack((M, tamp))

    # corrector steps
    for j in range(1, ny-1):
         # print (j)
        if xxx < E:
            detadx = 0
        if xxx > E:
            detadx = (1-j*d_eta)*math.tan(teta)/height(xxx)

        pdFdeps_1[j, 0] = detadx*(pF1[j-1, 0]-pF1[j, 0]
                                  )/d_eta+(1/height(xxx)
                                           )*(pG1[j-1, 0]-pG1[j, 0])/d_eta
        pdFdeps_2[j, 0] = detadx*(pF2[j-1, 0]-pF2[j, 0]
                                  )/d_eta+(1/height(xxx)
                                           )*(pG2[j-1, 0]-pG2[j, 0])/d_eta
        pdFdeps_3[j, 0] = detadx*(pF3[j-1, 0]-pF3[j, 0]
                                  )/d_eta+(1/height(xxx)
                                           )*(pG3[j-1, 0]-pG3[j, 0])/d_eta
        pdFdeps_4[j, 0] = detadx*(pF4[j-1, 0]-pF4[j, 0]
                                  )/d_eta+(1/height(xxx)
                                           )*(pG4[j-1, 0]-pG4[j, 0])/d_eta

        avdFdeps_1[j, 0] = 0.5*(dFdeps_1[j, 0]+pdFdeps_1[j, 0])
        avdFdeps_2[j, 0] = 0.5*(dFdeps_2[j, 0]+pdFdeps_2[j, 0])
        avdFdeps_3[j, 0] = 0.5*(dFdeps_3[j, 0]+pdFdeps_3[j, 0])
        avdFdeps_4[j, 0] = 0.5*(dFdeps_4[j, 0]+pdFdeps_4[j, 0])

        SF1 = (Cy*abs(pp[j+1, 0]-2*pp[j, 0]+pp[j-1, 0]
                      )/(pp[j+1, 0]+2*pp[j, 0]+pp[j-1, 0]
                         ))*(pF1[j+1, 0]-2*pF1[j, 0]+pF1[j-1, 0])
        SF2 = (Cy*abs(pp[j+1, 0]-2*pp[j, 0]+pp[j-1, 0]
                      )/(pp[j+1, 0]+2*pp[j, 0]+pp[j-1, 0]
                         ))*(pF2[j+1, 0]-2*pF2[j, 0]+pF2[j-1, 0])
        SF3 = (Cy*abs(pp[j+1, 0]-2*pp[j, 0]+pp[j-1, 0]
                      )/(pp[j+1, 0]+2*pp[j, 0]+pp[j-1, 0]
                         ))*(pF3[j+1, 0]-2*pF3[j, 0]+pF3[j-1, 0])
        SF4 = (Cy*abs(pp[j+1, 0]-2*pp[j, 0]+pp[j-1, 0]
                      )/(pp[j+1, 0]+2*pp[j, 0]+pp[j-1, 0]
                         ))*(pF4[j+1, 0]-2*pF4[j, 0]+pF4[j-1, 0])

        F1[j, i+1] = F1[j, i]+avdFdeps_1[j, 0]*d_eps + SF1
        F2[j, i+1] = F2[j, i]+avdFdeps_2[j, 0]*d_eps + SF2
        F3[j, i+1] = F3[j, i]+avdFdeps_3[j, 0]*d_eps + SF3
        F4[j, i+1] = F4[j, i]+avdFdeps_4[j, 0]*d_eps + SF4

        A[j, 0] = F3[j, i+1]**2/(2*F1[j, i+1]) - F4[j, i+1]
         # print (A[j, 0])
        B[j, 0] = (gamma/(gamma-1))*F1[j, i+1]*F2[j, i+1]
         # print (B[j, 0])
        C[j, 0] = -((gamma+1)/(2*(gamma-1)))*F1[j, i+1]**3
         # print (C[j, 0])

        rho[j, i+1] = (-B[j, 0]+(B[j, 0]**2-4*A[j, 0]*C[j, 0]
                                 )**0.5)/(2*A[j, 0])
         # print (rho[j, i+1])
        u[j, i+1] = F1[j, i+1]/rho[j, i+1]
        v[j, i+1] = F3[j, i+1]/F1[j, i+1]
        p[j, i+1] = F2[j, i+1] - F1[j, i+1]*u[j, i+1]
        T[j, i+1] = p[j, i+1]/(rho[j, i+1]*R)
        M[j, i+1] = (u[j, i+1]**2+v[j, i+1]**2)**0.5/(gamma*R*T[j, i+1])**0.5
         # print ("v[" +str(j)+", "+str(i+1)+"]"+str(v[j, i+1]))

    # boundary conditions for corrected values
    # bottom j = 0
    detadx = (1-0*d_eta)*math.tan(teta)/height(xxx)

    pdFdeps_1[0, 0] = detadx*(pF1[0, 0]-pF1[1, 0]
                              )/d_eta + (1/height(xxx)
                                         )*(pG1[0, 0]-pG1[1, 0])/d_eta
     # print ("pF1[1, 0] " +str(pF1[1, 0])+ "  pF1[0, 0] "+str(pF1[0, 0]))
    pdFdeps_2[0, 0] = detadx*(pF2[0, 0]-pF2[1, 0]
                              )/d_eta + (1/height(xxx)
                                         )*(pG2[0, 0]-pG2[1, 0])/d_eta
    pdFdeps_3[0, 0] = detadx*(pF3[0, 0]-pF3[1, 0]
                              )/d_eta + (1/height(xxx)
                                         )*(pG3[0, 0]-pG3[1, 0])/d_eta
    pdFdeps_4[0, 0] = detadx*(pF4[0, 0]-pF4[1, 0]
                              )/d_eta + (1/height(xxx)
                                         )*(pG4[0, 0]-pG4[1, 0])/d_eta

    avdFdeps_1[0, 0] = 0.5*(dFdeps_1[0, 0]+pdFdeps_1[0, 0])
    avdFdeps_2[0, 0] = 0.5*(dFdeps_2[0, 0]+pdFdeps_2[0, 0])
    avdFdeps_3[0, 0] = 0.5*(dFdeps_3[0, 0]+pdFdeps_3[0, 0])
    avdFdeps_4[0, 0] = 0.5*(dFdeps_4[0, 0]+pdFdeps_4[0, 0])

    F1[0, i+1] = F1[0, i]+avdFdeps_1[0, 0]*d_eps
     # print ("F1[0, i+1] " +str(F1[0, i+1])+ "  avdFdeps_1[0, 0] "+\
    # str(avdFdeps_1[0, 0])+ " d_eps " +str(d_eps))
    F2[0, i+1] = F2[0, i]+avdFdeps_2[0, 0]*d_eps
    F3[0, i+1] = F3[0, i]+avdFdeps_3[0, 0]*d_eps
    F4[0, i+1] = F4[0, i]+avdFdeps_4[0, 0]*d_eps

    A[0, 0] = F3[0, i+1]**2/(2*F1[0, i+1]) - F4[0, i+1]
     # print (A[0, 0])
    B[0, 0] = (gamma/(gamma-1))*F1[0, i+1]*F2[0, i+1]
     # print (B[0, 0])
    C[0, 0] = -((gamma+1)/(2*(gamma-1)))*F1[0, i+1]**3
     # print (C[0, 0])
    rho_cal = (-B[0, 0]+(B[0, 0]**2-4*A[0, 0]*C[0, 0])**0.5)/(2*A[0, 0])
     # print ("rho_cal correct " +str(rho_cal))
    u_cal = F1[0, i+1]/rho_cal
     # print ("u_cal correct " +str(u_cal))
    v_cal = F3[0, i+1]/F1[0, i+1]
     # print ("v_cal correct " +str(v_cal))
    p_cal = F2[0, i+1] - F1[0, i+1]*u_cal
     # print ("p_cal correct " +str(p_cal))
    T_cal = p_cal/(rho_cal*R)
     # print ("T_cal correct " +str(T_cal))
    M_cal = (u_cal**2+v_cal**2)**0.5/(gamma*R*T_cal)**0.5
     # print ("M_cal bottom correct " + str(M_cal))
    if xxx < E:
        psi = (math.atan(v_cal/u_cal))*180/math.pi
        phi = -psi
    else:
        psi = (math.atan(abs(v_cal)/u_cal))*180/math.pi
        phi = teta*180/math.pi - psi
     # print ("psi correct " +str(psi))
     # print ("phi correct " +str(phi))
    f_cal = angle(M_cal)
     # print ("f_cal correct " +str(f_cal))
    f_act = f_cal+phi
     # print ("f_act correct " +str(f_act))
    indice = find_nearest(tab_f_cal, f_act)
    M_act = machTable[indice]
     # print ("M_act " + str(M_act))
    p[0, i+1] = p_cal*((1+((gamma-1)/2)*M_cal**2
                        )/(1+((gamma-1)/2)*M_act**2))**(gamma/(gamma-1))
    T[0, i+1] = T_cal*((1+((gamma-1)/2)*M_cal**2)/(1+((gamma-1)/2)*M_act**2))
    rho[0, i+1] = p[0, i+1]/(R*T[0, i+1])
    u[0, i+1] = u_cal
    if xxx < E:
        v[0, i+1] = 0
    else:
        v[0, i+1] = -u[0, i+1]*math.tan(teta)
     # print ("v[:, i+1] " +str(v[:, i+1]))
    M[0, i+1] = M_act

    F1[0, i+1] = rho[0, i+1]*u[0, i+1]
    F2[0, i+1] = rho[0, i+1]*u[0, i+1]**2+p[0, i+1]
    F3[0, i+1] = rho[0, i+1]*u[0, i+1]*v[0, i+1]
    F4[0, i+1] = (gamma/(gamma-1)
                  )*(p[0, i+1]*u[0, i+1]
                     )+(rho[0, i+1]*u[0, i+1]
                        )*((u[0, i+1]**2+v[0, i+1]**2)/2)

    # top
    rho[ny-1, i+1] = 1.23
    u[ny-1, i+1] = 678
    v[ny-1, i+1] = 0
    p[ny-1, i+1] = 101000
    T[ny-1, i+1] = 286.1
    M[ny-1, i+1] = (u[ny-1, i+1]**2+v[ny-1, i+1]**2
                    )**0.5/(gamma*R*T[ny-1, i+1])**0.5

    F1[ny-1, i+1] = rho[ny-1, i+1]*u[ny-1, i+1]
    F2[ny-1, i+1] = rho[ny-1, i+1]*u[ny-1, i+1]**2+p[ny-1, i+1]
    F3[ny-1, i+1] = rho[ny-1, i+1]*u[ny-1, i+1]*v[ny-1, i+1]
    F4[ny-1, i+1] = (gamma/(gamma-1)
                     )*(p[ny-1, i+1]*u[ny-1, i+1]
                        )+(rho[ny-1, i+1]*u[ny-1, i+1]
                           )*((u[ny-1, i+1]**2+v[ny-1, i+1]**2)/2)
     # print ("M_top " + str(M[ny-1, i+1]))
   # end of boundary conditions

   # calculation of Gs at i+1
    G1 = np.hstack((G1, tamp))
    G2 = np.hstack((G2, tamp))
    G3 = np.hstack((G3, tamp))
    G4 = np.hstack((G4, tamp))
    G1[:, i+1] = rho[:, i+1]*v[:, i+1]
    G2[:, i+1] = rho[:, i+1]*u[:, i+1]*v[:, i+1]
    G3[:, i+1] = rho[:, i+1]*v[:, i+1]**2+p[:, i+1]
    G4[:, i+1] = (gamma/(gamma-1)
                  )*(p[:, i+1]*v[:, i+1]
                     )+(rho[:, i+1]*v[:, i+1]
                        )*((u[:, i+1]**2+v[:, i+1]**2)/2)
##    print ("G1[:, i+1] " +str(G1[:, i+1]))
##    print ("G2[:, i+1] " +str(G2[:, i+1]))
##    print ("G3[:, i+1] " +str(G3[:, i+1]))
##    print ("G4[:, i+1] " +str(G4[:, i+1]))
##
##    print ("F1[:, i+1] " +str(F1[:, i+1]))
##    print ("F2[:, i+1] " +str(F2[:, i+1]))
##    print ("F3[:, i+1] " +str(F3[:, i+1]))
##    print ("F4[:, i+1] " +str(F4[:, i+1]))

    i += 1
    xxx += d_eps
    x_coords = np.append(x_coords, xxx)

##    print ("\n  \n")
    print("loop " + str(i) + " dist " + str(xxx))

##################################plot grid
##plt.figure(1)
##for i in [1, 14, 24, 34, 44]:
##
##    plt.plot(u[:, i], range(ny), label = str(i))
##
##plt.legend()

x_new = x_coords.copy()
for i in range(ny-1):
    x_coords = np.vstack((x_coords, x_new))

y_coords = np.zeros(ny)
for i in range(len(x_new)-1):
    y_new = y_calc(x_new[i], E, height(x_new[i]), ny)
    y_coords = np.vstack((y_coords, y_new))

y_coords = y_coords.T

print ("x_coords shape "+str(x_coords.shape))
print ("y_coords shape "+str(y_coords.shape))

print ("u_shape   "+str(u.shape))
print ("v_shape   "+str(v.shape))

plt.figure(2)
plt.quiver(x_coords[::2, ::2], y_coords[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.contourf(x_coords, y_coords, (u**2+v**2)**0.5)
plt.colorbar()

plt.show()

####################################
####concatenate example
##a = np.zeros([3, 4, 5])
##print (a)
##b = np.ones([1, 4, 5])
##a = np.concatenate((a, b))
##print (a)
####################################

####################################
#############vstack exemple
##y = y_calc(60, 10, height(60), ny)
##print (y)
##y1 = y_calc(61, 10, height(61), ny)
##
##y = np.vstack((y, y1))
##print (y)
