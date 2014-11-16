# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 11:59:44 2014
exercice - chapter 9 - section 9.4
Incompressible Couette flow, pressure correction method
Anderson, John. Computational Fluid Dynamics. 1 edition.
New York: McGraw-Hill Science/Engineering/Math, 1995.
@author: ml971 hotmail.com
"""

from pylab import *


nx = 21  # for p
ny = 11  # for p
Ue = 1.
D = 0.01  # y-dist
L = 0.5  # x-dist
dx = L/(nx-1)
dy = D/(ny-1)
dt = 0.001
rho = 0.002377
mu = 0.0000003737
eps = 0.000000001  # precision

###grid
xx = np.linspace(0, L, nx)
yy = np.linspace(0, D, ny)
XX, YY = np.meshgrid(xx, yy)

###

######
#BC (-1 -> top and right; 0 -> bottom and left)
p = np.zeros((ny, nx))  # (rr, cc)  # p* in book
p_ = np.zeros_like(p)
u = np.zeros((ny, nx+1))
v = np.zeros((ny+1, nx+2))

u[-1, :] = Ue  # top
u[0, :] = 0.  # bottom

v[0, :] = 0.  # bottom
v[4, 14] = 0.5  # spike

p_[:, 0] = 0.  # inlet
v[:, 0] = 0.  # inlet

p_[:, -1] = 0.  # outlet

u_st = u.copy()  # u* at time step n
v_st = v.copy()

u_st_n = np.zeros_like(u)  # u* at time step n+1 (n=next)
v_st_n = np.zeros_like(v)
#end BC
########

#print ("p \n" + str(p)+ str(p.shape))
#print ("u \n" + str(u)+ str(u.shape))
#print ("v \n" + str(v)+ str(v.shape))

a = 2*(dt/dx**2 + dt/dy**2)
b = -dt/dx**2
c = -dt/dy**2


def v_(r, c):  # c=column; r=row
    return(0.5*(v[r, c]+v[r, c+1]))


def v__(r, c):
    return(0.5*(v[r+1, c] + v[r+1, c+1]))


def u_(r, c):
    return(0.5*(u[r-1, c-1] + u[r, c-1]))


def u__(r, c):
    return(0.5*(u[r-1, c] + u[r, c]))


def A_st(r, c):
    return(-((rho*u[r, c+1]**2 - rho*u[r, c-1]**2)/(2*dx) +
            ((rho*u[r+1, c]*v_(r+1, c))-(rho*u[r-1, c]*v__(r-1, c)))/(2*dy)) +
           mu*((u[r, c+1]-2*u[r, c]+u[r, c-1])/dx**2 +
            (u[r+1, c]-2*u[r, c]+u[r-1, c])/dy**2))


def B_st(r, c):
    return(-((rho*u_(r, c+1)*v[r, c+1] - rho*u__(r, c-1)*v[r, c-1])/(2*dx) +
            ((rho*v[r+1, c]**2) - (rho*v[r-1, c]**2))/(2*dy)) +
           mu*((v[r, c+1] - 2*v[r, c] + v[r, c-1])/dx**2 +
            (v[r+1, c] - 2*v[r, c] + v[r-1, c])/dy**2))


def rho_u(r, c):
    return(rho*u_st[r, c] + A_st(r, c)*dt - dt/dx * (p[r, c]-p[r, c-1]))


def rho_v(r, c):
    return(rho*v_st[r, c] + B_st(r, c)*dt - dt/dy*(p[r, c-1] - p[r-1, c-1]))


def p_calc(r, c):
    return(-1/a * (b*p_[r, c+1] + b*p_[r, c-1] +
                   c*p_[r+1, c] + c*p_[r-1, c] +
                   d_calc(r, c)))


def d_calc(r, c):
    return (1/dx*(rho*u_st_n[r, c+1]-rho*u_st_n[r, c]) +
            1/dy*(rho*v_st_n[r+1, c+1]-rho*v_st_n[r, c+1]))

figure(1)


it = 1
while it >= 1:
    print ("\n \n \n    iteration : " + str(it))

    for ll in range(1, ny-1):  # loop for u
        for cc in range(1, nx):
            u_st_n[ll, cc] = rho_u(ll, cc) / rho
    for ll in range(1, ny):   # loop for v
        for cc in range(1, nx+1):
            v_st_n[ll, cc] = rho_v(ll, cc) / rho

    u_st_n[:, 0] = u_st_n[:, 1]  # inlet
    u_st_n[:, -1] = u_st_n[:, -2]  # outlet
    v_st_n[:, -1] = v_st_n[:, -2]  # outlet
    u_st_n[-1, :] = Ue  # top
    #print ("u_st_n = u = u_st : \n" + str(u_st_n))
    #print ("v_st_n = v = v_st : \n" + str(v_st_n))

    #print ("p_ before loop : \n" + str(p_))
    print ("p_ max : " + str(p_.max()))
    cd = 1
    while cd >= 1:
        print ("internal loop : " + str(cd))
        buf = p_.copy()
        for ll in range(1, ny-1):
            for cc in range(1, nx-1):
                p_[ll, cc] = p_calc(ll, cc)
        print ("buf max : " + str(buf.max()))
        if abs(p_.max() - buf.max()) < eps:
            cd = -1
        cd += 1
    #print ("p_ after loop : \n" + str(p_))

    #print ("p before loop : \n" + str(p))
    tamp = p.copy()
    for ll in range(1, ny-1):
        for cc in range(1, nx-1):
            p[ll, cc] = p[ll, cc] + 0.1*p_[ll, cc]
    print ("p max : " + str(p.max()))
    print ("tamp max : " + str(tamp.max()))
    if abs(p.max() - tamp.max()) < eps:
        it = -1
    #print ("p after loop : \n" + str(p))

    u[:, :] = u_st_n[:, :].copy()
    v[:, :] = v_st_n[:, :].copy()
    u_st[:, :] = u_st_n[:, :].copy()
    v_st[:, :] = v_st_n[:, :].copy()

    if (it == 1 or it == 2 or it == 20 or it == 50 or it == 150 or it == 295):
        ###2D
        #figure(it)
        #quiver(XX, YY, u[:, :-1], v[:-1, :-2], label=it)
        #legend()
        ###1D
        plot(u[:, 10], yy, label=it)

    it += 1

legend()

#print ("p " + str(p)+ str(p.shape))
#print ("u " + str(u)+ str(u.shape))
#print ("v " + str(v)+ str(v.shape))

figure(2)
subplot(1, 1, 1)
plot(np.linspace(0, D, ny+1), v[:, 14])
show()
