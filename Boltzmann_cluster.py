# -*- coding: utf-8 -*-
"""
@author: Tobias Raum

Code used to calculate the evolution for one specific parameter set.
"""


import numpy as np
import matplotlib.pyplot as plt 
#import matplotlib
#import math
from scipy.integrate import ode
import scipy.integrate as intg
import scipy.special as spc
#import scipy.constants
from mpmath import apery
import pickle
import sys
import os

a = pickle.load(open('dofs_early_universe.p','rb'))   
c = pickle.load(open('dofs_early_universe_eps.p','rb'))
l = len(a[0]) 
b=[np.zeros(l),np.zeros(l)]
b_eps=[np.zeros(l),np.zeros(l)]
i=0
while i < l:
    b[0][i] = a[0][l-1-i]
    b[1][i] = a[1][l-i-1]
    b_eps[0][i] = c[0][l-1-i]
    b_eps[1][i] = c[1][l-i-1]
#    print(b[0][i],b[1][i])
    i+=1


def n_Zeq(x,M,g_starZ,m_Z):
    n = g_starZ * m_Z**2 * spc.kn(2,m_Z*x/M) / (2*np.pi**2*x/M)
    return n

def s(x,M,g_starS):
    s = 2*np.pi**2*g_starS/(45*(x/M)**3)
    return s

def Y_Zeq(x,M,g_starZ,m_Z,g_starS):
    Y = n_Zeq(x,M,g_starZ,m_Z)/s(x,M,g_starS)
    return Y

def Gamma_Z_to_NN(g_BL, m_Z, m_N):
    Gamma = (g_BL**2)/(24*np.pi*(m_Z**2)) * (m_Z**2 - 4*m_N**2)**(3/2)
#   
    return Gamma

def Gamma_Z_to_ff(g_BL,m_Z,m_f,q_BL,N_C): #Loop over all twelve fermions
    Gamma=0.0
    i=0
    while i<12:
        if 2*m_f[i]<m_Z:
            Gamma += (g_BL**2 * q_BL[i]**2 * N_C * (2*m_f[i]**2 + m_Z**2) * (m_Z**2 - 4*m_f[i]**2)**(1/2)/(12*np.pi*m_Z**2)) 
        i+=1
    return Gamma
        

def Gamma_phi_to_ZZ(m_phi,m_Z,alpha,v_BL2):
    Gamma = np.sqrt(m_phi**2 - 4* m_Z**2)/(16*np.pi*m_phi**2) * np.cos(alpha)**2 * (m_phi**4 - 4*m_phi**2*m_Z**2 + 12*m_Z**4)/(v_BL2)
    return Gamma

def p4EEsigmav_Zgamma(s,m_Z,g_BL,q_xf,q_f,m_f): #Variablen einfuegen! #Was ist q_xf? #eq28 #log-Funktion pruefen
    e=np.sqrt(4.*np.pi/137.)
    M2 = 0
    if s==m_Z**2 or s==4*m_f**2 or np.sqrt(s)==np.sqrt(s-4*m_f**2):
        M2 = 0
    else:    
        M2 = (g_BL*q_xf*e*q_f)**2 * 32/((s-m_Z**2)**2) * (np.sqrt(s)*(4*m_f**2*(s-m_Z**2) - 8*m_f**4 + m_Z**4 + s**2)/np.sqrt(s-4*m_f**2) * np.log((np.sqrt(s)+np.sqrt(s-4*m_f**2))/(np.sqrt(s)-np.sqrt(s-4*m_f**2))) - s*(4*m_f**2+s)-m_Z**4)
    res = 1./(16.*np.pi) * np.sqrt((s-4*m_f**2)/s) * M2 * 1./2. * (s-m_Z**2)/np.sqrt(s)
    return res

def p4EEsigmav_Zf(s,m_Z,g_BL,q_xf,q_f,m_f): #Variablen einfügen!
    e=np.sqrt(4.*np.pi/137.)
    M2=0
    if s==0 or m_f**2==s or s==(m_f-m_Z)**2 or s==(m_f+m_Z)**2 or s==m_Z**2-m_f**2+np.sqrt((s-(m_f-m_Z)**2)*(s-(m_f+m_Z)**2)) or s==m_Z**2-m_f**2-np.sqrt((s-(m_f-m_Z)**2)*(s-(m_f+m_Z)**2)):
        M2 = 0
    else:
        M2 = 8.*(g_BL*q_xf*e*q_f)**2/(s*(m_f**2-s)**2) * (-m_f**4 * (m_Z**2 + s) + m_f**2*s*(2*m_Z**2 + 15*s) + m_f**6 + s**2*(7*m_Z**2 + s) + 2*s**2 * (2*m_f**2*(m_Z**2 - 3*s)-3*m_f**4 - 2*m_Z**2*s + 2*m_Z**4+s**2)/np.sqrt((s-(m_f-m_Z)**2)*(s-(m_f+m_Z)**2)) * np.log((m_f**2-m_Z**2+s+np.sqrt((s-(m_f-m_Z)**2)*(s-(m_f+m_Z)**2)))/(m_f**2-m_Z**2+s-np.sqrt((s-(m_f-m_Z)**2)*(s-(m_f+m_Z)**2)))))
    res = 1./(16.*np.pi) * np.sqrt((s-m_f**2)/s) * M2 * 1./2. * np.sqrt((s-(m_f-m_Z)**2)*(s-(m_f+m_Z)**2)/s)
    return res

#def p4EEsigmanu_Zantif():
#    return 1#p4EEsigmanu_Zf()

def sigmav(x,M,m_Z,g_BL,q_xf,q_f,m_f,g_f,g_gamma,g_starZ):
    res = 0.
    i=0
    while i<12:
        res += 2*M/(32*x*np.pi**4)*intg.quad(lambda s: g_starZ*g_f[i]*p4EEsigmav_Zf(s,m_Z,g_BL,q_xf[i],q_f[i],m_f[i])*spc.kn(1,np.sqrt(s)*x/M),(m_Z+m_f[i])**2,np.inf,limit=50)[0]
        res += M/(32*x*np.pi**4)*intg.quad(lambda s: g_starZ*g_gamma*p4EEsigmav_Zgamma(s,m_Z,g_BL,q_xf[i],q_f[i],m_f[i])*spc.kn(1,np.sqrt(s)*x/M),max([m_Z**2,(2*m_f[i])**2]),np.inf,limit=50)[0]
        i+=1
    res += M/(32*x*np.pi**4)*intg.quad(lambda s: g_starZ*g_f[i]*p4EEsigmav_Zf(s,m_Z,g_BL,q_xf[i],q_f[i],m_f[i])*spc.kn(1,np.sqrt(s)*x/M),(m_Z+m_f[i])**2,np.inf,limit=50)[0]
    res += M/(32*x*np.pi**4)*intg.quad(lambda s: g_starZ*g_gamma*p4EEsigmav_Zgamma(s,m_Z,g_BL,q_xf[i],q_f[i],m_f[i])*spc.kn(1,np.sqrt(s)*x/M),max([m_Z**2,(2*m_f[i])**2]),np.inf,limit=50)[0]
    return res

def dt_dx(x,M,g_starS): #dt/dx = 1/Hx, H = (((pi^2 /30) * g_star * T^4)/(3*Mpl^2))**0.5
    H0 = (((np.pi**2/30)*g_starS*M**4)/(3*(2.44e18)**2))**0.5
#    print(H0,x,x/H0)
    return x/H0 

def dY_N_dt(x,M,m_Z,g_BL,m_N,g_starS,Y_Z):
    g_starS = np.interp(M/x, b[0], b[1])
    g_star_eps = np.interp(M/x, b_eps[0], b_eps[1])
    K_frac = 1.0
    if m_Z*x/M < 500.:
        K_frac = spc.kn(1,m_Z*x/M)/spc.kn(2,m_Z*x/M) #Bessel functions
#    print(K_frac)
    lGammar = Gamma_Z_to_NN(g_BL, m_Z, m_N) * K_frac
    dY = lGammar * Y_Z * dt_dx(x,M,g_starS)
    return dY

def dY_Z_dt(x,M,m_Z,g_BL,m_N,m_f,q_BL,q_f,N_C,alpha,g_starS,g_starZ,g_f,g_gamma,Y_Z):
#    v_BL2 = m_Z**2/(4.*g_BL**2)
    g_starS = np.interp(M/x, b[0], b[1])
    g_star_eps = np.interp(M/x, b_eps[0], b_eps[1])
    Y_Zeq_here = Y_Zeq(x,M,g_starZ,m_Z,g_starS)
    K_frac = 1.0
    if m_Z*x/M < 500.:
        K_frac = spc.kn(1,m_Z*x/M)/spc.kn(2,m_Z*x/M) #Bessel functions
#    print(m_Z*x/M, K_frac)
#    lGammar_phiZZ = Gamma_phi_to_ZZ(m_phi,m_Z,alpha,v_BL2)
    lGammar_Zij = (Gamma_Z_to_NN(g_BL,m_Z,m_N) + Gamma_Z_to_ff(g_BL,m_Z,m_f,q_BL,N_C))*K_frac
    sigmav_here = sigmav(x,M,m_Z,g_BL,q_BL,q_f,m_f,g_f,g_gamma,g_starZ)/n_Zeq(x,M,g_starZ,m_Z)
#    print([lGammar_Zij,sigmav_here,x])
    dY = - (lGammar_Zij + sigmav_here) * (Y_Z - Y_Zeq_here) * dt_dx(x,M,g_starS)
#    print(x,dY,Y_Z,Y_Zeq_here)
    return dY

def Y_eqf(T_i,T_x,g,g_starS):
    n = 3./4. * g * apery * T_i**3 / (np.pi**2)
    s = 2*np.pi**2*g_starS*T_x**3 / 45
    return n/s

def Y_eqb(T_i,T_x,g,g_starS):
    n = g * apery * T_i**3 / (np.pi**2)
    s = 2*np.pi**2*g_starS*T_x**3 / 45
    return n/s

#def Y_FD(T_i,T_x,g)


#begin of actual program
    

m_Z = float(sys.argv[2])
m_N = float(sys.argv[3])
g_BL = float(sys.argv[1])#2*1e-7#1e-10#2*1e-7
alpha = 0
q_BL = [1./3.,1./3.,1./3.,1./3.,1./3.,1./3.,-1,-1,-1,-1,-1,-1,-1]
m_f = [2.2e-3,4.7e-3,1.28,96.0e-3,173.1,4.18,0.511e-3,0.03e-9,105.66e-3,0.031e-9,1.777,0.059e-9,m_N]
q_el = [2./3.,-1./3.,2./3.,-1./3.,2./3.,-1./3.,-1,-1,-1,0,0,0,0]
#m_phi = 
N_C = 3
#g_starS = 106.75+2
g_star_eps = 116
g_Z = 3
g_gamma = 2
g_quark = 7./8. * 6
g_clepton = 7./8. * 2
g_neutrino = 7./8. * 2
g_f = [g_quark,g_quark,g_quark,g_quark,g_quark,g_quark,g_clepton,g_clepton,g_clepton,g_neutrino,g_neutrino,g_neutrino,g_neutrino]
T_rh = (30. * 3./(128. * np.pi**4 * g_star_eps))**0.25 * m_Z
T_P = T_rh/100.
x0_physical = 1./T_rh
M = 1.0/x0_physical

g_starS = np.interp(T_rh, b[0], b[1])
g_star_eps = np.interp(T_rh, b_eps[0], b_eps[1])
#print(M)

for temp_i in [0,1,2,3,4]:
    T_rh = (30. * 3./(128. * np.pi**4 * g_star_eps))**0.25 * m_Z
    g_star_eps = np.interp(T_rh, b_eps[0], b_eps[1])

def rhs(x,Y):
    return [dY_N_dt(x,M,m_Z,g_BL,m_N,g_starS,Y[1]),dY_Z_dt(x,M,m_Z,g_BL,m_N,m_f,q_BL,q_el,N_C,alpha,g_starS,g_Z,g_f,g_gamma,Y[1])]
#res = intg.solve_ivp(rhs,(0.01*M,0.1*M),(0.1e-10,2.0e-6))#100,4
#plt.loglog(res.t*M, res.y.T)
    


x0 = 1.
x1 = 1*M#1.*M#100*M
#res = intg.solve_ivp(rhs,(x0,x1),[1e-30,2.0e-14],t_eval=np.linspace(x0,x1,100000))#100,4
y0 = [0,0]#Y_Zeq(1.,M,g_Z,m_Z,g_starS)]#Y_eqb(T_P,T_rh,3,g_starS)]#Y_Zeq(1.,M,g_Z,m_Z,g_starS)]#Y_eqb(T_P,T_rh,3,g_starS)]#[0.1e-10,2.0e-6]#[0,0]#[Y_Zeq(1.,M,g_neutrino,m_N,g_starS),Y_Zeq(1.,M,g_Z,m_Z,g_starS)]#Y_eqf(T_P,T_rh,2,g_starS)
#r = ode(rhs).set_integrator('vode', method='bdf',rtol=1e-20,atol=1e-15)
r = ode(rhs).set_integrator('vode', method='bdf',rtol=1e-25,atol=1e-15)#30,20
r.set_initial_value(y0, x0)
xsol,ysol = [],[]
dt = 0.001#0.00001#0.001#0.000000000001
xsol.append(x0)
ysol.append(y0)
while r.successful() and r.t < x1:
    sol = r.integrate(r.t+dt)
#    print(r.t+dt, sol)
    xsol.append(r.t+dt)
    ysol.append(sol)
    if sol[0]>sol[1]*1e4:
        break
    
print(xsol[len(xsol)-1],ysol[len(xsol)-1])

testEq = Y_Zeq(np.array(xsol),M,g_Z,m_Z,g_starS)
#Y_Neq = Y_Zeq(np.array(xsol),M,g_neutrino,m_N,g_starS)
#print(Y_Neq)
#path = '/home/raum/Documents/Simulations/Data/Boltzmann/'
path = '/Data/'
#pickle.dump([xsol,ysol],open(path + 'Boltzmann_data.p','wb+'))

xsolM = []
for i in xsol:
    xsolM.append(i/M)

f = open(path + 'Boltzmann_data_g_' + str(g_BL) + '_mZ_' + str(m_Z) + '_dt_' + str(dt) + '_mN_' + str(m_N) + '.p','wb')
#with open(path+'Boltzmann_data_w.p','wb') as f:
pickle.dump([xsolM,ysol,testEq],f)
f.close()

#print(dY_Z_dt(1,1000,10,[10,100,1000],[3,4,5,6,7,8,1,1,1,0.01,0.01,0.01],[1./3.,1./3.,1./3.,1./3.,1./3.,1./3.,-1,-1,-1,-1,-1,-1],100000,3,0,130,10,1))
