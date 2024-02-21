# -*- coding: utf-8 -*-
"""
@author: Tobias Raum

Code used to perform the bisection search for parameter values with the correct dark matter abundance
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

def plotBoltzmann(g,mZ,dt,mN):
    myPath = '../Data/final_par/Boltzmann_data_g_'+str(g)+'_mZ_'+str(mZ)+'_dt_'+str(dt)+'_mN_'+str(mN)+'.p'
#print(data[1])
    if pth.isfile(myPath):
        plt.close()
        plt.rcParams['text.usetex'] = True
        data=pickle.load(open(myPath,'rb'))
#        fig = plt.figure()
#        a1 = fig.add_axes([0,0,1,1])
        plt.loglog(np.array(data[0])[1:],[i[0] for i in data[1][1:]],label=r'$Y_N$')
        plt.plot(np.array(data[0])[1:],np.array(data[2])[1:],label=r'$Y_{Z\prime}^{eq}$')
        plt.plot(np.array(data[0])[1:],[i[1] for i in data[1][1:]],label=r'$Y_{Z\prime}$')
        plt.axhline(y=0.12/2.8*1e-4, color='r',label=r'$\Omega h^2 = 0.12$')
#        a1.set_xlim(data[0][0],data[0][len(data[0])-1])
        plt.ylim(1e-12,1e-3)
        plt.xlabel(r'$\frac1T$ [$\frac1{GeV}$]')
        plt.ylabel(r'$Y_i = \frac{n_i}s$')
        plt.title(r'$g = $' + str(g) + r'; $m_{Z\prime} = $' + format(int(mZ),",") + r'GeV; $m_N = $' + format(int(mN*1e6),",") + r'keV')
        plt.legend()
        plt.savefig('../Plots/Boltzmann_rl/Boltzmann_plot_g_'+str(g)+'_mZ_'+str(mZ)+'_dt_'+str(dt)+'mN_'+str(mN)+'.png')
    else:
        print('Not a path: g=' + str(g)+'; mZ='+str(mZ))

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
    #return 0
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

def solve_Boltzmann(g_BL,m_Z,m_N):
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
#    T_P = T_rh/100.
    x0_physical = 1./T_rh
    M = 1.0/x0_physical
    
    g_starS = np.interp(T_rh, b[0], b[1])
    g_star_eps = np.interp(T_rh, b_eps[0], b_eps[1])
    #print(M)

    for tempi in [0,1,2,3,4]:
        T_rh = (30. * 3./(128. * np.pi**4 * g_star_eps))**0.25 * m_Z
        g_star_eps = np.interp(T_rh, b_eps[0], b_eps[1])
    
    def rhs(x,Y):
        return [dY_N_dt(x,M,m_Z,g_BL,m_N,g_starS,Y[1]),dY_Z_dt(x,M,m_Z,g_BL,m_N,m_f,q_BL,q_el,N_C,alpha,g_starS,g_Z,g_f,g_gamma,Y[1])]
    #res = intg.solve_ivp(rhs,(0.01*M,0.1*M),(0.1e-10,2.0e-6))#100,4
    #plt.loglog(res.t*M, res.y.T)
        
    
    
    x0 = 1.
    x1 = 1*M#1.*M#100*M
    #res = intg.solve_ivp(rhs,(x0,x1),[1e-30,2.0e-14],t_eval=np.linspace(x0,x1,100000))#100,4
    y0 = [0,0]#Y_Zeq(1.,M,g_Z,m_Z,g_starS)*1e-4]#Y_eqb(T_P,T_rh,3,g_starS)]#Y_Zeq(1.,M,g_Z,m_Z,g_starS)]#Y_eqb(T_P,T_rh,3,g_starS)]#[0.1e-10,2.0e-6]#[0,0]#[Y_Zeq(1.,M,g_neutrino,m_N,g_starS),Y_Zeq(1.,M,g_Z,m_Z,g_starS)]#Y_eqf(T_P,T_rh,2,g_starS)
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
            #print("Break sol[0]>sol[1]*1e4")
            break
        if sol[0] > 0.123/2.8 * 1e-4:
            break
    
    l = len(xsol)
    Omega_h_sq = m_N * ysol[l-1][0] * 2.8e8
        
    print("g_BL = " + str(g_BL) + "; m_Z = " + str(m_Z) + "; m_N = " + str(m_N), file=open('log/Boltzmann_scan_' + str(g_BL) + '.txt', 'a'))
    print(xsol[l-1],ysol[l-1],Omega_h_sq, file=open('log/Boltzmann_scan_' + str(g_BL) + '.txt','a'))
    
    testEq = Y_Zeq(np.array(xsol),M,g_Z,m_Z,g_starS)
    #Y_Neq = Y_Zeq(np.array(xsol),M,g_neutrino,m_N,g_starS)
    #print(Y_Neq)
    #path = '/home/raum/Documents/Simulations/Data/Boltzmann/'
    path = '../Data/final_par/'
    #pickle.dump([xsol,ysol],open(path + 'Boltzmann_data.p','wb+'))
    
    
    #fig, ax = plt.subplots(figsize = (10,6))
    #ax.plot(np.array(xsol)[1:]/M,[i[0] for i in ysol[1:]])
    #ax.plot(np.array(xsol)/M,testEq)
    #ax.plot(np.array(xsol)[1:]/M,[i[1] for i in ysol[1:]])
    #ax.plot(np.array(xsol)/M,Y_Neq)
    #ax.set_yscale("log")
    #ax.set_xscale("log")
    #ax.set_xlim(1/M,xsol[len(xsol)-1]/M)#3e-3)#1e-2
    #ax.set_ylim(1e-10,1e-3)#(1e-15,1e10)#
    #ax.savefig(path+'Boltzmann_plot',format='png')
    #note: Bessel functions get too close to zero for higher x-values than currently used, resulting in an error. We will need to adjust the code if we want to look at higher x values
    
    xsolM = []
    for i in xsol:
        xsolM.append(i/M)
    
    f = open(path + 'Boltzmann_data_g_' + str(g_BL) + '_mZ_' + str(m_Z) + '_dt_' + str(dt) + '_mN_' + str(m_N) + '.p','wb')
    #with open(path+'Boltzmann_data_w.p','wb') as f:
    pickle.dump([xsolM,ysol,testEq],f)
    f.close()
    
    #plotBoltzmann(g_BL,m_Z,dt,m_N)
    return(Omega_h_sq)

def bisection_Boltzmann(g_BL,m_N,m_Z_low=1e3,m_Z_high=1e7):
    print("Now scanning: g_BL = " + str(g_BL))
    Omega_low = solve_Boltzmann(g_BL,m_Z_low,m_N)
    Omega_high = solve_Boltzmann(g_BL,m_Z_high,m_N)
    if (Omega_low < 0.121 and Omega_low > 0.119):
        return([m_Z_low,Omega_low])
    if (Omega_high < 0.121 and Omega_high > 0.119):
        return([m_Z_high,Omega_high])
    if ((Omega_low > 0.121 and Omega_high > 0.121) or (Omega_low < 0.119 and Omega_high < 0.119)):
        print("Error! 0.12 not between the two values")
        return([0,0])
    if (Omega_low < 0.12 and Omega_high > 0.12):
        print("Error! Omega_low < 0.12 and Omega_high > 0.12")
        return ([0,0])
    
    m_Z_mid = 10**((np.log10(m_Z_low) + np.log10(m_Z_high))/2)
    Omega_mid = solve_Boltzmann(g_BL,m_Z_mid,m_N)
    testVar = 0
    if (Omega_mid < 0.121 and Omega_mid > 0.119):
        print("m_Z_mid found: " + str(m_Z_mid))
        testVar = 1
        #return([m_Z_mid,Omega_mid]) 
    while testVar == 0:
        #print("m_Z_mid_now = " + str(m_Z_mid) + "; Omega_mid = " + str(Omega_mid))
        if Omega_mid > 0.12:
            #print("low1")
            Omega_low = Omega_mid
            m_Z_low = m_Z_mid
            #print("low")
        else:
            #print("high1")
            Omega_high = Omega_mid
            m_Z_high = m_Z_mid
            #print("high")
        m_Z_mid = 10**((np.log10(m_Z_low) + np.log10(m_Z_high))/2)
        Omega_mid = solve_Boltzmann(g_BL,m_Z_mid,m_N)
        if (Omega_mid < 0.121 and Omega_mid > 0.119):
            testVar = 1
            print("For g_BL = " + str(g_BL) + " and m_N = " + str(m_N) + " found: m_Z = " + str(m_Z_mid) + "; Omega = " + str(Omega_mid),file=open('log/Boltzmann_scan_'+str(g_BL)+'.txt','a'))
    return([m_Z_mid,Omega_mid])
        
            
#bisection_Boltzmann(1e-6,1e-4,1e3,1e7)
#bisection_Boltzmann(1e-5,1e-4,1e5,1e7)
#bisection_Boltzmann(10**(-5.2),1e-4,1e5,1e7)
#bisection_Boltzmann(10**(-5.4),1e-4,1e5,1e7)
#bisection_Boltzmann(10**(-5.6),1e-4,1e4,1e6)
#bisection_Boltzmann(10**(-5.8),1e-4,1e4,1e6)
    
#bisection_Boltzmann(1e-5,1e-4,1e6,1e7)
#bisection_Boltzmann(1e-7,1e-4,600,1000)
#bisection_Boltzmann(1e-6,1e-4,1e4,1e5)

#m_Z = float(sys.argv[1])
#bisection_Boltzmann(10**(-5.3),1e-4,1e6,2.6e6)
