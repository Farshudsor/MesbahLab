"""
Created on March 4th 2019

@author: Farshud Sorourifar

plot multiple simulations of DualMPC.py from Results.csv
"""
import sys
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import dual2 as Dual
import pdb
import csv


n_sim = 20 #verify n_sim from DualMPC.py
run_time=20
n_st=7
n_ip =2
n_par=2
row_per = (1+n_st*2 + 1+n_ip*2 + 1+n_par*2)
discritize = 60

All_data =[]
res2_xk = np.zeros((n_sim*n_st,20) )
res3_xk = np.zeros((n_sim*n_st,20) )
res2_uk = np.zeros((n_sim*n_ip,19) )
res3_uk = np.zeros((n_sim*n_ip,19) )
res2_theta = np.zeros((n_sim*n_par,20) )
res3_theta = np.zeros((n_sim*n_par,20) )

case2 = 'CE'
case3 = 'Dual'

with open('Results_20.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    for row in readCSV:
    #    import pdb; pdb.set_trace()
        All_data += [row]

def Mkplots(n_sim, xk, uk, theta, case2, discritize=60):
    time = np.arange(20)
    time = time*60
    bl = np.ones(run_time)
    theta_act =  [3.0457*10**(-7)*discritize*.85, -323.05*.85] #  [ k0, dH ]
    theta_nom =  [3.0457*10**(-7)*discritize, -323.05] #  [ k0, dH ]

    plt.clf()
    plt.title(case2)
    plt.ylabel('$n_C$  [mol]')
    plt.ylim(0,2.5)
    plt.xlim(0,1200)
    plt.legend(['$n_C$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        plt.plot(time,res1_xk[0,:]*res1_xk[3,:],'b-')
    plt.savefig(case2+'_a.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$T \ [K]$')
    plt.ylim(320,326)
    plt.xlim(0,1200)
    plt.legend(['$T$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        plt.plot(time,res1_xk[4,:],'b-',time,bl*321,'r--',time,bl*325,'r--')
    plt.savefig(case2+'_b.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$V \ [m^3]$')
    plt.ylim(0,.014)
    plt.xlim(0,1200)
    plt.legend(['$V$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        plt.plot(time,res1_xk[0,:],'b-',time,bl*.007,'r--')
    plt.savefig(case2+'_g.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$\dot{V_{in}} \ [m^3/s]$')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-.1*10**-6, 9.1*10**-6))
    plt.ylim(-.1*10**-6, 9.1*10**-6)
    plt.xlim(0,1200)
    plt.legend(['$\dot{V}_{in}$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        plt.step(time[0:-1],res1_uk[0,:],'b-')
    plt.savefig(case2+'_c.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$T_{J,in-set}$ [K]')
    plt.ylim(278,352)
    plt.xlim(0,1200)
    plt.legend(['$T_{J,in-set}$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        plt.step(time[0:-1],res1_uk[1,:],'b-',time[0:-1],bl[0:-1]*280,'r--',time[0:-1],bl[0:-1]*350,'r--' )
    plt.savefig(case2+'_d.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$k_0 \ [m^3 \ mol^{-1} \ s^{-1}]$')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(3.0457*10**(-7)*discritize*.65, 3.0457*10**(-7)*discritize*1.2))
    plt.ylim(3.0457*10**(-7)*discritize*.65, 3.0457*10**(-7)*discritize*1.2)
    plt.xlim(0,1200)
    plt.legend(['$k_0$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        plt.plot(time,res1_theta[0,:],'b-',time,bl*theta_act[0],'g--',time,bl*theta_nom[0],'r--')
    plt.savefig(case2+'_e.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$\Delta H \ [kj \ mol^{-1}]$')
    plt.ylim(-323.05*1.2,-323.05*.65)
    plt.xlim(0,1200)
    plt.legend(['$\Delta H$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        plt.plot(time,res1_theta[1,:],'b-',time,bl*theta_act[1],'g--',time,bl*theta_nom[1],'r--')
    plt.savefig(case2+'_f.png')

def comp_par_error(n_sim, res2_theta, res3_theta):
    time = np.arange(20)
    discritize = (120/2)
    time = time*60
    bl = np.ones(run_time)
    theta_act =  [3.0457*10**(-7)*discritize*.85, -323.05*.85] #  [ k0, dH ]
    theta_nom =  [3.0457*10**(-7)*discritize, -323.05] #  [ k0, dH ]

    theta2 = np.zeros((n_par,run_time))
    theta3 = np.zeros((n_par,run_time))
    for i in np.arange(0, n_sim):
        #import pdb; pdb.set_trace()
        theta2 = theta2 + (res2_theta[i*n_par:(i+1)*n_par,:] - np.transpose(np.ones((run_time,n_par))*theta_act))**2
        theta3 = theta3 + (res3_theta[i*n_par:(i+1)*n_par,:] - np.transpose(np.ones((run_time,n_par))*theta_act))**2

    plt.clf()
    plt.title('Avg $k_0 \ error^2$')
    plt.ylabel('avg $k_0 \ error^2$')
    plt.xlim(0,1200)
    plt.legend(['CE','Dual'])
    plt.grid()
    plt.plot(time,theta2[0,:]/n_sim,time,theta3[0,:]/n_sim)
    plt.savefig('k0_error.png')

    plt.clf()
    plt.title('Avg $\Delta H \ error^2$')
    plt.ylabel('avg $\Delta H \ error^2$')
    plt.plot(time,theta2[1,:]/n_sim,time,theta3[1,:]/n_sim)
    plt.xlim(0,1200)
    plt.legend(['CE','Dual'])
    plt.grid()
    plt.savefig('dH_error.png')

for i in np.arange(0, n_sim):
    res2_xk[i*n_st,:] = All_data[i*row_per+1][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st,:] = All_data[i*row_per+2][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+1,:] = All_data[i*row_per+3][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+1,:] = All_data[i*row_per+4][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+2,:] = All_data[i*row_per+5][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+2,:] = All_data[i*row_per+6][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+3,:] = All_data[i*row_per+7][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+3,:] = All_data[i*row_per+8][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+4,:] = All_data[i*row_per+9][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+4,:] = All_data[i*row_per+10][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+5,:] = All_data[i*row_per+11][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+5,:] = All_data[i*row_per+12][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+6,:] = All_data[i*row_per+13][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+6,:] = All_data[i*row_per+14][2].replace('[','').replace(']','').replace('\n','').split()

    res2_theta[i*n_par,:] = All_data[i*row_per+16][2].replace('[','').replace(']','').replace('\n','').split()
    res3_theta[i*n_par,:] = All_data[i*row_per+17][2].replace('[','').replace(']','').replace('\n','').split()
    res2_theta[i*n_par+1,:] = All_data[i*row_per+18][2].replace('[','').replace(']','').replace('\n','').split()
    res3_theta[i*n_par+1,:] = All_data[i*row_per+19][2].replace('[','').replace(']','').replace('\n','').split()

    res2_uk[i*n_par,:] = All_data[i*row_per+21][2].replace('[','').replace(']','').replace('\n','').split()
    res3_uk[i*n_par,:] = All_data[i*row_per+22][2].replace('[','').replace(']','').replace('\n','').split()
    res2_uk[i*n_par+1,:] = All_data[i*row_per+23][2].replace('[','').replace(']','').replace('\n','').split()
    res3_uk[i*n_par+1,:] = All_data[i*row_per+24][2].replace('[','').replace(']','').replace('\n','').split()

Mkplots(n_sim, res2_xk, res2_uk, res2_theta, case2)
Mkplots(n_sim, res3_xk, res3_uk, res3_theta, case3)
comp_par_error(n_sim, res2_theta, res3_theta)
