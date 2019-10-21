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


n_sim = 10 #verify n_sim from DualMPC.py
run_time=10
n_st=7
n_ip =2
n_par=2
row_per = (1+n_st*6 + 1+n_ip*6 + 1+n_par*6)
discritize = 60*2

All_data =[]
res_xk = np.zeros((n_sim*n_st,run_time+1) )
res2_xk = np.zeros((n_sim*n_st,run_time+1) )
res4_xk = np.zeros((n_sim*n_st,run_time+1) )
res3_xk = np.zeros((n_sim*n_st,run_time+1) )
res5_xk = np.zeros((n_sim*n_st,run_time+1) )
res6_xk = np.zeros((n_sim*n_st,run_time+1) )

res_uk = np.zeros((n_sim*n_ip,run_time) )
res2_uk = np.zeros((n_sim*n_ip,run_time) )
res4_uk = np.zeros((n_sim*n_ip,run_time) )
res3_uk = np.zeros((n_sim*n_ip,run_time) )
res5_uk = np.zeros((n_sim*n_ip,run_time) )
res6_uk = np.zeros((n_sim*n_ip,run_time) )

res_theta = np.zeros((n_sim*n_par,run_time+1) )
res2_theta = np.zeros((n_sim*n_par,run_time+1) )
res4_theta = np.zeros((n_sim*n_par,run_time+1) )
res3_theta = np.zeros((n_sim*n_par,run_time+1) )
res5_theta = np.zeros((n_sim*n_par,run_time+1) )
res6_theta = np.zeros((n_sim*n_par,run_time+1) )

case2 = 'CE'
case3 = 'Dual'
case4 = 'Tube'
case5 = 'Tube+Dual'
case6 = 'Soft constraint Dual'


with open(file) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        All_data += [row]

def Mkplots(n_sim, xk, uk, theta, case2, x_opt, u_opt, t_opt):
    time = np.arange(run_time+1)
    time = time*discritize
    bl = np.ones(run_time+1)
    theta_act =  [3.0457*10**(-7)*discritize*.85, -323.05*.85] #  [ k0, dH ]
    theta_nom =  [3.0457*10**(-7)*discritize, -323.05] #  [ k0, dH ]

    uk_lb = [0,                     280]
    uk_ub = [9*10**(-6), 350]
    xk_lb = [0,      0,  0,   0,   321, 0,   0]
    xk_ub = [.007,  0, 0, 0, 325, 0, 0]

    opacity = .2

    plt.clf()
    plt.title(case2)
    plt.ylabel('$n_C$  [mol]')
    plt.ylim(0,2.0)
    plt.xlim(0,run_time*discritize)
    plt.legend(['$n_C$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        opt_xk = x_opt[i*n_st:(i+1)*n_st,:]
        opt_uk = u_opt[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        opt_t = t_opt[i*n_par:(i+1)*n_par,:]
        plt.plot(time,res1_xk[0,:]*res1_xk[3,:],'b-',alpha=opacity)
        plt.plot(time,opt_xk[0,:]*opt_xk[3,:],'p:')
    plt.savefig('./'+folder+case2+'_a.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$T \ [K]$')
    plt.ylim(320,330)
    plt.xlim(0,run_time*discritize)
    plt.legend(['$T$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        opt_xk = x_opt[i*n_st:(i+1)*n_st,:]
        opt_uk = u_opt[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        opt_t = t_opt[i*n_par:(i+1)*n_par,:]
        plt.plot(time,res1_xk[4,:],'b-',alpha=opacity)
        plt.plot(time, opt_xk[4,:],'p:')
        plt.plot(time,bl*321,'r-')
        plt.plot(time,bl*325,'r-')
        if sum(sum(opt_xk)) !=0:
            plt.plot(time,bl*bounds[0]*xk_ub[4],'r--')
            plt.plot(time,bl*bounds[1]*xk_lb[4],'r--')
    plt.savefig('./'+folder+case2+'_b.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$V \ [m^3]$')
    plt.ylim(0.003,.01)
    plt.xlim(0,run_time*discritize)
    plt.legend(['$V$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        opt_xk = x_opt[i*n_st:(i+1)*n_st,:]
        opt_uk = u_opt[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        opt_t = t_opt[i*n_par:(i+1)*n_par,:]
        plt.plot(time,res1_xk[0,:],'b-',alpha=opacity)
        plt.plot(time, opt_xk[0,:],'p:')
        plt.plot(time,bl*.007,'r-')
    plt.savefig('./'+folder+case2+'_g.png')

    if 0:
        plt.clf()
        plt.title(case2)
        plt.ylabel('$T_{J}$ [K]')
        plt.ylim(278,352)
        plt.xlim(0,run_time*discritize)
        plt.legend(['$T_{J}$'])
        plt.grid()
        for i in np.arange(0, n_sim):
            res1_xk = xk[i*n_st:(i+1)*n_st,:]
            res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
            opt_xk = x_opt[i*n_st:(i+1)*n_st,:]
            opt_uk = u_opt[i*n_ip:(i+1)*n_ip,:]
            res1_theta = theta[i*n_par:(i+1)*n_par,:]
            opt_t = t_opt[i*n_par:(i+1)*n_par,:]
            plt.step(time,res1_xk[5,:],'b-',alpha=opacity)
            plt.step(time, opt_xk[5,:],'p:')
            plt.step(time,bl*280,'r-' )
            plt.step(time,bl*350,'r-' )
        #plt.step(time,bl*bounds[2]*uk_ub[0],'r--')
        #plt.step(time,bl*bounds[3]*uk_lb[0],'r--')
        plt.savefig('./'+folder+case2+'_h.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$\dot{V_{in}} \ [m^3/s]$')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-.1*10**-6, 9.1*10**-6))
    plt.ylim(-.1*10**-6, 9.1*10**-6)
    plt.xlim(0,run_time*discritize)
    plt.legend(['$\dot{V}_{in}$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        opt_xk = x_opt[i*n_st:(i+1)*n_st,:]
        opt_uk = u_opt[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        opt_t = t_opt[i*n_par:(i+1)*n_par,:]
        plt.step(time[0:-1],res1_uk[0,:],'b-',alpha=opacity)
        plt.step(time[0:-1],opt_uk[0,:],'p:')
    plt.step(time,bl*uk_ub[0],'r-')
    plt.step(time,bl*uk_lb[0],'r-')
    if sum(sum(opt_xk)) !=0:
        plt.step(time,bl*uk_ub[0]*bounds[2],'r--')

    plt.savefig('./'+folder+case2+'_c.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$T_{J,in-set}$ [K]')
    plt.ylim(278,352)
    plt.xlim(0,run_time*discritize)
    plt.legend(['$T_{J,in-set}$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        opt_xk = x_opt[i*n_st:(i+1)*n_st,:]
        opt_uk = u_opt[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        opt_t = t_opt[i*n_par:(i+1)*n_par,:]
        plt.step(time[0:-1],res1_uk[1,:],'b-',alpha=opacity)
        plt.step(time[0:-1], opt_uk[1,:],'p:')
        plt.step(time[0:-1],bl[0:-1]*280,'r-' )
        plt.step(time[0:-1],bl[0:-1]*350,'r-' )

    plt.step(time,bl*bounds[2]*uk_ub[0],'r--')
    plt.step(time,bl*bounds[3]*uk_lb[0],'r--')
    plt.savefig('./'+folder+case2+'_d.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$k_0 \ [m^3 \ mol^{-1} \ s^{-1}]$')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(3.0457*10**(-7)*discritize*.65, 3.0457*10**(-7)*discritize*1.2))
    plt.ylim(3.0457*10**(-7)*discritize*.65, 3.0457*10**(-7)*discritize*1.2)
    plt.xlim(0,run_time*discritize)
    plt.legend(['$k_0$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        opt_xk = x_opt[i*n_st:(i+1)*n_st,:]
        opt_uk = u_opt[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        opt_t = t_opt[i*n_par:(i+1)*n_par,:]
        plt.plot(time,res1_theta[0,:],'b-',alpha=opacity)
        plt.plot(time,opt_t[0,:],'p:')
        plt.plot(time,bl*theta_act[0],'g--')
        plt.plot(time,bl*theta_nom[0],'r--')
    plt.savefig('./'+folder+case2+'_e.png')

    plt.clf()
    plt.title(case2)
    plt.ylabel('$\Delta H \ [kj \ mol^{-1}]$')
    plt.ylim(-340,-240)
    plt.xlim(0,run_time*discritize)
    plt.legend(['$\Delta H$'])
    plt.grid()
    for i in np.arange(0, n_sim):
        res1_xk = xk[i*n_st:(i+1)*n_st,:]
        res1_uk = uk[i*n_ip:(i+1)*n_ip,:]
        opt_xk = x_opt[i*n_st:(i+1)*n_st,:]
        opt_uk = u_opt[i*n_ip:(i+1)*n_ip,:]
        res1_theta = theta[i*n_par:(i+1)*n_par,:]
        opt_t = t_opt[i*n_par:(i+1)*n_par,:]
        plt.plot(time,res1_theta[1,:],'b-',alpha=opacity)
        plt.plot(time,opt_t[1,:],'p:')
        plt.plot(time,bl*theta_act[1],'g--')
        plt.plot(time,bl*theta_nom[1],'r--')
    plt.savefig('./'+folder+case2+'_f.png')

def comp_par_error(n_sim, res2_theta, res3_theta):
    time = np.arange(run_time)
    time = time* discritize
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
    plt.xlim(0,run_time*discritize)
    plt.legend(['CE','Dual'])
    plt.grid()
    plt.plot(time,theta2[0,:]/n_sim,time,theta3[0,:]/n_sim)
    plt.savefig('./'+folder+'k0_error.png')

    plt.clf()
    plt.title('Avg $\Delta H \ error^2$')
    plt.ylabel('avg $\Delta H \ error^2$')
    plt.plot(time,theta2[1,:]/n_sim,time,theta3[1,:]/n_sim)
    plt.xlim(0,run_time*discritize)
    plt.legend(['CE','Dual'])
    plt.grid()
    plt.savefig('./'+folder+'dH_error.png')

for i in np.arange(0, n_sim):
    ## States
    ##############
    res_xk[i*n_st,:] = All_data[i*row_per+1][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st,:] = All_data[i*row_per+2][2].replace('[','').replace(']','').replace('\n','').split()
    res4_xk[i*n_st,:] = All_data[i*row_per+3][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st,:] = All_data[i*row_per+4][2].replace('[','').replace(']','').replace('\n','').split()
    res5_xk[i*n_st,:] = All_data[i*row_per+5][2].replace('[','').replace(']','').replace('\n','').split()
    res6_xk[i*n_st,:] = All_data[i*row_per+6][2].replace('[','').replace(']','').replace('\n','').split()

    res_xk[i*n_st+1,:] = All_data[i*row_per+7][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+1,:] = All_data[i*row_per+8][2].replace('[','').replace(']','').replace('\n','').split()
    res4_xk[i*n_st+1,:] = All_data[i*row_per+9][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+1,:] = All_data[i*row_per+10][2].replace('[','').replace(']','').replace('\n','').split()
    res5_xk[i*n_st+1,:] = All_data[i*row_per+11][2].replace('[','').replace(']','').replace('\n','').split()
    res6_xk[i*n_st+1,:] = All_data[i*row_per+12][2].replace('[','').replace(']','').replace('\n','').split()

    res_xk[i*n_st+2,:] = All_data[i*row_per+13][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+2,:] = All_data[i*row_per+14][2].replace('[','').replace(']','').replace('\n','').split()
    res4_xk[i*n_st+2,:] = All_data[i*row_per+15][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+2,:] = All_data[i*row_per+16][2].replace('[','').replace(']','').replace('\n','').split()
    res5_xk[i*n_st+2,:] = All_data[i*row_per+17][2].replace('[','').replace(']','').replace('\n','').split()
    res6_xk[i*n_st+2,:] = All_data[i*row_per+18][2].replace('[','').replace(']','').replace('\n','').split()

    res_xk[i*n_st+3,:] = All_data[i*row_per+19][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+3,:] = All_data[i*row_per+20][2].replace('[','').replace(']','').replace('\n','').split()
    res4_xk[i*n_st+3,:] = All_data[i*row_per+21][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+3,:] = All_data[i*row_per+22][2].replace('[','').replace(']','').replace('\n','').split()
    res5_xk[i*n_st+3,:] = All_data[i*row_per+23][2].replace('[','').replace(']','').replace('\n','').split()
    res6_xk[i*n_st+3,:] = All_data[i*row_per+24][2].replace('[','').replace(']','').replace('\n','').split()

    res_xk[i*n_st+4,:] = All_data[i*row_per+25][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+4,:] = All_data[i*row_per+26][2].replace('[','').replace(']','').replace('\n','').split()
    res4_xk[i*n_st+4,:] = All_data[i*row_per+27][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+4,:] = All_data[i*row_per+28][2].replace('[','').replace(']','').replace('\n','').split()
    res5_xk[i*n_st+4,:] = All_data[i*row_per+29][2].replace('[','').replace(']','').replace('\n','').split()
    res6_xk[i*n_st+4,:] = All_data[i*row_per+30][2].replace('[','').replace(']','').replace('\n','').split()

    res_xk[i*n_st+5,:] = All_data[i*row_per+31][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+5,:] = All_data[i*row_per+32][2].replace('[','').replace(']','').replace('\n','').split()
    res4_xk[i*n_st+5,:] = All_data[i*row_per+33][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+5,:] = All_data[i*row_per+34][2].replace('[','').replace(']','').replace('\n','').split()
    res5_xk[i*n_st+5,:] = All_data[i*row_per+35][2].replace('[','').replace(']','').replace('\n','').split()
    res6_xk[i*n_st+5,:] = All_data[i*row_per+36][2].replace('[','').replace(']','').replace('\n','').split()

    res_xk[i*n_st+6,:] = All_data[i*row_per+37][2].replace('[','').replace(']','').replace('\n','').split()
    res2_xk[i*n_st+6,:] = All_data[i*row_per+38][2].replace('[','').replace(']','').replace('\n','').split()
    res4_xk[i*n_st+6,:] = All_data[i*row_per+39][2].replace('[','').replace(']','').replace('\n','').split()
    res3_xk[i*n_st+6,:] = All_data[i*row_per+40][2].replace('[','').replace(']','').replace('\n','').split()
    res5_xk[i*n_st+6,:] = All_data[i*row_per+41][2].replace('[','').replace(']','').replace('\n','').split()
    res6_xk[i*n_st+6,:] = All_data[i*row_per+42][2].replace('[','').replace(']','').replace('\n','').split()

    ## Parameters
    ##############
    res_theta[i*n_par,:] = All_data[i*row_per+44][2].replace('[','').replace(']','').replace('\n','').split()
    res2_theta[i*n_par,:] = All_data[i*row_per+45][2].replace('[','').replace(']','').replace('\n','').split()
    res4_theta[i*n_par,:] = All_data[i*row_per+46][2].replace('[','').replace(']','').replace('\n','').split()
    res3_theta[i*n_par,:] = All_data[i*row_per+47][2].replace('[','').replace(']','').replace('\n','').split()
    res5_theta[i*n_par,:] = All_data[i*row_per+48][2].replace('[','').replace(']','').replace('\n','').split()
    res6_theta[i*n_par,:] = All_data[i*row_per+49][2].replace('[','').replace(']','').replace('\n','').split()

    res_theta[i*n_par+1,:] = All_data[i*row_per+50][2].replace('[','').replace(']','').replace('\n','').split()
    res2_theta[i*n_par+1,:] = All_data[i*row_per+51][2].replace('[','').replace(']','').replace('\n','').split()
    res4_theta[i*n_par+1,:] = All_data[i*row_per+52][2].replace('[','').replace(']','').replace('\n','').split()
    res3_theta[i*n_par+1,:] = All_data[i*row_per+53][2].replace('[','').replace(']','').replace('\n','').split()
    res5_theta[i*n_par+1,:] = All_data[i*row_per+54][2].replace('[','').replace(']','').replace('\n','').split()
    res6_theta[i*n_par+1,:] = All_data[i*row_per+55][2].replace('[','').replace(']','').replace('\n','').split()

    ## Inputs
    ##############
    res_uk[i*n_par,:] = All_data[i*row_per+57][2].replace('[','').replace(']','').replace('\n','').split()
    res2_uk[i*n_par,:] = All_data[i*row_per+58][2].replace('[','').replace(']','').replace('\n','').split()
    res4_uk[i*n_par,:] = All_data[i*row_per+59][2].replace('[','').replace(']','').replace('\n','').split()
    res3_uk[i*n_par,:] = All_data[i*row_per+60][2].replace('[','').replace(']','').replace('\n','').split()
    res5_uk[i*n_par,:] = All_data[i*row_per+61][2].replace('[','').replace(']','').replace('\n','').split()
    res6_uk[i*n_par,:] = All_data[i*row_per+62][2].replace('[','').replace(']','').replace('\n','').split()

    res_uk[i*n_par+1,:] = All_data[i*row_per+63][2].replace('[','').replace(']','').replace('\n','').split()
    res2_uk[i*n_par+1,:] = All_data[i*row_per+64][2].replace('[','').replace(']','').replace('\n','').split()
    res4_uk[i*n_par+1,:] = All_data[i*row_per+65][2].replace('[','').replace(']','').replace('\n','').split()
    res3_uk[i*n_par+1,:] = All_data[i*row_per+66][2].replace('[','').replace(']','').replace('\n','').split()
    res5_uk[i*n_par+1,:] = All_data[i*row_per+67][2].replace('[','').replace(']','').replace('\n','').split()
    res6_uk[i*n_par+1,:] = All_data[i*row_per+68][2].replace('[','').replace(']','').replace('\n','').split()





Mkplots(n_sim, res2_xk, res2_uk, res2_theta, case2, res_xk*0, res_uk*0,res_theta*0)
Mkplots(n_sim, res3_xk, res3_uk, res3_theta, case3, res_xk*0, res_uk*0,res_theta*0)
Mkplots(n_sim, res4_xk, res4_uk, res4_theta, case4, res_xk, res_uk,res_theta)
Mkplots(n_sim, res5_xk, res5_uk, res5_theta, case5, res_xk, res_uk,res_theta)
Mkplots(n_sim, res6_xk, res6_uk, res6_theta, case6, res_xk*0, res_uk*0,res_theta*0)






#Mkplots(n_sim, res3_xk, res3_uk, res3_theta, case3)
#comp_par_error(n_sim, res2_theta, res3_theta)
