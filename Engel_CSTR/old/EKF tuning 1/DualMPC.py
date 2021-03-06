# -*- coding: utf-8 -*-
"""
Created on July 4th 2018

@author: Farshud Sorourifar
"""
import sys
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from scipy import linalg
from casadi.tools import *
import dual2 as Dual
import pdb
#import results as P
import csv

#########################################
##  Set-up
######################################
csv_file = "Results.csv"
save2csv = True
p_plots = True

# Run Case 1, 2, and 3
runC2 = True
runC3 = True

if save2csv:
    File = open(csv_file, 'wb')
    with File:
        writer = csv.writer(File)

run_time = 20
#Disturbance = 25 # time when k0 begins to be reduced
#end_Disturbance = 65 # k0 has been incresed by 10%
#slope_Disturbance = (7.2*(10**10)/10 - 7.92*(10**10)/10 ) / (end_Disturbance - Disturbance)

#Disturbance2 = 25 # time when Delta H begins to be reduced
#end_Disturbance2 = 65 #  Delta H has been reduced by 10%
#slope_Disturbance2 = ( -5.*(10**4) - -5.*(10**4) *1.1 ) / (end_Disturbance2 - Disturbance2)

Tsamp = 2
discritize = (120)/2 # [=] seconds/(2min) - discritze in two min
#Define the controller
n_pred = 10
n_ctrl = 10

#Define the problem size
n_st = 7
n_ip = 2
n_op = 2
n_par = n_op

#MySolver = "sqpmethod"
MySolver = "ipopt"
opts = {}
if MySolver == "sqpmethod":
    opts["qpsol"] = "qpoases"
    opts["qpsol_options"] = {"printLevel":"none"}
elif MySolver == "ipopt":
    opts = {"ipopt.print_level":5, "print_time": True, 'ipopt.max_iter':1000, 'output_file':'Main_out.txt'}

#Generate the random numbers
Sigmak_p = np.diag([10**-17, 5, 5, 5, .2, .2, .2])
vkp = (np.random.randn(run_time,n_st))
vkp = mtimes(vkp,Sigmak_p)

#Build results storage
time = np.zeros((run_time,1))
time_pred = np.zeros((n_pred,1))
Cost_opt = 0.0
Cost_ce = 0.0
Cost_dual = 0.0

    #Optimal control with known parameters
act_theta = np.zeros((run_time,n_par))
    #Case 2 - CE control with unknown parameters
res2_xk = np.zeros((run_time,n_st))
res2_theta = np.zeros((run_time,n_par))
res2_uk = np.zeros((run_time-1,n_ip))
    #Case 3 - Dual control with unknown parameters
res3_xk = np.zeros((run_time,n_st))
res3_theta = np.zeros((run_time,n_par))
res3_uk = np.zeros((run_time-1,n_ip))

# theta actual for plotting
#theta_act =  [3.0457*10**(-7)*discritize*.85, -323.05*.85] #  [ k0, dH ]
#theta_nom =  [3.0457*10**(-7)*discritize, -323.05] #  [ k0, dH ]

#for i in range(run_time):
#    if i >= Disturbance and i <= end_Disturbance :
#        theta_act[0] = theta_act[0] - slope_Disturbance
#    if i >= Disturbance2 and i <= end_Disturbance2 :
#        theta_act[1] = theta_act[1] - slope_Disturbance2
#    act_theta[i] = [theta_act[0],theta_act[1]]

def Mkplots(time, res1_xk, res1_uk, res1_theta, case, a,b,c, discritize=discritize):
    time = time*60
    bl = np.ones(run_time)
    theta_act =  [3.0457*10**(-7)*discritize*.85, -323.05*.85] #  [ k0, dH ]
    theta_nom =  [3.0457*10**(-7)*discritize, -323.05] #  [ k0, dH ]

    #import pdb; pdb.set_trace()
    if a:
        #a = plt.figure(set[0])
        #plt.clf()
        #plt.subplot(2,1,1)
        #plt.title(case)
        #plt.subplot(2, 1, 1)
        #plt.ylim(.45,.6)
        #plt.ylabel('$C_A$')
        #plt.plot(time,res1_xk[:,0], time, np.ones(run_time)*.485)
        #plt.legend(['$C_A$'])
        #plt.grid()
        #plt.subplot(2, 1, 2)
        #plt.ylim(340,360)
        #plt.ylabel('$T_r$')
        #plt.plot(time,res1_xk[:,1])
        #plt.legend(['$T_r$'])
        #plt.grid()
        a = plt.figure(1)
        plt.clf()
        plt.title(case)
        plt.ylabel('$n_C$  [mol]')
        plt.ylim(0,2.5)
        plt.xlim(0,1200)
        plt.plot(time,res1_xk[:,0]*res1_xk[:,3])
        plt.legend(['$n_C$'])
        plt.grid()
        plt.savefig(case+'_a.png')

        d = plt.figure(2)
        plt.clf()
        plt.title(case)
        plt.ylabel('$T \ [K]$')
        plt.ylim(320,326)
        plt.plot(time,res1_xk[:,4],time,bl*321,'r--',time,bl*325,'r--')
        plt.xlim(0,1200)
        plt.legend(['$T$'])
        plt.grid()
        plt.savefig(case+'_b.png')

    if b:
        b = plt.figure(3)
        plt.clf()
        plt.title(case)
        plt.ylabel('$\dot{V_{in}} \ [m^3/s]$')
        plt.ylim(-.1*10**-6, 9.1*10**-6)
        plt.step(time[0:-1],res1_uk[:,0])
        plt.xlim(0,1200)
        plt.legend(['$\dot{V}_{in}$'])
        plt.grid()
        plt.savefig(case+'_c.png')

        c = plt.figure(4)
        plt.clf()
        plt.title(case)
        plt.ylabel('$T_{J,in-set}$ [K]')
        plt.ylim(278,352)
        plt.step(time[0:-1],res1_uk[:,1],time[0:-1],bl[0:-1]*280,'r--',time[0:-1],bl[0:-1]*350,'r--' )
        plt.xlim(0,1200)
        plt.legend(['$T_{J,in-set}$'])
        plt.grid()
        plt.savefig(case+'_d.png')

    if c:
        e = plt.figure(5)
        plt.clf()
        plt.title(case)
        plt.ylabel('$k_0 \ [m^3 \ mol^{-1} \ s^{-1}]$')
        plt.ylim(3.0457*10**(-7)*discritize*.65, 3.0457*10**(-7)*discritize*1.2)
        plt.plot(time,res1_theta[:,0],time,bl*theta_act[0],'g--',time,bl*theta_nom[0],'r--')
        plt.xlim(0,1200)
        plt.legend(['$k_0$'])
        plt.grid()
        plt.savefig(case+'_e.png')

        f = plt.figure(6)
        plt.clf()
        plt.title(case)
        plt.ylabel('$\Delta H \ [kj \ mol^{-1}]$')
        plt.ylim(-323.05*1.2,-323.05*.65)
        plt.plot(time,res1_theta[:,1],time,bl*theta_act[1],'g--',time,bl*theta_nom[1],'r--')
        plt.xlim(0,1200)
        plt.legend(['$\Delta H$'])
        plt.grid()
        plt.savefig(case+'_f.png')



def writeData(uk_sp,xk_sp, theta_act,res2_uk,res2_xk,res2_theta,res3_uk,res3_xk,res3_theta,run_time,csv_file):
    File = open(csv_file, 'w')
    with File:
        writer = csv.writer(File)
        writer.writerow( ["Parameter", "Actual values", "Case 2 : CE", "Case3 : Dual", '' ,"diffrence in controlers", "CE offset", "Dual offset"])

        for i1 in range(run_time):
            writer.writerow( ['xk_1', xk_sp[0], res2_xk[i1,0], res3_xk[i1,0], '', res2_xk[i1,0]-res3_xk[i1,0], xk_sp[0]-res2_xk[i1,0], xk_sp[0]-res3_xk[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk_2', xk_sp[1], res2_xk[i1,1], res3_xk[i1,1], '', res2_xk[i1,1]-res3_xk[i1,1], xk_sp[1]-res2_xk[i1,1], xk_sp[1]-res3_xk[i1,1] ] )
        for i1 in range(run_time-1):
            writer.writerow( ['uk_1', uk_sp[0], res2_uk[i1,0], res3_uk[i1,0], '', res2_uk[i1,0]-res3_uk[i1,0], uk_sp[0]-res2_uk[i1,0], uk_sp[0]-res3_uk[i1,0] ] )
        for i1 in range(run_time-1):
            writer.writerow( ['uk_2', uk_sp[1], res2_uk[i1,1], res3_uk[i1,1], '', res2_uk[i1,1]-res3_uk[i1,1], uk_sp[1]-res2_uk[i1,1], uk_sp[1]-res3_uk[i1,1] ] )

        for i1 in range(run_time):
            writer.writerow( ['t1', theta_act[i1,0], res2_theta[i1,0], res3_theta[i1,0], '', res2_theta[i1,0]-res3_theta[i1,0], theta_act[i1,0]-res2_theta[i1,0], theta_act[i1,0]-res3_theta[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['t2', theta_act[i1,1], res2_theta[i1,1], res3_theta[i1,1], '', res2_theta[i1,1]-res3_theta[i1,1], theta_act[i1,1]-res2_theta[i1,1], theta_act[i1,1]-res3_theta[i1,1] ] )


def writeData_2(res2_uk,res2_xk,res2_theta,res3_uk,res3_xk,res3_theta,run_time,csv_file):
    File = open(csv_file, 'w')
    with File:
        writer = csv.writer(File)
        writer.writerow(['   States   '])
        writer.writerow([ 'CE', 'Vol' ,res2_xk[:,0]])
        writer.writerow([ 'Dual', 'Vol', res3_xk[:,0]])
        writer.writerow([])
        writer.writerow([ 'CE', 'Ca' ,res2_xk[:,1]])
        writer.writerow([ 'Dual', 'Ca', res3_xk[:,1]])
        writer.writerow([])
        writer.writerow([ 'CE', 'Cb' ,res2_xk[:,2]])
        writer.writerow([ 'Dual', 'Cb', res3_xk[:,2]])
        writer.writerow([])
        writer.writerow([ 'CE', 'Cc', res2_xk[:,3]])
        writer.writerow([ 'Dual', 'Cc', res3_xk[:,3]])
        writer.writerow([])
        writer.writerow([ 'CE', 'T', res2_xk[:,4]])
        writer.writerow([ 'Dual', 'T', res3_xk[:,4]])
        writer.writerow([])
        writer.writerow([ 'CE', 'Tj', res2_xk[:,5]])
        writer.writerow([ 'Dual', 'Tj', res3_xk[:,5]])
        writer.writerow([])
        writer.writerow([ 'CE', 'Tj_in', res2_xk[:,6]])
        writer.writerow([ 'Dual', 'Tj_in', res3_xk[:,6]])

        writer.writerow(['   Parameters   '])
        writer.writerow([ 'CE', 'k0', res2_theta[:,0]])
        writer.writerow([ 'Dual', 'k0', res3_theta[:,0]])
        writer.writerow([])
        writer.writerow([ 'CE', 'Delta H', res2_theta[:,1]])
        writer.writerow([ 'Dual', 'Delta H', res3_theta[:,1]])

        writer.writerow(['   Inputs   '])
        writer.writerow([ 'CE', 'V_in', res2_uk[:,0]])
        writer.writerow([ 'Dual', 'V_in', res3_uk[:,0]])
        writer.writerow([])
        writer.writerow([ 'CE', 'Tj_in', res2_uk[:,1]])
        writer.writerow([ 'Dual', 'Tj_in', res3_uk[:,1]])

#########################################
##  Begin Simulations
######################################
if runC3:
    #########################################
    ##  Define system
    ######################################
    #Define the system states
    xk3 = SX.sym('xk3',n_st)
    uk3 = SX.sym('uk3',n_ip)
    wk3 = SX.sym('wk3',n_st)
    vk3 = SX.sym('vk3',n_st)
    thetah3 = SX.sym('thetah3',n_par)
    zk3 = vertcat(xk3,thetah3)

    #######################
    #model
    ###########
    #Variables
    #V0 =x[0]                 %Volume
    #Ca = x[1]                %concentration
    #Cb = x[2]
    #Cc = x[3]
    #T0 = x[4]                 %Temp
    #Tj = x[5]
    #Tj_in = x[6]

    #Inputs
    #Vin= u[0]              %volumetric flow rate
    #Tjin = u[1]               %Coolant temp

    #Estimate parameters
    #k0 = theta[0]               %kinetic constant
    #dH = theta[1]               %reaction heat

    #known parameters
    rho =  1000            #density
    cp =  4.2              #heat capacity
    r = .092                #radius
    Vj = 2.22 * 10**(-3)
    Vjin = 9.167*10**(-5)*discritize
    U = .14844*discritize
    tau_c = 900/discritize
    Cbin = 3000
    Tin = 300              #inlet temp
    Vmax = 7*10**(-3)
    V0 =  3.5*10**(-3)               #volume
    Ca0 = 2000               #concentration of species A inlet
    Cb0 =  0
    Cc0 = 0
    T0 = 325
    Tj0 = 325
    Tj0_in = 325
    pi = 3.14159265

    theta_nom = np.array([3.0457*10**(-7)*discritize, -323.05]) #  [ k0, dH ]
    theta_act = np.array([3.0457*10**(-7)*discritize*.85, -323.05*.85]) #  [ k0, dH ]

    #Parameter statistics (2.3)/(5.3-5.4)
    Q = np.diag([[10**-17, 5, 5, 5, .2, .2, .2]])
    Qz = np.diag([10**-17, 5, 5, 5, .2, .2, .2, (3.0457*10**(-7))*discritize*4E-9, 323.05 * .9E1 ])
    R = np.eye(n_st)

    #Lower and upper bound on inputs and states
    uk_lb = [0,                     280]
    uk_ub = [9*10**(-6)*discritize, 350]
    xk_lb = [0,      0,  0,   0,   321, 0,   0]
    xk_ub = [.007,  inf, inf, inf, 325, inf, inf]


    #Define the system equations with unknown parameters
    de1 = uk3[0]
    de2 = -(uk3[0]/xk3[0])*xk3[1] - thetah3[0]*xk3[1]*xk3[2]
    de3 = (uk3[0]/xk3[0])*(Cbin-xk3[2]) - thetah3[0]*xk3[1]*xk3[2]
    de4 = -(uk3[0]/xk3[0])*xk3[3] + thetah3[0]*xk3[1]*xk3[2]
    de5 = (uk3[0]/xk3[0])*(Tin - xk3[4]) - (U*((2*xk3[0]/r)-pi*r**2)*(xk3[4]-xk3[5]))/(rho*cp*xk3[0]) - (thetah3[0]*xk3[1]*xk3[2]*thetah3[1])/(rho*cp)
    de6 = (Vjin/Vj) * (xk3[6]-xk3[4]) + (U*((2*xk3[0]/r)-pi*r**2)*(xk3[4]-xk3[5]))/(rho*cp*Vj)
    de7 = (1/tau_c)* (uk3[1]-xk3[6])
    dthetah1 = [0]*thetah3[0]
    dthetah2 = [0]*thetah3[1]


    sys_ode = vertcat(de1,de2,de3,de4,de5,de6,de7)
    mdl_ode =vertcat(de1,de2,de3,de4,de5,de6,de7,dthetah1,dthetah2)

    #Jacobians for predictions
    Jz3 = Function('Jz3',[xk3,uk3,thetah3],[jacobian((vertcat(xk3,thetah3)+mdl_ode),vertcat(xk3,thetah3))])
    Ju3 = Function('Ju3',[xk3,uk3,thetah3],[jacobian((vertcat(xk3,thetah3)+mdl_ode),uk3)])

    ode = {'x':xk3, 'p':vertcat(thetah3,uk3), 'ode':sys_ode}
    F_ode = integrator('F_ode', 'cvodes', ode)

    m_ode = {'x':zk3, 'p':uk3, 'ode':mdl_ode }
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    #Output equation (Observations)
    C = np.eye(n_st)
    fy = mtimes(C,xk3) + vk3
    h = Function('h',[xk3,vk3], [fy])
    Ch = Function('Ch',[vertcat(xk3,thetah3)],[jacobian(fy,vertcat(xk3,thetah3))])
    Chx = Function('Chx',[xk3],[jacobian(fy,xk3)])

    uk_opt = np.array([2.5*10**(-6)*discritize,280])
    xkp = np.array([V0, Ca0,Cb0,Cc0,T0,Tj0,Tj0_in])
    theta_par = theta_nom
    xkh0 = xkp
    zkh0 =vertcat(xkh0,theta_par)
    Sigmak_p = Qz

    res3_uk[0,:] = np.array([2.5*10**(-6),280])
    res3_xk[0,:] = xkp
    res3_theta[0,:] = theta_par

    for k in range(run_time-1):
        time[k+1] = k+1

        time_remain = run_time - k
        if time_remain < n_pred:
            n_pred = time_remain

        #if time[k] >= Disturbance and time[k] <= end_Disturbance :
        #    theta_act[0] = theta_act[0] -slope_Disturbance
        #    print('******************************************')
        #    print('******        k2 is reduced           ****')
        #    print('******************************************')
        #if time[k] >= Disturbance2 and time[k] <= end_Disturbance2 :
        #    theta_act[1] = theta_act[1] -slope_Disturbance2
        #    print('******************************************')
        #    print('******        Delta H is reduced           ****')
        #    print('******************************************')

        #Compute the measurement
        ykp = h(xkp,vkp[k,:].T)
        #KF update step/measurment
        Czh = Ch(zkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
        #import pdb; pdb.set_trace()
        zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
        xkh0 = zkh0[0:n_st]
        theta_par = zkh0[n_st:]
        Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)

        # generate CE control sequence for nominal trajectory
        Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(F_ode,n_pred,n_ctrl,n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk3,theta_par,uk3,Tsamp,xkh0,uk_opt)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        ce_res = res_mpc['x'].full().flatten()
        uk_opt1 = ce_res[n_st:n_st+n_ip]

        xce1 = ce_res[0::n_st+n_ip]
        xce2 = ce_res[1::n_st+n_ip]
        xce3 = ce_res[2::n_st+n_ip]
        xce4 = ce_res[3::n_st+n_ip]
        xce5 = ce_res[4::n_st+n_ip]
        xce6 = ce_res[5::n_st+n_ip]
        xce7 = ce_res[6::n_st+n_ip]
        xk_ce_3 = horzcat(xce1,xce2,xce3,xce4,xce5,xce6,xce7)
        uce1 = ce_res[7::n_st+n_ip]
        uce2 = ce_res[8::n_st+n_ip]
        uk_ce_3 = horzcat(uce1,uce2)
        #uk_opt = uk_ce_3[1,:]
        #pdb.set_trace()

        #Build and solve the dual optimization problem
        Jd, qu, op = Dual.gencost(n_st,n_ip,n_op,n_par,n_pred,Sigmak,mdl_ode,fy,xk3,uk3,thetah3,Tsamp,Q,Qz,R,uk_ce_3,xkh0,theta_par,xk_ce_3)
        nlp = {'x':vertcat(*qu), 'f': Jd, 'p': op}
        solver = nlpsol('solver', MySolver, nlp, {'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
        res = solver( x0=uk_opt1, p=xkh0, ubx=uk_ub, lbx=uk_lb, lbg=0, ubg=0)
        uk_opt = res['x'].full().flatten()

        #Simulate the system
        x_end = F_ode(x0=xkp, p=vertcat(theta_act,uk_opt))
        xkp = x_end['xf'].full()
        xkp = xkp.T #+ wkp[k,:]

        #KF prediction
        Az = Jz3(xkh0,uk_opt,theta_par)
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz

        if k < run_time-2:
            # Save results
            uk_opt[0]=uk_opt[0]/discritize
            res3_uk[k+1,:] = uk_opt
            res3_xk[k+1,:] = xkp
            res3_theta[k+1,:] = [theta_par[0],theta_par[1]]

        else:
            res3_xk[k+1,:] = xkp
            res3_theta[k+1,:] = [theta_par[0],theta_par[1]]

    Mkplots(time, res3_xk, res3_uk, res3_theta, 'Dual Control', 1,1,1)


if runC2:
    #########################################
    ##  Define system
    ######################################
    #Define the system states
    xk2 = SX.sym('xk2',n_st)
    uk2 = SX.sym('uk2',n_ip)
    wk2 = SX.sym('wk2',n_st)
    vk2 = SX.sym('vk2',n_st)
    thetah2 = SX.sym('thetah2',n_par)
    zk2 = vertcat(xk2,thetah2)

    #######################
    #model
    ###########
    #variables
    #V0 =x[0]
    #Ca = x[1]                %concentration of species A
    #Cb = x[2]
    #Cc = x[3]
    #T0 = x[4]
    #Tj = x[5]
    #Tj_in = x[6]

    #Inputs
    #Vin= u[0]              %volumetric flow rate
    #Tjin = u[1]               %Coolant temp

    # estimate parameters
    #k0 = theta[0]               %kinetic constant
    #dH = theta[1]               %reaction heat

    #known parameters
    rho =  1000            #density
    cp =  4.2              #heat capacity
    r = .092                #radius
    Vj = 2.22 * 10**(-3)
    Vjin = 9.167*10**(-5)*discritize
    U = .14844*discritize
    tau_c = 900/discritize
    Cbin = 3000
    Tin = 300              #inlet temp
    Vmax = 7*10**(-3)
    V0 =  3.5*10**(-3)               #volume
    Ca0 = 2000               #concentration of species A inlet
    Cb0 =  0
    Cc0 = 0
    T0 = 325
    Tj0 = 325
    Tj0_in = 325
    pi = 3.14159265

    theta_nom = np.array([3.0457*10**(-7)*discritize, -323.05]) #  [ k0, dH ]
    theta_act = np.array([3.0457*10**(-7)*discritize*.85, -323.05*.85]) #  [ k0, dH ]

    #Parameter statistics (2.3)/(5.3-5.4)
    Q = np.diag([[10**-17, 5, 5, 5, .2, .2, .2]])
    Qz = np.diag([10**-17, 5, 5, 5, .2, .2, .2, (3.0457*10**(-7))*discritize*2E-9, 323.05 * .9E1 ])
    R = np.eye(n_st)

    #Lower and upper bound on inputs and states
    uk_lb = [0,          280]
    uk_ub = [9*10**(-6)*discritize, 350]
    xk_lb = [0,      0,  0,   0,   321, 0,   0]
    xk_ub = [.007,  inf, inf, inf, 325, inf, inf]

    #Define the system equations with unknown parameters
    de1 = uk2[0]
    de2 = -(uk2[0]/xk2[0])*xk2[1] - thetah2[0]*xk2[1]*xk2[2]
    de3 = (uk2[0]/xk2[0])*(Cbin-xk2[2]) - thetah2[0]*xk2[1]*xk2[2]
    de4 = -(uk2[0]/xk2[0])*xk2[3] + thetah2[0]*xk2[1]*xk2[2]
    de5 = (uk2[0]/xk2[0])*(Tin - xk2[4]) - (U*((2*xk2[0]/r)-pi*r**2)*(xk2[4]-xk2[5]))/(rho*cp*xk2[0]) - (thetah2[0]*xk2[1]*xk2[2]*thetah2[1])/(rho*cp)
    de6 = (Vjin/Vj) * (xk2[6]-xk2[4]) + (U*((2*xk2[0]/r)-pi*r**2)*(xk2[4]-xk2[5]))/(rho*cp*Vj)
    de7 = (1/tau_c)* (uk2[1]-xk2[6])
    dthetah1 = [0]*thetah2[0]
    dthetah2 = [0]*thetah2[1]

    sys_ode = vertcat(de1,de2,de3,de4,de5,de6,de7)
    mdl_ode =vertcat(de1,de2,de3,de4,de5,de6,de7,dthetah1,dthetah2)

    #Jacobians for predictions
    Jx2 = Function('Jx',[xk2,uk2,thetah2],[jacobian((xk2+sys_ode),xk2)])
    Jz2 = Function('Jz',[xk2,uk2,thetah2],[jacobian((vertcat(xk2,thetah2)+mdl_ode),vertcat(xk2,thetah2))])
    Ju2 = Function('Ju',[xk2,uk2,thetah2],[jacobian((vertcat(xk2,thetah2)+mdl_ode),uk2)])

    ode2 = {'x':xk2, 'p':vertcat(thetah2,uk2), 'ode':sys_ode}
    F_ode = integrator('F_ode', 'cvodes', ode2)

    m_ode = {'x':zk2, 'p':uk2, 'ode':mdl_ode }
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    #Output equation (Observations)
    C = np.eye(n_st)
    fy = mtimes(C,xk2) + vk2
    h = Function('h',[xk2,vk2], [fy])
    Ch = Function('Ch',[vertcat(xk2,thetah2)],[jacobian(fy,vertcat(xk2,thetah2))])
    Chx = Function('Chx',[xk2],[jacobian(fy,xk2)])

    uk_opt = np.array([2.5*10**(-6),280])
    xkp = np.array([V0, Ca0,Cb0,Cc0,T0,Tj0,Tj0_in])
    theta_par = theta_nom
    xkh0 = xkp
    zkh0 =vertcat(xkh0,theta_par)
    Sigmak_p = Qz

    #storing the results
    res2_uk[0,:] = uk_opt
    res2_xk[0,:] = xkp
    res2_theta[0,:] = theta_par

    for k in range(run_time-1):
        time[k+1] = k+1

        time_remain = run_time - k
        if time_remain < n_pred:
            n_pred = time_remain

        #if time[k] >= Disturbance and time[k] <= end_Disturbance :
        #    theta_act[0] = theta_act[0] -slope_Disturbance
        #    print('******************************************')
        #    print('******        k0 is reduced           ****')
        #    print('******************************************')
        #if time[k] >= Disturbance2 and time[k] <= end_Disturbance2 :
        #    theta_act[1] = theta_act[1] -slope_Disturbance2
        #    print('******************************************')
        #    print('******        Delta H is reduced           ****')
        #    print('******************************************')

        #Compute the measurement
        ykp = h(xkp,vkp[k,:].T)

        #KF update step/measurment
        Czh = Ch(zkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
        zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
        xkh0 = zkh0[0:n_st]
        theta_par = zkh0[n_st:]
        Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)

        Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(F_ode,n_pred,n_ctrl,n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk2,theta_par,uk2,Tsamp,xkh0,uk_opt)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        uk_ce = res_mpc['x'].full().flatten()
        uk_opt = uk_ce[n_st:n_st+n_ip]

        #Simulate the system
        x_end = F_ode(x0=xkp, p=vertcat(theta_act,uk_opt))
        xkp = x_end['xf'].full()
        xkp = xkp.T #+ wkp[k,:]

        #KF prediction
        Az = Jz2(xkh0,uk_opt,theta_par)
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz

        if k < run_time-2:
            # Save results
            uk_opt[0] = uk_opt[0]/discritize
            res2_uk[k+1,:] =uk_opt # [uk_opt[0],uk_opt[1]]
            res2_xk[k+1,:] =xkp # [xkp[0,0],xkp[0,1]]
            res2_theta[k+1,:] =theta_par.T# [theta_par[0],theta_par[1]]

        else:
            res2_xk[k+1,:] =xkp# [xkp[0,0],xkp[0,1]]
            res2_theta[k+1,:] =theta_par.T# [theta_par[0],theta_par[1]]

    Mkplots(time, res2_xk, res2_uk, res2_theta, 'CE Control', 1,1,1)


#print('xk diff = ', res3_xk - res2_xk)
#print('uk diff = ', res3_uk - res2_uk)
print('theta3  = ', res3_theta)
print('theta2  = ', res2_theta)
print('Moles of C = ', res3_xk[:,3]*res3_xk[:,0])
#plt.show()
#import pdb; pdb.set_trace()
if save2csv:
    writeData_2(res2_uk,res2_xk,res2_theta,res3_uk,res3_xk,res3_theta,run_time,csv_file)
