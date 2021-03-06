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
import csv

#########################################
##  Set-up
######################################
csv_file = "./results/Results_"+run_name+".csv"
save2csv = True
p_plots = False
np.random.seed(seed)

# Run Case 1, 2, and 3
gen_nom = True
runC2 = True
runC3 = False

run_time = 10
Tsamp = 2
discritize = 60 * Tsamp # [=] seconds/(min) - discritze in two min
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
R = np.diag([10**-17, 5, 5, 5, .2, .2, .2])
vkp = (np.random.randn(run_time,n_st))
wkp = (np.random.randn(run_time,n_st+n_par))*0
vkp = mtimes(vkp,R)

#######################
# Model Parameters
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
Q = np.diag([0, 0, 0, 0,  0,  0,  0])
Qz = np.diag([0, 0, 0, 0,  0,  0,  0, ((3.0457*10**(-7))*discritize*.2)**2, (323.05*.2)**2 ])

#Lower and upper bound on inputs and states
uk_lb = [0,                     280]
uk_ub = [9*10**(-6)*discritize, 350*beta_u]
xk_lb = [0,      0,  0,   0,   321*alpha_l, 0,   0]
xk_ub = [.007,  inf, inf, inf, 325*alpha_u, inf, inf]

#Build results storage
time = np.zeros((run_time,1))
time_pred = np.zeros((n_pred,1))
Cost_ce = 0.0
Cost_dual = 0.0

#Case 1 -
res_xk = np.zeros((run_time+1,n_st))
res_theta = np.zeros((run_time+1,n_par))
res_uk = np.zeros((run_time,n_ip))
#Case 2 - CE control with unknown parameters
res2_xk = np.zeros((run_time+1,n_st))
res2_theta = np.zeros((run_time+1,n_par))
res2_uk = np.zeros((run_time,n_ip))
#Case 3 - Dual control with unknown parameters
res3_xk = np.zeros((run_time+1,n_st))
res3_theta = np.zeros((run_time+1,n_par))
res3_uk = np.zeros((run_time,n_ip))


def writeData_2(res_uk,res_xk,res_theta,res2_uk,res2_xk,res2_theta,res3_uk,res3_xk,res3_theta,run_time,csv_file):
    File = open(csv_file, 'a')
    with File:
        writer = csv.writer(File)
        writer.writerow(['   States   '])
        writer.writerow([ 'Opt', 'Vol' ,res_xk[:,0]])
        writer.writerow([ 'CE', 'Vol' ,res2_xk[:,0]])
        writer.writerow([ 'Dual', 'Vol', res3_xk[:,0]])
        writer.writerow([ 'Opt', 'Ca' , res_xk[:,1]])
        writer.writerow([ 'CE', 'Ca' , res2_xk[:,1]])
        writer.writerow([ 'Dual', 'Ca', res3_xk[:,1]])
        writer.writerow([ 'Opt', 'Cb' , res_xk[:,2]])
        writer.writerow([ 'CE', 'Cb' , res2_xk[:,2]])
        writer.writerow([ 'Dual', 'Cb', res3_xk[:,2]])
        writer.writerow([ 'Opt', 'Cc', res_xk[:,3]])
        writer.writerow([ 'CE', 'Cc', res2_xk[:,3]])
        writer.writerow([ 'Dual', 'Cc', res3_xk[:,3]])
        writer.writerow([ 'Opt', 'T', res_xk[:,4]])
        writer.writerow([ 'CE', 'T', res2_xk[:,4]])
        writer.writerow([ 'Dual', 'T', res3_xk[:,4]])
        writer.writerow([ 'Opt', 'Tj', res_xk[:,5]])
        writer.writerow([ 'CE', 'Tj', res2_xk[:,5]])
        writer.writerow([ 'Dual', 'Tj', res3_xk[:,5]])
        writer.writerow([ 'Opt', 'Tj_in', res_xk[:,6]])
        writer.writerow([ 'CE', 'Tj_in', res2_xk[:,6]])
        writer.writerow([ 'Dual', 'Tj_in', res3_xk[:,6]])

        writer.writerow(['   Parameters   '])
        writer.writerow([ 'Opt', 'k0', res_theta[:,0]])
        writer.writerow([ 'CE', 'k0', res2_theta[:,0]])
        writer.writerow([ 'Dual', 'k0', res3_theta[:,0]])
        writer.writerow([ 'Opt', 'Delta H', res_theta[:,1]])
        writer.writerow([ 'CE', 'Delta H', res2_theta[:,1]])
        writer.writerow([ 'Dual', 'Delta H', res3_theta[:,1]])

        writer.writerow(['   Inputs   '])
        writer.writerow([ 'Opt', 'V_in', res_uk[:,0]])
        writer.writerow([ 'CE', 'V_in', res2_uk[:,0]])
        writer.writerow([ 'Dual', 'V_in', res3_uk[:,0]])
        writer.writerow([ 'Opt', 'Tj_in', res_uk[:,1]])
        writer.writerow([ 'CE', 'Tj_in', res2_uk[:,1]])
        writer.writerow([ 'Dual', 'Tj_in', res3_uk[:,1]])

#generate nominal
if gen_nom:
    xk = SX.sym('xk',n_st)
    uk = SX.sym('uk',n_ip)
    wk = SX.sym('wk',n_st+n_par)
    vk = SX.sym('vk',n_st)
    thetah = SX.sym('thetah',n_par)
    zk = vertcat(xk,thetah)

    #Define the system equations with unknown parameters
    de1 = uk[0]
    de2 = -(uk[0]/xk[0])*xk[1] - thetah[0]*xk[1]*xk[2]
    de3 = (uk[0]/xk[0])*(Cbin-xk[2]) - thetah[0]*xk[1]*xk[2]
    de4 = -(uk[0]/xk[0])*xk[3] + thetah[0]*xk[1]*xk[2]
    de5 = (uk[0]/xk[0])*(Tin - xk[4]) - (U*((2*xk[0]/r)-pi*r**2)*(xk[4]-xk[5]))/(rho*cp*xk[0]) - (thetah[0]*xk[1]*xk[2]*thetah[1])/(rho*cp)
    de6 = (Vjin/Vj) * (xk[6]-xk[5]) + (U*((2*xk[0]/r)-pi*r**2)*(xk[4]-xk[5]))/(rho*cp*Vj)
    de7 = (1/tau_c)* (uk[1]-xk[6])
    dthetah1 = [0]*thetah[0]
    dthetah2 = [0]*thetah[1]

    sys_ode = vertcat(de1,de2,de3,de4,de5,de6,de7)
    mdl_ode =vertcat(de1,de2,de3,de4,de5,de6,de7,dthetah1,dthetah2)
    mdl_ode2 =vertcat(de1+wk[0],de2+wk[1],de3+wk[2],de4+wk[3],de5+wk[4],de6+wk[5],de7+wk[6],dthetah1,dthetah2)

    ode = {'x':xk, 'p':vertcat(thetah,uk), 'ode':sys_ode}
    F_ode = integrator('F_ode', 'cvodes', ode)

    m_ode = {'x':zk, 'p':uk, 'ode':mdl_ode }
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    #Jacobians for predictions
    Jx = Function('Jx',[xk,uk,thetah],[jacobian((xk+sys_ode),xk)])
    Jz = Function('Jz',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+mdl_ode),vertcat(xk,thetah))])
    Ju = Function('Ju',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+mdl_ode),uk)])
    L   = Function('L' ,[xk,uk,thetah,wk],[jacobian((vertcat(xk,thetah)+mdl_ode2),wk)])

    #Output equation (Observations)
    C = np.eye(n_st)
    fy = mtimes(C,xk) + vk
    h = Function('h',[xk,vk], [fy])
    Ch = Function('Ch',[vertcat(xk,thetah)],[jacobian(fy,vertcat(xk,thetah))])
    Chx = Function('Chx',[xk],[jacobian(fy,xk)])

    uk_opt = np.array([2.5*10**(-6)*discritize,280])
    xkp = np.array([V0, Ca0,Cb0,Cc0,T0,Tj0,Tj0_in])
    theta_par = theta_nom
    xkh0 = xkp
    zkh0 =vertcat(xkh0,theta_par)
    Sigmak_p = Qz

    #storing the results
    res_uk[0,:] = np.array([2.5*10**(-6),280])
    res_xk[0,:] = np.array([V0, Ca0,Cb0,Cc0,T0,Tj0,Tj0_in])
    res_theta[0,:] = theta_nom

    for k in range(run_time):
        time[k] = k

        time_remain = run_time - k
        if time_remain < n_pred:
            n_pred = time_remain

        #Compute the measurement
        ykp = h(xkp,0*vkp[k,:].T)
        #KF update step/measurment
        Czh = Ch(zkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
        zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
        xkh0 = zkh0[0:n_st]
        theta_par = zkh0[n_st:]
        Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)

        Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.opt_mpc(F_ode,n_pred,n_ctrl,n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,theta_par,uk,Tsamp,xkh0,uk_opt)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        uk_ce = res_mpc['x'].full().flatten()
        uk_opt = uk_ce[n_st:n_st+n_ip]
        #import pdb; pdb.set_trace()
        #Simulate the system
        x_end = F_ode(x0=xkp, p=vertcat(theta_act,uk_opt))
        xkp = x_end['xf'].full()
        xkp = xkp.T #+ wkp[k,:]

        #KF prediction
        Az = Jz(xkh0,uk_opt,theta_par)
        Jw = L(xkh0,uk_opt,theta_par,wkp[k,:])
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Jw*Qz*Jw.T


        if k < run_time-2:
            # Save results
            uk_opt[0] = uk_opt[0]/discritize
            res_uk[k+1,:] =uk_opt # [uk_opt[0],uk_opt[1]]
            uk_opt[0] = uk_opt[0]*discritize

            res_xk[k+1,:] =xkp # [xkp[0,0],xkp[0,1]]
            res_theta[k+1,:] =theta_par.T# [theta_par[0],theta_par[1]]
        else:
            res_xk[k+1,:] =xkp# [xkp[0,0],xkp[0,1]]
            res_theta[k+1,:] =theta_par.T# [theta_par[0],theta_par[1]]

if runC2:
    #########################################
    ##  Define system
    ######################################
    #Define the system states
    xk2 = SX.sym('xk2',n_st)
    uk2 = SX.sym('uk2',n_ip)
    wk2 = SX.sym('wk2',n_st+n_par)
    vk2 = SX.sym('vk2',n_st)
    thetah2 = SX.sym('thetah2',n_par)
    zk2 = vertcat(xk2,thetah2)

    #Define the system equations with unknown parameters
    de1 = uk2[0]
    de2 = -(uk2[0]/xk2[0])*xk2[1] - thetah2[0]*xk2[1]*xk2[2]
    de3 = (uk2[0]/xk2[0])*(Cbin-xk2[2]) - thetah2[0]*xk2[1]*xk2[2]
    de4 = -(uk2[0]/xk2[0])*xk2[3] + thetah2[0]*xk2[1]*xk2[2]
    de5 = (uk2[0]/xk2[0])*(Tin - xk2[4]) - (U*((2*xk2[0]/r)-pi*r**2)*(xk2[4]-xk2[5]))/(rho*cp*xk2[0]) - (thetah2[0]*xk2[1]*xk2[2]*thetah2[1])/(rho*cp)
    de6 = (Vjin/Vj) * (xk2[6]-xk2[5]) + (U*((2*xk2[0]/r)-pi*r**2)*(xk2[4]-xk2[5]))/(rho*cp*Vj)
    de7 = (1/tau_c)* (uk2[1]-xk2[6])
    dthetah1 = [0]*thetah2[0]
    dthetah2 = [0]*thetah2[1]

    sys_ode = vertcat(de1,de2,de3,de4,de5,de6,de7)
    mdl_ode =vertcat(de1,de2,de3,de4,de5,de6,de7,dthetah1,dthetah2)
    mdl_ode2 =vertcat(de1+wk2[0],de2+wk2[1],de3+wk2[2],de4+wk2[3],de5+wk2[4],de6+wk2[5],de7+wk2[6],dthetah1,dthetah2)

    #Jacobians for predictions
    Jx2 = Function('Jx2',[xk2,uk2,thetah2],[jacobian((xk2+sys_ode),xk2)])
    Jz2 = Function('Jz2',[xk2,uk2,thetah2],[jacobian((vertcat(xk2,thetah2)+mdl_ode),vertcat(xk2,thetah2))])
    Ju2 = Function('Ju2',[xk2,uk2,thetah2],[jacobian((vertcat(xk2,thetah2)+mdl_ode),uk2)])
    L2   = Function('L2' ,[xk2,uk2,thetah2,wk2],[jacobian((vertcat(xk2,thetah2)+mdl_ode2),wk2)])

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

    uk_opt = np.array([2.5*10**(-6)*discritize,280])
    xkp = np.array([V0, Ca0,Cb0,Cc0,T0,Tj0,Tj0_in])
    theta_par = theta_nom
    xkh0 = xkp
    zkh0 =vertcat(xkh0,theta_par)
    Sigmak_p = Qz

    #storing the results
    res2_uk[0,:] = np.array([2.5*10**(-6),280])
    res2_xk[0,:] = xkp
    res2_theta[0,:] = theta_par

    for k in range(run_time):
        time[k] = k

        time_remain = run_time - k
        if time_remain < n_pred:
            n_pred = time_remain

        #Compute the measurement
        ykp = h(xkp,vkp[k,:].T)
        #KF update step/measurment
        Czh = Ch(zkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
        zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
        xkh0 = zkh0[0:n_st]
        theta_par = zkh0[n_st:]
        Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)

        Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(F_ode,n_pred,n_ctrl,n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk2,theta_par,uk2,Tsamp,xkh0,uk_opt,res_xk,res_uk,k-1)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        uk_ce = res_mpc['x'].full().flatten()
        uk_opt = uk_ce[n_st:n_st+n_ip]
        #import pdb; pdb.set_trace()
        #Simulate the system
        x_end = F_ode(x0=xkp, p=vertcat(theta_act,uk_opt))
        xkp = x_end['xf'].full()
        xkp = xkp.T #+ wkp[k,:]

        #KF prediction
        Az = Jz2(xkh0,uk_opt,theta_par)
        Jw = L2(xkh0,uk_opt,theta_par,wkp[k,:])
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Jw*Qz*Jw.T

        if k < run_time-2:
            # Save results
            uk_opt[0] = uk_opt[0]/discritize
            res2_uk[k+1,:] =uk_opt # [uk_opt[0],uk_opt[1]]
            uk_opt[0] = uk_opt[0]*discritize

            res2_xk[k+1,:] =xkp # [xkp[0,0],xkp[0,1]]
            res2_theta[k+1,:] =theta_par.T# [theta_par[0],theta_par[1]]
        else:
            res2_xk[k+1,:] =xkp# [xkp[0,0],xkp[0,1]]
            res2_theta[k+1,:] =theta_par.T# [theta_par[0],theta_par[1]]

if runC3:
    #########################################
    ##  Define system
    ######################################
    #Define the system states
    xk3 = SX.sym('xk3',n_st)
    uk3 = SX.sym('uk3',n_ip)
    wk3 = SX.sym('wk3',n_st+n_par)
    vk3 = SX.sym('vk3',n_st)
    thetah3 = SX.sym('thetah3',n_par)
    zk3 = vertcat(xk3,thetah3)

    #Define the system equations with unknown parameters
    de1 = uk3[0]
    de2 = -(uk3[0]/xk3[0])*xk3[1] - thetah3[0]*xk3[1]*xk3[2]
    de3 = (uk3[0]/xk3[0])*(Cbin-xk3[2]) - thetah3[0]*xk3[1]*xk3[2]
    de4 = -(uk3[0]/xk3[0])*xk3[3] + thetah3[0]*xk3[1]*xk3[2]
    de5 = (uk3[0]/xk3[0])*(Tin - xk3[4]) - (U*((2*xk3[0]/r)-pi*r**2)*(xk3[4]-xk3[5]))/(rho*cp*xk3[0]) - (thetah3[0]*xk3[1]*xk3[2]*thetah3[1])/(rho*cp)
    de6 = (Vjin/Vj) * (xk3[6]-xk3[5]) + (U*((2*xk3[0]/r)-pi*r**2)*(xk3[4]-xk3[5]))/(rho*cp*Vj)
    de7 = (1/tau_c)* (uk3[1]-xk3[6])
    dthetah1 = [0]*thetah3[0]
    dthetah2 = [0]*thetah3[1]

    sys_ode = vertcat(de1,de2,de3,de4,de5,de6,de7)
    mdl_ode =vertcat(de1,de2,de3,de4,de5,de6,de7,dthetah1,dthetah2)
    mdl_ode2 =vertcat(de1+wk3[0],de2+wk3[1],de3+wk3[2],de4+wk3[3],de5+wk3[4],de6+wk3[5],de7+wk3[6],dthetah1,dthetah2)

    #Jacobians for predictions
    Jz3 = Function('Jz3',[xk3,uk3,thetah3],[jacobian((vertcat(xk3,thetah3)+mdl_ode),vertcat(xk3,thetah3))])
    Ju3 = Function('Ju3',[xk3,uk3,thetah3],[jacobian((vertcat(xk3,thetah3)+mdl_ode),uk3)])
    L3   = Function('L3'  ,[xk3,uk3,thetah3,wk3],[jacobian((vertcat(xk3,thetah3)+mdl_ode2),wk3)])

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
        #import pdb; pdb.set_trace()
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
        Jw = L3(xkh0,uk_opt,theta_par,wkp[k,:])
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Jw*Qz*Jw.T

        if k < run_time-2:
            uk_opt[0]=uk_opt[0]/discritize
            res3_uk[k+1,:] = uk_opt
            uk_opt[0]=uk_opt[0]*discritize

            res3_xk[k+1,:] = xkp
            res3_theta[k+1,:] = [theta_par[0],theta_par[1]]
        else:
            res3_xk[k+1,:] = xkp
            res3_theta[k+1,:] = [theta_par[0],theta_par[1]]



if runC2 and runC3 and p_plots:
    comp_par_error(time, res2_theta, res3_theta, 'CE Control', 'Dual Control')
    Mkplots(time, res2_xk, res2_uk, res2_theta, 'CE Control', 1,1,1)
    Mkplots(time, res3_xk, res3_uk, res3_theta, 'Dual Control', 1,1,1)
if save2csv:
    writeData_2(res_uk,res_xk,res_theta,res2_uk,res2_xk,res2_theta,res3_uk,res3_xk,res3_theta,run_time,csv_file)
