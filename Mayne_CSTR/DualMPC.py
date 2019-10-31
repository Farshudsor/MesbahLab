# -*- coding: utf-8 -*-
"""
Created on July 4th 2018
Last edit:

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
import time as Time


def writeData_2(solve_time, wkp,res_uk,res_xk,res4_uk,res4_xk,run_time,
                    alpha, beta):

    csv_file_x1 = "./results/x1.csv"
    csv_file_x2 = "./results/x2.csv"
    csv_file_x1_nom = "./results/Nom_x1.csv"
    csv_file_x2_nom = "./results/Nom_x2.csv"
    csv_file_u1 = "./results/u1.csv"
    csv_file_u1_nom = "./results/Nom_u1.csv"
    csv_file_rv = "./results/RandomParameters.csv"
    csv_file3 = "./results/time.csv"

    File = open(csv_file_x1, 'a')
    with File:
        writer = csv.writer(File)
        writer.writerow([ res4_xk[:,0]])

    File = open(csv_file_x2, 'a')
    with File:
        writer = csv.writer(File)
        writer.writerow([ res4_xk[:,1]])

    File = open(csv_file_x1_nom, 'a')
    with File:
        writer = csv.writer(File)
        writer.writerow([ res_xk[:,0]])

    File = open(csv_file_x2_nom, 'a')
    with File:
        writer = csv.writer(File)
        writer.writerow([ res_xk[:,1]])

    File = open(csv_file_u1, 'a')
    with File:
        writer = csv.writer(File)
        writer.writerow([res4_uk[:]])

    File = open(csv_file_u1_nom, 'a')
    with File:
        writer = csv.writer(File)
        writer.writerow([res_uk[:]])

    File2 = open(csv_file_rv, 'a')
    with File2:
        writer = csv.writer(File2)
        writer.writerow([ alpha , beta,wkp ])

    File3 = open(csv_file3, 'a')
    with File3:
        writer = csv.writer(File3)
        writer.writerow([ 'run', 'solve times' ])
        writer.writerow([ seed ,solve_time ])


#########################################
##  Set-up
######################################
save2csv = True
np.random.seed(seed)

# Run Case 1, 2, 3, and 4,
runC2 = 0
runC3 = 0
runC4 = 1
runC5 = 0

Tsamp = 3
run_time = 600
discritize = 60 * Tsamp # [=] seconds/(min) ; discritze in two min
#Define the controller
n_pred = 420
n_ctrl = 140

#Define the problem size
n_st = 2
n_ip = 1
n_op = 0

#MySolver = "sqpmethod"
MySolver = "ipopt"
opts = {}
if MySolver == "sqpmethod":
    opts["qpsol"] = "qpoases"
    opts["qpsol_options"] = {"printLevel":"none"}
elif MySolver == "ipopt":
    opts = {"ipopt.print_level":5, "print_time": True,
            'ipopt.max_iter':1000, 'output_file':'Main_out.txt'}


# Model Parameters
k0 = 300
theta = 20
xf = .3947
xc = .3816
M = 5
alpha = .117
x0 = [0.9831,0.3918]

#Lower and upper bound on inputs and states
uk_lb = [0];        uk_ub = [2]
xk_lb = [0, 0 ];    xk_ub = [1, 1]

# generate noise
A = np.random.uniform(0,.001,run_time*Tsamp).T
omega = np.random.uniform(0,1,run_time*Tsamp).T
wkp = A*np.sin(omega)

#Build results storage
time = np.zeros((run_time,1))

#Case 2 - CE control with unknown parameters
res2_xk = np.zeros((run_time+1,n_st))
res2_theta = np.zeros((run_time+1,n_par))
res2_uk = np.zeros((run_time,n_ip))
#Case 3 - Dual control with unknown parameters
res3_xk = np.zeros((run_time+1,n_st))
res3_theta = np.zeros((run_time+1,n_par))
res3_uk = np.zeros((run_time,n_ip))
#Case 4 - tube control with unknown parameters
res4_xk = np.zeros((run_time+1,n_st))
res4_theta = np.zeros((run_time+1,n_par))
res4_uk = np.zeros((run_time,n_ip))
#Case 5 - tube Dual control with unknown parameters
res5_xk = np.zeros((run_time+1,n_st))
res5_theta = np.zeros((run_time+1,n_par))
res5_uk = np.zeros((run_time,n_ip))
#Case 6 - tube Dual control with unknown parameters
res6_xk = np.zeros((run_time+1,n_st))
res6_theta = np.zeros((run_time+1,n_par))
res6_uk = np.zeros((run_time,n_ip))

solve_time = np.zeros((6,1))




if runC2:
    start = Time.time()

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

    n_pred=10

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

        Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(F_ode,n_pred,n_ctrl,n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk2,theta_par,uk2,Tsamp,xkh0,uk_opt)
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

    solve_time[1] =  Time.time() - start

if runC3:
    start = Time.time()

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

    n_pred = 10

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
    solve_time[2] =  Time.time() - start

if runC4:
    start = Time.time()
    n_pred = 140

    #########################################
    ##  Define system
    ######################################
    #Define the system states
    xk4 = SX.sym('xk4',n_st)
    uk4 = SX.sym('uk4',n_ip)
    wk4 = SX.sym('wk4',1)

    #Define the system equations with unknown parameters
    dx1= (1/theta)*(1-xk4[0])- k0*xk4[0]*exp(-M/xk4[1])
    dx2=(1/theta)*(xf-xk4[1])+ k0*xk4[0]*exp(-M/xk4[1]) - alpha*uk4[0]*(xk4[1]-xc)
    dx2w= (1/theta)*(xf-xk4[1])+k0*xk4[0]*exp(-M/xk4[1]) - alpha*uk4[0]*(xk4[1]-xc)+ wk4[0]

    sys_ode = vertcat(dx1,dx2w)
    mdl_ode =vertcat(dx1,dx2)

    f_ode = {'x':xk4, 'p':vertcat(uk4,wk4), 'ode':sys_ode}
    F_ode = integrator('F_ode', 'cvodes', f_ode)

    m_ode = {'x':xk4, 'p':uk4, 'ode':mdl_ode }
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    uk_opt = .71
    xkp = x0

    #storing the results
    res4_uk[0,:] = uk_opt
    res4_xk[0,:] = xkp

    for k in range(run_time):
        time[k] = k

        time_remain = run_time - k
        if time_remain < n_pred:
            n_pred = time_remain

        print('*****************************')
        print('********  ', k ,'  ****************')
        print('*****************************')

        if k%Tsamp ==0:
            res_xk_p = res_xk[k:k+n_pred,:]
            res_uk_p = res_uk[k:k+n_pred]
            Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.tube_mpc(M_ode,n_pred,n_ctrl,n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk4,uk4,Tsamp,xkp,uk_opt,res_xk_p,res_uk_p,k-1)
            qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
            solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
            res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
            uk_ce = res_mpc['x'].full().flatten()
            uk_opt = uk_ce[n_st:n_st+n_ip]

        x_end = F_ode(x0=xkp, p=vertcat(uk_opt,wkp[k]))
        xkp = x_end['xf']

        if k < run_time-2:         # Save results
            res4_uk[k+1,:] = uk_opt
            res4_xk[k+1,:] = xkp.T
        else:
            res4_xk[k+1,:] = xkp.T
    solve_time[3] =  Time.time() - start

if runC5:
    start = Time.time()

    #########################################
    ##  Define system
    ######################################
    #Define the system states
    xk5 = SX.sym('xk5',n_st)
    uk5 = SX.sym('uk5',n_ip)
    wk5 = SX.sym('wk5',n_st+n_par)
    vk5 = SX.sym('vk5',n_st)
    thetah5 = SX.sym('thetah5',n_par)
    zk5 = vertcat(xk5,thetah5)

    n_pred = 10


    #Define the system equations with unknown parameters
    de1 = uk5[0]
    de2 = -(uk5[0]/xk5[0])*xk5[1] - thetah5[0]*xk5[1]*xk5[2]
    de3 = (uk5[0]/xk5[0])*(Cbin-xk5[2]) - thetah5[0]*xk5[1]*xk5[2]
    de4 = -(uk5[0]/xk5[0])*xk5[3] + thetah5[0]*xk5[1]*xk5[2]
    de5 = (uk5[0]/xk5[0])*(Tin - xk5[4]) - (U*((2*xk5[0]/r)-pi*r**2)*(xk5[4]-xk5[5]))\
            /(rho*cp*xk5[0]) - (thetah5[0]*xk5[1]*xk5[2]*thetah5[1])/(rho*cp)
    de6 = (Vjin/Vj) * (xk5[6]-xk5[5]) + (U*((2*xk5[0]/r)-pi*r**2)*(xk5[4]-xk5[5]))/\
            (rho*cp*Vj)
    de7 = (1/tau_c)* (uk5[1]-xk5[6])
    dthetah1 = [0]*thetah5[0]
    dthetah2 = [0]*thetah5[1]

    sys_ode = vertcat(de1,de2,de3,de4,de5,de6,de7)
    mdl_ode =vertcat(de1,de2,de3,de4,de5,de6,de7,dthetah1,dthetah2)
    mdl_ode2 =vertcat(de1+wk5[0],de2+wk5[1],de3+wk5[2],de4+wk5[3],de5+wk5[4],
                        de6+wk5[5],de7+wk5[6],dthetah1,dthetah2)

    #Jacobians for predictions
    Jz5 = Function('Jz5',[xk5,uk5,thetah5],[jacobian((vertcat(xk5,thetah5)+mdl_ode),
                    vertcat(xk5,thetah5))])
    Ju5 = Function('Ju5',[xk5,uk5,thetah5],[jacobian((vertcat(xk5,thetah5)+mdl_ode),
                    uk5)])
    L5   = Function('L5'  ,[xk5,uk5,thetah5,wk5],[jacobian((vertcat(xk5,thetah5)
                    +mdl_ode2),wk5)])

    ode = {'x':xk5, 'p':vertcat(thetah5,uk5), 'ode':sys_ode}
    F_ode = integrator('F_ode', 'cvodes', ode)

    m_ode = {'x':zk5, 'p':uk5, 'ode':mdl_ode }
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    #Output equation (Observations)
    C = np.eye(n_st)
    fy = mtimes(C,xk5) + vk5
    h = Function('h',[xk5,vk5], [fy])
    Ch = Function('Ch',[vertcat(xk5,thetah5)],[jacobian(fy,vertcat(xk5,thetah5))])
    Chx = Function('Chx',[xk5],[jacobian(fy,xk5)])

    uk_opt = np.array([2.5*10**(-6)*discritize,280])
    xkp = np.array([V0, Ca0,Cb0,Cc0,T0,Tj0,Tj0_in])
    theta_par = theta_nom
    xkh0 = xkp
    zkh0 =vertcat(xkh0,theta_par)
    Sigmak_p = Qz

    res5_uk[0,:] = np.array([2.5*10**(-6),280])
    res5_xk[0,:] = xkp
    res5_theta[0,:] = theta_par

    for k in range(run_time):
        time[k] = k

        time_remain = run_time - k
        if time_remain < n_pred:
            n_pred = time_remain

        #Compute the measurement
        ykp = h(xkp,vkp[k,:].T)
        #KF update step/measurment
        Czh = Ch(zkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,
                mtimes(Sigmak_p,Czh.T)) + R)))
        #import pdb; pdb.set_trace()
        zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
        xkh0 = zkh0[0:n_st]
        theta_par = zkh0[n_st:]
        Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)

        # generate CE control sequence for nominal trajectory
        Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.tube_mpc(F_ode,n_pred,
                                                        n_ctrl,n_st,n_par,n_ip,uk_lb,
                                                        uk_ub,xk_lb,xk_ub,xk5,
                                                        theta_par,uk5,Tsamp,xkh0,
                                                        uk_opt,res_xk,res_uk,k-1)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':{'max_iter':1000,
                            "check_derivatives_for_naninf":'yes',
                            "print_user_options":'yes' }})
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
        Jd, qu, op = Dual.tube_dual(n_st,n_ip,n_op,n_par,n_pred,Sigmak,mdl_ode,
                                    fy,xk5,uk5,thetah5,Tsamp,Q,Qz,R,uk_ce_3,xkh0,
                                    theta_par,xk_ce_3,res_xk,res_uk,k)
        nlp = {'x':vertcat(*qu), 'f': Jd, 'p': op}
        solver = nlpsol('solver', MySolver, nlp, {'ipopt':{'max_iter':1000,
                        "check_derivatives_for_naninf":'yes', "print_user_options":
                        'yes' }})
        res = solver( x0=uk_opt1, p=xkh0, ubx=uk_ub, lbx=uk_lb, lbg=0, ubg=0)
        uk_opt = res['x'].full().flatten()

        #Simulate the system
        x_end = F_ode(x0=xkp, p=vertcat(theta_act,uk_opt))
        xkp = x_end['xf'].full()
        xkp = xkp.T #+ wkp[k,:]

        #KF prediction
        Az = Jz5(xkh0,uk_opt,theta_par)
        Jw = L5(xkh0,uk_opt,theta_par,wkp[k,:])
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Jw*Qz*Jw.T

        if k < run_time-2:
            uk_opt[0]=uk_opt[0]/discritize
            res5_uk[k+1,:] = uk_opt
            uk_opt[0]=uk_opt[0]*discritize

            res5_xk[k+1,:] = xkp
            res5_theta[k+1,:] = [theta_par[0],theta_par[1]]
        else:
            res5_xk[k+1,:] = xkp
            res5_theta[k+1,:] = [theta_par[0],theta_par[1]]

    solve_time[4] =  Time.time() - start


if save2csv:
    writeData_2(solve_time, wkp,res_uk,res_xk,res4_uk,res4_xk,
                run_time, alpha, beta)
