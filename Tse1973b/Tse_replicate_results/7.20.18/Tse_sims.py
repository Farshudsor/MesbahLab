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
import core
import Tse_fcn as Dual
import pdb
import Tse_comp_res as P
import csv


#########################################
##  Set-up
######################################
csv_file = "Results_1.csv"
save2csv = True

# Run Case 1, 2, and 3
runC1 = True
runC2 = True
runC3 = True

run_time = 6
Tsamp = 1

#Define the problem size
n_st = 3
n_ip = 1
n_op = 3
n_par_st = 3
n_par_ip = 3
n_par = n_par_st+n_par_ip

#Define the controller
n_pred = 5
n_ctrl = 5

#MySolver = "sqpmethod"
MySolver = "ipopt"

opts = {}
if MySolver == "sqpmethod":
    opts["qpsol"] = "qpoases"
    opts["qpsol_options"] = {"printLevel":"none"}
elif MySolver == "ipopt":
    opts = {"ipopt.print_level":5, "print_time": True, 'ipopt.max_iter':1000, 'output_file':'Main_out.txt'}












#########################################
##  Define system
######################################
#Define the system states
xk = SX.sym('xk',n_st)
uk = SX.sym('uk',n_ip)
x_act = SX.sym('x_act',n_st,run_time+1)
u_act = SX.sym('u_act',n_ip,run_time+1)
yk = SX.sym('yk',n_op)
wk = SX.sym('wk',n_st)
vk = SX.sym('vk',n_op)
theta_st = SX.sym('theta',n_par_st)
theta_ip = SX.sym('theta',n_par_ip)
thetah = SX.sym('thetah',n_par)
zk = vertcat(xk,thetah)
uk_opt = SX.sym('uk_opt',n_ctrl,n_ip)

#Define system models (5.2)
A = SX.sym('A',n_st,n_st)
A[0,:] = np.array([0, 1, 0]).T
A[1,:] = np.array([0, 0 ,1]).T
A[2,:] = theta_st.T
B = theta_ip

#Actual parameter values (5.5)
theta_act = [1.8, -1.01, .58, 0.3, 0.5, 1]
A_act = SX.sym('A',n_st,n_st)
A_act[0,:] = np.array([0, 1, 0]).T
A_act[1,:] = np.array([0, 0 ,1]).T
A_act[2,:] = np.array([0.1, 0.1, 0.01]).T
B_act = [ 0.01, 0.01, 0.1]

#Parameter statistics (2.3)/(5.3-5.4)
theta_h = [1, -0.6, 0.3, 0.1, 0.7, 1.5]
Sigma_tt = np.diag([0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
Q = np.diag([1,1,1])
Qz = np.diag([1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
R = np.diag([1,1,1])

#Initial states (5.6)
xkp = np.array([0,0,0])
x_act[:,0] = xkp

rho_v = np.array([0,0,20])
lamb =10**(-3)

#Define control objective
xk_sp = rho_v
Wx = rho_v
Wu = lamb

#Define the initial model condition
zkh0 = vertcat(xkp,theta_h)
Sigmak_p = np.diag([1,1,1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1,])

#Generate the random numbers
wkp = (np.random.randn(run_time,n_st) )
vkp = (np.random.randn(run_time,n_op) )

#Lower and upper bound on inputs
uk_lb = [-50]
uk_ub = [50]

#Lower and upper bound on states
xk_lb = [-50, -50, -50]
xk_ub =  [50,  50,  50]

#Define the system equations
de1 = xk[1] + thetah[3]*uk[0]
de2 = xk[2] + thetah[4]*uk[0]
de3 = thetah[0]*xk[0] + thetah[1]*xk[1] + thetah[2]*xk[2] +  thetah[5]*uk[0]
dthetah1 = [0]*thetah[0]
dthetah2 = [0]*thetah[1]
dthetah3 = [0]*thetah[2]
dthetah4 = [0]*thetah[3]
dthetah5 = [0]*thetah[4]
dthetah6 = [0]*thetah[5]

sys_ode = vertcat(de1,de2,de3)
mdl_ode =vertcat(de1,de2,de3,dthetah1,dthetah2,dthetah3,dthetah4,dthetah5,dthetah6)

Jz = Function('Jz',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+mdl_ode),vertcat(xk,thetah))])
Ju = Function('Ju',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+mdl_ode),uk)])

ode = {'x':xk, 'p':vertcat(thetah,uk), 'ode':sys_ode}
F_ode = integrator('F_ode', 'cvodes', ode)

m_ode = {'x':zk, 'p':uk, 'ode':mdl_ode }
M_ode = integrator('M_ode', 'cvodes', m_ode)

#Output equation (Observations)
C = np.matrix('0 0 1')
fy = mtimes(C,xk) + vk
h = Function('h',[xk,vk], [fy])
Ch = Function('Ch',[vertcat(xk,thetah)],[jacobian(fy,vertcat(xk,thetah))])


#Build results storage
time = np.zeros((run_time,1))
    #Case 1 - Optimal control with known parameters
res1_xk = np.zeros((run_time,n_st))
res1_uk = np.zeros((run_time-1,n_ip))
res1_xk[0,:]= xkp
    #Case 2 - CE control with unknown parameters
res2_xkp = np.zeros((run_time,n_st))
res2_ip = np.zeros((run_time,n_ip))
res2_par = np.zeros((run_time,n_par_st+n_par_ip))
res2_ykp = np.zeros((run_time,n_op))

res2_xk = np.zeros((run_time,n_st))
res2_theta = np.zeros((run_time,n_par_st+n_par_ip))
res2_uk = np.zeros((run_time-1,n_ip))
    #Case 3 - Dual control with unknown parameters
res3_xkp = np.zeros((run_time,n_st))
res3_ip = np.zeros((run_time,n_ip))
res3_par = np.zeros((run_time,n_par_st+n_par_ip))
res3_ykp = np.zeros((run_time,n_op))

res3_xk = np.zeros((run_time,n_st))
res3_theta = np.zeros((run_time,n_par_st+n_par_ip))
res3_uk = np.zeros((run_time,n_ip))







#########################################
##  Begin Simulations
######################################
if runC1 == True:
    #Build Case 1; the optimal solution with known parameters
    Jopt, qu_opt, lbq, ubq, g, lbg, ubg, qu_init = Dual.lb_mpc(run_time,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,thetah,uk,Tsamp,zkh0,xk_sp,Wx,Wu,wkp)
    nlp_opt = {'x':vertcat(*qu_opt), 'f':Jopt, 'g': vertcat(*g)}
    #pdb.set_trace()
    solver_opt = nlpsol('solver_opt', MySolver, nlp_opt)
    #Simulate case 1
    case1_res = solver_opt(x0=qu_init,  lbg=lbg, ubg=ubg)
    case1_res = case1_res['x'].full().flatten()
    #save Case 1 results
    for i1 in range(n_st):
        res1_xk[:,i1] += case1_res[i1::4]
    u_opt = case1_res[3::4]
    ukp = u_opt[0]

    P.Plot(res1_uk,res1_xk,run_time,'Case 1')

#Build Case 2; the CE MPC optimization problem
Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(mdl_ode,n_pred,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,thetah,uk,Tsamp,zkh0,xk_sp,Wx,Wu,uk_opt)
qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc)
print('******************************************')
print('******    Begin pre-loop CE   ****')
print('******************************************')

#pdb.set_trace()
case2_res = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
case2_res = case2_res['x'].full().flatten()
uk_ce =  case2_res[9::10]


#Build Case 3; the Dual Control MPC optimization problem
Jd, qu, op = Dual.gencost(n_st,n_ip,n_op,n_par,n_pred,Sigmak_p,mdl_ode,fy,xk,uk,thetah,Tsamp,xk_sp,Wx,Wu,Q,Qz,R,uk_ce)
nlp = {'x':vertcat(*qu), 'f': Jd, 'p': op}
solver = nlpsol('solver', MySolver, nlp)
print('******************************************')
print('******    Begin pre-loop DUAL       ****')
print('******************************************')
#pdb.set_trace()
res = solver( x0=ukp, p=zkh0, ubx=uk_ub, lbx=uk_lb, lbg=0, ubg=0)
ukp = res['x'].full().flatten()

for k in range(run_time-1):
    #Compute the measurement
    ykp = h(xkp,vkp[k,:].T)

    #EKF update step
    Czh = Ch(zkh0)
    Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
    zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
    xkh0 = zkh0[0:n_st]
    theta_par = zkh0[n_st:]
    Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)
    #Sigmak = np.array(Sigmakh)

    #storing the results
    res3_xk[k,:] = xkp
    res3_theta[k,:] = theta_par.T
    res3_uk[k,:] = ukp
    time[k,:] = k

    #Compute the LQR input gain
    #Az = Jz(xkh0,uk_opt,theta_par)
    #Bz = Ju(xkh0,uk_opt,theta_par)
    #Ah = Az[0:n_st,0:n_st]
    #Bh = Bz[0:n_st,0:n_ip]
    #Klqs, Pl = core.dlqr(np.array(Ah),np.array(Bh),Wx,Wu)

    #Generate the MPC profile
    #Update the initial condition, and constraints
    qu_init[0:n_st] = np.array(xkh0.full().tolist())
    #qu_init[0:n_st] = [xkh0[i].__float__() for i in range(n_st)]
    lbq[0:n_st] = np.array(xkh0.full().tolist())
    ubq[0:n_st] = np.array(xkh0.full().tolist())
    print('******************************************')
    print('******    Begin CE (loop ',k+1,')    ****')
    print('******************************************')
    #pdb.set_trace()
    res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
    uk_ce = res_mpc['x'].full().flatten()

    #save CE results
    res2_xk[k,:] = uk_ce[0:n_st]
    res2_theta[k,:] = uk_ce[n_st:n_st+n_par]
    res2_uk[k,:] = uk_ce[n_par+n_st:n_par+n_st+n_ip]

    print('******************************************')
    print('******    Begin Dual (loop ',k+1,')    ****')
    print('******************************************')
#    pdb.set_trace()
    res = solver( x0=ukp, p=zkh0,  ubx=uk_ub, lbx=uk_lb, lbg=0, ubg=0)
    ukp = res['x'].full().flatten()

    #Simulate the system
    x_end = F_ode(x0=xkp, p=vertcat(theta_h,ukp))
    xkp = x_end['xf'].full()
    #pdb.set_trace()
    xkp = xkp.T + wkp[k,:]

    #EKF prediction
    Az = Jz(xkh0,ukp,theta_par)
    z_end = M_ode(x0=zkh0, p = uk_ce[9])
    zkh0 = z_end['xf'].full()
    #pdb.set_trace()
    Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz

    print('******************************************')
    print('******    Run number ',k+1,' complete    ****')
    print('******************************************')

P.PlotC2(res2_uk,res2_xk,run_time,res2_theta,'Case 2')
P.PlotC2(res3_uk[0:-1],res3_xk,run_time,res3_theta,'Case 3')

if save2csv == True:
    #pdb.set_trace()
    P.writeData(res1_uk,res1_xk,theta_act,res2_uk,res2_xk,res2_theta,res3_uk,res3_xk,res3_theta,run_time,csv_file)
