# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:16:57 2017

@author: vinay
"""
import sys
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")

import numpy as NP
from casadi import *
import matplotlib.pyplot as plt
from scipy import linalg
from casadi.tools import *
import core
import dualfcn

MySolver = "ipopt"
opts = {}
if MySolver == "sqpmethod":
    opts["qpsol"] = "qpoases"
    opts["qpsol_options"] = {"printLevel":"none"}
elif MySolver == "ipopt":
    opts = {"ipopt.print_level":5, "print_time": False, 'ipopt.max_iter':1000}

#Define simulation time
run_time = 101

#Define the problem size
n_st = 4
n_ip = 2
n_op = 2
n_par = 2

#Define the controller
n_pred = 20
n_ctrl = 5


#Define the system states
xk = SX.sym('xk',n_st)
uk = SX.sym('uk',n_ip)
yk = SX.sym('yk',n_op)
wk = SX.sym('wk',n_st)
vk = SX.sym('vk',n_op)
thetah = SX.sym('thetah',n_par)
zk = vertcat(xk,thetah)
uk_opt = SX.sym('uk_opt',n_ctrl,n_ip)
#Define the system parameters
#k10 = 1.287e12
#k20 = 1.287e12
k30 = 9.043e09
ER1 = 9578.3
ER2 = 9578.3
ER3 = 8560.0
dHAB = 4.2
dHBC = -11.0
dHAD = -41.85
rho = 0.9342
Cp = 3.01
Cpj = 2.0
Ac = 0.215
V = 10.01
mj = 5.0
Tin = 130.0
kw = 4032.0
Tsamp = 0.005
CA_in = 5.0

Q = NP.diag((1e-4,1.2e-4,0.01,0.01))
Qz = NP.diag((1e-4,1.2e-4,0.01,0.01,1e-6,1e-6))
R = NP.diag((1e-5,0.002))

#Define the system ODEs
k1 = thetah[0]*1e12*NP.exp(-ER1/(xk[2] + 273.15))
k2 = thetah[1]*1e12*NP.exp(-ER2/(xk[2]+273.15))
k3 = k30*NP.exp(-ER3/(xk[2]+273.15))
dCA = uk[0]*(CA_in-xk[0]) - k1*xk[0] - k3*xk[0]**2
dCB = -uk[0]*xk[1] + k1*xk[0] - k2*xk[1]
dTR = uk[0]*(Tin - xk[2]) + kw*Ac*(xk[3]-xk[2])/(rho*Cp*V) - (k1*xk[0]*dHAB + k2*xk[1]*dHBC + k3*xk[0]**2*dHAD)/(rho*Cp)
dTJ = (uk[1] + kw*Ac*(xk[2]-xk[3]))/(mj*Cpj)

sys_ode = vertcat(dCA,dCB,dTR,dTJ)

#Develop augmented model
dthetah1 = [0]*thetah[0]
dthetah2 = [0]*thetah[1]
mdl_ode = vertcat(dCA,dCB,dTR,dTJ,dthetah1,dthetah2)
Jz = Function('Jz',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+Tsamp*mdl_ode),vertcat(xk,thetah))])
Ju = Function('Ju',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+Tsamp*mdl_ode),uk)])
ode = {'x':xk, 'p':vertcat(thetah,uk), 'ode':sys_ode}
F_ode = integrator('F_ode', 'cvodes', ode, {'tf':Tsamp})
m_ode = {'x':zk, 'p':uk, 'ode':mdl_ode}
M_ode = integrator('M_ode', 'cvodes', m_ode, {'tf':Tsamp})
#Output equation
C = NP.matrix('1.0 0 0 0;0 0 1.0 0')
fy = mtimes(C,xk) + vk
h = Function('h',[xk,vk], [fy])

#Define the initial system conditions
xkp = NP.array([0.8, 0.5, 134.14, 134.0])
uk_opt = NP.array([15.0,-1800.0])
theta = NP.array([1.290,1.280])

#Define control objective
xk_sp = NP.array([1.0, 0.9, 138.0, 134.0])
Wx = NP.diag([1.0, 100.0, 1e-5, 1e-5])
Wu = NP.diag([100.0, 0.10])

#Define the initial model condition
zkh0 = vertcat(0.95*xkp,1.05*theta)
Sigmak_p = NP.diag([0.01, 0.02, 0.1, 0.1, 0.002, 0.005])

#Define Jacobian functions for EKF
Ch = Function('Ch',[vertcat(xk,thetah)],[jacobian(fy,vertcat(xk,thetah))])

#Generate the random numbers
wkp = mtimes(NP.random.randn(run_time,n_st), linalg.sqrtm(Q))
vkp = mtimes(NP.random.randn(run_time,n_op), linalg.sqrtm(R))

#Store the results
res_xkp = NP.zeros((run_time,n_st))
res_ip = NP.zeros((run_time,n_ip))
res_par = NP.zeros((run_time,n_par))
res_ykp = NP.zeros((run_time,n_op))
time = NP.zeros((run_time,1))


#Lower and upper bound on inputs
uk_lb = [5.0, -5500.0]
uk_ub = [55.0, -1000.0]

#Lower and upper bound on states
xk_lb = [0.0, 0.0, 100.0, 100.0]
xk_ub = [5.0, 5.0, 140.0, 140.0]

#Build the CE MPC optimization problem
Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = dualfcn.ce_nmpc(mdl_ode,n_pred,n_ctrl,n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,thetah,uk,Tsamp,zkh0,xk_sp,Wx)
nlp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g': vertcat(*g)}
solver_mpc = nlpsol('solver_mpc', MySolver, nlp_mpc, opts)
#Build the dual optimization problem
Jd, qu, op = dualfcn.gencost(n_st,n_ip,n_op,n_par,n_pred,Sigmak_p,mdl_ode,fy,xk,uk,thetah,Tsamp,xk_sp,Wx,Wu,Q,Qz,R,uk_opt)
nlp = {'x':vertcat(*qu), 'f': Jd, 'p': op}
solver = nlpsol('solver', MySolver, nlp, opts)

for k in range(run_time):


    #Compute the measurement
    ykp = h(xkp,vkp[k,:].T)

    #EKF update step
    Czh = Ch(zkh0)
    Kkh = mtimes(Sigmak_p,mtimes(Czh.T,NP.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
    zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],NP.zeros((n_op,1)))))
    xkh0 = zkh0[0:n_st]
    theta_par = zkh0[n_st:]
    Sigmak = mtimes((NP.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)
    #Sigmak = NP.array(Sigmakh)

    #storing the results
    res_xkp[k,:] = xkp.T
    res_par[k,:] = theta_par.T
    res_ip[k,:] = uk_opt.T
    time[k,:] = k*Tsamp

    #Compute the LQR input gain
    #Az = Jz(xkh0,uk_opt,theta_par)
    #Bz = Ju(xkh0,uk_opt,theta_par)
    #Ah = Az[0:n_st,0:n_st]
    #Bh = Bz[0:n_st,0:n_ip]
    #Klqs, Pl = core.dlqr(NP.array(Ah),NP.array(Bh),Wx,Wu)

    #Generate the MPC profile
    #Update the initial condition, and constraints
    qu_init[0:n_st] = xkh0.full().tolist()
    lbq[0:n_st] = xkh0.full().tolist()
    ubq[0:n_st] = xkh0.full().tolist()
    res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq)
    uk_ce = res_mpc['x'].full().flatten

    res = solver(x0=uk_opt, p=NP.array(zkh0), lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
    uk_opt = res['x'].full().flatten()

    #Simulate the system
    x_end = F_ode(x0=xkp, p=vertcat(theta,uk_opt))
    xkp = x_end['xf'].full()
    xkp = xkp + wkp[k,:].T

    #EKF prediction
    Az = Jz(xkh0,uk_opt,theta_par)
    z_end = M_ode(x0=zkh0, p = uk_opt)
    zkh0 = z_end['xf'].full()
    Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz
    print(k)


plt.figure(num=1)
plt.plot(time,res_xkp[:,1], 'r-',linewidth=3.0)
plt.plot(time,xk_sp[1]*NP.ones((run_time,1)),'k-.',linewidth=2.0)

plt.figure(num=2)
plt.subplot(2, 1, 1)
plt.plot(time,res_ip[:,0],linewidth=2.0)
plt.subplot(2, 1, 2)
plt.plot(time,res_ip[:,1],linewidth=2.0)

plt.figure(num=3)
plt.plot(time,res_par[:,0],'r-x',linewidth=2.0)
plt.plot(time,res_par[:,1],'k-o',linewidth=2.0)
