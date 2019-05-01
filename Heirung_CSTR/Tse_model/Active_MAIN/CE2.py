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
import dual as Dual
import pdb
import results as P
import csv


#########################################
##  Set-up
######################################
csv_file = "Results_1.csv"
save2csv = False

# Run Case 1, 2, and 3
runC1 = True
runC2 = False
runC3 = False # N/A

run_time = 20
Tsamp = 1
#Define the controller
n_pred = 20
n_ctrl = 20

#Define the problem size
n_st = 3
n_ip = 1
n_op = 3
n_par_st = 3
n_par_ip = 3
n_par = n_par_st+n_par_ip


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
wk = SX.sym('wk',n_st)
vk = SX.sym('vk',n_op)
thetah = SX.sym('thetah',n_par)
zk = vertcat(xk,thetah)
#uk_opt = SX.sym('uk_opt',n_ctrl,n_ip)

#Actual parameter values (5.5)
theta_act = [1.8, -1.01, .58, 0.3, 0.5, 1]

#Parameter statistics (2.3)/(5.3-5.4)
theta_h = [1, -0.6, 0.3, 0.1, 0.7, 1.5]
theta_par = theta_h
Sigma_tt = np.diag([0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
Q = np.diag([1,1,1])
Qz = np.diag([1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
R = np.diag([1,1,1])

#Initial states (5.6)
xkp = np.array([0,0,0])
#x_act[:,0] = xkp

rho_v = np.array([0,0,20])
lamb =10**(-3)

#Define control objective
xk_sp = rho_v
Wx = rho_v
Wu = lamb

#Define the initial model condition
xkh0=xkp
zkh0 = vertcat(xkp,theta_h)
#Sigmak_p = np.diag([1,1,1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
Sigmak_p = np.diag([1,1,1])

#Generate the random numbers
wkp = (np.random.randn(run_time,n_st) )
vkp = (np.random.randn(run_time,n_op) )

#Lower and upper bound on inputs
uk_lb = [-inf]
uk_ub = [inf]

#Lower and upper bound on states
xk_lb = [-inf, -inf, -inf]
xk_ub =  [inf,  inf,  inf]

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

e1 = xk[1] + .3*uk[0]# + wkp[i1,0]
e2 = xk[2] + .5*uk[0]# + wkp[i1,1]
e3 = 1.8*xk[0] + -1.01*xk[1] + .58*xk[2] +  uk[0]# + wkp[i1,2]

c1_mdl = vertcat(e1,e2,e3)
sys_ode = vertcat(de1,de2,de3)
mdl_ode =vertcat(de1,de2,de3,dthetah1,dthetah2,dthetah3,dthetah4,dthetah5,dthetah6)

Jx = Function('Jx',[xk,uk],[jacobian((xk+c1_mdl),xk)])
Jz = Function('Jz',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+mdl_ode),vertcat(xk,thetah))])
Ju = Function('Ju',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+mdl_ode),uk)])

ode1 = {'x':xk, 'p':uk, 'ode':c1_mdl}
C_ode = integrator('F_ode', 'cvodes', ode1)

ode = {'x':xk, 'p':vertcat(thetah,uk), 'ode':sys_ode}
F_ode = integrator('F_ode', 'cvodes', ode)

m_ode = {'x':zk, 'p':uk, 'ode':mdl_ode }
M_ode = integrator('M_ode', 'cvodes', m_ode)

#Output equation (Observations)
C = np.matrix([0, 0, 1])
fy = mtimes(C,xk) + vk
h = Function('h',[xk,vk], [fy])
Ch = Function('Ch',[vertcat(xk,thetah)],[jacobian(fy,vertcat(xk,thetah))])
Chx = Function('Chx',[xk],[jacobian(fy,xk)])

#Build results storage
time = np.zeros((run_time,1))
Cost_opt = 0.0
Cost_ce = 0.0
    #Case 1 - Optimal control with known parameters
res1_xk = np.zeros((run_time,n_st))
res1_uk = np.zeros((run_time-1,n_ip))
#res1_xk[0,:]= xkp
    #Case 2 - CE control with unknown parameters
res2_xk = np.zeros((run_time,n_st))
res2_theta = np.zeros((run_time,n_par_st+n_par_ip))
res2_uk = np.zeros((run_time,n_ip))
#res2_xk[0,:]= xkp


f_uk = []
f_xk=np.zeros((run_time,n_st))
uk_opt= 0.0
















#########################################
##  Begin Simulations
######################################
if runC1 == True:
    for i2 in range(run_time):
        #Compute the measurement
        ykp = h(xkp,vkp[i2,:].T)
    #    ykp = h(xkp,np.zeros((n_st,1)))


        #KF update step/measurment
        #Czh = Ch(zkh0)
        Cxh = Chx(xkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Cxh.T,np.linalg.inv(mtimes(Cxh,mtimes(Sigmak_p,Cxh.T)) + R)))
        xkh0 = xkh0 + mtimes(Kkh,(ykp - h(xkh0,np.zeros((n_st,1)))))
    #    xkh0 = zkh0[0:n_st]
        theta_par = theta_act
        Sigmak = mtimes((np.eye(n_st) - mtimes(Kkh,Cxh)),Sigmak_p)
        #Sigmak = np.array(Sigmakh)

        time_remain = run_time - i2
        if time_remain < n_pred:
            n_pred = time_remain

        #Build Case 1; the optimal solution with known parameters
        Jopt = 0
        J=0
        qu_opt = []
        lbq = []
        ubq = []
        g = []
        lbg = []
        ubg = []
        q0 = []

        Xk = MX.sym('X_' + str(0), n_st)
        qu_opt += [Xk]
        lbq += [xkh0[i].__float__() for i in range(n_st)]
        ubq += [xkh0[i].__float__() for i in range(n_st)]
        q0 += [xkh0[i].__float__() for i in range(n_st)]


        for i1 in range(n_pred):
            # new NLP variable for the control
            Uk = MX.sym('U_' + str(i1), n_ip)
            qu_opt   += [Uk]
            lbq += [uk_lb[i] for i in range(n_ip)]
            ubq += [uk_ub[i] for i in range(n_ip)]
            q0  += [uk_opt]

            Fk = F_ode(x0=Xk, p=vertcat(theta_act,Uk))
            Xk_end = Fk['xf']

            # New NLriable for state at end of interval
            Xk = MX.sym('X_' + str(i1+1), n_st)
            qu_opt   += [Xk]
            lbq += [xk_lb[i] for i in range(n_st)]
            ubq += [xk_ub[i] for i in range(n_st)]
            q0 += [xkh0[i].__float__() for i in range(n_st)]

            # Add equality constraint
            g   += [Xk_end-Xk]
            lbg += [0]*n_st
            ubg += [0]*n_st

            #Cost function (5.8) - Soft Landing-Type example
            Jopt = Jopt + Uk**2*Wu
            if i1 == n_pred-2:
                Jopt = .5*(mtimes((Xk-Wx).T,(Xk-Wx)) + Jopt)


        nlp_opt = {'x':vertcat(*qu_opt), 'f':Jopt, 'g': vertcat(*g)}
        solver_opt = nlpsol('solver_opt', MySolver, nlp_opt)
        #Simulate case 1
        case1_res = solver_opt(x0=q0, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        case1_res = case1_res['x'].full().flatten()
        #save Case 1 results
        x1 = case1_res[0::4]
        x2 = case1_res[1::4]
        x3 = case1_res[2::4]
        u_opt = case1_res[3::4]

        if i2 < run_time-1:
            uk_opt = u_opt[0]
            #Simulate the system
            x_end = F_ode(x0=xkp, p=vertcat(theta_act,uk_opt))
            xkp = x_end['xf'].full().flatten()
            xkp = xkp.T + wkp[i2,:]
        #    xkp = xkp.T #+ wkp[k,:]

            #KF prediction
            Ax = Jx(xkh0,uk_opt)
            x_end = F_ode(x0=xkh0, p=vertcat(theta_act,uk_opt))
            xkh0 = x_end['xf'].full()
            Sigmak_p = mtimes(Ax,mtimes(Sigmak,Ax.T)) + Q

            #pdb.set_trace()

            f_uk += [u_opt[0]]
            f_xk[i2,:] = [x1[1],x2[1],x3[1]]
        #zkh0[0:n_st] =[res1_xk[1,i1]for i1 in range(n_st) ]

    #    opt_end = [res1_xk[-1,i1] for i1 in range(n_st)]
    #pdb.set_trace()

    Cost_opt = 0.5*(sum([Wu*f_uk[i1]**2 for i1 in range(run_time-1)]) +  sum((f_xk[-2,:]-Wx)**2) )
    print Cost_opt
    print('xkp final = ', f_xk[-1,:])
    P.Plot(f_uk[0:-1],f_xk[0:-1],run_time,'Case 1')

























if runC2 == True:
    Sigmak_p = np.diag([1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1])

    for k in range(run_time):
        time[k,:] = k
        #Compute the measurement
        ykp = h(xkp,vkp[k,:].T)
    #    ykp = h(xkp,np.zeros((n_st,1)))

        #KF update step/measurment
        Czh = Ch(zkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
        zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
        xkh0 = zkh0[0:n_st]
        theta_par = zkh0[n_st:]
        Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)
        #Sigmak = np.array(Sigmakh)


        time_remain = run_time - k
        if time_remain < n_pred:
            n_pred = time_remain

        print('******************************************')
        print('******    Begin CE (loop ',k+1,')    ****')
        print('******************************************')

        Jce = 0.0
        qu_ce = []
        lbq = []
        ubq = []
        g = []
        lbg = []
        ubg = []

        #xk = MX.sym('x_'+str(0),n_st)
        X0 = MX.sym('X0', n_st)
        qu_ce += [X0]

        lbq += [xkh0[i1].__float__() for i1 in range(n_st)]
        ubq += [xkh0[i1].__float__() for i1 in range(n_st)]
        qu_init = []
        qu_init += [xkh0[i1].__float__() for i1 in range(n_st)]
        xk = X0
        #xk_end = xkh0

        for i in range(n_pred):
            if i<=n_ctrl:
                uk = MX.sym('u_' + str(i),n_ip)
                qu_ce += [uk]
                lbq += uk_lb
                ubq += uk_ub
                qu_init += [uk_opt]

            x_end = F_ode(x0=xk, p=vertcat(theta_par,uk))
            #print(xk_end)
            xk_end = x_end['xf']

            xk = MX.sym('x_'+str(i+1),n_st)
            qu_ce += [xk]
            lbq += xk_lb
            ubq += xk_ub
            qu_init += [xkh0[i1].__float__() for i1 in range(n_st)]

            g += [xk_end-xk]
            lbg += [0]*(n_st)
            ubg += [0]*(n_st)

            #Cost function (5.8) - Soft Landing-Type example
            Jce = Jce +  uk**2*Wu
            if i == n_pred-1:
                #pdb.set_trace()
                Jce = .5*(mtimes(transpose(xk_end-Wx),(xk_end-Wx)) + Jce )

        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}

        #pdb.set_trace()
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc)
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        uk_ce = res_mpc['x'].full().flatten()

        uk_opt = uk_ce[n_st+n_ip-1]
    #    pdb.set_trace()

        #Simulate the system
        x_end = F_ode(x0=xkp, p=vertcat(theta_act,uk_opt))
        xkp = x_end['xf'].full()
        xkp = xkp.T + wkp[k,:]
    #    xkp = xkp.T #+ wkp[k,:]

        #KF prediction
        Az = Jz(xkh0,uk_opt,theta_par)
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz

        #storing the results
        res2_uk[k,:] = uk_opt
        res2_xk[k,:] = xkp
        res2_theta[k,:] = theta_par.T

        Cost_ce = Cost_ce + Wu*uk_opt**2

    #pdb.set_trace()
    Cost_ce = 0.5*(Cost_ce + mtimes((xkp-Wx),(xkp-Wx).T))

    P.PlotC2(res2_uk,res2_xk,run_time,res2_theta,'Case 2')

print 'Cost_ce = ',Cost_ce
print 'Cost_opt = ',Cost_opt
if save2csv == True:
    #pdb.set_trace()
    P.writeData(Cost_opt,Cost_ce,Cost_dual,u_opt,res1_xk,theta_act,res1_uk,res1_xk,theta_act,res2_uk,res2_xk,res2_theta,run_time,csv_file)
