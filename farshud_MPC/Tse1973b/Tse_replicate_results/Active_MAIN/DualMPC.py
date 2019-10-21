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
csv_file = "Results_8_C3.csv"
save2csv = True
p_plots = False

if save2csv:
    File = open(csv_file, 'wb')
    with File:
        writer = csv.writer(File)

# Run Case 1, 2, and 3
runC1 = True
runC2 = False
runC3 = False

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


#Generate the random numbers
wkp = (np.random.randn(run_time,n_st) )
vkp = (np.random.randn(run_time,n_op)*.1 )

#Build results storage
time = np.zeros((run_time,1))
Cost_opt = 0.0
Cost_ce = 0.0
Cost_dual = 0.0
    #Case 1 - Optimal control with known parameters
res1_xk = np.zeros((run_time,n_st))
res1_theta = np.zeros((run_time,n_par))
res1_uk = np.zeros((run_time-1,n_ip))
    #Case 2 - CE control with unknown parameters
res2_xk = np.zeros((run_time,n_st))
res2_theta = np.zeros((run_time,n_par_st+n_par_ip))
res2_uk = np.zeros((run_time,n_ip))
    #Case 3 - Dual control with unknown parameters
res3_xk = np.zeros((run_time,n_st))
res3_theta = np.zeros((run_time,n_par_st+n_par_ip))
res3_uk = np.zeros((run_time,n_ip))

uk_opt= 0.0


#########################################
##  Begin Simulations
######################################

if runC3:
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

    #########################################
    ##  Define system
    ######################################
    #Define the system states
    xk3 = SX.sym('xk3',n_st)
    uk3 = SX.sym('uk3',n_ip)
    wk3 = SX.sym('wk3',n_st)
    vk3 = SX.sym('vk3',n_op)
    thetah3 = SX.sym('thetah3',n_par)
    zk3 = vertcat(xk3,thetah3)
    #uk_opt = SX.sym('uk_opt',n_ctrl,n_ip)

    #Actual parameter values (5.5)
    theta_act = [1.8, -1.01, .58, 0.3, 0.5, 1]

    #Parameter statistics (2.3)/(5.3-5.4)
    theta_h = [1, -0.6, 0.3, 0.1, 0.7, 1.5]
    Sigma_tt = np.diag([0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
    Q = np.diag([1,1,1])
    Qz = np.diag([1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
    R = np.diag([1,1,1])


    rho_v = np.array([0,0,20])
    lamb =10**(-3)
    xk_sp = rho_v
    Wx = rho_v
    Wu = lamb

    #Lower and upper bound on inputs and states
    uk_lb = [-inf]
    uk_ub = [inf]
    xk_lb = [-inf, -inf, -inf]
    xk_ub =  [inf,  inf,  inf]

    #Define the system equations with unknown parameters
    de1 = xk3[1] + thetah3[3]*uk3[0]
    de2 = xk3[2] + thetah3[4]*uk3[0]
    de3 = thetah3[0]*xk3[0] + thetah3[1]*xk3[1] + thetah3[2]*xk3[2] +  thetah3[5]*uk3[0]
    dthetah1 = [0]*thetah3[0]
    dthetah2 = [0]*thetah3[1]
    dthetah3 = [0]*thetah3[2]
    dthetah4 = [0]*thetah3[3]
    dthetah5 = [0]*thetah3[4]
    dthetah6 = [0]*thetah3[5]
    #Define the system equations with known parameters
    e1 = xk3[1] + .3*uk3[0]
    e2 = xk3[2] + .5*uk3[0]
    e3 = 1.8*xk3[0] + -1.01*xk3[1] + .58*xk3[2] +  uk3[0]

    c1_mdl = vertcat(e1,e2,e3)
    sys_ode = vertcat(de1,de2,de3)
    mdl_ode =vertcat(de1,de2,de3,dthetah1,dthetah2,dthetah3,dthetah4,dthetah5,dthetah6)

    #Jacobians for predictions
    Jx3 = Function('Jx3',[xk3,uk3],[jacobian((xk3+c1_mdl),xk3)])
    Jz3 = Function('Jz3',[xk3,uk3,thetah3],[jacobian((vertcat(xk3,thetah3)+mdl_ode),vertcat(xk3,thetah3))])
    Ju3 = Function('Ju3',[xk3,uk3,thetah3],[jacobian((vertcat(xk3,thetah3)+mdl_ode),uk3)])

    ode = {'x':xk3, 'p':vertcat(thetah3,uk3), 'ode':sys_ode}
    F_ode = integrator('F_ode', 'cvodes', ode)

    m_ode = {'x':zk3, 'p':uk3, 'ode':mdl_ode }
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    c_ode = {'x':xk3, 'p':vertcat(uk3), 'ode':c1_mdl}
    C_ode =  integrator('C_ode', 'cvodes', c_ode)

    #Output equation (Observations)
    C = np.eye(3)
    fy = mtimes(C,xk3) + vk3
    h = Function('h',[xk3,vk3], [fy])
    Ch = Function('Ch',[vertcat(xk3,thetah3)],[jacobian(fy,vertcat(xk3,thetah3))])
    Chx = Function('Chx',[xk3],[jacobian(fy,xk3)])

    uk_opt = 0.0
    xkp = np.array([0,0,0])
    theta_par = theta_h
    xkh0=xkp
    zkh0 = vertcat(xkp,theta_par)
    Sigmak_p = np.diag([1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1])

    for k in range(run_time):
        #Compute the measurement
        ykp = h(xkp,vkp[k,:].T)

        #KF update step/measurment
        Czh = Ch(zkh0)
        import pdb; pdb.set_trace()

        Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
        zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
        xkh0 = zkh0[0:n_st]
        theta_par = zkh0[n_st:]
        Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)

        time_remain = run_time - k
        if time_remain < n_pred:
            n_pred = time_remain

        Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(F_ode,n_pred,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk3,theta_par,uk3,Tsamp,xkh0,xk_sp,Wx,Wu,uk_opt)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        ce_res = res_mpc['x'].full().flatten()
        uk_opt = ce_res[n_st+n_ip-1]

        #pdb.set_trace()
        xce1 = ce_res[0::4]
        xce2 = ce_res[1::4]
        xce3 = ce_res[2::4]
        xk_ce = horzcat(xce1,xce2,xce3)
        uk_ce = ce_res[3::4]
        uk_opt = uk_ce[0]

        print('******************************************')
        print('******    Begin Dual Cost (loop ',k+1,')    ****')
        print('******************************************')

        #import pdb; pdb.set_trace()
        #Build the dual optimization problem
        Jd, qu, op = Dual.gencost(n_st,n_ip,n_op,n_par,n_pred,Sigmak,mdl_ode,fy,xk3,uk3,vk3,thetah3,Tsamp,xk_sp,Wx,Wu,Q,Qz,R,uk_ce,xkh0,C_ode,theta_par,xk_ce)
        nlp = {'x':vertcat(*qu), 'f': Jd, 'p': op}
        solver = nlpsol('solver', MySolver, nlp, {'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
        res = solver( x0=uk_opt, p=xkh0)#, ubx=uk_ub, lbx=uk_lb, lbg=0, ubg=0)
        uk_opt = res['x'].full().flatten()

        #Simulate the system
        x_end = F_ode(x0=xkp, p=vertcat(theta_act,uk_opt))
        xkp = x_end['xf'].full()
        xkp = xkp.T #+ wkp[k,:]

        #storing the results
        res3_xk[k,:] = xkp
        res3_theta[k,:] = theta_par.T
        res3_uk[k,:] = uk_opt

        #KF prediction
        Az = Jz3(xkh0,uk_opt,theta_par)
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz

        #if k>15: import pdb; pdb.set_trace()

        Cost_dual = Cost_dual + Wu*uk_opt**2
        if k == run_time-1:
            Cost_dual = 0.5*(Cost_dual + (res3_xk[-1][0]**2+res3_xk[-1][1]**2+(res3_xk[-1][2]-Wx[2])**2))
        if p_plots:
            P.PlotC2(res3_uk,res3_xk,run_time,res3_theta,'Case 3')
if runC1:
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

    for i1 in range(run_time):
        for i2 in range(n_par):
            res1_theta[i1,i2] = theta_act[i2]


    #Parameter statistics (2.3)/(5.3-5.4)
    theta_h = [1, -0.6, 0.3, 0.1, 0.7, 1.5]
    Sigma_tt = np.diag([0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
    Q = np.diag([1,1,1])
    Qz = np.diag([1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
    R = np.diag([1,1,1])

    rho_v = np.array([0,0,20])
    lamb =10**(-3)
    xk_sp = rho_v
    Wx = rho_v
    Wu = lamb

    #Lower and upper bound on inputs and states
    uk_lb = [-inf]
    uk_ub = [inf]
    xk_lb = [-inf, -inf, -inf]
    xk_ub =  [inf,  inf,  inf]

    #Define the system equations with unknown parameters
    de1 = xk[1] + thetah[3]*uk[0]
    de2 = xk[2] + thetah[4]*uk[0]
    de3 = thetah[0]*xk[0] + thetah[1]*xk[1] + thetah[2]*xk[2] +  thetah[5]*uk[0]
    dthetah1 = [0]*thetah[0]
    dthetah2 = [0]*thetah[1]
    dthetah3 = [0]*thetah[2]
    dthetah4 = [0]*thetah[3]
    dthetah5 = [0]*thetah[4]
    dthetah6 = [0]*thetah[5]
    #Define the system equations with known parameters
    e1 = xk[1] + .3*uk[0]
    e2 = xk[2] + .5*uk[0]
    e3 = 1.8*xk[0] + -1.01*xk[1] + .58*xk[2] +  uk[0]

    c1_mdl = vertcat(e1,e2,e3)
    sys_ode = vertcat(de1,de2,de3)
    mdl_ode =vertcat(de1,de2,de3,dthetah1,dthetah2,dthetah3,dthetah4,dthetah5,dthetah6)

    #Jacobians for predictions
    Jx = Function('Jx',[xk,uk],[jacobian((xk+c1_mdl),xk)])
    Jz = Function('Jz',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+mdl_ode),vertcat(xk,thetah))])
    Ju = Function('Ju',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+mdl_ode),uk)])

    ode = {'x':xk, 'p':vertcat(thetah,uk), 'ode':sys_ode}
    F_ode = integrator('F_ode', 'cvodes', ode)

    m_ode = {'x':zk, 'p':uk, 'ode':mdl_ode }
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    #Output equation (Observations)
    C = np.eye(3,3)
    fy = mtimes(C,xk) + vk
    h = Function('h',[xk,vk], [fy])
    Ch = Function('Ch',[vertcat(xk,thetah)],[jacobian(fy,vertcat(xk,thetah))])
    Chx = Function('Chx',[xk],[jacobian(fy,xk)])

    uk_opt = 0.0
    xkp = np.array([0,0,0])
    theta_par = theta_act
    xkh0=xkp
    Sigmak_p = np.diag([1, 1, 1])

    for i2 in range(run_time):

        #Compute the measurement
        ykp = h(xkp,vkp[i2,:].T)

        #KF update step/measurment
        Cxh = Chx(xkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Cxh.T,np.linalg.inv(mtimes(Cxh,mtimes(Sigmak_p,Cxh.T)) + R)))
        xkh0 = xkh0 + mtimes(Kkh,(ykp - h(xkh0,np.zeros((n_st,1)))))
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
        import pdb; pdb.set_trace()
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
            xkp = xkp.T #+ wkp[k,:]

            #KF prediction
            Ax = Jx(xkh0,uk_opt)
            x_end = F_ode(x0=xkh0, p=vertcat(theta_act,uk_opt))
            xkh0 = x_end['xf'].full()
            Sigmak_p = mtimes(Ax,mtimes(Sigmak,Ax.T)) + Q

            res1_uk[i2] += [u_opt[0]]
            res1_xk[i2,:] = [x1[1],x2[1],x3[1]]


    Cost_opt = 0.5*(sum([Wu*res1_uk[i1]**2 for i1 in range(run_time-1)]) +  sum((res1_xk[-2,:]-Wx)**2) )
    if p_plots:
        P.Plot(f_uk[0:-1],f_xk[0:-1],run_time,'Case 1')
if runC2:
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

    ########################################
    ##  Define system
    ######################################
    #Define the system states
    xk2 = SX.sym('xk2',n_st)
    uk2 = SX.sym('uk2',n_ip)
    wk2 = SX.sym('wk2',n_st)
    vk2 = SX.sym('vk2',n_op)
    thetah2 = SX.sym('thetah2',n_par)
    zk2 = vertcat(xk2,thetah2)
    #uk_opt = SX.sym('uk_opt',n_ctrl,n_ip)

    #Actual parameter values (5.5)
    theta_act = [1.8, -1.01, .58, 0.3, 0.5, 1]

    #Parameter statistics (2.3)/(5.3-5.4)
    theta_h = [1, -0.6, 0.3, 0.1, 0.7, 1.5]
    Sigma_tt = np.diag([0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
    Q = np.diag([1,1,1])
    Qz = np.diag([1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
    R = np.diag([1,1,1])

    rho_v = np.array([0,0,20])
    lamb =10**(-3)
    xk_sp = rho_v
    Wx = rho_v
    Wu = lamb

    #Lower and upper bound on inputs and states
    uk_lb = [-inf]
    uk_ub = [inf]
    xk_lb = [-inf, -inf, -inf]
    xk_ub =  [inf,  inf,  inf]

    #Define the system equations with unknown parameters
    de1 = xk2[1] + thetah2[3]*uk2[0]
    de2 = xk2[2] + thetah2[4]*uk2[0]
    de3 = thetah2[0]*xk2[0] + thetah2[1]*xk2[1] + thetah2[2]*xk2[2] +  thetah2[5]*uk2[0]
    dthetah1 = [0]*thetah2[0]
    dthetah2 = [0]*thetah2[1]
    dthetah3 = [0]*thetah2[2]
    dthetah4 = [0]*thetah2[3]
    dthetah5 = [0]*thetah2[4]
    dthetah6 = [0]*thetah2[5]
    #Define the system equations with known parameters
    e1 = xk2[1] + .3*uk2[0]
    e2 = xk2[2] + .5*uk2[0]
    e3 = 1.8*xk2[0] + -1.01*xk2[1] + .58*xk2[2] +  uk2[0]

    c1_mdl = vertcat(e1,e2,e3)
    sys_ode = vertcat(de1,de2,de3)
    mdl_ode =vertcat(de1,de2,de3,dthetah1,dthetah2,dthetah3,dthetah4,dthetah5,dthetah6)

    #Jacobians for predictions
    Jx2 = Function('Jx2',[xk2,uk2],[jacobian((xk2+c1_mdl),xk2)])
    Jz2 = Function('Jz2',[xk2,uk2,thetah2],[jacobian((vertcat(xk2,thetah2)+mdl_ode),vertcat(xk2,thetah2))])
    Ju2 = Function('Ju2',[xk2,uk2,thetah2],[jacobian((vertcat(xk2,thetah2)+mdl_ode),uk2)])

    ode = {'x':xk2, 'p':vertcat(thetah2,uk2), 'ode':sys_ode}
    F_ode = integrator('F_ode', 'cvodes', ode)

    m_ode = {'x':zk2, 'p':uk2, 'ode':mdl_ode }
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    c_ode = {'x':xk2, 'p':vertcat(uk2), 'ode':c1_mdl}
    C_ode =  integrator('C_ode', 'cvodes', c_ode)

    #Output equation (Observations)
    C = np.eye(3)
    fy = mtimes(C,xk2) + vk2
    h = Function('h',[xk2,vk2], [fy])
    Ch = Function('Ch',[vertcat(xk2,thetah2)],[jacobian(fy,vertcat(xk2,thetah2))])
    Chx = Function('Chx',[xk2],[jacobian(fy,xk2)])

    uk_opt = 0.0
    xkp = np.array([0,0,0])
    theta_par = theta_h
    xkh0=xkp
    zkh0 = vertcat(xkp,theta_par)
    Sigmak_p = np.diag([1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1])
    Sigmak = 0

    for k in range(run_time):

        time[k,:] = k
        #Compute the measurement
        ykp = h(xkp,vkp[k,:].T)

        #KF update step/measurment
        Czh = Ch(zkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
        zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
        xkh0 = zkh0[0:n_st]
        theta_par = zkh0[n_st:]
        Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)

        time_remain = run_time - k
        if time_remain < n_pred:
            n_pred = time_remain

        print('******************************************')
        print('******    Begin CE (loop ',k+1,')    ****')
        print('******************************************')

        Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(F_ode,n_pred,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk2,theta_par,uk2,Tsamp,xkh0,xk_sp,Wx,Wu,uk_opt)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        uk_ce = res_mpc['x'].full().flatten()
        uk_opt = uk_ce[n_st+n_ip-1]
        #pdb.set_trace()

        #Simulate the system
        x_end = F_ode(x0=xkp, p=vertcat(theta_act,uk_opt))
        xkp = x_end['xf'].full()
        xkp = xkp.T #+ wkp[k,:]

        #KF prediction
        Az = Jz2(xkh0,uk_opt,theta_par)
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz

        #storing the results
        res2_uk[k,:] = uk_opt
        res2_xk[k,:] = xkp
        res2_theta[k,:] = theta_par.T

        Cost_ce = Cost_ce + Wu*(uk_opt**2)

    #pdb.set_trace()
    Cost_ce = 0.5*(Cost_ce + sum((res2_xk[-1,:]-Wx)**2))
    if p_plots:
        P.PlotC2(res2_uk,res2_xk,run_time,res2_theta,'Case 2')

print 'Cost_dual = ',Cost_dual
#print '        est error  = ' , (res3_theta[-1,:] - theta_act)**2
#print '            final stages = ', res3_xk[-1,:], res3_xk[-2,:]
#print '             control variation (u_ce - u_dual) = ', Dual_vs_CE
print 'Cost_ce = ', Cost_ce
#print '        est error  = ' , (res2_theta[-1,:] - theta_act)**2

print' Relative Error (ce - dual) = ' ,(res2_theta[-1,:] - theta_act)**2 - (res3_theta[-1,:] - theta_act)**2
print'negative is bad, larger values are better'


print 'Cost_opt = ',Cost_opt


def writeDataOpt1(Cost_ce,res2_uk,res2_xk,res2_theta,run_time,csv_file,Case):
    File = open(csv_file, 'wb')
    with File:
        writer = csv.writer(File)

        writer.writerow( ["Parameter", Case] )
        writer.writerow( ["Cost",    Cost_ce ] )

        for i1 in range(run_time):
            writer.writerow( ['xk1' , res2_xk[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk2' , res2_xk[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk3' , res2_xk[i1,2] ] )
        for i1 in range(run_time-1):
            writer.writerow( ['uk'  , res2_uk[i1,0] ] )


        for i1 in range(run_time):
            writer.writerow( ['t1' , res2_theta[0] ] )
        for i1 in range(run_time):
            writer.writerow( ['t2' , res2_theta[1] ] )
        for i1 in range(run_time):
            writer.writerow( ['t3' , res2_theta[2] ] )
        for i1 in range(run_time):
            writer.writerow( ['t4' , res2_theta[3] ] )
        for i1 in range(run_time):
            writer.writerow( ['t5' , res2_theta[4] ] )
        for i1 in range(run_time):
            writer.writerow( ['t6' , res2_theta[5] ] )

def writeDataSingle(Cost_ce,res2_uk,res2_xk,res2_theta,run_time,csv_file,Case):
    File = open(csv_file, 'wb')
    with File:
        writer = csv.writer(File)

        writer.writerow( ["Parameter", Case] )
        writer.writerow( ["Cost",    Cost_ce ] )

        for i1 in range(run_time):
            writer.writerow( ['xk1' , res2_xk[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk2' , res2_xk[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk3' , res2_xk[i1,2] ] )
        for i1 in range(run_time-1):
            writer.writerow( ['uk'  , res2_uk[i1,0] ] )


        for i1 in range(run_time):
            writer.writerow( ['t1' , res2_theta[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['t2' , res2_theta[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['t3' , res2_theta[i1,2] ] )
        for i1 in range(run_time):
            writer.writerow( ['t4' , res2_theta[i1,3] ] )
        for i1 in range(run_time):
            writer.writerow( ['t5' , res2_theta[i1,4] ] )
        for i1 in range(run_time):
            writer.writerow( ['t6' , res2_theta[i1,5] ] )

def writeDataOpt(opt,ce,dual,uk1,xk1,theta1,uk2,xk2,theta2,uk3,xk3,theta3,run_time,csv_file,Case1,Case2,Case3):
    File = open(csv_file, 'a')
    with File:
        writer = csv.writer(File)

        writer.writerow( ["Parameter", Case1, Case2, Case3] )
        writer.writerow( ["Cost",  opt[0],ce, dual[0]] )

        for i1 in range(run_time):
            writer.writerow( ['xk1' , xk1[i1,0], xk2[i1,0], xk3[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk2' , xk1[i1,1], xk2[i1,1], xk3[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk3' , xk1[i1,2], xk2[i1,2], xk3[i1,2] ] )
        for i1 in range(run_time-1):
            writer.writerow( ['uk'  , uk1[i1,0], uk2[i1,0], uk3[i1,0] ] )


        for i1 in range(run_time):
            writer.writerow( ['t1' , theta1[0], theta2[i1,0], theta3[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['t2' , theta1[1], theta2[i1,1], theta3[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['t3' , theta1[2], theta2[i1,2], theta3[i1,2] ] )
        for i1 in range(run_time):
            writer.writerow( ['t4' , theta1[3], theta2[i1,3], theta3[i1,3] ] )
        for i1 in range(run_time):
            writer.writerow( ['t5' , theta1[4], theta2[i1,4], theta3[i1,4] ] )
        for i1 in range(run_time):
            writer.writerow( ['t6' , theta1[5], theta2[i1,5], theta3[i1,5] ] )


if save2csv:
    if runC1 and runC2 and runC3:
        writeDataOpt(Cost_opt,Cost_ce,Cost_dual,res1_uk,res1_xk,theta_act,res2_uk,res2_xk,res2_theta,res3_uk,res3_xk,res3_theta,run_time,csv_file,"Case Opt","Case CE","Case Dual")
    #pdb.set_trace()
    elif runC1:
        writeDataOpt1(Cost_opt,res1_uk,res1_xk,theta_act,run_time,csv_file,"Case Opt")
    elif runC2:
        writeDataSingle(Cost_ce,res2_uk,res2_xk,res2_theta,run_time,csv_file,"Case CE")
    elif runC3:
        writeDataSingle(Cost_dual,res3_uk,res3_xk,res3_theta,run_time,csv_file,"Case Dual")
    #P.writeDataCE(Cost_opt,Cost_ce,f_uk,f_xk,theta_act,res2_uk,res2_xk,res2_theta,run_time,csv_file)
    #P.writeData(Cost_opt,Cost_ce,Cost_dual,u_opt,res1_xk,theta_act,res1_uk,res1_xk,theta_act,res2_uk,res2_xk,res2_theta,run_time,csv_file)
