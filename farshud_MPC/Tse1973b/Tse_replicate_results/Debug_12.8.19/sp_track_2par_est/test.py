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
runC2 = True
runC3 = True # N/A

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
x_act = SX.sym('x_act',n_st,run_time+1)
u_act = SX.sym('u_act',n_ip,run_time+1)
yk = SX.sym('yk',n_op)
wk = SX.sym('wk',n_st)
vk = SX.sym('vk',n_op)
theta_st = SX.sym('theta',n_par_st)
theta_ip = SX.sym('theta',n_par_ip)
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
x_act[:,0] = xkp

rho_v = np.array([0,0,20])
lamb =10**(-3)

#Define control objective
xk_sp = rho_v
Wx = rho_v
Wu = lamb

#Define the initial model condition
zkh0 = vertcat(xkp,theta_h)
Sigmak_p = np.diag([1,1,1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1])

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

sys_ode = vertcat(de1,de2,de3)
mdl_ode =vertcat(de1,de2,de3,dthetah1,dthetah2,dthetah3,dthetah4,dthetah5,dthetah6)

Jz = Function('Jz',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+mdl_ode),vertcat(xk,thetah))])
Ju = Function('Ju',[xk,uk,thetah],[jacobian((vertcat(xk,thetah)+mdl_ode),uk)])

ode = {'x':xk, 'p':vertcat(thetah,uk), 'ode':sys_ode}
F_ode = integrator('F_ode', 'cvodes', ode)
#7F_ode = Function('F_ode', [xk, uk, thetah], [sys_ode])

m_ode = {'x':zk, 'p':uk, 'ode':mdl_ode }
M_ode = integrator('M_ode', 'cvodes', m_ode)

#Output equation (Observations)
C = np.matrix('0 0 1')
fy = mtimes(C,xk) + vk
h = Function('h',[xk,vk], [fy])
Ch = Function('Ch',[vertcat(xk,thetah)],[jacobian(fy,vertcat(xk,thetah))])


#Build results storage
time = np.zeros((run_time,1))

res3_xk = np.zeros((run_time,n_st))
res3_theta = np.zeros((run_time,n_par_st+n_par_ip))
res3_uk = np.zeros((run_time,n_ip))

uk_opt=0
xkh0 = np.zeros((n_st,1))
theta_par = [1.8, -1.01, .58, 0.3, 0.5, 1]

for k in range(run_time):
    time[k,:] = k

    res3_uk[k,:] = uk_opt
    res3_xk[k,:] = xkh0.T
    res3_theta[k,:] = theta_par

    time_remain = run_time - k
    if time_remain < n_pred:
        n_pred = time_remain

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

    lbq += [xkh0[i1] for i1 in range(n_st)]
    ubq += [xkh0[i1] for i1 in range(n_st)]
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

        Jce = Jce + uk**2*Wu
        #Define the system equations
        de1 = xk[1] + theta_par[3]*uk[0]
        de2 = xk[2] + theta_par[4]*uk[0]
        de3 = theta_par[0]*xk[0] + theta_par[1]*xk[1] + theta_par[2]*xk[2] +  theta_par[5]*uk[0]
        sys_ode = vertcat(de1,de2,de3)

        m_ode = {'x':xk, 'p':uk, 'ode':sys_ode}
        M_ode = integrator('M_ode', 'cvodes', m_ode)

        x_end = M_ode(x0=xk, p=uk)
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
    #    if i1 == n_pred:
    #        Jce = .5*(mtimes((xk-Wx).T,(xk-Wx)) + Jce)


#    Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(sys_ode,n_pred,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,theta_par,uk,Tsamp,xkh0,xk_sp,Wx,Wu,uk_opt)
#    pdb.set_trace()
    Jce = .5*(mtimes((xk-Wx).T,(xk-Wx)) + Jce)
    qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}

    solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc)
    res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
    uk_ce = res_mpc['x'].full().flatten()
    uk_opt = uk_ce[n_st+n_ip-1]
#    pdb.set_trace()

    #Simulate the system
    x_end = M_ode(x0=xkh0, p=uk_opt)
    xkp = x_end['xf'].full()
#    xkp = xkp.T + wkp[k,:]
    xkh0 = xkp# + wkp[k,:]

    print('******************************************')
    print('******    Run number ',k+1,' complete    ****')
    print('******************************************')
    print Jce
    print(Jce.shape)
    print uk_ce

res3_xk[k,:] = xkh0.T
res3_theta[k,:] = theta_par
#P.PlotC2(res2_uk,res2_xk,run_time,res2_theta,'Case 2')
P.PlotC2(res3_uk,res3_xk,run_time,res3_theta,'Case 3')
