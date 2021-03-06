"""
Created on March 4th 2019

@author: Farshud Sorourifar

Execute multiple simulations of DualMPC.py
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

n_sim = 2

multi_sweep = 1
gen_nom = 1
plots = 0

#problem run time
Tsamp = 3
run_time = 600
discritize = 60 * Tsamp # [=] seconds/(min) - discritze in two min
#Define the controller
#n_pred = 420
n_ctrl = 140
#Define the problem size
n_st = 2;   n_ip = 1; n_par = 0

#backoffs; alpha for upper coinstraint, beta for Lower
# for alpha and beta, [0] for u1, [1] for x1, [2] for x2
alpha = np.random.uniform(0, .15 ,3)
beta = np.random.uniform(0,.15, 3)
run_name = "random_backoffs"
testname = 'Test_'

if multi_sweep:
    if gen_nom:
        MySolver = "ipopt"
        opts = {}
        opts = {"ipopt.print_level":5, "print_time": True,
                'ipopt.max_iter':1000, 'output_file':'Main_out.txt'}

        #Build results storage
        time = np.zeros((run_time,1))

        # Model Parameters
        k0 = 300
        theta = 20
        xf = .3947
        xc = .3816
        M = 5
        alpha = 0.117
        x0 = [0.9831,0.3918]
        xe  = [.2632,.6519]

        #Lower and upper bound on inputs and states
        uk_lb = [.12];        uk_ub = [1.999]
        #uk_lb = [.0];        uk_ub = [inf]
        xk_lb = [0, 0 ];    xk_ub = [1, 1]

        #########################################
        ##  Define system
        ######################################
        #Define the system states
        xk4 = SX.sym('xk4',n_st)
        xks = xe
        uk4 = SX.sym('uk4',n_ip)
        uks = .7585
        wk4 = SX.sym('wk4',1)

        #Define the system equations with unknown parameters
        dx1= (1-xk4[0])/theta -  k0*xk4[0]*exp(-M/xk4[1])
        dx2= (xf-xk4[1])/theta + k0*xk4[0]*exp(-M/xk4[1]) - alpha*uk4[0]*(xk4[1]-xc)

        mdl_ode = vertcat(dx1,dx2)
        m_ode = {'x':xk4, 'p':uk4, 'ode':mdl_ode }
        M_ode = integrator('M_ode', 'cvodes', m_ode)

        fun = Function('fun', [xk4,uk4], [mdl_ode])

        Jx = Function('Jx', [xk4,uk4],[jacobian(mdl_ode,xk4)])
        Ju = Function('Ju', [xk4,uk4],[jacobian(mdl_ode,uk4)])

        lin_mdl = mtimes(Jx(xe,uks),(xk4-xks)) + mtimes(Ju(xe,uks),(uk4-uks))
        lin_fun = Function('lin_fun', [xk4,uk4], [lin_mdl])

        test1 = lin_fun(x0,.8305)
        test2 = lin_fun(xe,.7585)
        l_ode = {'x':xk4, 'p':uk4, 'ode':lin_mdl }
        L_ode = integrator('M_ode', 'cvodes', l_ode)

        #M_ode = L_ode

        xt = xe
        x_t = np.zeros((run_time,n_st))
        if False: #verify the nominal trajectory
            for i9 in range(run_time):
                if i9%Tsamp == 0:
                    run = i9/Tsamp
                time+=[i9]
                xt = M_ode(x0=xt, p=2)
                xt = xt['xf']
                x_t[i9,:] = xt.T

            #plt.plot(range(len(xce1)),xce1)
            #plt.plot(range(len(xce1)),xce2)
            plt.plot(range(run_time), x_t[:,0], ':')
            plt.plot(range(run_time), x_t[:,1], ':')
            #plt.plot(range(len(xce1)),xe[0]*np.ones((len(xce1),1)))
            #plt.plot(range(len(xce1)),xe[1]*np.ones((len(xce1),1)))
            plt.show()

        import pdb; pdb.set_trace()
        uk_opt = .55

        Jce,qu_ce,lbq,ubq,g,lbg,ubg,qu_init=Dual.ce_mpc(M_ode,run_time,n_ctrl,
                                            n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,
                                            xk_ub,xk4,uk4,Tsamp,x0,uk_opt)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':
                            {'max_iter':1000,"check_derivatives_for_naninf":
                            'yes', "print_user_options":'yes' }})
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        res = res_mpc['x'].full().flatten()

        xres = res[0:(run_time+1)*2]
        xce1 = xres[0::n_st]
        xce2 = xres[1::n_st]
        uce1 = res[(run_time+1)*2:]

        xt = x0
        x_t = np.zeros((run_time,n_st))
        if True: #verify the nominal trajectory
            for i9 in range(run_time):
                if i9%Tsamp == 0:
                    run = i9/Tsamp
                time+=[i9]
                xt = M_ode(x0=xt, p=uce1[run])
                xt = xt['xf']
                x_t[i9,:] = xt.T

            plt.plot(range(len(xce1)),xce1)
            plt.plot(range(len(xce1)),xce2)
            #plt.plot(range(run_time), x_t[:,0], ':')
            #plt.plot(range(run_time), x_t[:,1], ':')
            plt.plot(range(len(xce1)),xe[0]*np.ones((len(xce1),1)))
            plt.plot(range(len(xce1)),xe[1]*np.ones((len(xce1),1)))
            plt.show()
            import pdb; pdb.set_trace()

        res_xk = horzcat(xce1,xce2)
        res_uk = uce1

    for i in range(n_sim):
        seed = i
        sys.argv = ['DualMPC.py','alpha','beta',
         'run_name' ,'seed', 'res_xk','res_uk','testname']
        execfile('DualMPC.py')

if plots:
    folder = 'results/'+testname+'/'
    file = "results/"+testname+"_"+run_name+".csv"
    sys.argv = ['plotting_script.py', 'folder', 'file', 'bounds', 'testname' ]
    execfile('plotting_script.py')
