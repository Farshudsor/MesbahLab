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

n_sim = 5

multi_sweep = 1
gen_nom = 0
plots = 1

#problem run time
run_time = 10
Tsamp = 2
discritize = 60 * Tsamp # [=] seconds/(min) - discritze in two min
#Define the controller
n_pred = 10
n_ctrl = 10
#Define the problem size
n_st = 7;   n_ip = 2
n_op = 2;   n_par = n_op

res_xk = np.zeros((run_time+1,n_st));   res_theta = np.zeros((run_time+1,n_par))
res_uk = np.zeros((run_time,n_ip))

x_u = np.round(.998,3); x_l = np.round(1.00,3)
u_u = np.round(1.00,3); u_l = np.round(1.00,3)
bounds = [x_u,x_l,u_u,u_l]

alpha_u = x_u;  alpha_l = x_l
beta_u = u_u;   beta_l = u_l
run_name = str(alpha_u)+'alpha_u_'+str(alpha_l)+'alpha_l'+str(beta_u)+'beta_u_'+str(beta_l)+'beta_l'

testname = 'w_1000'

if multi_sweep:
    if gen_nom:
        MySolver = "ipopt"
        opts = {}
        opts = {"ipopt.print_level":5, "print_time": True, 'ipopt.max_iter':1000, 'output_file':'Main_out.txt'}

        #Generate the random numbers
        R = np.diag([10**-17, 5, 5, 5, .2, .2, .2])
        wkp = (np.random.randn(run_time,n_st+n_par))*0
        vkp = (np.random.randn(run_time,n_st))
        vkp = mtimes(vkp,R)

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
        #theta_act = theta_nom

        #Parameter statistics (2.3)/(5.3-5.4)
        Q = np.diag([0, 0, 0, 0,  0,  0,  0])
        Qz = np.diag([0, 0, 0, 0,  0,  0,  0, ((3.0457*10**(-7))*discritize*.2)**2, (323.05*.2)**2 ])

        #Build results storage
        time = np.zeros((run_time,1))

        # Tighten constraints for tube MPC
        uk_lb = [0,                     280]
        uk_ub = [9*10**(-6)*discritize*beta_u, 350]
        xk_lb = [0,      0,  0,   0,   321*alpha_l, 0,   0]
        xk_ub = [.007,  inf, inf, inf, 325*alpha_u, inf, inf]

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

    for i in range(n_sim):
        seed = i
        sys.argv = ['DualMPC.py','alpha_u','alpha_l','beta_u','beta_l',
         'run_name' ,'seed', 'res_xk','res_uk','res_theta','testname']
        execfile('DualMPC.py')

if plots:
    folder = 'results/'+testname+'/'
    file = "results/"+testname+"_"+run_name+".csv"
    sys.argv = ['plotting_script.py', 'folder', 'file', 'bounds', 'testname' ]
    execfile('plotting_script.py')
