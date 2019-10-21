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

n_sim = 10

multi_sweep = 0
gen_nom = 1
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
run_name = str(alpha_u)+'alpha_u_'+str(alpha_l)+'alpha_l'+str(beta_u)+'beta_u_'\
            +str(beta_l)+'beta_l'

testname = 'U0.998_L1.0'

if multi_sweep:
    if gen_nom:
        MySolver = "ipopt"
        opts = {}
        opts = {"ipopt.print_level":5, "print_time": True,
                'ipopt.max_iter':1000, 'output_file':'Main_out.txt'}

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
        T0 = 325*alpha_u
        Tj0 = 325
        Tj0_in = 325
        pi = 3.14159265

        theta_nom = np.array([3.0457*10**(-7)*discritize, -323.05]) #[k0,dH]
        theta_act = theta_nom

        #Parameter statistics (2.3)/(5.3-5.4)
        Q = np.diag([0, 0, 0, 0,  0,  0,  0])
        Qz = np.diag([0, 0, 0, 0,  0,  0,  0,
                    ((3.0457*10**(-7))*discritize*.2)**2, (323.05*.2)**2 ])

        #Build results storage
        time = np.zeros((run_time,1))

        # Tighten constraints for tube MPC
        uk_lb = [0,                     280]
        uk_ub = [9*10**(-6)*discritize, 350]
        xk_lb = [0,      0,  0,   0,   321*alpha_l, 0,   0]
        xk_ub = [.007,  inf, inf, inf, 325*alpha_u, inf, inf]

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
        de5 = (uk2[0]/xk2[0])*(Tin - xk2[4]) - (U*((2*xk2[0]/r)-pi*r**2)*(xk2[4]
                -xk2[5]))/(rho*cp*xk2[0]) - (thetah2[0]*xk2[1]*xk2[2]*
                thetah2[1])/(rho*cp)
        de6 = (Vjin/Vj) * (xk2[6]-xk2[5]) + (U*((2*xk2[0]/r)-pi*r**2)*(xk2[4]-
                xk2[5]))/(rho*cp*Vj)
        de7 = (1/tau_c)* (uk2[1]-xk2[6])
        dthetah1 = [0]*thetah2[0]
        dthetah2 = [0]*thetah2[1]

        sys_ode = vertcat(de1,de2,de3,de4,de5,de6,de7)
        mdl_ode = vertcat(de1,de2,de3,de4,de5,de6,de7,dthetah1,dthetah2)
        mdl_ode2 = vertcat(de1+wk2[0],de2+wk2[1],de3+wk2[2],de4+wk2[3],de5+wk2[4]
                            ,de6+wk2[5],de7+wk2[6],dthetah1,dthetah2)

        #Jacobians for predictions
        Jx2 = Function('Jx2',[xk2,uk2,thetah2],[jacobian((xk2+sys_ode),xk2)])
        Jz2 = Function('Jz2',[xk2,uk2,thetah2],[jacobian((vertcat(xk2,thetah2)+
                        mdl_ode),vertcat(xk2,thetah2))])
        Ju2 = Function('Ju2',[xk2,uk2,thetah2],[jacobian((vertcat(xk2,thetah2)+
                        mdl_ode),uk2)])
        L2   = Function('L2' ,[xk2,uk2,thetah2,wk2],[jacobian((vertcat(xk2,
                        thetah2)+mdl_ode2),wk2)])

        ode2 = {'x':xk2, 'p':vertcat(thetah2,uk2), 'ode':sys_ode}
        F_ode = integrator('F_ode', 'cvodes', ode2)

        m_ode = {'x':zk2, 'p':uk2, 'ode':mdl_ode }
        M_ode = integrator('M_ode', 'cvodes', m_ode)

        #Output equation (Observations)
        C = np.eye(n_st)
        fy = mtimes(C,xk2) + vk2
        h = Function('h',[xk2,vk2], [fy])
        Ch = Function('Ch',[vertcat(xk2,thetah2)],[jacobian(fy,vertcat(xk2,
                        thetah2))])
        Chx = Function('Chx',[xk2],[jacobian(fy,xk2)])

        uk_opt = np.array([2.5*10**(-6)*discritize,280])
        xkp = np.array([V0, Ca0,Cb0,Cc0,T0,Tj0,Tj0_in])
        theta_par = theta_nom
        xkh0 = xkp
        zkh0 =vertcat(xkh0,theta_par)
        Sigmak_p = Qz

        Jce,qu_ce,lbq,ubq,g,lbg,ubg,qu_init=Dual.ce_mpc(F_ode,n_pred,n_ctrl,
                                            n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,
                                            xk_ub,xk2,theta_par,uk2,Tsamp,xkh0,
                                            uk_opt)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':
                            {'max_iter':1000,"check_derivatives_for_naninf":
                            'yes', "print_user_options":'yes' }})
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        res = res_mpc['x'].full().flatten()

        xce1 = res[0::n_st+n_ip]
        xce2 = res[1::n_st+n_ip]
        xce3 = res[2::n_st+n_ip]
        xce4 = res[3::n_st+n_ip]
        xce5 = res[4::n_st+n_ip]
        xce6 = res[5::n_st+n_ip]
        xce7 = res[6::n_st+n_ip]
        res_xk = horzcat(xce1,xce2,xce3,xce4,xce5,xce6,xce7)

    import pdb; pdb.set_trace()

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
