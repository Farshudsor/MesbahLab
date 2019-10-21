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
import dual as Dual
import pdb
#import results as P
import csv

#########################################
##  Set-up
######################################
csv_file = "Results.csv"
save2csv = True
p_plots = True

if save2csv:
    File = open(csv_file, 'wb')
    with File:
        writer = csv.writer(File)

# Run Case 1, 2, and 3
runC2 = True
runC3 = True

run_time = 100
Disturbance = 25 # time when k0 begins to be reduced
end_Disturbance = 45 # k0 has been incresed by 10%
slope_Disturbance = (7.2*(10**10)/10 - 7.92*(10**10)/10 ) / (end_Disturbance - Disturbance)

Tsamp = .1
#Define the controller
n_pred = 10
n_ctrl = 10

#Define the problem size
n_st = 2
n_ip = 2
n_op = 2
n_par_st = 1
n_par_ip = 1
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
xk_sp = np.array([.5, 350 ])
Sigmak_p = np.diag([10**-2, 10**-1])
wkp = (np.random.randn(run_time,n_st) )
vkp = (np.random.randn(run_time,n_op))
vkp = mtimes(vkp,Sigmak_p*2)
Wx = np.diag([(1./.5),(1./350*0)])*10
Wu = np.diag([(1./900),(1./300)])*.01

#import pdb; pdb.set_trace()

#Build results storage
time = np.zeros((run_time,1))
time_pred = np.zeros((n_pred,1))
Cost_opt = 0.0
Cost_ce = 0.0
Cost_dual = 0.0

    #Optimal control with known parameters
act_theta = np.zeros((run_time,n_par))
    #Case 2 - CE control with unknown parameters
res2_xk = np.zeros((run_time,n_st))
res2_theta = np.zeros((run_time,n_par_st+n_par_ip))
res2_uk = np.zeros((run_time-1,n_ip))
    #Case 3 - Dual control with unknown parameters
res3_xk = np.zeros((run_time,n_st))
res3_theta = np.zeros((run_time,n_par_st+n_par_ip))
res3_uk = np.zeros((run_time-1,n_ip))

# theta iactual for plotting purpouse
theta_act = [7.2*(10**10)/10, -5.*(10**4)] #  [ k0, dH ]
for i in range(run_time):
    if i >= Disturbance and i <= end_Disturbance :
        theta_act[0] = theta_act[0] - slope_Disturbance
    act_theta[i] = [theta_act[0],theta_act[1]]



def Mkplots(time, res1_xk, res1_uk, res1_theta, case, a,b,c,set, T_act = act_theta):
    time = time*.1
    bl = np.ones(run_time)
    #import pdb; pdb.set_trace()
    if a:
        #a = plt.figure(set[0])
        #plt.clf()
        #plt.subplot(2,1,1)
        #plt.title(case)
        #plt.subplot(2, 1, 1)
        #plt.ylim(.45,.6)
        #plt.ylabel('$C_A$')
        #plt.plot(time,res1_xk[:,0], time, np.ones(run_time)*.485)
        #plt.legend(['$C_A$'])
        #plt.grid()
        #plt.subplot(2, 1, 2)
        #plt.ylim(340,360)
        #plt.ylabel('$T_r$')
        #plt.plot(time,res1_xk[:,1])
        #plt.legend(['$T_r$'])
        #plt.grid()
        a = plt.figure(set[0])
        plt.clf()
        plt.title(case)
        plt.ylim(.475,.535)
        plt.ylabel('$C_A$')
        plt.plot(time,res1_xk[:,0], time, bl*.485)
        plt.legend(['$C_A$', 'Lower Constraint'])
        plt.grid()
        plt.savefig(case+'_a.png')

    if b:
        b = plt.figure(set[1])
        plt.clf()
        plt.subplot(2,1,1)
        plt.title(case)
        plt.subplot(2, 1, 1)
        plt.ylim(35,125)
        plt.ylabel('q')
        plt.step(time[0:-1],res1_uk[:,0],time[0:-1],bl[0:-1]*40,time[0:-1],bl[0:-1]*120)
        #plt.legend(['q', 'lower bound', 'upper bound'])
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.ylim(285,325)
        plt.ylabel('$T_c$')
        plt.step(time[0:-1],res1_uk[:,1], time[0:-1],bl[0:-1]*290,time[0:-1],bl[0:-1]*320)
        #plt.legend(['$T_c$', 'lower bound', 'upper bound'])
        plt.grid()
        plt.savefig(case+'_b.png')

    if c:
        c = plt.figure(set[2])
        plt.clf()
        plt.subplot(2,1,1)
        plt.title(case)
        plt.subplot(2, 1, 1)
        plt.ylabel('$k_0$ %error')
        plt.plot(time,((res1_theta[:,0]- act_theta[:,0])/act_theta[:,0]))
        #plt.legend(['$k_0$ estimate'])
        plt.grid()
        plt.subplot(2, 1, 2)
        #plt.ylim(-5.*(10**4) - 1, -5.*(10**4)+1)
        plt.ylabel('$\Delta H$ %error')
        plt.plot(time,((res1_theta[:,1]-act_theta[:,1])/act_theta[:,1]))
        #plt.legend(['$\Delta H_r$'])
        plt.grid()
        plt.savefig(case+'_c.png')


def writeData(uk_sp,xk_sp, theta_act,res2_uk,res2_xk,res2_theta,res3_uk,res3_xk,res3_theta,run_time,csv_file):
    File = open(csv_file, 'w')
    with File:
        writer = csv.writer(File)
        writer.writerow( ["Parameter", "Actual values", "Case 2 : CE", "Case3 : Dual", '' ,"diffrence in controlers", "CE offset", "Dual offse"])

        for i1 in range(run_time):
            writer.writerow( ['xk_1', xk_sp[0], res2_xk[i1,0], res3_xk[i1,0], '', res2_xk[i1,0]-res3_xk[i1,0], xk_sp[0]-res2_xk[i1,0], xk_sp[0]-res3_xk[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk_2', xk_sp[1], res2_xk[i1,1], res3_xk[i1,1], '', res2_xk[i1,1]-res3_xk[i1,1], xk_sp[1]-res2_xk[i1,1], xk_sp[1]-res3_xk[i1,1] ] )
        for i1 in range(run_time-1):
            writer.writerow( ['uk_1', uk_sp[0], res2_uk[i1,0], res3_uk[i1,0], '', res2_uk[i1,0]-res3_uk[i1,0], uk_sp[0]-res2_uk[i1,0], uk_sp[0]-res3_uk[i1,0] ] )
        for i1 in range(run_time-1):
            writer.writerow( ['uk_2', uk_sp[1], res2_uk[i1,1], res3_uk[i1,1], '', res2_uk[i1,1]-res3_uk[i1,1], uk_sp[1]-res2_uk[i1,1], uk_sp[1]-res3_uk[i1,1] ] )

        for i1 in range(run_time):
            writer.writerow( ['t1', theta_act[i1,0], res2_theta[i1,0], res3_theta[i1,0], '', res2_theta[i1,0]-res3_theta[i1,0], theta_act[i1,0]-res2_theta[i1,0], theta_act[i1,0]-res3_theta[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['t2', theta_act[i1,1], res2_theta[i1,1], res3_theta[i1,1], '', res2_theta[i1,1]-res3_theta[i1,1], theta_act[i1,1]-res2_theta[i1,1], theta_act[i1,1]-res3_theta[i1,1] ] )



#########################################
##  Begin Simulations
######################################

if runC3:

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

    #######################
    #model
    ###########
    #variables
    #Ca = x[0]                %concentration of species A
    #Tr = x[1]               %reactor temp

    #Inputs
    #qdot = u[0]              %volumetric flow rate
    #Tc = u[1]               %Coolant temp

    # estimate parameters
    #k0 = theta[0]               %kinetic constant
    #dH = theta[1]               %reaction heat

    #known parameters
    Cain = 1.              #concentration of species A inlet
    Tin = 350.              #inlet temp
    EaR = 8750.              #%Energy of activation
    UA = 5.*(10**4) / 10               #overall heat transfer coefficient* heat trans. area
    cp =  .239              #heat capacity
    rho =  1000.           #density
    V =  100.               #volume
    theta_act = [7.2*(10**10)/10, -5.*(10**4)] #  [ k0, dH ]

    #Parameter statistics (2.3)/(5.3-5.4)
    Q = np.diag([1./.5,1./350*0])
    Qz = np.diag([10**-2, 10**-1, 9*10**18,10**-1])# 10**-1, 9*10**18, 10**-1])
    R = np.diag([1,1])

    uk_sp = np.array([100.,300.])
    xk_sp = np.array([.5, 350. ])
    #Wx = np.diag([(1./.5),(1./350)*0])*10
    #Wu = np.diag([(1./900),(1./300)])*.1

    #Lower and upper bound on inputs and states
    uk_lb = [40, 290]
    uk_ub = [120, 320]
    xk_lb = [.485, 0]
    xk_ub =  [inf,  inf]

    #Define the system equations with unknown parameters
    de1 = (uk3[0]/10/V)*(Cain-xk3[0]) - thetah3[0]*exp(-EaR/(xk3[1]))*xk3[0]
    de2 = (uk3[0]/10/V)*(Tin-xk3[1]) - thetah3[1]/(rho*cp)*thetah3[0]*exp(-EaR/(xk3[1]))*xk3[0] + UA/(rho*cp*V)*(uk3[1] - xk3[1] )
    dthetah1 = [0]*thetah3[0]
    dthetah2 = [0]*thetah3[1]

    sys_ode = vertcat(de1,de2)
    mdl_ode =vertcat(de1,de2,dthetah1,dthetah2)

    #Jacobians for predictions
    Jz3 = Function('Jz3',[xk3,uk3,thetah3],[jacobian((vertcat(xk3,thetah3)+mdl_ode),vertcat(xk3,thetah3))])
    Ju3 = Function('Ju3',[xk3,uk3,thetah3],[jacobian((vertcat(xk3,thetah3)+mdl_ode),uk3)])

    ode = {'x':xk3, 'p':vertcat(thetah3,uk3), 'ode':sys_ode}
    F_ode = integrator('F_ode', 'cvodes', ode)

    m_ode = {'x':zk3, 'p':uk3, 'ode':mdl_ode }
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    #Output equation (Observations)
    C = np.eye(2)
    fy = mtimes(C,xk3) + vk3
    h = Function('h',[xk3,vk3], [fy])
    Ch = Function('Ch',[vertcat(xk3,thetah3)],[jacobian(fy,vertcat(xk3,thetah3))])
    Chx = Function('Chx',[xk3],[jacobian(fy,xk3)])

    uk_opt = np.array([100,300])
    xkp = np.array([.50,350])
    theta_par = theta_act
    xkh0 = xkp
    zkh0 =vertcat(xkh0,theta_par)
    Sigmak_p = np.diag([10**-2, 10**-1, 9*10**18, 10**-1]) # 10**-2, 9*10**18, 10**-1])

    res3_uk[0,:] = uk_opt
    res3_xk[0,:] = xkp
    res3_theta[0,:] = [theta_par[0],theta_par[1]]

    for k in range(run_time-1):
        time[k+1] = k+1

        if time[k] >= Disturbance and time[k] <= end_Disturbance :
            theta_act[0] = theta_act[0] -slope_Disturbance
            print('******************************************')
            print('******        k2 is reduced           ****')
            print('******************************************')


        #Compute the measurement
        ykp = h(xkp,vkp[k,:].T)
        #KF update step/measurment
        Czh = Ch(zkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
        zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
        xkh0 = zkh0[0:n_st]
        theta_par = zkh0[n_st:]
        Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)
        # generate CE control sequence for nominal trajectory
        Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(F_ode,n_pred,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk3,theta_par,uk3,Tsamp,xkh0,xk_sp,uk_sp,Wx,Wu,uk_opt)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        ce_res = res_mpc['x'].full().flatten()
        #uk_opt = ce_res[n_st+n_ip-1]
        uk_opt1 = ce_res[n_st:n_st+n_ip]

        xce1 = ce_res[0::4]
        xce2 = ce_res[1::4]
        xk_ce_3 = horzcat(xce1,xce2)
        uce1 = ce_res[2::4]
        uce2 = ce_res[3::4]
        uk_ce_3 = horzcat(uce1,uce2)
        #uk_opt = uk_ce_3[1,:]
        #pdb.set_trace()

        #Build and solve the dual optimization problem
        Jd, qu, op = Dual.gencost(n_st,n_ip,n_op,n_par,n_pred,Sigmak,mdl_ode,fy,xk3,uk3,thetah3,Tsamp,xk_sp,uk_sp,Wx,Wu,Q,Qz,R,uk_ce_3,xkh0,theta_par,xk_ce_3)
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
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz

        if k < run_time-2:
            # Save results
            res3_uk[k+1,:] = [uk_opt[0],uk_opt[1]]
            res3_xk[k+1,:] = [xkp[0,0],xkp[0,1]]
            res3_theta[k+1,:] = [theta_par[0],theta_par[1]]

        else:
            res3_xk[k+1,:] = [xkp[0,0],xkp[0,1]]
            res3_theta[k+1,:] = [theta_par[0],theta_par[1]]

    Mkplots(time, res3_xk, res3_uk, res3_theta, 'Dual Control', 1,1,1, [4,5,6])


if runC2:
    #########################################
    ##  Define system
    ######################################
    #Define the system states
    xk2 = SX.sym('xk2',n_st)
    uk2 = SX.sym('uk2',n_ip)
    wk2 = SX.sym('wk2',n_st)
    vk2 = SX.sym('vk2',n_op)
    thetah2 = SX.sym('thetah2',n_par)
    zk2 = vertcat(xk2,thetah2)

    #######################
    #model
    ###########
    #variables
    #Ca = x[0]                %concentration of species A
    #Tr = x[1]               %reactor temp

    #Inputs
    #qdot = u[0]              %volumetric flow rate
    #Tc = u[1]               %Coolant temp

    # estimate parameters
    #k0 = theta[0]               %kinetic constant
    #dH = theta[1]               %reaction heat

    #known parameters
    Cain = 1.              #concentration of species A inlet
    Tin = 350.              #inlet temp
    EaR = 8750.              #%Energy of activation
    UA = 5.*(10**4) / 10               #overall heat transfer coefficient* heat trans. area
    cp =  .239              #heat capacity
    rho =  1000.           #density
    V =  100.               #volume
    theta_act = [7.2*(10**10)/10, -5.*(10**4)] #  [ k0, dH ]

    #Parameter statistics (2.3)/(5.3-5.4)
    Q = np.diag([1./.5,1./350*0])
    Qz = np.diag([10**-2, 10**-1, 9*10**18,10**-1])# 10**-2, 9*10**18, 10**-1])n
    R = np.diag([1,1])

    uk_sp = np.array([100,300])
    xk_sp = np.array([.5, 350 ])
#    Wx = np.diag([(1./.5),(1./350)*0])*10
#    Wu = np.diag([(1./900),(1./300)])*.1

    #Lower and upper bound on inputs and states
    uk_lb = [40, 290]
    uk_ub = [120, 320]
    xk_lb = [.485, 0]
    xk_ub =  [inf,  inf]

    #Define the system equations with unknown parameters
    de1 = (uk2[0]/10/V)*(Cain-xk2[0]) - thetah2[0]*exp(-EaR/xk2[1])*xk2[0]
    de2 = (uk2[0]/10/V)*(Tin-xk2[1]) - (thetah2[1]/(rho*cp))*thetah2[0]*exp(-EaR/xk2[1])*xk2[0] + (UA/(rho*cp*V))*(uk2[1]-xk2[1])
    dthetah1 = [0]*thetah2[0]
    dthetah2 = [0]*thetah2[1]

    sys_ode = vertcat(de1,de2)
    mdl_ode =vertcat(de1,de2,dthetah1,dthetah2)

    #Jacobians for predictions
    Jx2 = Function('Jx',[xk2,uk2,thetah2],[jacobian((xk2+sys_ode),xk2)])
    Jz2 = Function('Jz',[xk2,uk2,thetah2],[jacobian((vertcat(xk2,thetah2)+mdl_ode),vertcat(xk2,thetah2))])
    Ju2 = Function('Ju',[xk2,uk2,thetah2],[jacobian((vertcat(xk2,thetah2)+mdl_ode),uk2)])

    ode2 = {'x':xk2, 'p':vertcat(thetah2,uk2), 'ode':sys_ode}
    F_ode = integrator('F_ode', 'cvodes', ode2)

    m_ode = {'x':zk2, 'p':uk2, 'ode':mdl_ode }
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    #Output equation (Observations)
    C = np.eye(2)
    fy = mtimes(C,xk2) + vk2
    h = Function('h',[xk2,vk2], [fy])
    Ch = Function('Ch',[vertcat(xk2,thetah2)],[jacobian(fy,vertcat(xk2,thetah2))])
    Chx = Function('Chx',[xk2],[jacobian(fy,xk2)])

    uk_opt = np.array([100.,300.])
    xkp = np.array([.50,350])
    theta_par = theta_act
    xkh0 = xkp
    zkh0 = vertcat(xkp,theta_par)
    Sigmak_p = np.diag([10**-2, 10**-1, 9*10**18,10**-1])# 10**-2, 9*10**18, 10**-1])

    #storing the results
    res2_uk[0,:] = uk_opt
    res2_xk[0,:] = xkp
    res2_theta[0,:] = theta_par

    for k in range(run_time-1):
        time[k+1] = k+1

        if time[k] >= Disturbance and time[k] <= end_Disturbance :
            theta_act[0] = theta_act[0] -slope_Disturbance
            print('******************************************')
            print('******        k0 is reduced           ****')
            print('******************************************')

        #Compute the measurement
        ykp = h(xkp,vkp[k,:].T)

        #KF update step/measurment
        Czh = Ch(zkh0)
        Kkh = mtimes(Sigmak_p,mtimes(Czh.T,np.linalg.inv(mtimes(Czh,mtimes(Sigmak_p,Czh.T)) + R)))
        zkh0 = zkh0 + mtimes(Kkh,(ykp - h(zkh0[0:n_st],np.zeros((n_st,1)))))
        xkh0 = zkh0[0:n_st]
        theta_par = zkh0[n_st:]
        Sigmak = mtimes((np.eye(n_st+n_par) - mtimes(Kkh,Czh)),Sigmak_p)

        Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(F_ode,n_pred,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk2,theta_par,uk2,Tsamp,xkh0,xk_sp,uk_sp,Wx,Wu,uk_opt)
        qp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}
        solver_mpc = nlpsol('solver_mpc', MySolver, qp_mpc,{'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
        res_mpc = solver_mpc(x0=qu_init, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        uk_ce = res_mpc['x'].full().flatten()
        uk_opt = uk_ce[n_st:n_st+n_ip]

        #Simulate the system
        x_end = F_ode(x0=xkp, p=vertcat(theta_act,uk_opt))
        xkp = x_end['xf'].full()
        xkp = xkp.T #+ wkp[k,:]

        #KF prediction
        Az = Jz2(xkh0,uk_opt,theta_par)
        z_end = M_ode(x0=zkh0, p=uk_opt)
        zkh0 = z_end['xf'].full()
        Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz

        if k < run_time-2:
            # Save results
            res2_uk[k+1,:] = [uk_opt[0],uk_opt[1]]
            res2_xk[k+1,:] = [xkp[0,0],xkp[0,1]]
            res2_theta[k+1,:] = [theta_par[0],theta_par[1]]

        else:
            res2_xk[k+1,:] = [xkp[0,0],xkp[0,1]]
            res2_theta[k+1,:] = [theta_par[0],theta_par[1]]

    Mkplots(time, res2_xk, res2_uk, res2_theta, 'CE Control', 1,1,1, [1,2,3])

print('xk diff = ', res3_xk - res2_xk)
print('uk diff = ', res3_uk - res2_uk)
print('theta diff = ', res3_theta - res2_theta)
#plt.show()
if save2csv:
    writeData(uk_sp,xk_sp, act_theta, res2_uk,res2_xk,res2_theta,res3_uk,res3_xk,res3_theta,run_time,csv_file)
