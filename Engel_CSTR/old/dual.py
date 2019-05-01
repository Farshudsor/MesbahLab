# -*- coding: utf-8 -*-
"""
Created on July 4th 2018

@author: Farshud Sorourifar
"""
import sys
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")
from casadi import *
import numpy as NP
import pdb
from scipy import linalg
#import core


##########
## Case2 Function
##################
def ce_mpc(F_ode,n_pred,n_ctrl,n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,theta_par,uk,Tsamp,xkh0,uk_opt):

    Jce = 0.0
    qu_ce = []
    lbq = []
    ubq = []
    g = []
    lbg = []
    ubg = []

    X0 = MX.sym('X0', n_st)
    qu_ce += [X0]
    lbq += [xkh0[i1].__float__() for i1 in range(n_st)]
    ubq += [xkh0[i1].__float__() for i1 in range(n_st)]
    qu_init = []
    qu_init += [xkh0[i1].__float__() for i1 in range(n_st)]
    Xk = X0

    for i in range(n_pred):
        if i<=n_ctrl:
            if i != 0:
                Uk_ = Uk
            else:
                Uk_ = uk_opt
            Uk = MX.sym('U_' + str(i),n_ip)
            qu_ce += [Uk]
            lbq += [uk_lb[i1] for i1 in range(n_ip)]
            ubq += [uk_ub[i1] for i1 in range(n_ip)]
            qu_init += [uk_opt[i1] for i1 in range(n_ip)]

        x_end = F_ode(x0=Xk, p=vertcat(theta_par,Uk))
        xk_end = x_end['xf']
        Jce = Jce - Xk[0]*Xk[3] + .1*(Uk[0]-Uk_[0])**2 + 10**(-6)*(Uk[1]-Uk_[1])**2

        Xk = MX.sym('X_'+str(i+1),n_st)
        qu_ce += [Xk]
        lbq += [xk_lb[i1] for i in range(n_st)]
        ubq += [xk_ub[i1] for i in range(n_st)]
        qu_init += [xkh0[i1].__float__() for i1 in range(n_st)]

        g += [xk_end-Xk]
        lbg += [0]*(n_st)
        ubg += [0]*(n_st)

    return Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init



##########
## Case3 Function
##################
def gencost(n_st,n_ip,n_op,n_par,n_pred,Sigmak,mdl_ode,fy,xk,uk,thetah,Tsamp,Q,Qz,R,uk_ce,xkh0,theta_par,xk_ce):

    n_tot = n_st+n_par
    discritize = (60)/2 # [=] seconds/(2min) - discritze in two min


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

    #Define the system equations with unknown parameters
    de1 = uk[0]
    de2 = -(uk[0]/xk[0])*xk[1] - theta_par[0]*xk[1]*xk[2]
    de3 = (uk[0]/xk[0])*(Cbin-xk[2]) - theta_par[0]*xk[1]*xk[2]
    de4 = -(uk[0]/xk[0])*xk[3] + theta_par[0]*xk[1]*xk[2]
    de5 = (uk[0]/xk[0])*(Tin - xk[4]) - (U*((2*xk[0]/r)-pi*r**2)*(xk[4]-xk[5]))/(rho*cp*xk[0]) - (theta_par[0]*xk[1]*xk[2]*theta_par[1])/(rho*cp)
    de6 = (Vjin/Vj) * (Tin-xk[4]) + (U*((2*xk[0]/r)-pi*r**2)*(xk[4]-xk[5]))/(rho*cp*Vj)
    de7 = (1/tau_c)* (uk[1]-xk[6])
    dthetah1 = [0]*theta_par[0]
    dthetah2 = [0]*theta_par[1]

    sys_ode = vertcat(de1,de2,de3,de4,de5,de6,de7)
    mdl_ode =vertcat(de1,de2,de3,de4,de5,de6,de7,dthetah1,dthetah2)

    qu = [uk]
    discrete_fcn = vertcat(xk,theta_par)+ mdl_ode
    Fx = Function('Fx',[xk,uk],[discrete_fcn])
    m_ode = {'x':vertcat(xk), 'p':uk, 'ode':sys_ode}
    M_ode = integrator('M_ode', 'cvodes', m_ode)

    #Creating the augmented states
    zk = vertcat(xk,thetah)
    zkh0= vertcat(xkh0,theta_par)
    e_base = SX.eye(n_st)
    e_base_aug = SX.eye(n_st+n_par)

    ##############################################################################################################################################
    #Output equation (Observations)
    C = np.eye(n_st)
    fy = mtimes(C,xk)
    h = Function('h',[xk], [fy])
    Ch = Function('Ch',[vertcat(xk)],[jacobian(fy,vertcat(xk))])
    Chx = Function('Chx',[xk],[jacobian(fy,xk)])
    ###################################################################################################################################

    Jz = Function('Jz',[xk,uk,thetah],[jacobian(discrete_fcn,vertcat(xk,thetah))])
    Jx = Function('Jx',[xk,uk,thetah],[jacobian(discrete_fcn,xk)])
    Ju = Function('Ju',[xk,uk,thetah],[jacobian(discrete_fcn,uk)])

    #free parameters of the optimization problem
    op = vertcat(xk)
    Jd = 0.0

    #Generate the nominal trajectory
    res_xkh_nom = SX.zeros((n_pred,n_st))
    res_xkh_nomi = SX.zeros((n_pred,n_st))
    uk_pred_nom = SX.zeros((n_pred,n_ip))
    uk_pred = SX.zeros((n_pred,n_ip))

    for i in range(n_pred):
        if i==0:
            uk_nom = uk  #uk is the input that we will optimize to generate an input wiith dual intent
            xkh_nom = Fx(xkh0,uk)
            xkh0 = xkh_nom[0:n_st]
        else:
            uk_nom = uk_ce[i]

        uk_pred_nom[i,:] = uk_nom.T
        res_xkh_nom[i,:] = xkh0.T

        #Generate the CE trajectory of the states
        xkh_nom = Fx(xkh0,uk_nom)
        res_xkh_nomi[i,:] = xkh_nom[0:n_st].T
        xkh0 = xkh_nom[0:n_st]

    K0 = SX.zeros((n_pred*n_st,n_st))
    Wg = SX.zeros((n_pred*(n_st+n_par),(n_st+n_par)))
    Kg = SX.zeros((n_pred*(n_st+n_par),(n_st+n_par)))

    #This loop generates the matrices related to Eqn. 12-23 in the write-up
    for i in reversed(range(n_pred)):
        if i==(n_pred-1):
            Wxn = np.zeros([n_st,n_st])
            Wx = np.zeros([n_st,n_st])
            Wx[0,0] = 1
            Wx[0,3] = 1
            xk_sp = np.array( [1,0,0,1,0,0,0])
            Wu = np.diag([10**(-1), 10**(-6)])
#            import pdb; pdb.set_trace()
            p0 = -mtimes(Wxn,xk_sp) #3.7
            K0[i*n_st:(i+1)*n_st,:] = Wxn #3.6
            Ktx = SX.zeros((n_par,n_st)) #3.15
            Ktt = SX.zeros((n_par,n_par)) #3.16
            D = SX.eye(n_par)

            Wgi = horzcat(vertcat(Wxn,SX.zeros((n_par,n_st))),SX.zeros((n_st+n_par,n_par)))
            Wg[i*(n_st+n_par):(i+1)*(n_st+n_par),:] = Wgi

        else:
            Az = Jz(res_xkh_nom[i,:],uk_pred_nom[i,:],theta_par)
            Ax = Jx(res_xkh_nom[i,:],uk_pred_nom[i,:],theta_par)
            Ah = Az[0:n_st,0:n_st]
            At = Az[0:n_st,n_st:]
            Bh = Ju(res_xkh_nom[i,:],uk_pred_nom[i,:],theta_par)
            Bh = Bh[0:n_st,0:n_ip]
            #(eq3.5)
            mu_0i = (Wu + mtimes(Bh.T,mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],Bh)))
            mu_0 = solve(mu_0i,SX.eye(mu_0i.size1()))
            e_base_sum = SX.zeros((n_par,n_ip))


            Ktth = mtimes(Bh.T,(mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],At) + mtimes(Ktx.T,D))) + e_base_sum.T #whats in {} from eq 3.16
            #import pdb; pdb.set_trace()
            Ktt = mtimes(mtimes(At.T,K0[(i+1)*n_st:(i+2)*n_st,:]),At) +mtimes(D.T,mtimes(Ktx,At)) + mtimes(mtimes(At.T,Ktx.T),D) + mtimes(mtimes(D.T,Ktt),D) - mtimes(mu_0,mtimes(Ktth.T,Ktth)) #3.16
            Ktxh1 = mtimes((mtimes(At.T,K0[(i+1)*n_st:(i+2)*n_st,:]) + mtimes(D.T,Ktx)),Bh) + e_base_sum #whats in the first {} from eq 3.15
            Ktxh2 = mtimes(Bh.T,mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],Ah)) #whats in the second {} from eq 3.15
            Ktxh = mtimes(Ktxh1,Ktxh2)
            Ktx = mtimes((mtimes(At.T,K0[(i+1)*n_st:(i+2)*n_st,:]) + mtimes(D,Ktx)),Ah) - mtimes(mu_0,Ktxh) #3.15
            #(eq 3.4)
            uk_pred[i+1,:] = -mtimes(mu_0,mtimes(Bh.T,(mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],mtimes(Ah,res_xkh_nomi[i,:].T))+p0)))
            #(eq 3.7)
            import pdb; pdb.set_trace()
            p0 = mtimes(Ah.T,mtimes((SX.eye(n_st)- mtimes(mu_0,mtimes( K0[(i+1)*n_st:(i+2)*n_st,:], mtimes(Bh,Bh.T)))),p0)) - mtimes(Wx,xk_sp)

            #(eq 3.6)
            #K0[i*n_st:(i+1)*n_st,:] = mtimes(Ah.T,mtimes((SX.eye(n_st) - mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],mtimes(mtimes(Bh,mu_0),Bh.T))),mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],Ah))) + Wx
            K0[i*n_st:(i+1)*n_st,:] = mtimes(mtimes(Ah.T,(SX.eye(n_st) - mtimes(mu_0,mtimes( K0[(i+1)*n_st:(i+2)*n_st,:], mtimes(Bh,Bh.T))))), mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],Ah)) + Wx

            #eq. 3.17
            Wgi = horzcat(vertcat(Wx,SX.zeros((n_par,n_st))),SX.zeros((n_st+n_par,n_par)))
            Wg[i*(n_st+n_par):(i+1)*(n_st+n_par),:] = Wgi

        #(eq 3.18)
        p0x = mtimes(K0[i*n_st:(i+1)*n_st,:],res_xkh_nom[i,:].T) + p0
        #(eq 3.13)
        Kg[i*(n_st+n_par):(i+1)*(n_st+n_par),:] = horzcat(vertcat(K0[i*n_st:(i+1)*n_st,:],Ktx),vertcat(Ktx.T,Ktt))


    #This loop generates the predictions denoted by Eqs. (7)-(12) in the write-up
    for j in range(n_pred):
        if j==0:
            #This generates the predictions using the dual input uk
            Az = Jz(res_xkh_nom[j,:].T,uk_pred_nom[j,:],theta_par)
            Bh = Ju(res_xkh_nom[j,:].T,uk_pred_nom[j,:],theta_par)
            Ah = Az[0:n_st,0:n_st]
            Bh = Bh[0:n_st,0:n_ip]
            e_base_vec = NP.zeros((n_st,1))
            zkh0 = (res_xkh_nom[j,:], theta_par)

            #Hessian computation of the augmented space
            for b1 in range(n_st):
                Hx =  jacobian(jacobian(discrete_fcn[b1],zk[b1]),zk[b1])
                e_base_vec = e_base_vec + .5*mtimes(e_base[:,b1],trace((mtimes(Hx,Sigmak[0:n_st,0:n_st]))))

            x_end = Fx(res_xkh_nom[j,:],uk_pred_nom[j,:])
            x_end = x_end[0:n_st] + e_base_vec
            xk = x_end[0:n_st]

            e_base_mat = SX.zeros(((n_st+n_par),(n_st+n_par)))
            for b1 in range(n_st+n_par):
                Hz1 = jacobian(jacobian(discrete_fcn[b1],zk),zk)
                for b2 in range(n_st+n_par):
                    Hz2 = jacobian(jacobian(discrete_fcn[b2],zk),zk)
                    e_base_mat = e_base_mat + mtimes(mtimes(e_base_aug[:,b1],e_base_aug[:,b2].T),trace(mtimes(mtimes(Hz1,Sigmak),mtimes(Hz2,Sigmak))))

            Cx = Ch(x_end[0:n_st])
            #pdb.set_trace()
            Cz = horzcat(Cx,np.zeros((n_st,n_par)))

            #EKF-type propagation of the covariance of the augmented state-space
            Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz
            Kee = mtimes(Cz,mtimes(Sigmak_p,Cz.T)) + R
            Keei = solve(Kee,SX.eye(Kee.size1()))

            Kk = mtimes(mtimes(Sigmak_p,Cz.T),Keei)
            Sigmak = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmak_p)
            Sigmakxx = vertcat(horzcat(Sigmak[0:n_st,0:n_st], SX.zeros((n_st,n_par))),SX.zeros((n_par,n_st+n_par)))

            #import pdb; pdb.set_trace()
            #generate the cost function based on Eq. 24(e) of the write-up
            Jd = 0.5*mtimes((uk).T,mtimes(Wu,(uk))) + 0.5*mtimes((xk).T,mtimes(K0[j*n_st:(j+1)*n_st,:],(xk))) + mtimes(p0.T,xk) + 0.5*trace(mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmak)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:]))
            #From Klenske
            #Jd = 0.5*mtimes((xk).T,mtimes(K0[j*n_st:(j+1)*n_st,:],(xk))) + 0.5*trace(mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmakxx)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:]))

        else:
            #This generates the predictions for the remainder of the horizon using the C.E. input trajectory
            Az = Jz(res_xkh_nom[j,:].T,uk_pred_nom[j,:].T,theta_par)
            Bh = Ju(res_xkh_nom[j,:].T,uk_pred_nom[j,:].T,theta_par)
            Ah = Az[0:n_st,0:n_st]
            Bh = Bh[0:n_st,0:n_ip]
            e_base_vec = NP.zeros((n_st,1))

            #Hessian computation of the augmented space
            for b1 in range(n_st):
                Hx =  jacobian(jacobian(discrete_fcn[b1],zk[b1]),zk[b1])
                e_base_vec = e_base_vec + .5*mtimes(e_base[:,b1],trace((mtimes(Hx,Sigmak[0:n_st,0:n_st]))))

            x_end = Fx(res_xkh_nom[j,:],uk_pred_nom[j,:])
            x_end = x_end[0:n_st] + e_base_vec
            xk = x_end[0:n_st]

            Cx = Ch(x_end[0:n_st])
            #pdb.set_trace()
            Cz = horzcat(Cx,np.zeros((n_st,n_par)))

            #EKF-type propagation of the covariance of the augmneted state-space
            Sigmak_p = mtimes(Az.T,mtimes(Sigmak,Az)) + Qz
            Kee = mtimes(Cz,mtimes(Sigmak_p,Cz.T)) + R
            Keei = solve(Kee,SX.eye(Kee.size1()))
            Kk = mtimes(mtimes(Sigmak_p,Cz.T),Keei)
            Sigmak = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmak_p)
            Sigmakxx = vertcat(horzcat(Sigmak[0:n_st,0:n_st], SX.zeros((n_st,n_par))),SX.zeros((n_par,n_st+n_par)))

            #generate the cost function based on Eq. 24(e) of the write-up
            Jd = Jd + 0.5*trace(mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmakxx)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:]))

    return Jd, qu, op
