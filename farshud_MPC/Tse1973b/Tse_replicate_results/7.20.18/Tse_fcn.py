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
import core

#pdb.set_trace()







##########
## Case1 Function
def lb_mpc(n_pred,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,thetah,uk,Tsamp,zkh0,xk_sp,Wx,Wu,wkp):
    Jopt = 0
    J=0
    qu_opt = []
    lbq = []
    ubq = []
    g = []
    lbg = []
    ubg = []
    q0 = []

    xk = SX.sym('xk',n_st)
    Xk = SX.sym('X_' + str(0), n_st)
    qu_opt += [Xk]
    lbq += xk_lb
    ubq += xk_ub
    q0 += [zkh0[i].__float__() for i in range(n_st)]

    #Actual parameter values (5.5)
    A_act = np.matrix('0 1 0 ; 0 0 1 ; 1.8 -1.01, .58')
    A_act[0,:] = np.array([0, 1, 0]).T
    A_act[1,:] = np.array([0, 0 ,1]).T
    A_act[2,:] = np.array([1.8, -1.01, .58]).T
    B_act = np.array([ 0.3, 0.5, 1])

    for i1 in range(n_pred-1):
        # new NLP variable for the control
        Uk = SX.sym('U_' + str(i1), n_ip)
        qu_opt   += [Uk]
        lbq += [uk_lb[i] for i in range(n_ip)]
        ubq += [uk_ub[i] for i in range(n_ip)]
        q0  += [0]

        if i1 == 0:
            de1 =  .3*Uk[0] + wkp[i1,0]
            de2 =  .5*Uk[0] + wkp[i1,1]
            de3 =     Uk[0] + wkp[i1,2]
            Xk_end =vertcat(de1,de2,de3)
        elif i1>0:
            de1 = Xk[1] + .3*Uk[0] + wkp[i1,0]
            de2 = Xk[2] + .5*Uk[0] + wkp[i1,1]
            de3 = 1.8*Xk[0] + -1.01*Xk[1] + .58*Xk[2] +  Uk[0] + wkp[i1,2]
            Xk_end = vertcat(de1,de2,de3)

        # New NLriable for state at end of interval
        Xk = SX.sym('X_' + str(i1+1), n_st)
        qu_opt   += [Xk]
        lbq += [xk_lb[i] for i in range(n_st)]
        ubq += [xk_ub[i] for i in range(n_st)]
        q0 += [zkh0[i].__float__() for i in range(n_st)]

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0]*n_st
        ubg += [0]*n_st

        ####### Choose Simulation type
        #Cost function (5.7) - Intercept example
        #Jce = .5*((xk[2]-rho)**2 + sum( [lamb*uk_opt[i2,:]**2 for i2 in range(n_ctrl)]) )

        #Cost function (5.8) - Soft Landing-Type example
        if i1 <= n_pred-2:
            Jopt = Jopt + Uk.T*Wu*Uk
        if i1 == n_pred-2:
            Jopt = .5*(mtimes((Xk-Wx).T,(Xk-Wx)) + Jopt)
        #pdb.set_trace()

    return Jopt, qu_opt, lbq, ubq, g, lbg, ubg, q0




















##########
## Case2 Function
##################
def ce_mpc(mdl_ode,n_pred,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,thetah,uk,Tsamp,zkh0,xk_sp,Wx,Wu,uk_opt):
    #Wx = vertcat(Wx,0,0,0,0,0,0)
    n_par = n_par_ip+n_par_st
    Jce = 0.0
    qu_ce = []
    lbq = []
    ubq = []
    g = []
    lbg = []
    ubg = []
#    zk_lb = -inf,-inf,-inf,-inf,-inf,-inf,-inf,-inf,-inf
#    zk_ub =  inf, inf, inf, inf, inf, inf, inf, inf, inf
    zk_lb = [-50, -50, -50, -50, -50, -50, -50, -50, -50]
    zk_ub =  [50,  50,  50,  50,  50,  50,  50,  50,  50]

    zk = vertcat(xk,thetah)
    m_ode = {'x':zk, 'p':uk, 'ode':mdl_ode}
    M_ode = integrator('M_ode', 'cvodes', m_ode)

#    zk = SX.sym('zk_'+str(0),n_st+n_par)

    #xk = zk[0:n_st]
    #thetah = zk[n_st:n_par+n_st]
    #zk = vertcat(xk,thetah)

    #theta_st = thetah[0:n_par_st]
#    theta_ip = thetah[n_par_st:n_par]

    #qu_ce += [zk]
    xk = MX.sym('x_'+str(0),n_st)
    thetah = MX.sym('thetah_'+str(0),n_par)
    #xk = SX.sym('xk',n_st)
    #thetah = SX.sym('thetah',n_par)
    qu_ce += [vertcat(xk, thetah)]
    #F = Function('F', [vertcat(xk, thetah),uk],[mdl_ode])

    lbq += zk_lb
    ubq += zk_ub
    qu_init = []
    qu_init += [zkh0[i1].__float__() for i1 in range(n_st+n_par)]


    for i in range(n_pred):
        if i<=n_ctrl:
            uk = MX.sym('u_' + str(i),n_ip)
            qu_ce += [uk]
            lbq += uk_lb
            ubq += uk_ub
            qu_init += [0]

        z_end = M_ode(x0=vertcat(xk,thetah), p=uk)
        zk_end = z_end['xf']

        xk = MX.sym('x_'+str(i+1),n_st)
        thetah = MX.sym('thetah_'+str(i+1),n_par)

        #qu_ce += [zk]
        qu_ce += [vertcat(xk,thetah)]
        lbq += zk_lb
        ubq += zk_ub
        qu_init += [zkh0[i1].__float__() for i1 in range(n_st+n_par)]

        g += [zk_end-vertcat(xk, thetah)]
        lbg += [0]*(n_st+n_par)
        ubg += [0]*(n_st+n_par)

        #Cost function (5.8) - Soft Landing-Type example
        Jce = Jce + uk.T*Wu*uk
        if i == n_pred-1:
            #Jce = .5*(mtimes((zk[0:3]-Wx).T,(zk[0:3]-Wx)) + Jce)
            Jce = .5*(mtimes((xk-Wx).T,(xk-Wx)) + Jce )

        #pdb.set_trace()

    return Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init

























##########
## Case3 Function
##################
def gencost(n_st,n_ip,n_op,n_par,n_pred,Sigmak,mdl_ode,fy,xk,uk,thetah,Tsamp,xk_sp,Wx,Wu,Q,Qz,R,uk_ce):

    Wx = np.diag(Wx)

    n_tot = n_st+n_par

    qu = [uk]
    discrete_fcn =vertcat(xk,thetah)+ mdl_ode
    Fx = Function('Fx',[xk,uk,thetah],[discrete_fcn])
    #m_ode = {'x':vertcat(xk,thetah), 'p':uk, 'ode':mdl_ode}
    #F_ode = integrator('F_ode', 'cvodes', m_ode, {'tf':Tsamp})

    #Creating the augmented states
    zk = vertcat(xk,thetah)
    e_base = SX.eye(n_st)
    e_base_aug = SX.eye(n_st+n_par)

    Cz = jacobian(fy,vertcat(xk,thetah))
    Jz = Function('Jz',[xk,uk,thetah],[jacobian(discrete_fcn,vertcat(xk,thetah))])
    #Jx = Function('Jx',[xk,uk,thetah],[jacobian(discrete_fcn,xk)])
    Ju = Function('Ju',[xk,uk,thetah],[jacobian(discrete_fcn,uk)])

    #free parameters of the optimization problem
    op = vertcat(xk,thetah)
    Jd = 0.0

    #Generate the nominal trajectory
    res_xkh_nom = SX.zeros((n_pred,n_st))
    res_xkh_nomi = SX.zeros((n_pred,n_st))
    uk_pred_nom = SX.zeros((n_pred,n_ip))
    uk_pred = SX.zeros((n_pred,n_ip))

    for i in range(n_pred):
        if i==0:
            uk_nom = uk  #uk is the input that we will optimize to generate an input wiith dual intent
        else:
            uk_nom = uk_ce[i]

        #pdb.set_trace()
        uk_pred_nom[i,:] = uk_nom.T
        res_xkh_nom[i,:] = xk.T

        #Generate the CE trajectory of the states
        xkh_nom = Fx(xk,uk_nom,thetah)
        res_xkh_nomi[i,:] = xkh_nom[0:n_st].T
        xk = xkh_nom[0:n_st]

    #This loop generates the matrices related to Eqn. 12-23 in the write-up
    K0 = SX.zeros((n_pred*n_st,n_st))
    Wg = SX.zeros((n_pred*(n_st+n_par),(n_st+n_par)))
    Kg = SX.zeros((n_pred*(n_st+n_par),(n_st+n_par)))

    for i in reversed(range(n_pred)):
        if i==(n_pred-1):
             p0 = -mtimes(Wx,xk_sp)
             K0[i*n_st:(i+1)*n_st,:] = Wx
             Ktx = SX.zeros((n_par,n_st))
             Ktt = SX.zeros((n_par,n_par))
            # p0x = mtimes(K0,xk)+p0
            # p0x = mtimes(K0[i*n_st:(i+1)*n_st,:],xk) + p0
        else:
            Az = Jz(res_xkh_nom[i,:],uk_pred_nom[i,:],thetah)
        #    Ax = Jx(res_xkh_nom[i,:],uk_pred_nom[i,:],thetah)
            Ah = Az[0:n_st,0:n_st]
            At = Az[0:n_st,n_st:]
            Bh = Ju(res_xkh_nom[i,:],uk_pred_nom[i,:],thetah)
            Bh = Bh[0:n_st,0:n_ip]
            mu_0i = (Wu + mtimes(Bh.T,mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],Bh)))
            mu_0 = solve(mu_0i,SX.eye(mu_0i.size1()))
            e_base_sum = SX.zeros((n_par,n_ip))
            #for b1 in range(n_st):
            #    a = mtimes(e_base[:,b1].T,p0x)
            #    b = mtimes(a,Bh[b1,:])
            #    e_base_sum = e_base_sum[:,-1] + b.T
            #for b1 in range(n_st):
            #   e_base_sum = e_base_sum + mtimes(mtimes(e_base[:,b1].reshape((1,n_st)),p0x),Bh[b1,:]).T

            #pdb.set_trace()

            #sum1(mtimes(e_base,p0x)*Bh)
            Ktth = mtimes(Bh.T,(mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],At) + Ktx.T)) + e_base_sum.T
            Ktt = mtimes(mtimes(At.T,K0[(i+1)*n_st:(i+2)*n_st,:]),At) + mtimes(Ktx,At) + mtimes(At.T,Ktx.T) + Ktt - mtimes(mu_0,mtimes(Ktth.T,Ktth))
            Ktxh1 = mtimes((mtimes(At.T,K0[(i+1)*n_st:(i+2)*n_st,:]) + Ktx),Bh) + e_base_sum#sum1(mtimes(e_base,p0x)*Bh).T
            Ktxh2 = mtimes(Bh.T,mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],Ah))
            Ktxh = mtimes(Ktxh1,Ktxh2)
            Ktx = mtimes((mtimes(At.T,K0[(i+1)*n_st:(i+2)*n_st,:]) + Ktx),Ah) - mtimes(mu_0,Ktxh)

            uk_pred[i+1,:] = -mtimes(mu_0,mtimes(Bh.T,(mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],mtimes(Ah,res_xkh_nomi[i,:].T))+p0)))
            p0 = mtimes(Ah.T,mtimes((SX.eye(n_st)- mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],mtimes(mtimes(Bh,mu_0),Bh.T))),p0)) - mtimes(Wx,xk_sp)

            #mtimes(Ah.T,mtimes((SX.eye(n_st)- mu_0*mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],mtimes(Bh.T,Bh))),mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],Ah)))

            K0[i*n_st:(i+1)*n_st,:] = mtimes(Ah.T,mtimes((SX.eye(n_st) - mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],mtimes(mtimes(Bh,mu_0),Bh.T))),mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],Ah))) + Wx

            Wgi = horzcat(vertcat(Wx,SX.zeros((n_par,n_st))),SX.zeros((n_st+n_par,n_par)))
            Wg[i*(n_st+n_par):(i+1)*(n_st+n_par),:] = Wgi

        p0x = mtimes(K0[i*n_st:(i+1)*n_st,:],res_xkh_nom[i,:].T) + p0
        Kg[i*(n_st+n_par):(i+1)*(n_st+n_par),:] = horzcat(vertcat(K0[i*n_st:(i+1)*n_st,:],Ktx),vertcat(Ktx.T,Ktt))

    #Generate term 1 & 3 of the trace in eq.24.e of writeup
    #the loop is a little excessive, as it could be compacted into the previous loop, but it works
#    term3 = SX.sym('term3',n_tot*(n_pred-1),n_st)
#    term1 =  SX.sym('term1',n_tot*(n_pred-1),n_tot)


    #This loop generates the predictions denoted by Eqs. (7)-(12) in the write-up
    for j in range(n_pred):
        if j==0:
            #This generates the predictions using the dual input uk
            Az = Jz(res_xkh_nom[j,:].T,uk,thetah)
            #Ax = Jx(res_xkh_nom[j,:].T,uk,thetah)
            Bh = Ju(res_xkh_nom[j,:].T,uk,thetah)
            Ah = Az[0:n_st,0:n_st]
            Bh = Bh[0:n_st,0:n_ip]
            e_base_vec = NP.zeros((n_st,1))
            #Hessian computation of the augmented space
            for b1 in range(n_st):
                Hx =  jacobian(jacobian(discrete_fcn[b1],zk[b1]),zk[b1])
                e_base_vec = e_base_vec + .5*mtimes(e_base[:,b1],trace((mtimes(Hx,Sigmak[0:n_st,0:n_st]))))

            x_end = Fx(res_xkh_nom[j,:].T,uk,thetah) #mtimes(Ah,res_xkh_nom[j,:].T) + mtimes(Bh,uk)
            x_end = x_end[0:n_st] + e_base_vec
            xk = x_end[0:n_st]

            e_base_mat = SX.zeros(((n_st+n_par),(n_st+n_par)))
            for b1 in range(n_st+n_par):
                Hz1 = jacobian(jacobian(discrete_fcn[b1],zk),zk)
                for b2 in range(n_st+n_par):
                    Hz2 = jacobian(jacobian(discrete_fcn[b2],zk),zk)
                    e_base_mat = e_base_mat + mtimes(mtimes(e_base_aug[:,b1],e_base_aug[:,b2].T),trace(mtimes(mtimes(Hz1,Sigmak),mtimes(Hz2,Sigmak))))

            #EKF-type propagation of the covariance of the augmented state-space
            Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz + 0.5*e_base_mat
            Kee = mtimes(Cz,mtimes(Sigmak_p,Cz.T)) + R
            Keei = solve(Kee,SX.eye(Kee.size1()))
            Kk = mtimes(mtimes(Sigmak_p,Cz.T),Keei)
            Sigmak = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmak_p)

            #Sigmaj_p = mtimes(Az.T, mtimes(Sigmak, Az))+Qz
            #Sigmaj = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmaj_p)


            #t3 = sum([term3[(z1)*n_tot:(z1+1)*n_tot,:] for z1 in reversed(range(n_pred-1-j))])
            #t1 = sum([term1[(z1)*n_tot:(z1+1)*n_tot,:] for z1 in reversed(range(n_pred-1-j))])

            #generate the cost function based on Eq. 24(e) of the write-up
            # with term 3
            #Jd = 0.5*mtimes(uk.T,mtimes(Wu,uk)) + 0.5*mtimes(xk.T,mtimes(K0[j*n_st:(j+1)*n_st,:],xk)) + mtimes(p0.T,xk) + 0.5*trace((t1)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:])*t3)
            # Original
            Jd = 0.5*mtimes(uk.T,mtimes(Wu,uk)) + 0.5*mtimes(xk.T,mtimes(K0[j*n_st:(j+1)*n_st,:],xk)) + mtimes(p0.T,xk) + 0.5*trace(mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmak)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:]))
            # Simple Cost
        #    Jd = Jd + 0.5*mtimes(uk.T,mtimes(Wu,uk)) + 0.5*mtimes(xk.T,mtimes(K0[j*n_st:(j+1)*n_st,:],xk))
        else:
            #This generates the predictions for the remainder of the horizon using the C.E. input trajectory
            Az = Jz(xk,uk_pred[j,:].T,thetah)
            Bh = Ju(xk,uk_pred[j,:].T,thetah)
            Ah = Az[0:n_st,0:n_st]
            Bh = Bh[0:n_st,0:n_ip]
            x_end = Fx(xk,uk_pred[j,:].T,thetah)#mtimes(Ah,xk) + mtimes(Bh,uk_pred[j,:].T)
            xk = x_end[0:n_st]

            #EKF-type propagation of the covariance of the augmneted state-space
            Sigmak_p = mtimes(Az.T,mtimes(Sigmak,Az)) + Qz
            Kee = mtimes(Cz,mtimes(Sigmak_p,Cz.T)) + R
            Keei = solve(Kee,SX.eye(Kee.size1()))
            Kk = mtimes(mtimes(Sigmak_p,Cz.T),Keei)

            #Jd = Jd + 0.5*trace(mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmak))
            Sigmak = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmak_p)

            #t3 = sum([term3[(z1)*n_tot:(z1+1)*n_tot,:] for z1 in reversed(range(n_pred-j))])
            #t1 = sum([term3[(z1)*n_tot:(z1+1)*n_tot,:] for z1 in reversed(range(n_pred-j))])

            #generate the cost function based on Eq. 24(e) of the write-up
            #  full
            #Jd = Jd + 0.5*trace((t1)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:])*t3)
            #  from Vinays
            Jd = Jd + 0.5*trace(mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmak)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:]))

    return Jd, qu, op


##########
## Case3 Function
##################
#def gencost(n_st,n_ip,n_op,n_par,n_pred,Sigmak,mdl_ode,fy,xk,uk,thetah,Tsamp,xk_sp,Wx,Wu,Q,Qz,R,uk_ce):
