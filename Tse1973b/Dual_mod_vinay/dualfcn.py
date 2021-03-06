# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:47:06 2017

@author: vinay
"""
import sys
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")
from casadi import *
import numpy as NP
import pdb
#from scipy import linalg
#import core

def gencost(n_st,n_ip,n_op,n_par,n_pred,Sigmak,mdl_ode,fy,xk,uk,thetah,Tsamp,xk_sp,Wx,Wu,Q,Qz,R,uk_ce):


    qu = [uk]
    discrete_fcn = vertcat(xk,thetah)+Tsamp*mdl_ode
    Fx = Function('Fx',[xk,uk,thetah],[discrete_fcn])
    #m_ode = {'x':vertcat(xk,thetah), 'p':uk, 'ode':mdl_ode}
    #F_ode = integrator('F_ode', 'cvodes', m_ode, {'tf':Tsamp})

    #Creating the augmented states
    zk = vertcat(xk,thetah)
    e_base = SX.eye(n_st)
    e_base_aug = SX.eye(n_st+n_par)

    Cz = jacobian(fy,vertcat(xk,thetah))
    Jz = Function('Jz',[xk,uk,thetah],[jacobian(discrete_fcn,vertcat(xk,thetah))])
    Jx = Function('Jx',[xk,uk,thetah],[jacobian(discrete_fcn,xk)])
    Ju = Function('Ju',[xk,uk,thetah],[jacobian(discrete_fcn,uk)])

    #free parameters of the optimization problem
    op = vertcat(xk,thetah,uk)
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
            uk_nom = uk_ce


        uk_pred_nom[i,:] = uk_nom.T
        res_xkh_nom[i,:] = xk.T

        #Azh = Jz(xk,uk,thetah)
        #Axh = Azh[0:n_st,0:n_st]
        #Bu = Ju(xk,uk,thetah)
        #Bu = Bu[0:n_st,0:n_ip]
        #xkh_nom = mtimes(Axh,xk) + mtimes(Bu,uk)

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
             p0x = mtimes(K0[i*n_st:(i+1)*n_st,:],xk) + p0
        else:
            Az = Jz(res_xkh_nom[i,:],uk_pred_nom[i,:],thetah)
            Ax = Jx(res_xkh_nom[i,:],uk_pred_nom[i,:],thetah)
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
            for b1 in range(n_st):
               e_base_sum = e_base_sum + mtimes(mtimes(e_base[:,b1].reshape((1,4)),p0x),Bh[b1,:]).T

            #sum1(mtimes(e_base,p0x)*Bh)
            Ktth = mtimes(Bh.T,(mtimes(K0[(i+1)*n_st:(i+2)*n_st,:],At) + Ktx.T)) + e_base_sum
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
    term3 = SX.sym('term3',6*(n_pred-1),2)
    term1 =  SX.sym('term1',6*(n_pred-1),6)
    for j in range(n_pred):
        if j==0:
            #This generates the predictions using the dual input uk
            Az = Jz(res_xkh_nom[j,:].T,uk,thetah)
            Ax = Jx(res_xkh_nom[j,:].T,uk,thetah)
            Bh = Ju(res_xkh_nom[j,:].T,uk,thetah)
            Ah = Az[0:n_st,0:n_st]
            Bh = Bh[0:n_st,0:n_ip]
            e_base_vec = NP.zeros((n_st,1))
            #Hessian computation of the augmented space
            for b1 in range(n_st):
                Hx = jacobian(jacobian(discrete_fcn[b1],zk[b1]),zk[b1])
                e_base_vec = e_base_vec + .5*mtimes(e_base[:,b1],trace((mtimes(Hx,Sigmak[0:n_st,0:n_st]))))

            x_end = Fx(res_xkh_nom[j,:].T,uk,thetah) #mtimes(Ah,res_xkh_nom[j,:].T) + mtimes(Bh,uk)
            x_end = x_end[0:n_st] + e_base_vec
            xk = x_end[0:n_st]

            e_base_mat = SX.zeros(((n_st+n_par),(n_st+n_par)))
            for b1 in range(n_st+n_par):
                Hz1 = jacobian(jacobian(discrete_fcn[b1],zk),zk)
                for b2 in range(n_st+n_par):
                    Hz2 = jacobian(jacobian(discrete_fcn[b2],zk),zk)
                    e_base_mat = e_base_mat +.5*mtimes(mtimes(e_base_aug[:,b1],e_base_aug[:,b2].T),trace(mtimes(mtimes(Hz1,Sigmak),mtimes(Hz2,Sigmak))))

            #EKF-type propagation of the covariance of the augmented state-space
            Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz + 0.5*e_base_mat
            Kee = mtimes(Cz,mtimes(Sigmak_p,Cz.T)) + R
            Keei = solve(Kee,SX.eye(Kee.size1()))
            Kk = mtimes(mtimes(Sigmak_p,Cz.T),Keei)
            Sigmak = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmak_p)

            term3[j*6:(j+1)*6,:] = mtimes((Sigmak_p - Sigmak),Kk)
            term1[j*6:(j+1)*6,:] = mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmak)

        elif j < (n_pred-1):
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
            Sigmak = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmak_p)

            term3[j*6:(j+1)*6,:] = mtimes((Sigmak_p - Sigmak),Kk)
            term1[j*6:(j+1)*6,:] = mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmak)


    #This loop generates the predictions denoted by Eqs. (7)-(12) in the write-up
    for j in range(n_pred):
        if j==0:
            #This generates the predictions using the dual input uk
            Az = Jz(res_xkh_nom[j,:].T,uk,thetah)
            Ax = Jx(res_xkh_nom[j,:].T,uk,thetah)
            Bh = Ju(res_xkh_nom[j,:].T,uk,thetah)
            Ah = Az[0:n_st,0:n_st]
            Bh = Bh[0:n_st,0:n_ip]
            e_base_vec = NP.zeros((n_st,1))
            #Hessian computation of the augmented space
            for b1 in range(n_st):
                Hx = jacobian(jacobian(discrete_fcn[b1],zk[b1]),zk[b1])
                e_base_vec = e_base_vec + .5*mtimes(e_base[:,b1],trace((mtimes(Hx,Sigmak[0:n_st,0:n_st]))))

            x_end = Fx(res_xkh_nom[j,:].T,uk,thetah) #mtimes(Ah,res_xkh_nom[j,:].T) + mtimes(Bh,uk)
            x_end = x_end[0:n_st] + e_base_vec
            xk = x_end[0:n_st]

            e_base_mat = SX.zeros(((n_st+n_par),(n_st+n_par)))
            for b1 in range(n_st+n_par):
                Hz1 = jacobian(jacobian(discrete_fcn[b1],zk),zk)
                for b2 in range(n_st+n_par):
                    Hz2 = jacobian(jacobian(discrete_fcn[b2],zk),zk)
                    e_base_mat = e_base_mat +.5*mtimes(mtimes(e_base_aug[:,b1],e_base_aug[:,b2].T),trace(mtimes(mtimes(Hz1,Sigmak),mtimes(Hz2,Sigmak))))

            #EKF-type propagation of the covariance of the augmented state-space
            Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz + 0.5*e_base_mat
            Kee = mtimes(Cz,mtimes(Sigmak_p,Cz.T)) + R
            Keei = solve(Kee,SX.eye(Kee.size1()))
            Kk = mtimes(mtimes(Sigmak_p,Cz.T),Keei)
            Sigmak = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmak_p)

            Sigmaj_p = mtimes(Az.T, mtimes(Sigmak, Az))+Qz
            Sigmaj = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmaj_p)


            t3 = sum([term3[(z1)*6:(z1+1)*6,:] for z1 in reversed(range(n_pred-1-j))])
            t1 = sum([term1[(z1)*6:(z1+1)*6,:] for z1 in reversed(range(n_pred-1-j))])

            #generate the cost function based on Eq. 24(e) of the write-up
            # with term 3
            Jd = Jd + 0.5*mtimes(uk.T,mtimes(Wu,uk)) + 0.5*mtimes(xk.T,mtimes(K0[j*n_st:(j+1)*n_st,:],xk)) + mtimes(p0.T,xk) + 0.5*trace((t1)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:])*t3)
            # Original
            #Jd = Jd + 0.5*mtimes(uk.T,mtimes(Wu,uk)) + 0.5*mtimes(xk.T,mtimes(K0[j*n_st:(j+1)*n_st,:],xk)) + mtimes(p0.T,xk) + 0.5*trace(mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmak)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:]))
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

            t3 = sum([term3[(z1)*6:(z1+1)*6,:] for z1 in reversed(range(n_pred-j))])
            t1 = sum([term3[(z1)*6:(z1+1)*6,:] for z1 in reversed(range(n_pred-j))])

            #generate the cost function based on Eq. 24(e) of the write-up
            #  full
            Jd = Jd + 0.5*trace((t1)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:])*t3)
            #  from Vinays
            #Jd = Jd + 0.5*mtimes(uk.T,mtimes(Wu,uk)) + 0.5*mtimes(xk.T,mtimes(K0[j*n_st:(j+1)*n_st,:],xk)) + mtimes(p0.T,xk) + 0.5*trace(mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmak)+mtimes((Sigmak_p - Sigmak),Kg[j*(n_st+n_par):(j+1)*(n_st+n_par),:]))

    return Jd, qu, op

def ce_nmpc(mdl_ode,n_pred,n_ctrl,n_st,n_par,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,thetah,uk,Tsamp,zkh0,xk_sp,Wx,uk_opt):
    Jce = 0.0
    zk = vertcat(xk,thetah)
    m_ode = {'x':zk, 'p':uk, 'ode':mdl_ode}
    M_ode = integrator('M_ode', 'cvodes', m_ode, {'tf':Tsamp})
    qu_ce = []
    lbq = []
    ubq = []
    g = []
    lbg = []
    ubg = []

    xk = MX.sym('xk',n_st)
    thetah = MX.sym('thetah',n_par)
    qu_ce += [vertcat(xk,thetah)]
    #lbq += [vertcat(xk,thetah)]
    #ubq += [vertcat(xk,thetah)]
    #lbq += xk_lb
    #ubq += xk_ub
    lbq += 0.0, 0.0, 100.0, 100.0,-inf,-inf
    ubq += 5.0, 5.0, 140.0, 140.0,inf,inf

    qu_init = []
    qu_init += [zkh0[i].__float__() for i in range(6)]


    for i in range(n_pred):
        if i<=n_ctrl:
            uk = MX.sym('u_' + str(i),n_ip)
            qu_ce += [uk]
            lbq += uk_lb
            ubq += uk_ub
            qu_init += [uk_opt[i2] for i2 in range(2)]

        z_end = M_ode(x0=vertcat(xk,thetah), p=uk)
        zk_end = z_end['xf']

        Jce += mtimes((xk_sp - zk_end[0:n_st]).T,mtimes(Wx,(xk_sp - zk_end[0:n_st])))
        xk = MX.sym('x_'+str(i+1),n_st)
        thetah = MX.sym('thetah_'+str(i+1),n_par)
        qu_ce += [vertcat(xk,thetah)]
        lbq += 0.0, 0.0, 100.0, 100.0,-inf,-inf
        ubq += 5.0, 5.0, 140.0, 140.0,inf,inf
        qu_init += [zkh0[i].__float__() for i in range(6)]
        g += [zk_end-vertcat(xk,thetah)]
        lbg += [0]*(n_st+n_par)
        ubg += [0]*(n_st+n_par)

        #pdb.set_trace()


    return Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init
