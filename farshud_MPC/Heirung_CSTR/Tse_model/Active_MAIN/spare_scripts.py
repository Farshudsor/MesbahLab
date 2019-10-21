


    for j in range(n_pred):
        if j==0:
            #This generates the predictions using the dual input uk
            Az = Jz(res_xkh_nom[j,:].T,uk,thetah)
        #    Ax = Jx(res_xkh_nom[j,:].T,uk,thetah)
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
                    #test2+=Hz2
            #EKF-type propagation of the covariance of the augmented state-space
            Sigmak_p = mtimes(Az,mtimes(Sigmak,Az.T)) + Qz + 0.5*e_base_mat
            Kee = mtimes(Cz,mtimes(Sigmak_p,Cz.T)) + R
            Keei = solve(Kee,SX.eye(Kee.size1()))
            Kk = mtimes(mtimes(Sigmak_p,Cz.T),Keei)
            Sigmak = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmak_p)

            #term3[j*n_tot:(j+1)*n_tot,:] = mtimes((Sigmak_p - Sigmak),Kk)
            #term1[j*n_tot:(j+1)*n_tot,:] = mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmak)

        #elif j < (n_pred-1):
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
            #Sigmak = mtimes((SX.eye(n_st+n_par) - mtimes(Kk,Cz)),Sigmak_p)

            #term3[j*n_tot:(j+1)*n_tot,:] = mtimes((Sigmak_p - Sigmak),Kk)
            #term1[j*n_tot:(j+1)*n_tot,:] = mtimes(Wg[j*(n_st+n_par):(j+1)*(n_st+n_par),:],Sigmak)





























if runC1 == True:
    #Build Case 1; the optimal solution with known parameters
    Jopt, qu_opt, lbq, ubq, g, lbg, ubg, qu_init = Dual.lb_mpc(run_time,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,thetah,uk,Tsamp,zkh0,xk_sp,Wx,Wu,wkp)
    lp_opt = {'x':vertcat(*qu_opt), 'f':Jopt, 'g': vertcat(*g)}
    pdb.set_trace()
    solver_opt = nlpsol('solver_opt', MySolver, lp_opt)
    #Simulate case 1
    case1_res = solver_opt(x0=qu_init,  lbg=lbg, ubg=ubg)
    case1_res = case1_res['x'].full().flatten()
    #save Case 1 results
    for i1 in range(n_st):
        res1_xk[:,i1] += case1_res[i1::4]
    res1_uk[:,0] += case1_res[3::4]

    P.Plot(res1_uk,res1_xk,run_time,'Case 1')


if runC2 == True:
    #Build Case 2; the CE MPC optimization problem
    Jce, qu_ce, lbq, ubq, g, lbg, ubg, qu_init = Dual.ce_mpc(mdl_ode,n_pred,n_ctrl,n_st,n_par_st,n_par_ip,n_ip,uk_lb,uk_ub,xk_lb,xk_ub,xk,thetah,uk,Tsamp,zkh0,xk_sp,Wx,Wu,uk_opt)
    nlp_mpc = {'x':vertcat(*qu_ce), 'f':Jce, 'g':vertcat(*g)}


    solver_mpc = nlpsol('solver_mpc', MySolver, nlp_mpc)
    #Simulate case 2
    case2_res = solver_mpc(x0=qu_init, lbg=lbg, ubg=ubg)
    case2_res = case2_res['x'].full().flatten()
    #save Case 2 results
    for i1 in range(n_st):
        res2_xk[:,i1] += case2_res[i1::10]
    #pdb.set_trace()
    for i1 in range(n_par):
        res2_theta[:,i1] += case2_res[3+i1::10]
    res2_uk[:,0] += case2_res[9::10]

    P.PlotC2(res2_uk,res2_xk,run_time,res2_theta,'Case 2')

if runC3 == True:
    Jd, qu, op = dualfcn.gencost(n_st,n_ip,n_op,n_par,n_pred,Sigmak_p,mdl_ode,fy,xk,uk,thetah,Tsamp,xk_sp,Wx,Wu,Q,Qz,R,uk_opt)
    nlp = {'x':vertcat(*qu), 'f': Jd, 'p': op[0:9]}
    solver = nlpsol('solver', MySolver, nlp, {'ipopt':{'max_iter':1000,"check_derivatives_for_naninf":'yes', "print_user_options":'yes' }})
