####    I've attached a few plots from the script, and the results do not look correct, as they do not approach the set points. This could be due to the changes I've made, or what appears to me, to be incorrect equations. Is it possible that some script you sent me is an old/incomplete version? I ask because some of the lines in the script resulted in incorrect dimensions, which I don't think it could be a result of the syntax of previous versions of casadi/ipopt/python.

####   Currently, the results of the script are not correct(fails to approach the state set points). This could be due to the changes I've made, or what appears to me, to be incorrect equations. Is it possible that some script you sent me is an old/incomplete version? I ask because some of the lines in the script resulted in incorrect dimensions, which I don't think it could be a result of the syntax of previous versions of casadi/ipopt/python.










#    #res = solver(x0=vertcat(xkp,theta), p=NP.array(zkh0), lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
#    #res = solver(x0=NP.array(zkh0) ,p=uk_opt, lbx=lbq[-7:-1], ubx=ubq[-7:-1], lbg=0, ubg=0)
    #res = solver(p=uk_opt, x0=NP.array(zkh0), ubx=[5.0, 5.0, 140.0, 140.0,inf,inf], lbx=[0.0, 0.0, 100.0, 100.0,-inf,-inf], lbg=0, ubg=0)
    #res = solver( x0=NP.array(zkh0), ubx=[5.0, 5.0, 140.0, 140.0,inf,inf], lbx=[0.0, 0.0, 100.0, 100.0,-inf,-inf], lbg=0, ubg=0)
#    res = solver( x0=qu_ce, ubx=[5.0, 5.0, 140.0, 140.0,inf,inf], lbx=[0.0, 0.0, 100.0, 100.0,-inf,-inf], lbg=0, ubg=0)


1 = my qu_init
2 = their qu_init

