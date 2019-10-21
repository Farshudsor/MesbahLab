"""
Created on March 4th 2019

@author: Farshud Sorourifar

Execute multiple simulations of DualMPC.py
"""
import sys
import numpy as np
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")

n_sim = 20

x_u = np.arange(.995,1.001,.001)
x_l = np.arange(1.00,1.0055,.001)
sweep_x = len(x_l)
u_u = np.arange(.995,1.001,.001)
u_l = np.arange(1.00,1.0055,.001)
#import pdb; pdb.set_trace()

for i in range(n_sim):
    sim = i

#    for j1 in range(sweep_x): # scale x upper
#        alpha_x = x_u[j1]
#        run_name = str(alpha_x)+'alpha_x'
#        beta_x = 1
#        alpha_u = 1
#        beta_u = 1
#        sys.argv = ['DualMPC.py','sim','alpha_x','beta_x','alpha_x','beta_x', 'run_name' ]
#        execfile('DualMPC.py')

#    for j2 in range(sweep_x): # scale x_lower
#        alpha_x = 1
#        beta_x = x_l[j2]
#        alpha_u = 1
#        beta_u = 1
#        run_name = str(beta_x)+'beta_x'
#        sys.argv = ['DualMPC.py','sim','alpha_x','beta_x','alpha_x','beta_x','run_name']
#        execfile('DualMPC.py')

    for j3 in range(sweep_x): #scale u_uapper
        alpha_x = 1
        beta_x = 1
        alpha_u = u_u[j3]
        beta_u = 1
        run_name = str(alpha_u)+'alpha_u'
        sys.argv = ['DualMPC.py','sim','alpha_x','beta_x','alpha_x','beta_x','run_name']
        execfile('DualMPC.py')

    for j4 in range(sweep_x): # scale u_lower
        alpha_x = 1
        beta_x = 1
        alpha_u = 1
        beta_u = u_l[j4]
        run_name = str(beta_u)+'beta_u'
        sys.argv = ['DualMPC.py','sim','alpha_x','beta_x','alpha_x','beta_x','run_name']
        execfile('DualMPC.py')
