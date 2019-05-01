"""
Created on March 4th 2019

@author: Farshud Sorourifar

Execute multiple simulations of DualMPC.py
"""
import sys
import numpy as np
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")

plots = False
n_sim = 10

multi_sweep = True

x_u = np.round(.998,3)
x_l = np.round(1.00,3)
u_u = np.round(.98,3)
u_l = np.round(1.00,3)
bounds = [x_u,x_l,u_u,u_l]

if multi_sweep:
    alpha_u = x_u
    alpha_l = x_l
    beta_u = u_u
    beta_l = u_l
    run_name = str(alpha_u)+'alpha_u_'+str(alpha_l)+'alpha_l'+str(beta_u)+'beta_u_'+str(beta_l)+'beta_l'
#    for i in range(n_sim):
#        seed = i
#        sys.argv = ['DualMPC.py','alpha_u','alpha_l','beta_u','beta_l', 'run_name' ,'seed']
#        execfile('DualMPC.py')

    folder = 'results/'+run_name+'/'
    file = "results/Results_"+run_name+".csv"
    sys.argv = ['plotting_script.py', 'folder', 'file', 'bounds' ]
    execfile('plotting_script.py')
