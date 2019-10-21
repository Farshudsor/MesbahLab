"""
Created on March 4th 2019

@author: Farshud Sorourifar

Execute multiple simulations of DualMPC.py
"""
import sys
import numpy as np
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")

plots = False
n_sim = 20

sweepxu = False
sweepxl = False
sweepuu = False
sweepul = False

multi_sweep = True

x_u = np.round(0.996,3)
x_l = np.round(1.00,3)
u_u = np.round(0.98,3)
u_l = np.round(1.00,3)
bounds = [x_u,x_l,u_u,u_l]
#import pdb; pdb.set_trace()


if multi_sweep:
    alpha_u = x_u
    alpha_l = x_l
    beta_u = u_u
    beta_l = u_l
    run_name = str(alpha_u)+'alpha_u_'+str(alpha_l)+'alpha_l'+str(beta_u)+'beta_u_'+str(beta_l)+'beta_l'
#    for i in range(n_sim):
#        seed = i
#        sys.argv = ['DualMPC.py','alpha_u','alpha_l','beta_u','beta_l', 'run_name','seed' ]
#        execfile('DualMPC.py')

    folder = 'results/'+run_name+'/'
    file = "results/Results_"+run_name+".csv"
    sys.argv = ['plotting_script.py', 'folder', 'file', 'bounds' ]
    execfile('plotting_script.py')


if sweepxu:
    for j1 in range(len(x_u)): # scale x upper
        for i in range(n_sim):
            alpha_u = x_u[j1]
            run_name = str(alpha_u)+'alpha+'
            alpha_l = 1
            beta_u = 1
            beta_l = 1
            sys.argv = ['DualMPC.py','alpha_u','alpha_l','alpha_u','alpha_l', 'run_name' ]
            execfile('DualMPC.py')

if sweepxl:
    for j2 in range(len(x_l)): # scale x_lower
        for i in range(n_sim):
            alpha_u = 1
            alpha_l = x_l[j2]
            beta_u = 1
            beta_l = 1
            run_name = str(alpha_l)+'alpha-'
            sys.argv = ['DualMPC.py','alpha_u','alpha_l','alpha_u','alpha_l','run_name']
            execfile('DualMPC.py')

if sweepuu:
    for j3 in range(len(u_u)): #scale u_upper
        for i in range(n_sim):
            alpha_u = 1
            alpha_l = 1
            beta_u = u_u[j3]
            beta_l = 1
            run_name = str(beta_u)+'beta+'
            sys.argv = ['DualMPC.py','alpha_u','alpha_l','alpha_u','alpha_l','run_name']
            execfile('DualMPC.py')

if sweepul:
    for j4 in range(len(u_l)): # scale u_lower
        for i in range(n_sim):
            alpha_u = 1
            alpha_l = 1
            beta_u = 1
            beta_l = u_l[j4]
            run_name = str(beta_l)+'beta-'
            sys.argv = ['DualMPC.py','alpha_u','alpha_l','alpha_u','alpha_l','run_name']
            execfile('DualMPC.py')


if plots:
    if True:
        for j1 in range(len(x_u)): # scale x upper
            alpha_u = x_u[j1]
            run_name = str(alpha_u)+'alpha+'

            folder = 'results/'+run_name+'/'
            file = "results/Results_"+run_name+".csv"
            sys.argv = ['plotting_script.py', 'folder', 'file' ]
            execfile('plotting_script.py')

    if True:
        for j2 in range(len(x_l)): # scale x_lower
            alpha_l = x_l[j2]
            run_name = str(alpha_l)+'alpha-'

            folder = 'results/'+run_name+'/'
            file = "results/Results_"+run_name+".csv"
            sys.argv = ['plotting_script.py', 'folder', 'file' ]
            execfile('plotting_script.py')

    if True:
        for j3 in range(len(u_u)): #scale u_upper
            beta_u = u_u[j3]
            run_name = str(beta_u)+'beta+'

            folder = 'results/'+run_name+'/'
            file = "results/Results_"+run_name+".csv"
            sys.argv = ['plotting_script.py', 'folder', 'file' ]
            execfile('plotting_script.py')

    if True:
        for j4 in range(len(u_l)): # scale u_lower
            beta_l = u_l[j4]
            run_name = str(beta_l)+'beta-'

            folder = 'results/'+run_name+'/'
            file = "results/Results_"+run_name+".csv"
            sys.argv = ['plotting_script.py', 'folder', 'file' ]
            execfile('plotting_script.py')
