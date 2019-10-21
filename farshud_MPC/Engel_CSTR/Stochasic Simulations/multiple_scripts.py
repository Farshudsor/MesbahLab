"""
Created on March 4th 2019

@author: Farshud Sorourifar

Execute multiple simulations of DualMPC.py
"""
import sys
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")

n_sim = 20
for i in range(n_sim):
    execfile("DualMPC.py")
