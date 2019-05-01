# -*- coding: utf-8 -*-
"""
Created on July 25th 2018

@author: Farshud Sorourifar

"""
import sys
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from scipy import linalg
from casadi.tools import *
import core
import pdb
import csv
import results as P
import pylab


trials = 20
run_time = 20

xN1,xN2,ip1,ip2,cost,t1error_case2,t2error_case2,t3error_case2,t4error_case2,t5error_case2,t6error_case2 = P.Compute_resultsCE(trials,run_time)

#average Cost
costavg1 = sum([cost[i1,0] for i1 in range(trials)])/trials
costavg2 = sum([cost[i1,1] for i1 in range(trials)])/trials

#average miss distance squared
xN1miss = (sum([(xN1[0,i1]**2) for i1 in range(trials)]) + sum([(xN1[1,i1]**2) for i1 in range(trials)]) + sum([((xN1[2,i1]-20)**2) for i1 in range(trials)]))/trials
xN2miss = (sum([(xN2[0,i1]**2) for i1 in range(trials)]) + sum([(xN2[1,i1]**2) for i1 in range(trials)]) + sum([((xN2[2,i1]-20)**2) for i1 in range(trials)]))/trials

t1avg = np.zeros((run_time-1))
t2avg = np.zeros((run_time-1))
t3avg = np.zeros((run_time-1))
t4avg = np.zeros((run_time-1))
t5avg = np.zeros((run_time-1))
t6avg = np.zeros((run_time-1))

time = np.zeros((run_time-1))

ip1_cum = np.zeros((run_time-1))
ip2_cum = np.zeros((run_time-1))

ip1_avg = 0.0
ip2_avg = 0.0

for i1 in range(run_time-1):
    #average error in parameter estimation
    t1avg[i1] = (sum(t1error_case2[i1,:])/trials)**2
    t2avg[i1] = (sum(t2error_case2[i1,:])/trials)**2
    t3avg[i1] = (sum(t3error_case2[i1,:])/trials)**2
    t4avg[i1] = (sum(t4error_case2[i1,:])/trials)**2
    t5avg[i1] = (sum(t5error_case2[i1,:])/trials)**2
    t6avg[i1] = (sum(t6error_case2[i1,:])/trials)**2

    time[i1] = i1

    #average control energy -( lambda* sum(uk**2) )
    #still needs cumsum
    ip1_k = (sum([ip1[i1,i2]**2 for i2 in range(trials)]))*(10**(-3))
    ip2_k = (sum([ip2[i1,i2]**2 for i2 in range(trials)]))*(10**(-3))

    ip1_tot = ip1_tot + ip1_k
    ip2_tot = ip2_tot + ip2_k

    ip1_cum[i1] = ip1_tot/trials
    ip2_cum[i1] = ip2_tot/trials



print('Case 1 average cost = ', costavg1)
print('Case 2 average cost = ', costavg2)
print('Case 1 average miss distance  = ', xN1miss)
print('Case 2 average miss distance  = ', xN2miss)
print('Case 1 weighted cumulative control energy  = ', ip1_tot/trials)
print('Case 2 weighted cumulative control energy  = ', ip2_tot/trials)

pdb.set_trace()

a = plt.figure(1)
plt.clf()
plt.title('Avg. estimation error squared for theta 1, 2, & 3 ')
plt.plot(time,t1avg)
plt.plot(time,t2avg)
plt.plot(time,t3avg)
plt.yscale('log')
plt.legend(['Theata 1', 'Theta 2', 'Theta 3'])
plt.grid()
a.show()

b = plt.figure(2)
plt.clf()
plt.title('Avg. estimation error squared for theta 4, 5 & 6 ')
plt.plot(time,t4avg)
plt.plot(time,t5avg)
plt.plot(time,t6avg)
plt.yscale('log')
plt.legend(['Theata 4', 'Theta 5', 'Theta 6'])
plt.grid()
b.show()

c = plt.figure(3)
plt.clf()
plt.title('Avg. cumulative control energy')
plt.plot(time,ip2_cum)
plt.yscale('log')
plt.legend(['CE control'])
plt.grid()
c.show()

raw_input()
