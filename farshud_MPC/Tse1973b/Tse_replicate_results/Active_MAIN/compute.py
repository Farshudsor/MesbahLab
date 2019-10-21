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


trials = 10
run_time = 20
time = np.zeros(run_time)
for i in range(run_time):
    time[i] = i+1
t1C2, t2C2, t3C2, t4C2, t5C2, t6C2, t1C3, t2C3, t3C3, t4C3, t5C3, t6C3, cost_avg1,cost_avg2,cost_avg3,test = P.Compute_results(trials,run_time,3,6,1)

t1C2_avg = np.zeros(run_time)
t2C2_avg = np.zeros(run_time)
t3C2_avg = np.zeros(run_time)
t4C2_avg = np.zeros(run_time)
t5C2_avg = np.zeros(run_time)
t6C2_avg = np.zeros(run_time)

t1C3_avg = np.zeros(run_time)
t2C3_avg = np.zeros(run_time)
t3C3_avg = np.zeros(run_time)
t4C3_avg = np.zeros(run_time)
t5C3_avg = np.zeros(run_time)
t6C3_avg = np.zeros(run_time)

for i2 in range(run_time):
    #pdb.set_trace()
    t1C2_avg[i2] = (sum(t1C2[i2,:])/trials)**2
    t2C2_avg[i2] = (sum(t2C2[i2,:])/trials)**2
    t3C2_avg[i2] = (sum(t3C2[i2,:])/trials)**2
    t4C2_avg[i2] = (sum(t4C2[i2,:])/trials)**2
    t5C2_avg[i2] = (sum(t5C2[i2,:])/trials)**2
    t6C2_avg[i2] = (sum(t6C2[i2,:])/trials)**2

    t1C3_avg[i2] = (sum(t1C3[i2,:])/trials)**2
    t2C3_avg[i2] = (sum(t2C3[i2,:])/trials)**2
    t3C3_avg[i2] = (sum(t3C3[i2,:])/trials)**2
    t4C3_avg[i2] = (sum(t4C3[i2,:])/trials)**2
    t5C3_avg[i2] = (sum(t5C3[i2,:])/trials)**2
    t6C3_avg[i2] = (sum(t6C3[i2,:])/trials)**2

print('Case 1 average cost = ', cost_avg1)
print('Case 2 average cost = ', cost_avg2)
print('Case 3 average cost = ', cost_avg3)



a = plt.figure(1)
plt.clf()
plt.title('Avg. estimation error squared for theta 1, 2, & 3 ')
plt.plot(time,abs(t1C2_avg),'r')
plt.plot(time,abs(t2C2_avg),'b')
plt.plot(time,abs(t3C2_avg),'g')
plt.plot(time,abs(t1C3_avg),':r')
plt.plot(time,abs(t2C3_avg),':b')
plt.plot(time,abs(t3C3_avg),':g')
plt.yscale('log')
plt.legend(['Theta 1 CE', 'Theta 2 CE', 'Theta 3 CE','Theta 1 Dual', 'Theta 2 Dual', 'Theta 3 Dual'])
plt.grid()
a.show()




b = plt.figure(2)
plt.clf()
plt.title('Avg. estimation error squared for theta 4, 5 & 6 ')
plt.plot(time,abs(t4C2_avg),'r')
plt.plot(time,abs(t5C2_avg),'b')
plt.plot(time,abs(t6C2_avg),'g')
plt.plot(time,abs(t4C3_avg),':r')
plt.plot(time,abs(t5C3_avg),':b')
plt.plot(time,abs(t6C3_avg),':g')
plt.yscale('log')
plt.legend(['Theta 4 CE', 'Theta 5 CE', 'Theta 6 CE','Theta 4 Dual', 'Theta 5 Dual', 'Theta 6 Dual'])
plt.grid()
b.show()

#c = plt.figure(3)
#plt.clf()
#plt.title('Avg. cumulative control energy')
#plt.plot(time,ip2_cum)
#plt.yscale('log')
#plt.legend(['CE control'])
#plt.grid()
#c.show()

raw_input()
