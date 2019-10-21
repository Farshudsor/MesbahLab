# -*- coding: utf-8 -*-
"""
Created on July 4th 2018

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


import csv
with open('Results_1.csv') as f:
    reader = csv.reader(f)
    F = [r for r in reader]
    F.pop(0) # remove header



#########################################
##  Compute comparative results from a set of simulations
######################################
def Compute_results(trials,run_time,n_st,n_par,n_ip):
    run_time = run_time + 1
    #Case 2 parameter errors
    t1error_case2 = np.zeros((run_time, trials))
    t2error_case2 = np.zeros((run_time, trials))
    t3error_case2 = np.zeros((run_time, trials))
    t4error_case2 = np.zeros((run_time, trials))
    t5error_case2 = np.zeros((run_time, trials))
    t6error_case2 = np.zeros((run_time, trials))
    #Case 3 parameter errors
    t1error_case3 = np.zeros((run_time, trials))
    t2error_case3 = np.zeros((run_time, trials))
    t3error_case3 = np.zeros((run_time, trials))
    t4error_case3 = np.zeros((run_time, trials))
    t5error_case3 = np.zeros((run_time, trials))
    t6error_case3 = np.zeros((run_time, trials))
    #Cost
    cost = np.zeros(trials, 3)


    for i1 in range(trials):
        with open('Results_'+str(i1)+'.csv') as f:
            reader = csv.reader(f)
            F = [r for r in reader]
            F.pop(0) # remove header

            for i2 in range(run_time):
                #Error in parameter estimation
                t1error_case2[i2,i1] = float(F[i2+run_time*4][1])-float(F[i2+run_time*4][2])
                t1error_case3[i2,i1] = float(F[i2+run_time*4][1])-float(F[i2+run_time*4][3])

                t2error_case2[i2,i1] = float(F[i2+run_time*5][1])-float(F[i2+run_time*5][2])
                t2error_case3[i2,i1] = float(F[i2+run_time*5][1])-float(F[i2+run_time*5][3])

                t3error_case2[i2,i1] = float(F[i2+run_time*6][1])-float(F[i2+run_time*6][2])
                t3error_case3[i2,i1] = float(F[i2+run_time*6][1])-float(F[i2+run_time*6][3])

                t4error_case2[i2,i1] = float(F[i2+run_time*7][1])-float(F[i2+run_time*7][2])
                t4error_case3[i2,i1] = float(F[i2+run_time*7][1])-float(F[i2+run_time*7][3])

                t5error_case2[i2,i1] = float(F[i2+run_time*8][1])-float(F[i2+run_time*8][2])
                t5error_case3[i2,i1] = float(F[i2+run_time*8][1])-float(F[i2+run_time*8][3])

                t6error_case2[i2,i1] = float(F[i2+run_time*9][1])-float(F[i1+run_time*9][2])
                t6error_case3[i2,i1] = float(F[i2+run_time*9][1])-float(F[i1+run_time*9][3])



            #collect Cost
            cost[0] = sum([float(F[run_time*3+i3][1]) for i3 in range(run_time)]) + float(F[run_time*3-1][1])
            cost[1] = sum([float(F[run_time*3+i3][2]) for i3 in range(run_time)]) + float(F[run_time*3-1][2])
            cost[2] = sum([float(F[run_time*3+i3][3]) for i3 in range(run_time)]) + float(F[run_time*3-1][3])

    #compute averages






















#########################################
##  Organize data to write to CSV
######################################
def writeData(res1_uk,res1_xk,theta_act,res2_uk,res2_xk,res2_theta,res3_uk,res3_xk,res3_theta,run_time,csv_file):
    File = open(csv_file, 'w')
    with File:
        writer = csv.writer(File)
        writer.writerow( ["Parameter","Case 1", "Case 2", "Case3"])

        for i1 in range(run_time):
            writer.writerow( ['xk1' , res1_xk[i1,0], res2_xk[i1,0], res3_xk[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk2' , res1_xk[i1,1], res2_xk[i1,1], res3_xk[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk3' , res1_xk[i1,2], res2_xk[i1,2], res3_xk[i1,2] ] )
        for i1 in range(run_time-1):
            writer.writerow( ['uk'  , res1_uk[i1,0], res2_uk[i1,0], res3_uk[i1,0] ] )


        for i1 in range(run_time):
            writer.writerow( ['t1', theta_act[0], res2_theta[i1,0], res3_theta[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['t2' , theta_act[1], res2_theta[i1,1], res3_theta[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['t3' , theta_act[2], res2_theta[i1,2], res3_theta[i1,2] ] )
        for i1 in range(run_time):
            writer.writerow( ['t4' , theta_act[3], res2_theta[i1,3], res3_theta[i1,3] ] )
        for i1 in range(run_time):
            writer.writerow( ['t5' , theta_act[4], res2_theta[i1,4], res3_theta[i1,4] ] )
        for i1 in range(run_time):
            writer.writerow( ['t6' , theta_act[5], res2_theta[i1,5], res3_theta[i1,5] ] )


















#########################################
##  Plotting scripts for Case1, and for Case 2 & 3
######################################
#For Case 1
def Plot(res1_uk,res1_xk,run_time, case):
    time=[]
    for i in range(run_time):
        time+=[i]
    utime=[]
    for i in range(run_time-1):
        utime+=[i]

    a = plt.figure(1)
    plt.clf()
    plt.title(case)
    plt.step(utime,res1_uk[:,0], 'r-')
    plt.ylabel('Control Input')
    plt.legend(['uk'])
    plt.grid()
    a.show()

    b = plt.figure(2)
    plt.clf()
    plt.subplot(2,1,1)
    plt.title(case)
    plt.subplot(2, 1, 1)
    plt.ylabel('States 1 & 2')
    plt.plot(time,res1_xk[:,0])
    plt.plot(time,res1_xk[:,1])
    plt.legend(['x1', 'x2'])
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.ylabel('States 3')
    plt.plot(time,res1_xk[:,2])
    plt.legend(['x3'])
    plt.grid()
    b.show()

    #raw_input()

#For Case 2 & 3
def PlotC2(res1_uk,res1_xk,run_time,res1_theta, case):
    time=[]
    for i in range(run_time):
        time+=[i]
    utime=[]
    for i in range(run_time-1):
        utime+=[i]

    a = plt.figure(1)
    plt.clf()
    plt.title(case)
    plt.step(utime,res1_uk[:,0], 'r-')
    plt.ylabel('Control Input')
    plt.legend(['uk'])
    plt.grid()
    a.show()

    b = plt.figure(2)
    plt.clf()
    plt.subplot(2,1,1)
    plt.title(case)
    plt.subplot(2, 1, 1)
    plt.ylabel('States 1 & 2')
    plt.plot(time,res1_xk[:,0])
    plt.plot(time,res1_xk[:,1])
    plt.legend(['x1', 'x2'])
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.ylabel('States 3')
    plt.plot(time,res1_xk[:,2])
    plt.legend(['x3'])
    plt.grid()
    b.show()

    c = plt.figure(3)
    plt.clf()
    plt.title(case)
    plt.plot(time,res1_theta[:,0])
    plt.plot(time,res1_theta[:,1])
    plt.plot(time,res1_theta[:,2])
    plt.ylabel('Theta 1, 2, & 3 ')
    plt.legend(['Theata 1', 'Theta 2', 'Theta 3'])
    plt.grid()
    c.show()

    d = plt.figure(4)
    plt.clf()
    plt.title(case)
    plt.plot(time,res1_theta[:,3])
    plt.plot(time,res1_theta[:,4])
    plt.plot(time,res1_theta[:,5])
    plt.ylabel('Theta 4, 5, & 6 ')
    plt.legend(['Theta 4', 'Theta 5', 'Theta 6'])
    plt.grid()
    d.show()

    #raw_input()
