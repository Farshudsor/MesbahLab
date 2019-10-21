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




#########################################
##  Compute comparative results from a set of simulations
######################################
def Compute_results(trials,run_time,n_st,n_par,n_ip):
    run_time = run_time #+ 1
    #Case 2 parameter errors
    t1C2 = np.zeros((run_time, trials))
    t2C2 = np.zeros((run_time, trials))
    t3C2 = np.zeros((run_time, trials))
    t4C2 = np.zeros((run_time, trials))
    t5C2 = np.zeros((run_time, trials))
    t6C2 = np.zeros((run_time, trials))
    #Case 3 parameter errors
    t1C3 = np.zeros((run_time, trials))
    t2C3 = np.zeros((run_time, trials))
    t3C3 = np.zeros((run_time, trials))
    t4C3 = np.zeros((run_time, trials))
    t5C3 = np.zeros((run_time, trials))
    t6C3 = np.zeros((run_time, trials))
    test = {}

    #Cost
    cost = np.zeros((trials, 3))


    for i1 in range(trials):
        with open('R'+str(i1+1)+'.csv') as f:
    #    with open('Results/10_24_18/Results_'+str(i1+1)+'_All.csv') as f:
            reader = csv.reader(f)
            F = [r for r in reader]
            F.pop(0) # remove header

            for i2 in range(run_time):
                #Error in parameter estimation
                t1C2[i2,i1] = float(F[i2+run_time*4][1])-float(F[i2+run_time*4][2])
                t1C3[i2,i1] = float(F[i2+run_time*4][1])-float(F[i2+run_time*4][3])

                t2C2[i2,i1] = float(F[i2+run_time*5][1])-float(F[i2+run_time*5][2])
                t2C3[i2,i1] = float(F[i2+run_time*5][1])-float(F[i2+run_time*5][3])

                t3C2[i2,i1] = float(F[i2+run_time*6][1])-float(F[i2+run_time*6][2])
                t3C3[i2,i1] = float(F[i2+run_time*6][1])-float(F[i2+run_time*6][3])

                t4C2[i2,i1] = float(F[i2+run_time*7][1])-float(F[i2+run_time*7][2])
                t4C3[i2,i1] = float(F[i2+run_time*7][1])-float(F[i2+run_time*7][3])

                t5C2[i2,i1] = float(F[i2+run_time*8][1])-float(F[i2+run_time*8][2])
                t5C3[i2,i1] = float(F[i2+run_time*8][1])-float(F[i2+run_time*8][3])

                t6C2[i2,i1] = float(F[i2+run_time*9][1])-float(F[i2+run_time*9][2])
                t6C3[i2,i1] = float(F[i2+run_time*9][1])-float(F[i2+run_time*9][3])
                #test[i2] = F[i2+run_time*9+1][0]

            #import pdb; pdb.set_trace()
            cost[i1,0] = float(F[0][1])
            cost[i1,1] = float(F[0][2])
            cost[i1,2] = float(F[0][3])



            #collect Cost
            cost_avg1 = sum(cost[:,0])/trials
            cost_avg2 = sum(cost[:,1])/trials
            cost_avg3 = sum(cost[:,2])/trials

    return t1C2, t2C2, t3C2, t4C2, t5C2, t6C2, t1C3, t2C3, t3C3, t4C3, t5C3, t6C3, cost_avg1,cost_avg2,cost_avg3,test


def Compute_resultsCE(trials,run_time):
    n_st = 3
    n_par = 6
    n_ip = 1
    #run_time = run_time + 1

    #Case 2 parameter errors
    t1error_case2 = np.zeros((run_time, trials))
    t2error_case2 = np.zeros((run_time, trials))
    t3error_case2 = np.zeros((run_time, trials))
    t4error_case2 = np.zeros((run_time, trials))
    t5error_case2 = np.zeros((run_time, trials))
    t6error_case2 = np.zeros((run_time, trials))

    #control sequence for control cost
    ip1 = np.zeros((run_time, trials))
    ip2 = np.zeros((run_time, trials))
    #final stage state
    xN1 = np.zeros((n_st, trials))
    xN2 = np.zeros((n_st, trials))

    #Cost
    cost = np.zeros((trials, 2))


    for i1 in range(trials):
        with open('./Results/Results_'+str(i1)+'.csv') as f:
            reader = csv.reader(f)
            F = [r for r in reader]

            F.pop(0) # remove header

            cost[i1, :] = float(F[0][1]), float(F[0][2])

            F.pop(0) # remve "costs" line

            xN1[0,i1] = float(F[run_time-1][1])
            xN1[1,i1] = float(F[run_time*2-1][1])
            xN1[2,i1] = float(F[run_time*3-1][1])

            xN2[0,i1] = float(F[run_time-1][2])
            xN2[1,i1] = float(F[run_time*2-1][2])
            xN2[2,i1] = float(F[run_time*3-1][2])

            for i2 in range(run_time-1):
                #final stage state

                #control sequence
                ip1[i2,i1] = float(F[run_time*3+i2][1])
                ip2[i2,i1] = float(F[run_time*3+i2][2])

                #pdb.set_trace()
                #Error in parameter estimation
                t1error_case2[i2,i1] = float(F[i2+run_time*4-1][1])-float(F[i2+run_time*4-1][2])
                t2error_case2[i2,i1] = float(F[i2+run_time*5-1][1])-float(F[i2+run_time*5-1][2])
                t3error_case2[i2,i1] = float(F[i2+run_time*6-1][1])-float(F[i2+run_time*6-1][2])
                t4error_case2[i2,i1] = float(F[i2+run_time*7-1][1])-float(F[i2+run_time*7-1][2])
                t5error_case2[i2,i1] = float(F[i2+run_time*8-1][1])-float(F[i2+run_time*8-1][2])
                t6error_case2[i2,i1] = float(F[i2+run_time*9-1][1])-float(F[i2+run_time*9-1][2])

    return xN1,xN2, ip1, ip2, cost, t1error_case2, t2error_case2, t3error_case2, t4error_case2, t5error_case2, t6error_case2




#########################################
##  Organize data to write to CSV
######################################
def writeDataSingle(Cost_ce,res2_uk,res2_xk,res2_theta,run_time,csv_file,Case):
    File = open(csv_file, 'w')
    with File:
        writer = csv.writer(File)

        writer.writerow( ["Parameter","Case 1"] )
        writer.writerow( ["Cost",    Case ] )

        for i1 in range(run_time):
            writer.writerow( ['xk1' , res2_xk[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk2' , res2_xk[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk3' , res2_xk[i1,2] ] )
        for i1 in range(run_time-1):
            writer.writerow( ['uk'  , res2_uk[i1,0] ] )


        for i1 in range(run_time):
            writer.writerow( ['t1' , res2_theta[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['t2' , res2_theta[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['t3' , res2_theta[i1,2] ] )
        for i1 in range(run_time):
            writer.writerow( ['t4' , res2_theta[i1,3] ] )
        for i1 in range(run_time):
            writer.writerow( ['t5' , res2_theta[i1,4] ] )
        for i1 in range(run_time):
            writer.writerow( ['t6' , res2_theta[i1,5] ] )


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


def writeDataCE(Cost_opt,Cost_ce,res1_uk,res1_xk,theta_act,res2_uk,res2_xk,res2_theta,run_time,csv_file):
    File = open(csv_file, 'w')
    with File:
        writer = csv.writer(File)

        writer.writerow( ["Parameter","Case 1", "Case 2"] )
        writer.writerow( ["Cost",    Cost_opt,   Cost_ce] )

        for i1 in range(run_time):
            writer.writerow( ['xk1' , res1_xk[i1,0], res2_xk[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk2' , res1_xk[i1,1], res2_xk[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['xk3' , res1_xk[i1,2], res2_xk[i1,2] ] )
        for i1 in range(run_time-1):
            writer.writerow( ['uk'  , res1_uk[i1], res2_uk[i1,0] ] )


        for i1 in range(run_time):
            writer.writerow( ['t1' , theta_act[0], res2_theta[i1,0] ] )
        for i1 in range(run_time):
            writer.writerow( ['t2' , theta_act[1], res2_theta[i1,1] ] )
        for i1 in range(run_time):
            writer.writerow( ['t3' , theta_act[2], res2_theta[i1,2] ] )
        for i1 in range(run_time):
            writer.writerow( ['t4' , theta_act[3], res2_theta[i1,3] ] )
        for i1 in range(run_time):
            writer.writerow( ['t5' , theta_act[4], res2_theta[i1,4] ] )
        for i1 in range(run_time):
            writer.writerow( ['t6' , theta_act[5], res2_theta[i1,5] ] )














#########################################
##  Plotting scripts for Case1, and for Case 2 & 3
######################################
#For Case 1
def Plot(res1_uk,res1_xk,run_time, case):
    time=[]
    for i in range(run_time-1):
        time+=[i]
    utime=[]
    for i in range(run_time-2):
        utime+=[i]

    #print('u opt = ', len(res1_uk))
    #print('u time = ', len(utime))

    a = plt.figure(1)
    plt.clf()
    plt.title(case)
    plt.step(utime,res1_uk, 'r-')
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

    raw_input()

#For Case 2 & 3
def PlotC2(res1_uk,res1_xk,run_time,res1_theta, case):
    time=[]
    for i in range(run_time):
        time+=[i]
    utime=[]
    for i in range(run_time):
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
    plt.subplot(2, 1, 1)
    plt.title(case)
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
    plt.ylabel('Theta 1, 2, & 3 ')
    plt.legend(['Theata 1', 'Theata 5'])
    plt.grid()
    c.show()

#    c = plt.figure(3)
#    plt.clf()
#    plt.title(case)
#    plt.plot(time,res1_theta[:,0])
#    plt.plot(time,res1_theta[:,1])
#    plt.plot(time,res1_theta[:,2])
#    plt.ylabel('Theta 1, 2, & 3 ')
#    plt.legend(['Theata 1', 'Theta 2', 'Theta 3'])
#    plt.grid()
#    c.show()

#    d = plt.figure(4)
#    plt.clf()
#    plt.title(case)
#    plt.plot(time,res1_theta[:,3])
#    plt.plot(time,res1_theta[:,4])
#    plt.plot(time,res1_theta[:,5])
#    plt.ylabel('Theta 4, 5, & 6 ')
#    plt.legend(['Theta 4', 'Theta 5', 'Theta 6'])
#    plt.grid()
#    d.show()

    raw_input()


def PlotC2C3(res1_uk,res1_xk,res1_theta,case1, res2_uk,res2_xk,res2_theta, case2, run_time):

    time=[]
    for i in range(run_time):
        time+=[i]
    utime=[]
    for i in range(run_time):
        utime+=[i]

    a = plt.figure(1)
    plt.clf()
    plt.title(' ')
    plt.step(utime,res1_uk[:,0], 'r-')
    plt.step(utime,res2_uk[:,0], 'b')
    plt.ylabel('Control Input')
    plt.legend([case1, case2])
    plt.grid()
    a.show()

    b = plt.figure(2)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.title(' ')
    plt.ylabel('States 1 & 2')
    plt.plot(time,res1_xk[:,0], 'r-')
    plt.plot(time,res1_xk[:,1], 'o:')
    plt.plot(time,res2_xk[:,0], 'b-')
    plt.plot(time,res2_xk[:,1], 'g:')
    plt.legend(['x1 '+case1, 'x2 '+case1, 'x1 '+case2, 'x2 '+case2])
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.ylabel('States 3')
    plt.plot(time,res1_xk[:,2], 'r-')
    plt.plot(time,res2_xk[:,2], 'b-')
    plt.legend(['x3 '+case1, 'x3 '+case2])
    plt.grid()
    b.show()

    c = plt.figure(3)
    plt.clf()
    plt.title(' ')
    plt.plot(time,res1_theta[:,0], 'r-')
    plt.plot(time,res2_theta[:,0], 'b-')
    plt.ylabel('Theta 1 error squared ')
    plt.legend(['Theata 1 '+case1, 'Theata 1 '+case2])
    plt.grid()
    c.show()

    d = plt.figure(4)
    plt.clf()
    plt.title(' ')
    plt.plot(time,res1_theta[:,1], 'r-')
    plt.plot(time,res2_theta[:,1], 'b-')
    plt.ylabel('Theta 5 error squared ')
    plt.legend(['Theata 5 '+case1,'Theata 5 '+case2])
    plt.grid()
    d.show()

    raw_input()
