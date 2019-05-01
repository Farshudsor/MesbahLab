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

    raw_input()

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
