#Author : Farshud Sorourifar,
#               modified code from Dougan
#Date : 1/3/18

#import packages
import sys
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")
from casadi import *
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random as ran

#### problem parameters, formulation of process model####
#system size
nx = 3          # concentrations
nu = 1          # rate

#simulation time
Nsim = 20      #number of simulation time steps
Delta = .5      #sampling time [hr]
Np = 3          #preditcion horizon

# declare casadi variable graph
x = MX.sym('x',nx) #symbolic vector [1x3]
u = MX.sym('u') #symbolic scalar [1x1]

####### NL SYSTEM DYNAMICS
# States, inputs, and disturbances are physically:
#x[0] = X       # g/L           # biomass concentration
#x[1] = S       # g/L           # substrate concentration
#x[2] = P       # g/L           # product concentration
#u[0] = D       # 1/hr          # dilution rate

# fixed parameters
Yxs = 0.4       # g/g           # yield of biomass per substrate consumed
alpha = 2.2     # g/g           # yield parameter
beta = 0.2      # 1/hr          # yield parameter
K_M = 1.2       # g/L           # affinity constant
K_I = 22.0      # g/L           # inhibition constant
mu_max0 = 0.48  # 1/hr          # maximum growth rate
P_M = 50.0      # g/L           # product saturation constant
S_f = 20.0      # g/L           # concentration of substrate in inlet feed

# growth rate
mu = mu_max0*(1-x[2]/P_M)*x[1]/(K_M + x[1] + x[1]**2/K_I)

# differential equations
dx1 = -u[0]*x[0] + mu*x[0]
dx2 = u[0]*(S_f-x[1]) - 1/Yxs*mu*x[0]
dx3 = -u[0]*x[2] + (alpha*mu+beta)*x[0]

#### Set up OPC #####
# get steady state values
uss = [0.15]
xguess=[ 7.038, 2.404, 24.869]
xss = [6.06778, 4.83056, 19.4169]
#  [6.06778, 4.83056, 19.4169],[0.15]
# concatenate differential equations
xdot = vertcat(dx1,dx2,dx3)
# create casadi function


#define quadratic objective function
obj = (x[2]-xss[2])**2+(x[1]-xss[1])**2+(x[0]-xss[0])**2+10*(u[0]-uss[0])**2

# define integrator
ode = {'x':x, 'p':u, 'ode':xdot, 'quad':obj}
opts = {'abstol':1e-10, 'reltol':1e-10, 'tf':Delta}
I = integrator('I', 'cvodes', ode, opts)
f = Function('f', [x,u], [xdot])


F = f.jacobian()
FJ1 = F([1,6,0],[.15],0)

def fjac(x,u):

    df1dx1= -u[0] + (mu_max0*(1-x[2]/P_M)*x[1]/(K_M + x[1] + x[1]**2/K_I))
    df1dx2= (mu_max0*K_I*x[0]*(P_M-x[2])*(K_M*K_I-x[1]**2)) / (P_M*(K_M*K_I + x[1]*(K_I+x[1]))**2)
    df1dx3= (mu_max0*K_I*x[0]*x[1]) / (P_M*(K_M*K_I+x[1]*(K_I+x[1])))

    df2dx1= (mu_max0*K_I*x[1]*(P_M-x[2])) / (P_M*Yxs*(K_M*K_I+x[1]*(K_I+x[1])))
    df2dx2= -(mu_max0*x[0]*(1-x[2]/P_M)) / (Yxs*(K_M+x[1]+x[1]**2/K_I))  +  (mu_max0*x[0]*x[1]*(1-x[2]/P_M)*(2*x[1]/K_I+1)) / (Yxs*(K_M+x[1]+x[1]**2/K_I)**2) - u[0]
    df2dx3= (mu_max0*K_I*x[0]*x[1]) / (P_M*K_M*K_I*Yxs + P_M*K_I*Yxs*x[1] + P_M*Yxs*x[1]**2)

    df3dx1= (mu_max0*alpha*x[1]*(1-x[2]/P_M)) / (K_M+x[1]+x[1]**2/K_I) + beta
    df3dx2= (mu_max0*K_I*alpha*x[0]*(P_M-x[2])*(K_M*K_I-x[1]**2)) / (P_M*(K_M*K_I+x[1]*(K_I+x[1]))**2)
    df3dx3= -(mu_max0*alpha*x[0]*x[1]) / (P_M*(K_M+x[1]+x[1]**2/K_I)) - u[0]

    return horzcat(vertcat(df1dx1,-df2dx1,df3dx1),vertcat(df1dx2,df2dx2,df3dx2),vertcat(-df1dx3,df2dx3,df3dx3))
FJ2 = fjac([1,6,0],[0.15])

Fk = I(x0=x, p=.15)
#Fk = I(x0=E, p=Uk)
Xk_end = Fk['xf']
print('Xk_end = ', Xk_end)
#J = J + Fk['qf']


print(FJ1[0:nx,0:nx], FJ2)


tgrid = []
tgrid += [0]

D = []
a = []
b = []
c = []
u0 = []

D = [1,0,0]
a += [D[0]]
b += [D[1]]
c += [D[2]]
for i in range(1000):
    u = (np.random.normal(.0,.2))
    Fk = I(x0=D, p=.2)
    D = Fk['xf']# + u
    a += [D[0]]
    b += [D[1]]
    c += [D[2]]
    u0 += [.15]
    tgrid += [i*Delta]

print(a[-1],b[-1],c[-1])

u0+=[0]
plt.figure()
plt.clf()
plt.plot(tgrid, a, '-')
plt.plot(tgrid, b, '--')
plt.plot(tgrid, c, ':')
plt.step(tgrid, u0, '-')

plt.xlabel('Time [hr]')
plt.ylabel('Concentration [g/L]')
#plt.title('y0')
plt.legend(['x1','x2','x3'])
plt.grid()

plt.show()
