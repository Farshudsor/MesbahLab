
#not write bytecode to maintain clean directories
import sys
sys.dont_write_bytecode = True

# Imports required packages.
sys.path.append(r"/home/fsorourifar/casadi-py27-v3.4.4")
from casadi import *
import numpy as NP
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.ticker import MaxNLocator
from scipy import linalg
from scipy import signal
from numpy import random

# Import core code.
import core


### Problem parameters
# system size
nx = 3
nu = 1
ny = 3
nw = 3
nv = ny

# simulation time
Nsim = 120       # number of simulation time steps
Delta = 0.5      # sampling time, hr

# parameters for MPC
N = 10           # prediction horizon

# declare casadi variable graph
x = MX.sym('x',nx) #symbolic vector [1x3]
u = MX.sym('u',nu) #symbolic scalar [1x1]
w = MX.sym('w',nw) #symbolic vector [1x3]
						#FS. why not declare variables y & v
						#Why MX.sym() over SX.sym() -- Ans: MX is more economical (see 2x2 example- must perform 8 operations vs. 2 operations)


####### NL SYSTEM DYNAMICS
# States, inputs, and disturbances are physically:
#x[0] = X       # g/L           # biomass concentration
#x[1] = S       # g/L           # substrate concentration
#x[2] = P       # g/L           # product concentration
#u[0] = D       # 1/hr          # dilution rate
										#FS. this means x is a concentration vector, u is a rate

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
dx1 = -u[0]*x[0] + mu*x[0] + w[0]
dx2 = u[0]*(S_f-x[1]) - 1/Yxs*mu*x[0] + w[1]
dx3 = -u[0]*x[2] + (alpha*mu+beta)*x[0] + w[2]
							#FS. differentials w.r.t variable x,
							# is dx1, dx2, and dx3 the first, second and third derivatives of variable x[:]
							# or is dx1 the dif.eq to x[0]; dx2 to x[1]; dx3 to x[2]

# concatenate differential equations
xdot = vertcat(dx1,dx2,dx3)
# create casadi function
f = Function("f", [x,u,w], [xdot])
				#FS.
####################################

# measurement matrix (linear for now)
C = NP.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# steady state values
uss = [0.15]
wss = [0.0, 0.0, 0.0]
xss = core.rootFinder(f, nx, args=[uss,wss], lbx=[7.0, 2.0, 24.0], ubx=100.0)

# get linear dynamics around ss (in deviation variables)
lin = core.getLinearizedModel(f, [xss,uss,wss], ["A","B","G"], Delta=Delta)
A = lin["A"]; B = lin["B"]; G = 0.5*NP.eye(nw)#G = lin["G"]

# specify number of artificial disturbanes + matrices
nd = ny
Bd = NP.zeros((nx,nd))
Cd = NP.eye(nd)

# controlled variable matrix, z=Hy (nz <= nu)
H = NP.array([[0.0, 0.0, 1.0]])

# setpoint on controlled variables (in deviation variables)
ztar = NP.array([0.0])

# get observer gain for augmented system [x ; d]
At = NP.zeros((nx+nd,nx+nd)); At[0:nx,0:nx] = A; At[0:nx,nx:] = Bd; At[nx:,nx:] = NP.eye(nd)
Bt = NP.zeros((nx+nd,nu)); Bt[0:nx,:] = B
Ct = NP.concatenate((C,Cd),axis=1)
Lt, Pt = core.dlqe(At,Ct,NP.eye(nx+nd),0.1*NP.eye(ny))
tarMat = NP.zeros((nx+nu,nx+nu)); tarMat[0:nx,0:nx] = NP.eye(nx)-A; tarMat[0:nx,nx:] = -B; tarMat[nx:,0:nx] = H.dot(C)

# mpc weights
# Qx = NP.array([[1.0/xss[0]**2, 0, 0], [0, 1.0/xss[1]**2, 0], [0, 0, 1.0/xss[2]**2]])
Qz = NP.array([[1.0/xss[2]**2]])
Ru = NP.array([[0.1/uss[0]**2]])
Qx = core.mtimes(C.T,H.T,Qz,H,C)

# stage cost (in deviation variables)
Lstage = mtimes(x.T,Qx,x) + mtimes(u.T,Ru,u)

# create casadi graph of deviation variables to dynamics and stage cost
MPC_dynamics = Function("MPC_dynamics", [x,u], [core.mtimes(A,x) + core.mtimes(B,u), Lstage])

# bounds on the states
X_lb = -inf;     X_ub = 7.2
S_lb = -inf;     S_ub = inf
P_lb = -inf;     P_ub = inf
x_lb = [X_lb, S_lb, P_lb]
x_ub = [X_ub, S_ub, P_ub]

# bounds on the control inputs
D_lb = 0.0;     D_ub = 1.0
u_lb = [D_lb]
u_ub = [D_ub]
u0 = [0.5] # initial guess for inputs

# initial condition for states (in deviation variables)
# x0 = [xss[i]-xss[i] for i in range(nx)]
xss_new = core.rootFinder(f, nx, args=[0.2,wss], lbx=[1.0, 0.1, 5.0], ubx=10000.0)
x0 = [xss_new[i]-xss[i] for i in range(nx)]

# noise covariances
# Qw = 1*NP.diag([0.0001, 0.001, 0.01])
# Rv = 1*NP.diag([0.001, 0.002, 0.003])
Qw = 0.0005*NP.diag([xss[0], xss[1], xss[2]])
Rv = 0.0001*NP.diag([xss[0], xss[1], xss[2]])
if NP.trace(Qw) > 0:
    Qw_half = linalg.cholesky(Qw,lower=True)
else:
    Qw_half = Qw
if NP.trace(Rv) > 0:
    Rv_half = linalg.cholesky(Rv,lower=True)
else:
    Rv_half = Rv

# pick solver
# MySolver = "ipopt"
# MySolver = "worhp"
MySolver = "sqpmethod"

# define integrator for simulation
ode = {'x':x, 'p':vertcat(u,w), 'ode':xdot}
opts = {'abstol':1e-10, 'reltol':1e-10, 'tf':Delta}
I = integrator('I', 'cvodes', ode, opts)

# number of monte carlo runs
Nmc = 5

# save figures (0=NO, 1=YES)
save_fig = 0

# export to matlab (0=NO, 1=YES)
export_to_matlab = 0
export_name = 'MPC_data_v3'



### Build instance of MPC problem
# start with an empty NLP
q = []
q0 = []
lbq = []
ubq = []
J = 0
g = []
lbg = []
ubg = []
# "lift" initial conditions
X0 = MX.sym('X0', nx)
q += [X0]
lbq += [0]*nx
ubq += [0]*nx
q0 += [0]*nx
# formulate the QP
Xk = X0
for k in range(N):
    # new NLP variable for the control
    Uk = MX.sym('U_' + str(k), nu)
    q   += [Uk]
    lbq += [u_lb[i]-uss[i] for i in range(nu)]
    ubq += [u_ub[i]-uss[i] for i in range(nu)]
    q0  += [0]*nu
    # next step dynamics and stage cost
    Fk, Lk = MPC_dynamics(Xk,Uk)
    Xk_end = Fk
    J = J + Lk
    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k+1), nx)
    q   += [Xk]
    lbq += [x_lb[i]-xss[i] for i in range(nx)]
    ubq += [x_ub[i]-xss[i] for i in range(nx)]
    q0  += [0]*nx
    # Add equality constraint
    g   += [Xk_end-Xk]
    lbg += [0]*nx
    ubg += [0]*nx
# set solver options
opts = {}
if MySolver == "sqpmethod":
  opts["qpsol"] = "qpoases"
  opts["qpsol_options"] = {"printLevel":"none"}
# create NLP solver for MPC problem
prob = {'f':J, 'x':vertcat(*q), 'g':vertcat(*g)}
solver = nlpsol('solver', MySolver, prob, opts)

### Now simulate.
X = []
U = []
W = []
V = []
# load data to keep consistent noise realizations
# matFile = scipy.io.loadmat('SMPC_data_v2.mat',squeeze_me=True)
# W = matFile["disturbances"]
# V = matFile["noise"]
# time over simulation
Tgrid = [Delta*k for k in range(Nsim+1)]
for i_mc in range(Nmc):
    # get noise realization over simulation
    W += [Qw_half.dot(random.randn(nw,Nsim)).T]
    V += [Rv_half.dot(random.randn(nv,Nsim+1)).T]
    # preallocate (in deviation variables)
    X += [NP.zeros((Nsim+1,nx))]
    Y = NP.zeros((Nsim+1,ny))
    U += [NP.zeros((Nsim,nu))]
    Xhat = NP.zeros((Nsim+1,nx))
    Dhat = NP.zeros((Nsim+1,nd))
    Xtar = NP.zeros((Nsim,nx))
    Utar = NP.zeros((Nsim,nu))
    # initialize
    X[i_mc][0,:] = NP.array(x0)
    Xhat[0,:] = NP.array(x0)
    for k in range(Nsim):

        if k > 50:
            ### change target
            ztar_k = ztar + NP.array([4.0])
        else:
            ztar_k = ztar
        # if k > 1000:
        #     ### add deterministic step change to process noise
        #     W[i_mc][k,:] += NP.array([0.0])

        # target calculator
        #btar = NP.concatenate((Bd.dot(Dhat[k,:].T),-H.dot(Cd).dot(Dhat[k,:].T)+ztar_k),axis=1)
        btar = NP.concatenate((Bd.dot(Dhat[k,:].T),-H.dot(Cd).dot(Dhat[k,:].T)+ztar_k),axis=0)
        tarSol = NP.linalg.solve(tarMat, btar)
        Xtar[k,:] = tarSol[0:nx]
        Utar[k,:] = tarSol[nx:]

        ### update mpc problem data
            # Follows Rawlings "Disturbance Models for Offset-Free Model-Predictive Control"
            # regulator problem 21 where states are defined as deviation from target
        # initial condition
        q0[0:nx] = Xhat[k,:]-Xtar[k,:]
        lbq[0:nx] = q0[0:nx]


        ubq[0:nx] = q0[0:nx]
        # bounds
        u_lb_update = [u_lb[i]-uss[i]-Utar[k,i] for i in range(nu)]
        u_ub_update = [u_ub[i]-uss[i]-Utar[k,i] for i in range(nu)]
        # ONLY WORKS WHEN ALL STATES ARE MEASURED AND C=I
        x_lb_update = [x_lb[i]-xss[i]-Xtar[k,i]-Dhat[k,i] for i in range(nx)]
        x_ub_update = [x_ub[i]-xss[i]-Xtar[k,i]-Dhat[k,i] for i in range(nx)]
        for i in range(N):
            lbq[nx+i*(nx+nu):nx+nu+i*(nx+nu)] = u_lb_update
            ubq[nx+i*(nx+nu):nx+nu+i*(nx+nu)] = u_ub_update
            lbq[nx+nu+i*(nx+nu):2*nx+nu+i*(nx+nu)] = x_lb_update
            ubq[nx+nu+i*(nx+nu):2*nx+nu+i*(nx+nu)] = x_ub_update

        # solve MPC OCP
        sol = solver(x0=q0, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
        q_opt = sol['x'].full().flatten()
        u_opt = q_opt[nx::nx+nu]
        U[i_mc][k,:] = u_opt[0:nu] + Utar[k,:]

        # give first input to true system and simulate one step forward
        # res = I(x0=X[i_mc][k,:]+NP.array(xss), p=vertcat(U[i_mc][k,:]+NP.array(uss), W[i_mc][k,:]+NP.array(wss)))
        # X[i_mc][k+1,:] = NP.squeeze(res['xf']) - NP.array(xss)
        res = I(x0=X[i_mc][k,:]+NP.array(xss), p=vertcat(U[i_mc][k,:]+NP.array(uss), W[i_mc][k,:]+NP.array(wss)))
        X[i_mc][k+1,:] =  A.dot(X[i_mc][k,:]) + B.dot(U[i_mc][k,:]) + G.dot(W[i_mc][k,:])

        # get measurements
        Y[k+1,:] = C.dot(X[i_mc][k+1,:]) + V[i_mc][k+1,:]

        # update estimator
        stack = NP.concatenate((Xhat[k,:],Dhat[k,:]),axis=0).T
        pred = At.dot(stack) + Bt.dot(U[i_mc][k,:].T)
        err = Y[k+1,:] - Ct.dot(pred).T
        update = pred + Lt.dot(err)
        Xhat[k+1,:] = update[0:nx]
        Dhat[k+1,:] = update[nx:]

    # convert back from deviation to true values
    X[i_mc][0,:] += NP.array(xss)
    for k in range(Nsim):
        U[i_mc][k,:] += NP.array(uss)
        W[i_mc][k,:] += NP.array(wss)
        X[i_mc][k+1,:] += NP.array(xss)

print 'uss:', uss
print 'xss:', xss
print 'xss_new', xss_new
Pf_mean = 0.0
for i in range(Nmc):
    Pf_mean += X[i][-1,-1]
Pf_mean = Pf_mean/Nmc
print 'Pf_mean:', Pf_mean
print 'A:', A
print 'B:', B
print 'G:', G
print 'C:', C
print 'H:', H
print 'Qw:', Qw
print 'Rv:', Rv
print 'Qx:', Qx
print 'Ru:', Ru
print 'x0 (deviation):', x0

### PLOTS
# states versus time
plt.ion()
plt.figure(1)
for i in range(nx):
    plot = plt.subplot(nx, 1, i+1)
    for k in range(Nmc):
        plt.plot(Tgrid, X[k][:,i])
    plt.plot(Tgrid, x_ub[i]*NP.ones(Nsim+1), linestyle='--', color='k')
    plt.ylabel('x_'+str(i+1))
    plt.xlabel("Time")
    plt.grid()
    plot.yaxis.set_major_locator(MaxNLocator(4))
plt.show()

# control inputs versus time
plt.ion()
plt.figure(2)
for i in range(nu):
    plot = plt.subplot(nu, 1, i+1)
    for k in range(Nmc):
        plt.step(Tgrid, vertcat(DM.nan(1), U[k][:,i]))
    plt.ylabel('u_'+str(i+1))
    plt.xlabel("Time")
    plt.grid()
    plot.yaxis.set_major_locator(MaxNLocator(4))
plt.show()

# disturbances versus time
plt.ion()
plt.figure(3)
for i in range(nw):
    plot = plt.subplot(nw, 1, i+1)
    for k in range(Nmc):
        plt.step(Tgrid, vertcat(DM.nan(1), W[k][:,i]))
    plt.ylabel('w_'+str(i+1))
    plt.xlabel("Time")
    plt.grid()
    plot.yaxis.set_major_locator(MaxNLocator(4))
plt.show()

if save_fig == 1:
    plt.figure(1)
    plt.savefig("states_versus_time_MC.pdf", bbox_inches='tight')
    plt.figure(2)
    plt.savefig("inputs_versus_time_MC.pdf", bbox_inches='tight')
    plt.figure(3)
    plt.savefig("add_disturbances_versus_time_MC.pdf", bbox_inches='tight')

if export_to_matlab == 1:
    export_dict = {
    "states":X,
    "inputs":U,
    "disturbances":W,
    "noise":V,
    "time": Tgrid,
    }
    scipy.io.savemat(export_name, mdict=export_dict)

raw_input("Press Enter to exit...") # pauses script so that figures can be viewed
