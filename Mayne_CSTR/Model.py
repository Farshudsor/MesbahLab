n_sim = 420
n_pred = 140
n_st = 2
n_ip = 1

xk = SX.sym('xk',n_st)
uk = SX.sym('uk',n_ip)
wk = SX.sym('wk',1)


k = 300
theta = 20
xf = .3947
xc = .3816
M = 5
alpha = .117
x0 = [0.9831,0.3918]

dx1= (1/theta)*(1-xk[0])- k*xk[0]*exp(-M/xk[1])
dx2=(1/theta)*(xf-xk[1])- k*xk[0]*exp(-M/xk[1]) - alpha*uk*(xk[1]-xc) + w

uk_lb = [0];        uk_ub = [2]
xk_lb = [0, 0 ];    xk_ub = [1, 1]

v=[.12, 1.99]

# Cost Function
L = .5*(dot(xk,xk))**.5 + .5*(dot(uk,uk))

# generate noise
A = np.random.uniform(0,.001,n_sim)
omega = np.random.uniform(0,1,n_sim)
wkp = A*sin(omega)
