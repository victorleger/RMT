import numpy as np
from scipy.optimize import fsolve
from numpy.linalg import inv
import scipy.stats as stat

def test_perf(p,nl,nu,rho_l,rho_u,beta,sim,Tt,norm=True,naive=True,opt=True,opt_alpha=True,alpha_tilde=2,alpha_min=1,alpha_max=10):
    """
    Return classification error for naive scores, optimal scores, and optimal scores/alpha
    naive : If True, return error for naive scores
    opt : If True, return error for optimal scores
    opt_alpha : If True, return error for optimal scores/alpha
    """
    k = 2
    cl = p/nl
    cu = p/nu
    y_naive = np.array([-1,1,-1,1])

    Lambda = np.ones((k,k))
    Lambda[0,1] = beta
    Lambda[1,0] = beta
    A = np.kron(Lambda,np.eye(p))

    MU = np.zeros((k,2,p))
    mu = np.random.multivariate_normal(np.zeros(p),np.eye(p))/np.sqrt(p)
    mu_ortho = np.random.multivariate_normal(np.zeros(p),np.eye(p))/np.sqrt(p)
    MU[0,0] = -mu
    MU[0,1] = mu
    MU[1,0] = -(sim*mu + np.sqrt(1-sim**2)*mu_ortho)
    MU[1,1] = sim*mu + np.sqrt(1-sim**2)*mu_ortho

    MUc = MU.copy()
    for  t in range(k):
        for j in range(2):
            MUc[t,j] -= ((nu*rho_u[t][0]+nl*rho_l[t][0])*MU[t,0] + (nu*rho_u[t][1]+nl*rho_l[t][1])*MU[t,1])/np.sum(nu*rho_u[t]+nl*rho_l[t])
    M = np.array([MUc[0,0],MUc[0,1],MUc[1,0],MUc[1,1]]).T
    M_cal = M.T@M

    Zl, Zu = create_data(p,nl,nu,k,rho_l,rho_u,MU)
    N = np.linalg.norm(Zu.T@A@Zu,ord=2)/(k*p)

    alpha = alpha_tilde*N
    Lambda_tilde = Lambda/alpha

    if naive:
        m_naive, sigma_naive = stats(Lambda_tilde,cl,cu,k,rho_l,rho_u,M_cal,y_naive)
        e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th = perf(p,k,nl,nu,rho_l,rho_u,Zl,Zu,y_naive,Lambda_tilde,Tt,0,m_naive,sigma_naive)
    else:
        e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th = 0,0,0,0

    if opt:
        y_opt = scores_opti(Lambda_tilde,cl,cu,k,rho_l,rho_u,M_cal,y_naive)
        m_opt, sigma_opt = stats(Lambda_tilde,cl,cu,k,rho_l,rho_u,M_cal,y_opt)
        e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th = perf(p,k,nl,nu,rho_l,rho_u,Zl,Zu,y_opt,Lambda_tilde,Tt,(m_opt[2*Tt]+m_opt[2*Tt+1])/2,m_opt,sigma_opt)
    else:
        e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th = 0,0,0,0

    if opt_alpha:
        alpha = alpha_opti(N,Lambda,cl,cu,k,rho_l,rho_u,M_cal,y_naive,Tt,alpha_min,alpha_max)
        Lambda_tilde = Lambda/alpha
        y_opt = scores_opti(Lambda_tilde,cl,cu,k,rho_l,rho_u,M_cal,y_naive)
        m_opt, sigma_opt = stats(Lambda_tilde,cl,cu,k,rho_l,rho_u,M_cal,y_opt)
        e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th = perf(p,k,nl,nu,rho_l,rho_u,Zl,Zu,y_opt,Lambda_tilde,Tt,(m_opt[2*Tt]+m_opt[2*Tt+1])/2,m_opt,sigma_opt)
        alpha_real = alpha/N
    else:
        e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th = 0,0,0,0
        alpha_real = 0

    return e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th, e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th, e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th, alpha_real

def create_data(p,nl,nu,k,rho_l,rho_u,MU):
    """
    Generate data matrices Zl and Zu for a given setting
    """
    Xl = np.zeros((p,nl))
    i1 = 0
    i2 = 0
    for t in range(k):
        for j in range(2):
            i2 += int(nl*rho_l[t,j])
            Xl[:,i1:i2] = np.random.multivariate_normal(MU[t,j],np.eye(p),size=int(nl*rho_l[t,j])).T
            i1 = i2

    Xu = np.zeros((p,nu))
    i1 = 0
    i2 = 0
    for t in range(k):
        for j in range(2):
            i2 += int(nu*rho_u[t,j])
            Xu[:,i1:i2] = np.random.multivariate_normal(MU[t,j],np.eye(p),size=int(nu*rho_u[t,j])).T
            i1 = i2

    il1 = 0
    il2 = 0
    iu1 = 0
    iu2 = 0
    for t in range(k):
        nlt = int(np.sum(nl*rho_l[t]))
        nut = int(np.sum(nu*rho_u[t]))
        il2 += nlt
        iu2 += nut
        Pl = (Xl[:,il1:il2]@np.ones((nlt,nlt))+Xu[:,iu1:iu2]@np.ones((nut,nlt))).copy()/(nlt+nut)
        Pu = (Xl[:,il1:il2]@np.ones((nlt,nut))+Xu[:,iu1:iu2]@np.ones((nut,nut))).copy()/(nlt+nut)
        Xl[:,il1:il2] -= Pl
        Xu[:,iu1:iu2] -= Pu
        il1 = il2
        iu1 = iu2

    Zl = np.zeros((k*p,nl))
    i1 = 0
    i2 = 0
    for t in range(k):
        i2 += int(np.sum(nl*rho_l[t]))
        Zl[t*p:(t+1)*p,i1:i2] = Xl[:,i1:i2]
        i1 = i2

    Zu = np.zeros((k*p,nu))
    i1 = 0
    i2 = 0
    for t in range(k):
        i2 += int(np.sum(nu*rho_u[t]))
        Zu[t*p:(t+1)*p,i1:i2] = Xu[:,i1:i2]
        i1 = i2
    return Zl, Zu

def stats(Lambda_tilde,cl,cu,k,rho_l,rho_u,M_cal,y):
    """
    Computes statistics (mean and variance) of the decision function for a given choice of scores y and hyperparameter matrix Lambda_tilde
    """
    def f(x):
        return x - np.diag(Lambda_tilde+Lambda_tilde@inv(np.diag((k*cu*(1-x))/np.sum(rho_u,axis=1))-Lambda_tilde)@Lambda_tilde)/k

    delta = fsolve(f,[0.5,0.5])
    delta_bar = np.sum(rho_u,axis=1)/(k*cu*(1-delta))
    A_cal = Lambda_tilde+Lambda_tilde@inv(np.diag(1/delta_bar)-Lambda_tilde)@Lambda_tilde
    delta_tilde = np.repeat(1/(k*cu*(1-delta)),2)
    delta_tilde = delta_tilde*rho_u.flatten()

    D_tilde = np.diag(delta_tilde)
    D_rho_l = np.diag(rho_l.flatten())
    D_rho_u = np.diag(rho_u.flatten())

    Gamma = inv(np.eye(2*k)-D_tilde@(np.kron(A_cal,np.ones((2,2)))*M_cal))

    T_bar = A_cal**2/k
    d = np.sum(rho_u,axis=1)/(k*cu*(1-delta)**2)
    T = T_bar@inv(np.eye(k)-np.diag(d)@T_bar)

    Diag_tilde = np.kron(np.eye(k),np.ones(2)).T
    for t in range(k):
        Diag_tilde[t] += d*T[t]

    V = np.zeros((k,k,k))
    for t in range(k):
        V[t] = A_cal@np.diag(Diag_tilde[t])@A_cal

    H = np.zeros((2,2*k,2*k))
    for t in range(k):
        H[t] = Gamma.T@(np.kron(V[t],np.ones((2,2)))*M_cal)@Gamma + k*cu*(Gamma.T-np.eye(2*k))@np.diag(np.repeat(T[t],2)/rho_u.flatten())@(Gamma.T-np.eye(2*k))

    m = cu/cl*inv(D_rho_u)@(Gamma-np.eye(2*k))@D_rho_l@y

    sigma = np.zeros(k)
    for t in range(k):
        sigma[t] = 1/(k*cl*(1-delta[t]))**2*np.dot(H[t]@D_rho_l@y,D_rho_l@y) + 1/(k*cl*(1-delta[t])**2)*np.dot(D_rho_l@np.diag(np.repeat(T[t],2))@y,y)
    sigma = np.sqrt(sigma)
    return m, sigma

def scores_opti(Lambda_tilde,cl,cu,k,rho_l,rho_u,M_cal,y):
    """
    Returns optimal scores for a given hyperparameter matrix Lambda_tilde
    """
    def f(x):
        return x - np.diag(Lambda_tilde+Lambda_tilde@inv(np.diag((k*cu*(1-x))/np.sum(rho_u,axis=1))-Lambda_tilde)@Lambda_tilde)/k

    delta = fsolve(f,[0.5,0.5])
    delta_bar = np.sum(rho_u,axis=1)/(k*cu*(1-delta))
    A_cal = Lambda_tilde+Lambda_tilde@inv(np.diag(1/delta_bar)-Lambda_tilde)@Lambda_tilde
    delta_tilde = np.repeat(1/(k*cu*(1-delta)),2)
    delta_tilde = delta_tilde*rho_u.flatten()

    D_tilde = np.diag(delta_tilde)
    D_rho_l = np.diag(rho_l.flatten())
    D_rho_u = np.diag(rho_u.flatten())

    Gamma = inv(np.eye(2*k)-D_tilde@(np.kron(A_cal,np.ones((2,2)))*M_cal))

    T_bar = A_cal**2/k
    d = np.sum(rho_u,axis=1)/(k*cu*(1-delta)**2)
    T = T_bar@inv(np.eye(k)-np.diag(d)@T_bar)

    Diag_tilde = np.kron(np.eye(k),np.ones(2)).T
    for t in range(k):
        Diag_tilde[t] += d*T[t]

    V = np.zeros((k,k,k))
    for t in range(k):
        V[t] = A_cal@np.diag(Diag_tilde[t])@A_cal

    H = np.zeros((2,2*k,2*k))
    for t in range(k):
        H[t] = Gamma.T@(np.kron(V[t],np.ones((2,2)))*M_cal)@Gamma + k*cu*(Gamma.T-np.eye(2*k))@np.diag(np.repeat(T[t],2)/rho_u.flatten())@(Gamma.T-np.eye(2*k))

    y_opt = np.zeros(2*k)
    y_opt[2*Tt] = -1
    y_opt[2*Tt+1] = 1
    y_opt = inv(H[Tt]@D_rho_l/(k*cl)+np.diag(np.repeat(T[Tt],2)))@(Gamma.T-np.eye(2*k))@inv(D_rho_u)@y_opt
    return y_opt

def alpha_opti(N,Lambda,cl,cu,k,rho_l,rho_u,M_cal,y,Tt,alpha_min,alpha_max):
    """
    Returns optimal alpha for a given setting
    """
    ALPHA = N*np.exp(np.linspace(np.log(alpha_min),np.log(alpha_max),101))[1::]
    E = []
    for alpha in ALPHA:
        Lambda_tilde = Lambda/alpha
        m, sigma = stats(Lambda_tilde,cl,cu,k,rho_l,rho_u,M_cal,y)
        if (sigma>0).all():
            E.append(stat.norm.sf((m[2*Tt+1]-m[2*Tt])/2,0,sigma[Tt]))
        else:
            E.append(1)
    alpha = ALPHA[np.argmin(E)]
    return alpha

def perf(p,k,nl,nu,rho_l,rho_u,Zl,Zu,y,Lambda_tilde,Tt,threshold,m,sigma):
    """
    Computes theoretical and empirical error of the method for a given choice of scores y and hyperparameter matrix Lambda_tilde
    """
    A = np.kron(Lambda_tilde,np.eye(p))
    Q_prime = inv(np.eye(nu)-(Zu.T@A@Zu)/(k*p))
    Yl = np.zeros(np.sum(nl))
    i1 = 0
    i2 = 0
    for t in range(k):
        for j in range(2):
            i2 += int(nl*rho_l[t,j])
            Yl[i1:i2] = y[2*t+j]
            i1 = i2

    Fu = (Q_prime@Zu.T@A@Zl@Yl)/(k*p)

    n_inf = 0
    for i in range(Tt):
        n_inf += int(nu*rho_u[i,0])+int(nu*rho_u[i,1])

    Y1 = Fu[n_inf:n_inf+int(nu*rho_u[Tt,0])]
    Y2 = Fu[n_inf+int(nu*rho_u[Tt,0]):n_inf+int(nu*rho_u[Tt,0])+int(nu*rho_u[Tt,1])]
    e1_emp = np.mean(Y1>threshold)
    e2_emp = np.mean(Y2<threshold)
    e1_th = np.mean(1-stat.norm.cdf(threshold,m[2*Tt],sigma[Tt]))
    e2_th = np.mean(stat.norm.cdf(threshold,m[2*Tt+1],sigma[Tt]))
    return e1_emp, e2_emp, e1_th, e2_th