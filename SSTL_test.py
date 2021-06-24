import matplotlib.pyplot as plt

plt.ion()
plt.show()

# rho_l : (k,2)-array, proportion of data in each task and class for unlabelled data, total sum must be 1
# rho_u : (k,2)-array, proportion of data in each task and class for labelled data, total sum must be 1
# beta : hyperparameter controlling correlation between tasks
# sim : similarity between task 1 and 2 when generating data
# alpha_tilde = alpha / ||Wuu||: hyperparameter controlling the balance between supervised and unsupervised learning
#       1 < alpha_tilde < +\infty
# alpha_min/alpha_max : range for the search of optimal alpha_tilde
# Tt : Target task
# nb_iter : Number of realisations to plot the graphs

## Influence of nu

p = 100
nl = 100
rho_l = np.array([[0.25,0.25],[0.25,0.25]])
rho_u = np.array([[0.25,0.25],[0.25,0.25]])
beta = 0.5
sim = 1
alpha_tilde = 2
alpha_min = 1
alpha_max = 10
Tt = 0

nb_iter = 100

NU = [100,200,300,400,500,600,700,800,900,1000]
n = len(NU)

E_naive_emp, E_naive_th, E_opt_emp, E_opt_th, E_alpha_emp, E_alpha_th = np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter))

for i in range(n):
    for j in range(nb_iter):
        e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th, e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th, e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th, alpha_real = test_perf(p,nl,NU[i],rho_l,rho_u,beta,sim,Tt,alpha_tilde=alpha_tilde,alpha_min=alpha_min,alpha_max=alpha_max)
        E_naive_emp[i,j] = (e1_naive_emp+e2_naive_emp)/2
        E_naive_th[i,j] = (e1_naive_th+e2_naive_th)/2
        E_opt_emp[i,j] = (e1_opt_emp+e2_opt_emp)/2
        E_opt_th[i,j] = (e1_opt_th+e2_opt_th)/2
        E_alpha_emp[i,j] = (e1_alpha_emp+e2_alpha_emp)/2
        E_alpha_th[i,j] = (e1_alpha_th+e2_alpha_th)/2

m_naive_emp = np.mean(E_naive_emp,axis=1)
s_naive_emp = np.sqrt(np.var(E_naive_emp,axis=1))
m_naive_th = np.mean(E_naive_th,axis=1)
s_naive_th = np.sqrt(np.var(E_naive_th,axis=1))
m_opt_emp = np.mean(E_opt_emp,axis=1)
s_opt_emp = np.sqrt(np.var(E_opt_emp,axis=1))
m_opt_th = np.mean(E_opt_th,axis=1)
s_opt_th = np.sqrt(np.var(E_opt_th,axis=1))
m_alpha_emp = np.mean(E_alpha_emp,axis=1)
s_alpha_emp = np.sqrt(np.var(E_alpha_emp,axis=1))
m_alpha_th = np.mean(E_alpha_th,axis=1)
s_alpha_th = np.sqrt(np.var(E_alpha_th,axis=1))

f = stat.norm.isf(0.01,0,1)/np.sqrt(nb_iter)

plt.figure(1)
plt.clf()
plt.title('alpha_tilde = {}, alpha_min = {}'.format(alpha_tilde,alpha_min))
plt.plot(NU, m_naive_emp, 'o-', label='Binary labels (emp)', color='b')
plt.plot(NU, m_naive_th, label='Binary labels (th)', color='b')
plt.fill_between(NU, m_naive_emp-f*s_naive_emp, m_naive_emp+f*s_naive_emp, label='99% confidence interval', color='b', alpha=0.1)
plt.plot(NU, m_opt_emp, 'o-', label='Optimal labels (emp)', color='r')
plt.plot(NU, m_opt_th, label='Optimal labels (th)', color='r')
plt.fill_between(NU, m_opt_emp-f*s_opt_emp, m_opt_emp+f*s_opt_emp, label='99% confidence interval', color='r', alpha=0.1)
plt.plot(NU, m_alpha_emp, 'o-', label='Optimal (emp)', color='g')
plt.plot(NU, m_alpha_th, label='Optimal (th)', color='g')
plt.fill_between(NU, m_alpha_emp-f*s_alpha_emp, m_alpha_emp+f*s_alpha_emp, label='99% confidence interval', color='g', alpha=0.1)
plt.xlabel('nu')
plt.legend()

## Influence of nl

p = 100
nu = 100
rho_l = np.array([[0.25,0.25],[0.25,0.25]])
rho_u = np.array([[0.25,0.25],[0.25,0.25]])
beta = 0.5
sim = 1
alpha_tilde = 2
alpha_min = 1
alpha_max = 10
Tt = 0

nb_iter = 100

NL = [100,200,300,400,500,600,700,800,900,1000]
n = len(NL)

E_naive_emp, E_naive_th, E_opt_emp, E_opt_th, E_alpha_emp, E_alpha_th = np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter))

for i in range(n):
    for j in range(nb_iter):
        e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th, e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th, e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th, alpha_real = test_perf(p,NL[i],nu,rho_l,rho_u,beta,sim,Tt,alpha_tilde=alpha_tilde,alpha_min=alpha_min,alpha_max=alpha_max)
        E_naive_emp[i,j] = (e1_naive_emp+e2_naive_emp)/2
        E_naive_th[i,j] = (e1_naive_th+e2_naive_th)/2
        E_opt_emp[i,j] = (e1_opt_emp+e2_opt_emp)/2
        E_opt_th[i,j] = (e1_opt_th+e2_opt_th)/2
        E_alpha_emp[i,j] = (e1_alpha_emp+e2_alpha_emp)/2
        E_alpha_th[i,j] = (e1_alpha_th+e2_alpha_th)/2

m_naive_emp = np.mean(E_naive_emp,axis=1)
s_naive_emp = np.sqrt(np.var(E_naive_emp,axis=1))
m_naive_th = np.mean(E_naive_th,axis=1)
s_naive_th = np.sqrt(np.var(E_naive_th,axis=1))
m_opt_emp = np.mean(E_opt_emp,axis=1)
s_opt_emp = np.sqrt(np.var(E_opt_emp,axis=1))
m_opt_th = np.mean(E_opt_th,axis=1)
s_opt_th = np.sqrt(np.var(E_opt_th,axis=1))
m_alpha_emp = np.mean(E_alpha_emp,axis=1)
s_alpha_emp = np.sqrt(np.var(E_alpha_emp,axis=1))
m_alpha_th = np.mean(E_alpha_th,axis=1)
s_alpha_th = np.sqrt(np.var(E_alpha_th,axis=1))

f = stat.norm.isf(0.01,0,1)/np.sqrt(nb_iter)

plt.figure(2)
plt.clf()
plt.title('alpha_tilde = {}, alpha_min = {}'.format(alpha_tilde,alpha_min))
plt.plot(NL, m_naive_emp, 'o-', label='Binary labels (emp)', color='b')
plt.plot(NL, m_naive_th, label='Binary labels (th)', color='b')
plt.fill_between(NL, m_naive_emp-f*s_naive_emp, m_naive_emp+f*s_naive_emp, label='99% confidence interval', color='b', alpha=0.1)
plt.plot(NL, m_opt_emp, 'o-', label='Optimal labels (emp)', color='r')
plt.plot(NL, m_opt_th, label='Optimal labels (th)', color='r')
plt.fill_between(NL, m_opt_emp-f*s_opt_emp, m_opt_emp+f*s_opt_emp, label='99% confidence interval', color='r', alpha=0.1)
plt.plot(NL, m_alpha_emp, 'o-', label='Optimal (emp)', color='g')
plt.plot(NL, m_alpha_th, label='Optimal (th)', color='g')
plt.fill_between(NL, m_alpha_emp-f*s_alpha_emp, m_alpha_emp+f*s_alpha_emp, label='99% confidence interval', color='g', alpha=0.1)
plt.xlabel('nl')
plt.legend()

## Influence of alpha

p = 100
nl = 100
nu = 100
rho_l = np.array([[0.25,0.25],[0.25,0.25]])
rho_u = np.array([[0.25,0.25],[0.25,0.25]])
beta = 0.5
sim = 1
Tt = 0

nb_iter = 100

Alpha_tilde = np.exp(np.linspace(0,np.log(10),21))[1::]
n = len(Alpha_tilde)

E_naive_emp, E_naive_th, E_opt_emp, E_opt_th = np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter))

for i in range(n):
    for j in range(nb_iter):
        e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th, e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th, e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th, alpha_real = test_perf(p,nl,nu,rho_l,rho_u,beta,sim,Tt,opt_alpha=False,alpha_tilde=Alpha_tilde[i])
        E_naive_emp[i,j] = (e1_naive_emp+e2_naive_emp)/2
        E_naive_th[i,j] = (e1_naive_th+e2_naive_th)/2
        E_opt_emp[i,j] = (e1_opt_emp+e2_opt_emp)/2
        E_opt_th[i,j] = (e1_opt_th+e2_opt_th)/2

m_naive_emp = np.mean(E_naive_emp,axis=1)
s_naive_emp = np.sqrt(np.var(E_naive_emp,axis=1))
m_naive_th = np.mean(E_naive_th,axis=1)
s_naive_th = np.sqrt(np.var(E_naive_th,axis=1))
m_opt_emp = np.mean(E_opt_emp,axis=1)
s_opt_emp = np.sqrt(np.var(E_opt_emp,axis=1))
m_opt_th = np.mean(E_opt_th,axis=1)
s_opt_th = np.sqrt(np.var(E_opt_th,axis=1))
m_alpha_emp = np.mean(E_alpha_emp,axis=1)
s_alpha_emp = np.sqrt(np.var(E_alpha_emp,axis=1))
m_alpha_th = np.mean(E_alpha_th,axis=1)
s_alpha_th = np.sqrt(np.var(E_alpha_th,axis=1))

f = stat.norm.isf(0.01,0,1)/np.sqrt(nb_iter)

plt.figure(3)
plt.clf()
plt.title('nu = {}, nl = {}, p = {}'.format(nu,nl,p))
# plt.plot(Alpha_tilde, m_naive_emp, 'o-', label='Binary labels (emp)', color='b')
plt.plot(Alpha_tilde, m_naive_th, label='Binary labels (th)', color='b')
# plt.fill_between(Alpha_tilde, m_naive_emp-f*s_naive_emp, m_naive_emp+f*s_naive_emp, label='99% confidence interval', color='b', alpha=0.1)
# plt.plot(Alpha_tilde, m_opt_emp, 'o-', label='Optimal labels (emp)', color='r')
plt.plot(Alpha_tilde, m_opt_th, label='Optimal labels (th)', color='r')
# plt.fill_between(Alpha_tilde, m_opt_emp-f*s_opt_emp, m_opt_emp+f*s_opt_emp, label='99% confidence interval', color='r', alpha=0.1)
plt.xlabel('alpha_tilde')
plt.legend()

## Influence of beta

p = 100
nl = 100
nu = 100
rho_l = np.array([[0.25,0.25],[0.25,0.25]])
rho_u = np.array([[0.25,0.25],[0.25,0.25]])
sim = 0.5
alpha_tilde = 2
alpha_min = 1.5
alpha_max = 10
Tt = 0

nb_iter = 100

BETA = np.linspace(-1,1,21)-1e-5
n = len(BETA)

E_naive_emp, E_naive_th, E_opt_emp, E_opt_th, E_alpha_emp, E_alpha_th = np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter))

for i in range(n):
    for j in range(nb_iter):
        e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th, e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th, e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th, alpha_real = test_perf(p,nl,nu,rho_l,rho_u,BETA[i],sim,Tt,alpha_tilde=alpha_tilde,alpha_min=alpha_min,alpha_max=alpha_max)
        E_naive_emp[i,j] = (e1_naive_emp+e2_naive_emp)/2
        E_naive_th[i,j] = (e1_naive_th+e2_naive_th)/2
        E_opt_emp[i,j] = (e1_opt_emp+e2_opt_emp)/2
        E_opt_th[i,j] = (e1_opt_th+e2_opt_th)/2
        E_alpha_emp[i,j] = (e1_alpha_emp+e2_alpha_emp)/2
        E_alpha_th[i,j] = (e1_alpha_th+e2_alpha_th)/2

m_naive_emp = np.mean(E_naive_emp,axis=1)
s_naive_emp = np.sqrt(np.var(E_naive_emp,axis=1))
m_naive_th = np.mean(E_naive_th,axis=1)
s_naive_th = np.sqrt(np.var(E_naive_th,axis=1))
m_opt_emp = np.mean(E_opt_emp,axis=1)
s_opt_emp = np.sqrt(np.var(E_opt_emp,axis=1))
m_opt_th = np.mean(E_opt_th,axis=1)
s_opt_th = np.sqrt(np.var(E_opt_th,axis=1))
m_alpha_emp = np.mean(E_alpha_emp,axis=1)
s_alpha_emp = np.sqrt(np.var(E_alpha_emp,axis=1))
m_alpha_th = np.mean(E_alpha_th,axis=1)
s_alpha_th = np.sqrt(np.var(E_alpha_th,axis=1))

f = stat.norm.isf(0.01,0,1)/np.sqrt(nb_iter)

plt.figure(4)
plt.clf()
plt.title('sim = {}, alpha_tilde = {}'.format(sim,alpha_tilde))
plt.plot(BETA, m_naive_emp, 'o-', label='Binary labels (emp)', color='b')
plt.plot(BETA, m_naive_th, label='Binary labels (th)', color='b')
plt.fill_between(BETA, m_naive_emp-f*s_naive_emp, m_naive_emp+f*s_naive_emp, label='99% confidence interval', color='b', alpha=0.1)
plt.plot(BETA, m_opt_emp, 'o-', label='Optimal labels (emp)', color='r')
plt.plot(BETA, m_opt_th, label='Optimal labels (th)', color='r')
plt.fill_between(BETA, m_opt_emp-f*s_opt_emp, m_opt_emp+f*s_opt_emp, label='99% confidence interval', color='r', alpha=0.1)
plt.plot(BETA, m_alpha_emp, 'o-', label='Optimal (emp)', color='g')
plt.plot(BETA, m_alpha_th, label='Optimal (th)', color='g')
plt.fill_between(BETA, m_alpha_emp-f*s_alpha_emp, m_alpha_emp+f*s_alpha_emp, label='99% confidence interval', color='g', alpha=0.1)
plt.xlabel('Beta')
plt.legend()

## Optimal alpha as a function of nl

p = 100
nu = 100
rho_l = np.array([[0.25,0.25],[0.25,0.25]])
rho_u = np.array([[0.25,0.25],[0.25,0.25]])
beta = 0.5
sim = 1
alpha_min = 1
alpha_max = 10
Tt = 0

nb_iter = 100

NL = np.array([20,50,100,200,500])
n = len(NL)

Alpha_real = np.zeros((n,nb_iter))

for i in range(n):
    for j in range(nb_iter):
        e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th, e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th, e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th, alpha_real = test_perf(p,NL[i],nu,rho_l,rho_u,beta,sim,Tt,naive=False,opt=False,alpha_min=alpha_min,alpha_max=alpha_max)
        Alpha_real[i,j] = alpha_real

m_alpha_real = np.mean(Alpha_real,axis=1)

plt.figure(5)
plt.clf()
plt.title('p = {}, nu = {}'.format(p,nu))
plt.plot(NL, m_alpha_real, 'o-', label='Optimal (emp)', color='g')
plt.xlabel('nl')
plt.ylabel('Optimal alpha_tilde')
plt.legend()

## Optimal alpha as a function of nu

p = 100
nl = 100
rho_l = np.array([[0.25,0.25],[0.25,0.25]])
rho_u = np.array([[0.25,0.25],[0.25,0.25]])
beta = 0.5
sim = 1
alpha_min = 1
alpha_max = 10
Tt = 0

nb_iter = 100

NU = np.array([100,200,500,1000,2000])
n = len(NU)

Alpha_real = np.zeros((n,nb_iter))

for i in range(n):
    for j in range(nb_iter):
        e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th, e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th, e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th, alpha_real = test_perf(p,nl,NU[i],rho_l,rho_u,beta,sim,Tt,naive=False,opt=False,alpha_min=alpha_min,alpha_max=alpha_max)
        Alpha_real[i,j] = alpha_real

m_alpha_real = np.mean(Alpha_real,axis=1)

plt.figure(6)
plt.clf()
plt.title('p = {}, nl = {}'.format(p,nu))
plt.plot(NU, m_alpha_real, 'o-', label='Optimal (emp)', color='g')
plt.xlabel('nu')
plt.ylabel('Optimal alpha_tilde')
plt.legend()

## Influence of sim with beta = sim

p = 100
nl = 100
nu = 100
rho_l = np.array([[0.25,0.25],[0.25,0.25]])
rho_u = np.array([[0.25,0.25],[0.25,0.25]])
alpha_tilde = 2
alpha_min = 1.5
alpha_max = 10
Tt = 0

nb_iter = 100

BETA = np.linspace(-1,1,21)-1e-5
BETA[0] = -1+1e-5
n = len(BETA)

E_naive_emp, E_naive_th, E_opt_emp, E_opt_th, E_alpha_emp, E_alpha_th = np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter))

for i in range(n):
    for j in range(nb_iter):
        e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th, e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th, e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th, alpha_real = test_perf(p,nl,nu,rho_l,rho_u,BETA[i],BETA[i],Tt,alpha_tilde=alpha_tilde)
        E_naive_emp[i,j] = (e1_naive_emp+e2_naive_emp)/2
        E_naive_th[i,j] = (e1_naive_th+e2_naive_th)/2
        E_opt_emp[i,j] = (e1_opt_emp+e2_opt_emp)/2
        E_opt_th[i,j] = (e1_opt_th+e2_opt_th)/2
        E_alpha_emp[i,j] = (e1_alpha_emp+e2_alpha_emp)/2
        E_alpha_th[i,j] = (e1_alpha_th+e2_alpha_th)/2

m_naive_th = np.mean(E_naive_th,axis=1)
m_opt_th = np.mean(E_opt_th,axis=1)
m_alpha_th = np.mean(E_alpha_th,axis=1)

plt.figure(7)
plt.clf()
plt.title('alpha_tilde = {}'.format(alpha_tilde))
plt.plot(BETA, m_naive_th, 'o-', label='Binary labels', color='b')
plt.plot(BETA, m_opt_th, 'o-', label='Optimal labels', color='r')
plt.plot(BETA, m_alpha_th, 'o-', label='Optimal', color='g')
plt.xlabel('Beta = sim')
plt.legend()

## Influence of sim with constant beta

p = 100
nl = 100
nu = 100
rho_l = np.array([[0.25,0.25],[0.25,0.25]])
rho_u = np.array([[0.25,0.25],[0.25,0.25]])
alpha_tilde = 2
alpha_min = 1.5
alpha_max = 10
Tt = 0

nb_iter = 100

BETA = np.linspace(-1,1,21)-1e-5
BETA[0] = -1+1e-5
n = len(BETA)

E_naive_emp, E_naive_th, E_opt_emp, E_opt_th, E_alpha_emp, E_alpha_th = np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter)), np.zeros((n,nb_iter))

for i in range(n):
    for j in range(nb_iter):
        e1_naive_emp, e2_naive_emp, e1_naive_th, e2_naive_th, e1_opt_emp, e2_opt_emp, e1_opt_th, e2_opt_th, e1_alpha_emp, e2_alpha_emp, e1_alpha_th, e2_alpha_th, alpha_real = test_perf(p,nl,nu,rho_l,rho_u,1,BETA[i],Tt,alpha_tilde=alpha_tilde)
        E_naive_emp[i,j] = (e1_naive_emp+e2_naive_emp)/2
        E_naive_th[i,j] = (e1_naive_th+e2_naive_th)/2
        E_opt_emp[i,j] = (e1_opt_emp+e2_opt_emp)/2
        E_opt_th[i,j] = (e1_opt_th+e2_opt_th)/2
        E_alpha_emp[i,j] = (e1_alpha_emp+e2_alpha_emp)/2
        E_alpha_th[i,j] = (e1_alpha_th+e2_alpha_th)/2

m_naive_th = np.mean(E_naive_th,axis=1)
m_opt_th = np.mean(E_opt_th,axis=1)
m_alpha_th = np.mean(E_alpha_th,axis=1)

plt.figure(8)
plt.clf()
plt.title('beta = {}, alpha_tilde = {}'.format(1,alpha_tilde))
plt.plot(BETA, m_naive_th, 'o-', label='Binary labels', color='b')
plt.plot(BETA, m_opt_th, 'o-', label='Optimal labels', color='r')
plt.plot(BETA, m_alpha_th, 'o-', label='Optimal', color='g')
plt.xlabel('sim')
plt.legend()