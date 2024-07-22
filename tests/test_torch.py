import datadrivenlinearization as ddl
import numpy as np
import torch
from torch.autograd.functional import hessian, jacobian
from scipy.integrate import solve_ivp

def Amatrix(k1,k2,c1,c2):
    return np.array([[0, 1, 0, 0], [-(k1+k2), -c1 - c2, k2, c2], [0, 0,0,1], [k2, c2, -(k1+k2), -c1 - c2]])
def shawpierre(t, x, k1,k2,c1,c2,gamma, eps, Omega):
    return Amatrix(k1,k2,c1,c2)@x + np.array([0, -gamma * x[0]**3 + eps * np.cos(Omega * t) , 0, 0])
def shawpierreFreq(k1,k2,c1,c2):
    AA = Amatrix(k1,k2,c1,c2)
    w, v = np.linalg.eig(AA)
    return w
beta_alpha =1#10 * np.sqrt(3)
c = 0.03
k = 1
gamma = 0.5

k1 = 1
k2 = 3+0.325
c1 = 0.03
factor = beta_alpha / 2
c2 = c1* factor

eps0 = 1.5
ic = [0.09009306, 1.60106112, 0.02575738, 2.74465314]
Omega = 1.19237334
period0= 2*np.pi/Omega#;%5.90496506;%;5.900447076663105;
teval = np.arange(0, 100*period0, 0.01*period0)




np.random.seed(3)
solTrans = solve_ivp(shawpierre, [0,1000*period0], ic, method = 'DOP853', rtol = 1e-8, args=(k1, k2, c1, c2,gamma, eps0, Omega)) # transients
solDecay1 = solve_ivp(shawpierre, [0,100*period0], solTrans.y[:,-1],
                      t_eval = teval, method = 'DOP853', rtol = 1e-8, args=(k1, k2, c1, c2, gamma, 0, Omega)) # transients


