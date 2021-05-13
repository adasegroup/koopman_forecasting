import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

import torch
from sklearn.model_selection import train_test_split

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name, noise = 0.0, theta=2.4):
    if name == 'pendulum_lin':
        return pendulum_lin(noise)      
    if name == 'pendulum':
        return pendulum(noise, theta)    
    else:
        raise ValueError('dataset {} not recognized'.format(name))


def rescale(Xsmall, Xsmall_test):
    #******************************************************************************
    # Rescale data
    #******************************************************************************
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test

constant_unknown_1 = 9.81
number_of_steps = 2200
step_size = 0.1


def sol(t,theta0):
    S = np.sin(0.5*(theta0) )
    K_S = ellipk(S**2)
    omega_0 = np.sqrt(constant_unknown_1)
    sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
    theta = 2.0*np.arcsin( S*sn )
    d_sn_du = cn*dn
    d_sn_dt = -omega_0 * d_sn_du
    d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
    return np.stack([theta, d_theta_dt],axis=1)



def pendulum(noise, theta=2.4):
    
    np.random.seed(1)

    anal_ts = np.arange(0, number_of_steps*step_size, step_size)
    X = sol(anal_ts, theta)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    Xclean = Xclean.T.dot(Q.T)
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    X_train, X_test, X_train_clean, X_test_clean = train_test_split(X, y, test_size=0.23, random_state=42)     
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, 64, 1
