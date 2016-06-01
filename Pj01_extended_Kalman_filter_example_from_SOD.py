# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:36:51 2016

@author: AshivD
"""
# Learning Extended Kalman filter
# Ashiv Dhondea, RRSG, UCT
# 28 May 2016
# Validated: 1 June 2016
# Based on example 4.8.2 Spring-Mass Problem and 4.15 on pg 272
# Statistical Orbit Determination. Tapley, Schutz, Born. Abbrev.: SOD.

# Import the required libraries ###############################################
import numpy as np
import math
#from numpy import linalg
import scipy.linalg
#from numpy.random import randn
import matplotlib.pyplot as plt

## Declare parameters for simulation###########################################
dt = 1 # stepsize in [s]
t = np.arange(0,20,dt,dtype=float) # [s]
# Declare system parameters i.e. constants from the spring-mass problem
m = 1.5 # [kg]
k1 = 2.5 # [N/m]
k2 = 3.7 # [N/m]
h = 5.4 # [m]
x0 = 3.0 # [m]
v0 = 0 # [m]
omega2 = (k1+k2)/m 
omega = np.sqrt(omega2) # [rad/s]

## Define functions used in this code #########################################
# Define nonlinear observation model function
def fnG(x_state,h): # Observation function
    rho = np.sqrt(np.square(x_state[0]) + np.square(h))
    rhodot = x_state[0]*x_state[1]/rho
    return np.array([rho,rhodot])

# Define linearized observation model function
def fnHtilde(x_state,h):
    rho = np.sqrt(np.square(x_state[0]) + np.square(h))
    H = np.zeros([x_state.shape[0],x_state.shape[0] ],dtype=float)
    H[0,0] = x_state[0]/rho
    H[1,0] = x_state[1]/rho - np.square(x_state[0])*x_state[1]/np.power(rho,3)
    H[1,1] = x_state[0]/rho
    return H
# Define State Transition Matrix of Dynamical Model
def fnSTM(omega,zeta):
    Phi = np.array([[np.cos(omega*zeta), np.sin(omega*zeta)/omega],
                    [-omega*np.sin(omega*zeta),np.cos(omega*zeta)]],
                     dtype=float)
    return Phi

# Define state sensitivity matrix of dynamical model
def fnAmat(omega):
    return np.array([[0,1],[-np.square(omega),0] ],dtype=float)

## Declare a priori values for Gauss-Newton filter
X0hat = np.array([4.0,0.2],dtype=float)

P0 = np.diag([1000,100]);
## Load observational data
x_state = np.zeros([2,len(t)],dtype=float)
# Process noise. No covariance matrix in this example.
Q = np.diag([np.square(0),np.square(0)]);
sigma_rho = 0.25
sigma_rho_dot = 0.1
# Measurement covariance matrix
R = np.diag([np.square(sigma_rho),np.square(sigma_rho_dot)]);
Rinv = np.diag([np.square(1/sigma_rho),np.square(1/sigma_rho_dot)]);

# Y = measurement vectors from the radar sensor.
Y = np.zeros([x_state.shape[0],x_state.shape[1]],dtype=float)
# Initialize Y
Y[:,0] = X0hat;
# Generate real data and sensor data
# Generate trajectory and observations
Phi = fnSTM(omega,dt);
for index in range (0,len(t)):
    wn = np.random.multivariate_normal([0,0],Q).transpose();
    x_state[:,index] = np.dot(Phi,x_state[:,index-1]) + wn;
    # Create observations of this trajectory
    vn = np.random.multivariate_normal([0,0],R).transpose()
    Y[:,index] = fnG(x_state[:,index],h) + vn;

# Plot sensor data
fig = plt.figure()
plt.plot(t,Y[0,:],'b.')
plt.ylabel('position in [m]')
plt.xlabel('time in [s]')
plt.show()

fig = plt.figure()
plt.plot(t,Y[1,:],'b.')
plt.ylabel('velocity in [m/s]')
plt.xlabel('time in [s]')
plt.show()

# total observation covariance matrix. represented by R sans serif in TFE
def fnCreateConcatenatedRmat(R,Rinv,stacklen):
    L = [R]; Linv = [Rinv];
    for index in range (0,stacklen):
        L.append(R);
        Linv.append(Rinv);
    ryn = scipy.linalg.block_diag(*L);
    ryninv = scipy.linalg.block_diag(*Linv);
    return ryn,ryninv;

## extended Kalman filter functions ###########################################
def fnEKF_predict( F,A, m, P, Q):
    # fnKF_predict implements the extended Kalman Filter predict step.
    # F is the nonlinear dynamics function.
    # A is the Jacobian of the function F evaluated at m.
    # m is the mean, P is the covariance matrix.
    # process noise: Q matrix
    m_pred = np.dot(F,m);
    P_pred = np.dot(np.dot(A,P),np.transpose(A)) + Q;
    # m_pred and P_pred are the predicted mean state vector and covariance
    # matrix at the current time step before seeing the measurement.
    return m_pred, P_pred

def fnEKF_update(m_minus, P_minus, y,H,M, R ):
    # m_minus,P_minus: state vector and covariance matrix
    # y is the measurement vector. H is the nonlinear measurement function and
    # M is its Jacobian. R is the measurement covariance matrix.
    innovation_mean = H;#np.dot(H,m_minus);
    prediction_covariance = (R + np.dot(M,np.dot(P_minus,np.transpose(M))));
    KalmanGain = np.dot(np.dot(P_minus,np.transpose(M)),np.linalg.inv(prediction_covariance));

    # Calculate estimated mean state vector and its covariance matrix.
    m = m_minus + np.dot(KalmanGain , (y - innovation_mean));
    
    P = P_minus - np.dot(np.dot(KalmanGain,prediction_covariance),np.transpose(KalmanGain));
    return m,P
   
 ## Filter the sensor data  ###################################################
x_state_hat = np.zeros([2,len(t)],dtype=float)
x_state_hat[:,0] = X0hat;
P_hat = np.zeros([2,2,len(t)],dtype=float);
P_hat[:,:,0] = P0;

for index in (1,x_state_hat.shape[1]-1):
    m_pred,P_pred = fnEKF_predict( fnSTM(omega,dt),fnAmat(omega),x_state_hat[:,index-1], P_hat[:,:,index-1], Q);
    x_state_hat[:,index],P_hat[:,:,index] = fnEKF_update(m_pred, P_pred, Y[:,index],fnG(m_pred,h),fnHtilde(m_pred,h), R );

## Plot results ###############################################################
# Comparison of filter estimates with true state vectors.
fig = plt.figure()
plt.plot(t,x_state_hat[0,:],'bo')
plt.plot(t,x_state[0,:],'r.')
plt.ylabel('position in [m]')
plt.xlabel('time in [s]')
plt.show()
    
fig = plt.figure()
plt.plot(t,x_state_hat[1,:],'bo')
plt.plot(t,x_state[1,:],'r.')
plt.ylabel('velocity in [m/s]')
plt.xlabel('time in [s]')
plt.show()

## Compute the RMS Error ######################################################
epsilon = np.zeros_like(Y,dtype=float);
squared_error = np.zeros_like(t,dtype=float);
for index in (0,x_state_hat.shape[1]-1):
    epsilon[:,index] = np.subtract(Y[:,index],fnG(x_state_hat[:,index],h));
    squared_error[index] = np.dot(np.dot(np.transpose(epsilon[:,index]),Rinv),epsilon[:,index]);
mse = np.sum(squared_error)/float(np.shape(epsilon)[0]*np.shape(squared_error)[0]);
rmse = math.sqrt(mse);
print rmse
