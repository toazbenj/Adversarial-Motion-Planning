#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:43:21 2021

@author: robotics-labs
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp

#%% Analyse the theta

# Load the reference.
theta_r = np.load('theta_ref_step_PID.npy')
# Load the theta
theta = np.load('theta_step_PID.npy')
# Load the position.
pos = np.load('position_ver2.npy')
pos = np.reshape(pos,(pos.shape[0],pos.shape[1]))
#%% Plotting parameters.
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 22
legend_size = 20
FONT_SIZE = 30
AXIS_SIZE = 25
TICK_SIZE = 25

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=AXIS_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=legend_size)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

line_plt = ['b','g','k','y','r','c','m','w']

marker_list = ['o','v','D','H','P','h','d','p','P','*','h','H','+','x','X','D','d','|','_']
lines = ['--', '-.', ':']

#%% Plot.

plt.figure()
figure = plt.gcf()
figure.set_size_inches(8,6)
time_vec = np.linspace(0,len(theta_r)/20.0,num=len(theta_r))
plt.plot(time_vec[200:],theta_r[200:],\
             linewidth=2.5, \
             linestyle='-', label='ref')
plt.plot(time_vec[200:],theta[200:],\
             linewidth=2.5, \
             linestyle='-', label='mea')
plt.xlabel('Time')
plt.ylabel('Theta')

#%% Solve the LQR problem.
def dare(A,B,Q,R):
    '''
    Solve the LQR problem.
    '''
    # Solve ricatti equation.
    X = np.matrix(sp.solve_discrete_are(A, B, Q, R))
    # Compute the LQR gain.
    K = -np.matrix(sp.inv(B.T*X*B + R)*(B.T*X*A))
    
    return K

dt = 0.2
# Discrete time dynamics.
A = [[1.0,0.0],[0.0,1.0]]
B = [[dt,0.0],[0.0,dt]]
A = np.asmatrix(A)
B = np.asmatrix(B)

# State and control weights.
Q = np.eye((2))
R = np.eye((2))

# Solve the LQR gain.
K = dare(A,B,Q,R)

#%% Plot trajectory.
# Initial position.
init_pos = pos[0,:].reshape(2,1)
str_LQR = []
str_LQR.append(init_pos)

while np.linalg.norm(init_pos) >= 1e-3:
    x_next = A@init_pos + B@K@init_pos
    init_pos =np.array(x_next)
    str_LQR.append(init_pos)
    
str_LQR = np.hstack(str_LQR)
#%% Plotting.
    
plt.figure()
figure = plt.gcf()
figure.set_size_inches(8,6)
time_vec = np.linspace(0,len(theta_r)/20.0,num=len(theta_r))
plt.plot(str_LQR[0,:],str_LQR[1,:], '-',\
             linewidth=2.5, \
              label='ref')
plt.plot(pos[:,0],pos[:,1], '-o',\
             linewidth=2.5, \
              label='mea')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
