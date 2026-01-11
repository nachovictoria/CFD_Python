# 12 steps to Navier Stokes
# Lesson 1: 1D Linear Convection
# Source: https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/

# Library import
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Simulation variables
L=2.0 # Length of the domain
nx=81 # Number of spacial divisions
dx=L/(nx-1) # Distance between points
dt=0.00025 # Time step
t=5 # Total time
nt=int(t/dt)+1 # Number of time steps
C=1.0 # Assumed wave speed

# Wave initialization
u=np.ones(nx)
u[int(0.5/dx):int(1/dx+1)]=2.0 # u=2 for 0.5<=x<=1

#Plot initial contion
plt.plot(np.linspace(0,L,nx),u, label='Initial condition')
# Solver implementation
un=np.ones(nx) # Initializa previous timestep container
for n in range(nt):
    un=u.copy() # Solve previous timestep
    for i in range(1,nx): # Starts aat one to use the backwards in space (vector starts at 0)
        u[i]=un[i]-C*dt/dx*(un[i]-un[i-1])
    if n%100==0:
        plt.plot(np.linspace(0,L,nx),u, label=f'Time step {n}')

# Plotting
plt.title('1D Linear Convection')
plt.xlabel('x')
plt.ylabel('u')
plt.legend(loc='upper right')
plt.show()

#Playing with grid spacing and time step
nx=np.array([41,81,161])
dt=np.array([0.00025,0.000125,0.000625])
counter=1
for n_i in range(len(nx)):
    dx=L/(nx[n_i]-1) # Distance between points
    u=np.ones(nx[n_i])
    u[int(0.5/dx):int(1/dx+1)]=2.0 # u=2 for 0.5<=x<=1
    for n_j in range(len(dt)):
        nt=int(t/dt[n_j])+1 # Number of time steps
        for n in range(nt):
            un=u.copy() # Solve previous timestep
            for i in range(1,nx[n_i]): # Starts aat one to use the backwards in space (vector starts at 0)
                u[i]=un[i]-C*dt[n_j]/dx*(un[i]-un[i-1])
            if n%100==0:
                plt.subplot(len(nx),len(dt),counter)
                plt.plot(np.linspace(0,L,nx[n_i]),u, label=f'Time step {n}')
                plt.title(f'nx={nx[n_i]}, dt={dt[n_j]}')
        counter+=1
plt.show()
