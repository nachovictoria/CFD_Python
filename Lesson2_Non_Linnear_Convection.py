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
t=10 # Total time (increased from 5 to 10 for more dramatic effect)
nt=int(t/dt)+1 # Number of time steps
C=1.0 # Assumed wave speed

# Wave initialization (ENHANCED: u=5 instead of u=2)
u=np.ones(nx)
u[int(0.5/dx):int(1/dx+1)]=5.0 # u=5 for 0.5<=x<=1 (increased for stronger non-linearity)
u2=np.ones(nx)
u2[int(0.5/dx):int(1/dx+1)]=5.0 # u=5 for 0.5<=x<=1

#Plot initial contion
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0,L,nx),u, 'k-', linewidth=2, label='Initial condition')
# Solver implementation
un=np.ones(nx) # Initializa previous timestep container
un2=np.ones(nx) # Initializa previous timestep container
for n in range(nt):
    un=u.copy() # Solve previous timestep
    un2=u2.copy() # Solve previous timestep (FIXED: was copying from u)
    for i in range(1,nx): # Starts aat one to use the backwards in space (vector starts at 0)
        u[i]=un[i]-un[i]*dt/dx*(un[i]-un[i-1])
        u2[i]=un2[i]-C*dt/dx*(un2[i]-un2[i-1])
    # Plot at strategic times to show evolution clearly
    if n in [1000, 5000, 10000, 20000, 40000]:  # At t=0.25s, 1.25s, 2.5s, 5s, 10s
        time_val = n*dt
        plt.plot(np.linspace(0,L,nx),u, '-', linewidth=1.5, label=f't={time_val:.2f}s (Non-Linear)', alpha=0.8)
        plt.plot(np.linspace(0,L,nx),u2,'--', linewidth=1.5, label=f't={time_val:.2f}s (Linear)', alpha=0.8)

# Plotting
plt.title('1D Non-Linear Convection vs Linear Convection\n(Notice wave steepening in non-linear case)', fontsize=14)
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('u [m/s]', fontsize=12)
plt.legend(loc='upper right', fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()