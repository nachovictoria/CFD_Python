# 12 steps to Navier Stokes
# Lesson 3: 1D Diffusion
# Source: https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/

# Library import
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Simulation variables
nu=0.3 # Viscosity
L=2.0 # Length of the domain
nx=81 # Number of spacial divisions
dx=L/(nx-1) # Distance between points
sigma=0.2
dt=sigma*dx**2/nu # Time step
t=10 # Total time (increased from 5 to 10 for more dramatic effect)
nt=int(t/dt)+1 # Number of time steps
C=1.0 # Assumed wave speed

# Wave initialization (ENHANCED: u=5 instead of u=2)
u=np.ones(nx)
u[int(0.5/dx):int(1/dx+1)]=2.0 # u=2 for 0.5<=x<=1

#Plot initial contion
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0,L,nx),u, 'k-', linewidth=2, label='Initial condition')
# Solver implementation
un=np.ones(nx) # Initializa previous timestep container
for n in range(nt):
    un=u.copy() # Solve previous timestep
    for i in range(1,nx-1): # Starts aat one to use the backwards in space (vector starts at 0) end in nx-1 to use the forward as well
        u[i]=un[i]+nu*dt/dx**2*(un[i+1]-2*un[i]+un[i-1])
    # Plot at strategic times to show evolution clearly
    if n in [1000, 5000, 10000, 20000, 40000]:  # At t=0.25s, 1.25s, 2.5s, 5s, 10s
        time_val = n*dt
        plt.plot(np.linspace(0,L,nx),u, '-', linewidth=1.5, label=f't={time_val:.2f}s', alpha=0.8)

# Plotting
plt.title('1D Diffusion', fontsize=14)
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('u ', fontsize=12)
plt.legend(loc='upper right', fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()