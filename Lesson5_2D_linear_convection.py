# 12 steps to Navier Stokes
# Lesson 5: 2D Linear Convection
# Source: https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/

# Library import
from matplotlib.dates import num2timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D 
import sympy
import time
import sys

###variable declarations
nx = 81
ny = 81
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx))
un = np.ones((ny, nx))

# Apply initial conditons
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')  # Modern matplotlib syntax
X, Y = np.meshgrid(x, y)                            
surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
plt.show()

# Configure the solver

for n in range(nt + 1):  # Time marches forward +1 to reach the last time step
    un = u.copy()
    for j in range(1,ny-1): # Loop over rows except last and first for boundary conditions
        for i in range(1,nx-1): # Loop over columns except last and first for boundary conditions
            u[j,i] = un[j,i] - c * dt / dx * (un[j,i] - un[j,i-1]) - c * dt / dy * (un[j,i] - un[j-1,i]) # Solved with nested for loops
    
        # Enforce Dirichlet boundary conditions (u=1 on all boundaries)
        u[0, :] = 1   # Bottom boundary
        u[-1, :] = 1  # Top boundary
        u[:, 0] = 1   # Left boundary
        u[:, -1] = 1  # Right boundary
    

# Plot final result
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u')
ax.set_title(f'2D Linear Convection at t = {nt*dt:.2f}s')
plt.show()

# Solved with array operations
u = np.ones((ny, nx))
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
for n in range(nt + 1):  # Time marches forward +1 to reach the last time step
    un = u.copy()
    u[1:, 1:] = un[1:, 1:] - c * dt / dx * (un[1:, 1:] - un[1:, :-1]) - c * dt / dy * (un[1:, 1:] - un[:-1, 1:]) # Solved with array operations
    # Enforce Dirichlet boundary conditions (u=1 on all boundaries)
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
surf2 = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u')
ax.set_title(f'2D Linear Convection at t = {nt*dt:.2f}s')
plt.show()