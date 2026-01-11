# 12 steps to Navier Stokes
# Lesson 6: 2D Non Linear Convection
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
nx = 101
ny = 101
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx))
v = np.ones((ny, nx))
un = np.ones((ny, nx))
vn = np.ones((ny, nx))

# Apply initial conditons
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
v[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')  # Modern matplotlib syntax
X, Y = np.meshgrid(x, y)                            
surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
plt.show()

# Solved with array operations
for n in range(nt + 1):  # Time marches forward +1 to reach the last time step
    un = u.copy()
    vn = v.copy()
    u[1:, 1:] = (un[1:, 1:] - (un[1:, 1:] * c * dt / dx * (un[1:, 1:] - un[1:, :-1])) - (vn[1:, 1:] * c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))
    v[1:, 1:] = (vn[1:, 1:] - (un[1:, 1:] * c * dt / dx * (vn[1:, 1:] - vn[1:, :-1])) - (vn[1:, 1:] * c * dt / dy * (vn[1:, 1:] - vn[:-1, 1:])))
    
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
surf2 = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u')
ax.set_title(f'2D Non Linear Convection u at t = {nt*dt:.2f}s')
plt.show()

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
surf2 = ax.plot_surface(X, Y, v[:], cmap=cm.viridis)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u')
ax.set_title(f'2D Non Linear Convection v at t = {nt*dt:.2f}s')
plt.show()