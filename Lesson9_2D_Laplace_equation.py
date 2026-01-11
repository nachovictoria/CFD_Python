# 12 steps to Navier Stokes
# Lesson 9: 2D Laplace Equation
# Source: https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/

# Library import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

### Variable declarations
nx = 31              # Grid points in x
ny = 31              # Grid points in y
dx = 2 / (nx - 1)    # Spatial step in x
dy = 2 / (ny - 1)    # Spatial step in y

# Grid
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Initialize solution array
p = np.zeros((ny, nx))

# Boundary conditions
p[:, 0] = 0          # Left boundary: p = 0 at x=0
p[:, -1] = y         # Right boundary: p = y (linear variation) at x=2
p[0, :] = p[1, :]    # Bottom boundary: Neumann (∂p/∂y = 0) at y=0
p[-1, :] = p[-2, :]  # Top boundary: Neumann (∂p/∂y = 0) at y=2

# Plot initial condition
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, p, cmap=cm.viridis)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('p')
ax.set_title('Initial Condition')
plt.show()

def laplace2d(p, max_iter=10000, tol=1e-4):
    
    pn = np.empty_like(p)
    
    for iter in range(max_iter):
        pn = p.copy()
        
        # Jacobi iteration: central difference in 2D
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) + 
                          dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1])) / 
                         (2 * (dx**2 + dy**2)))
        
        # Enforce boundary conditions
        p[:, 0] = 0              # Left: p = 0
        p[:, -1] = y             # Right: p = y
        p[0, :] = p[1, :]        # Bottom: Neumann ∂p/∂y = 0
        p[-1, :] = p[-2, :]      # Top: Neumann ∂p/∂y = 0
        
        # Check convergence
        diff = np.linalg.norm(p - pn)
        if diff < tol:
            print(f"Converged after {iter+1} iterations (diff = {diff:.2e})")
            break
    else:
        print(f"Did not converge after {max_iter} iterations (diff = {diff:.2e})")
    
    return p

# Solve Laplace equation
p = laplace2d(p.copy(), max_iter=10000, tol=1e-4)

# Plot final solution
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, p, cmap=cm.viridis)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('p')
ax.set_title('2D Laplace Equation Solution')
plt.show()

# Also plot as contour
fig, ax = plt.subplots(figsize=(11, 7))
contour = ax.contourf(X, Y, p, levels=20, cmap=cm.viridis)
ax.contour(X, Y, p, levels=20, colors='black', linewidths=0.5)
plt.colorbar(contour)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Laplace Equation - Contour Plot')
plt.axis('equal')
plt.show()