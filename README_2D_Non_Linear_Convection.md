# Lesson 6: 2D Non-Linear Convection

## Overview

This lesson extends non-linear convection from 1D (Lesson 2) to **2 dimensions**, introducing a **coupled system** where two velocity components `u` and `v` transport themselves in both x and y directions. The implementation uses **array operations** (vectorization) for computational efficiency, achieving dramatic speedups over nested loops.

## Physical Background

### The 2D Non-Linear Convection Equations

The governing equations form a coupled system:

```
∂u/∂t + u·∂u/∂x + v·∂u/∂y = 0
∂v/∂t + u·∂v/∂x + v·∂v/∂y = 0
```

Where:
- `u(x,y,t)` is the velocity component in the x-direction
- `v(x,y,t)` is the velocity component in the y-direction
- `t` is time
- `x, y` are spatial coordinates

**Physical Meaning**: This is a **2D velocity field** that transports itself. Each component (`u`, `v`) is advected by the full velocity vector `(u, v)`.

### Comparison with Other Equations

| Equation | Formula | Key Feature |
|----------|---------|------------|
| **1D Non-Linear Convection** | `∂u/∂t + u·∂u/∂x = 0` | Self-advection in 1D |
| **2D Linear Convection** | `∂u/∂t + c·∂u/∂x + c·∂u/∂y = 0` | Constant velocity `c` |
| **2D Non-Linear Convection** | `∂u/∂t + u·∂u/∂x + v·∂u/∂y = 0`<br>`∂v/∂t + u·∂v/∂x + v·∂v/∂y = 0` | **Coupled, self-advecting** |
| **2D Burgers' (next step)** | Add diffusion: `+ ν·∇²u` | Non-linear + viscosity |

### Physical Interpretation

#### 1. **Self-Advecting Velocity Field**
- The velocity field `(u, v)` determines its own evolution
- Fast-moving regions catch up to slow-moving regions
- **Wave steepening** occurs in 2D (like 1D, but in all directions)

#### 2. **Coupling Between Components**
- `u` is advected by both `u` (x-direction) and `v` (y-direction)
- `v` is advected by both `u` (x-direction) and `v` (y-direction)
- The two equations are **interdependent** - must solve simultaneously

#### 3. **Non-Linear Effects**
- **Shock formation**: Steep gradients can develop
- **Vortex stretching**: Rotation can intensify
- **Energy cascade**: Non-linearity transfers energy between scales

#### 4. **Real-World Applications**
- **Shallow water waves**: 2D water surface dynamics (inviscid limit)
- **Traffic flow**: 2D vehicle dynamics on road networks
- **Gas dynamics**: 2D compressible flow (simplified)
- **Precursor to Navier-Stokes**: Same structure as momentum equations

### Extension from 1D to 2D Non-Linear

| Aspect | 1D Non-Linear | 2D Non-Linear |
|--------|--------------|---------------|
| **Variables** | `u` only | `u` and `v` (coupled) |
| **Self-advection** | `u·∂u/∂x` | `u·∂u/∂x + v·∂u/∂y` |
| **Equations** | 1 equation | **2 coupled equations** |
| **Wave steepening** | Along x-axis | **In all directions** |
| **Computational cost** | O(nx·nt) | **O(nx·ny·nt)** |

## Mathematical Formulation

### Discretization Using Backward Differences

For the **u-component**:
```
∂u/∂t + u·∂u/∂x + v·∂u/∂y = 0

Discretized:
u[j,i]ⁿ⁺¹ = u[j,i]ⁿ - u[j,i]ⁿ·Δt/Δx·(u[j,i]ⁿ - u[j,i-1]ⁿ) 
                      - v[j,i]ⁿ·Δt/Δy·(u[j,i]ⁿ - u[j-1,i]ⁿ)
```

For the **v-component**:
```
∂v/∂t + u·∂v/∂x + v·∂v/∂y = 0

Discretized:
v[j,i]ⁿ⁺¹ = v[j,i]ⁿ - u[j,i]ⁿ·Δt/Δx·(v[j,i]ⁿ - v[j,i-1]ⁿ) 
                      - v[j,i]ⁿ·Δt/Δy·(v[j,i]ⁿ - v[j-1,i]ⁿ)
```

**Key observations**:
1. Both equations use the **same velocity field** `(u, v)` for advection
2. Each component has its own **spatial gradient**
3. Backward differences in both x and y directions
4. **Explicit time-stepping** (forward Euler)

## Array Operations Implementation

### Why Array Operations?

For 2D problems with two coupled fields:
- **Grid size**: 101 × 101 = 10,201 points
- **Per timestep**: 20,402 updates (u and v)
- **Total operations**: 100 steps × 20,402 ≈ 2 million operations

Using **nested loops** in Python would be prohibitively slow. **Array operations** (vectorization) provide 20-100× speedup!

### Complete Array-Based Solver

```python
# Initialize
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial conditions
u[int(.5/dy):int(1/dy+1), int(.5/dx):int(1/dx+1)] = 2
v[int(.5/dy):int(1/dy+1), int(.5/dx):int(1/dx+1)] = 2

# Time loop
for n in range(nt + 1):
    un = u.copy()  # Store previous u
    vn = v.copy()  # Store previous v
    
    # Update u using array operations
    u[1:, 1:] = (un[1:, 1:] 
                 - un[1:, 1:] * dt/dx * (un[1:, 1:] - un[1:, :-1])  # u·∂u/∂x
                 - vn[1:, 1:] * dt/dy * (un[1:, 1:] - un[:-1, 1:])) # v·∂u/∂y
    
    # Update v using array operations  
    v[1:, 1:] = (vn[1:, 1:]
                 - un[1:, 1:] * dt/dx * (vn[1:, 1:] - vn[1:, :-1])  # u·∂v/∂x
                 - vn[1:, 1:] * dt/dy * (vn[1:, 1:] - vn[:-1, 1:])) # v·∂v/∂y
    
    # Enforce boundary conditions
    u[0, :] = 1; u[-1, :] = 1; u[:, 0] = 1; u[:, -1] = 1
    v[0, :] = 1; v[-1, :] = 1; v[:, 0] = 1; v[:, -1] = 1
```

### Array Slicing Breakdown for u-Component

Let's analyze the u-equation update:

```python
u[1:, 1:] = (un[1:, 1:] 
             - un[1:, 1:] * dt/dx * (un[1:, 1:] - un[1:, :-1])
             - vn[1:, 1:] * dt/dy * (un[1:, 1:] - un[:-1, 1:]))
```

**Term by term**:

| Slice | Meaning | Role |
|-------|---------|------|
| `un[1:, 1:]` | Current `u` values | Base value |
| `un[1:, :-1]` | Left neighbors (i-1) | Backward difference in x |
| `un[:-1, 1:]` | Down neighbors (j-1) | Backward difference in y |
| `vn[1:, 1:]` | Current `v` values | Advection velocity in y |

**Physical interpretation**:
```python
u[1:, 1:] = un[1:, 1:]                              # Start with current u
            - un[1:, 1:] * dt/dx * ∂u/∂x            # Self-advection in x
            - vn[1:, 1:] * dt/dy * ∂u/∂y            # Cross-advection in y
```

### Visual Explanation of Slicing

For a 5×5 grid, updating u[1:, 1:]:

```
Interior points to update: u[1:, 1:]
       j→  0    1    2    3    4
    i↓
    0:     [.]  [.]  [.]  [.]  [.]
    1:     [.]  [U]  [U]  [U]  [U]
    2:     [.]  [U]  [U]  [U]  [U]
    3:     [.]  [U]  [U]  [U]  [U]
    4:     [.]  [U]  [U]  [U]  [U]

Left neighbors for ∂u/∂x: un[1:, :-1]
       0    1    2    3    4
    0: [.]  [.]  [.]  [.]  [.]
    1: [U]  [U]  [U]  [U]  [.]
    2: [U]  [U]  [U]  [U]  [.]
    3: [U]  [U]  [U]  [U]  [.]
    4: [U]  [U]  [U]  [U]  [.]
       ↑ shifted left by 1

Down neighbors for ∂u/∂y: un[:-1, 1:]
       0    1    2    3    4
    0: [.]  [U]  [U]  [U]  [U]
    1: [.]  [U]  [U]  [U]  [U]
    2: [.]  [U]  [U]  [U]  [U]
    3: [.]  [U]  [U]  [U]  [U]
    4: [.]  [.]  [.]  [.]  [.]
       ↑ shifted down by 1

Current v values: vn[1:, 1:]
       0    1    2    3    4
    0: [.]  [.]  [.]  [.]  [.]
    1: [.]  [V]  [V]  [V]  [V]
    2: [.]  [V]  [V]  [V]  [V]
    3: [.]  [V]  [V]  [V]  [V]
    4: [.]  [V]  [V]  [V]  [V]
```

When you compute:
```python
un[1:, 1:] - un[1:, :-1]  # Backward difference in x → ∂u/∂x
un[1:, 1:] - un[:-1, 1:]  # Backward difference in y → ∂u/∂y
```

Each interior point gets the correct spatial derivatives!

## Simulation Parameters

```python
nx = 101             # Grid points in x
ny = 101             # Grid points in y  
nt = 100             # Time steps
c = 1                # Reference speed [m/s] (not used in non-linear)
dx = 2 / (nx - 1)    # Spatial step x = 0.02 m
dy = 2 / (ny - 1)    # Spatial step y = 0.02 m
sigma = 0.2          # Stability parameter
dt = sigma * dx      # Time step = 0.004 s
```

### Initial Conditions

Both velocity components start with a **square pulse**:

```python
u[int(.5/dy):int(1/dy+1), int(.5/dx):int(1/dx+1)] = 2
v[int(.5/dy):int(1/dy+1), int(.5/dx):int(1/dx+1)] = 2
```

- `u = 2, v = 2` for `0.5 ≤ x ≤ 1.0` and `0.5 ≤ y ≤ 1.0`
- `u = 1, v = 1` everywhere else

**Physical meaning**: A region of fluid moving at velocity `(2, 2)` embedded in a background moving at `(1, 1)`.

### Boundary Conditions

**Dirichlet boundaries** on all four edges:

```python
u[0, :] = 1   # Bottom    v[0, :] = 1
u[-1, :] = 1  # Top       v[-1, :] = 1
u[:, 0] = 1   # Left      v[:, 0] = 1
u[:, -1] = 1  # Right     v[:, -1] = 1
```

Both velocity components are fixed at `1` on all boundaries.

## Stability Analysis

### 2D Non-Linear Stability Condition

The CFL-like condition for non-linear convection:

```
u_max·Δt/Δx + v_max·Δt/Δy ≤ 1
```

**More conservative form**:
```
(|u|_max + |v|_max)·Δt / min(Δx, Δy) ≤ 1
```

### Current Simulation Stability

With initial `u_max = v_max = 2`:

```
Δt = sigma·Δx = 0.2 × 0.02 = 0.004 s

CFL_u = 2 × 0.004 / 0.02 = 0.4
CFL_v = 2 × 0.004 / 0.02 = 0.4

CFL_total = 0.4 + 0.4 = 0.8 < 1 ✓
```

**Status**: ✓ **Stable** (but close to limit!)

> [!IMPORTANT]
> Non-linear convection can develop **higher velocities** through wave steepening. The maximum velocity may increase during the simulation, potentially violating stability. Monitor `max(u)` and `max(v)` if running for many timesteps!

## Performance: Array Operations vs Nested Loops

### Theoretical Comparison

For `nx = ny = 101, nt = 100`:

**Nested loops implementation**:
```python
for n in range(nt):
    for j in range(1, ny-1):  # 99 iterations
        for i in range(1, nx-1):  # 99 iterations
            # Update u[j,i]
            # Update v[j,i]
```
- **Total iterations**: 100 × 99 × 99 × 2 ≈ 2 million
- **Python overhead**: ~1-10 μs per iteration
- **Estimated time**: 2-20 seconds

**Array operations**:
```python
for n in range(nt):
    u[1:, 1:] = ...  # One NumPy call
    v[1:, 1:] = ...  # One NumPy call
```
- **Total NumPy calls**: 100 × 2 = 200
- **NumPy overhead**: ~0.1-1 ms per call
- **Estimated time**: 0.02-0.2 seconds

**Speedup**: **20-100×** faster with arrays!

### Memory Efficiency

**Arrays needed**:
```
u, v, un, vn = 4 × (101 × 101) × 8 bytes = 326 KB
```

**Temporary arrays from slicing**:
```
un[1:, 1:], un[1:, :-1], etc. ≈ 6 temporary views  
Additional ~200 KB (views don't copy data)
```

**Total**: ~500 KB (negligible for modern computers)

### Why Array Operations are Dramatically Faster

1. **Compiled C code**: NumPy operations run in optimized C libraries
2. **Vectorized**: CPU SIMD instructions process multiple elements simultaneously
3. **Cache efficient**: Contiguous memory access patterns
4. **No Python overhead**: Avoids interpreter overhead for each element

### Nested Loop Implementation (For Comparison)

<details>
<summary>Click to see nested loop version (not recommended!)</summary>

```python
for n in range(nt + 1):
    un = u.copy()
    vn = v.copy()
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            # Update u
            u[j,i] = (un[j,i] 
                      - un[j,i] * dt/dx * (un[j,i] - un[j,i-1])
                      - vn[j,i] * dt/dy * (un[j,i] - un[j-1,i]))
            
            # Update v
            v[j,i] = (vn[j,i]
                      - un[j,i] * dt/dx * (vn[j,i] - vn[j,i-1])
                      - vn[j,i] * dt/dy * (vn[j,i] - vn[j-1,i]))
    
    # Boundaries
    u[0, :] = 1; u[-1, :] = 1; u[:, 0] = 1; u[:, -1] = 1
    v[0, :] = 1; v[-1, :] = 1; v[:, 0] = 1; v[:, -1] = 1
```

**Use only for learning purposes!** Production code should always use arrays.

</details>

## Expected Results

### Evolution of Velocity Field

1. **t = 0**: Square region with `(u, v) = (2, 2)` in center, `(1, 1)` elsewhere

2. **Early times** (t ≈ 0-0.1s):
   - High-velocity region begins moving faster than background
   - Wave steepening starts on leading edges
   - Gradients sharpen

3. **Mid times** (t ≈ 0.1-0.3s):
   - Significant steepening and distortion
   - Pulse elongates and deforms
   - Non-linear effects prominent

4. **Late times** (t = 0.4s):
   - Complex wave shapes
   - Possible shock-like features
   - Strong gradients near boundaries

### Differences from Linear 2D Convection

| Linear Convection | Non-Linear Convection |
|------------------|----------------------|
| Shape preserved | **Shape distorts** |
| Uniform translation | **Differential motion** |
| No shock formation | **Shocks can form** |
| Velocity constant | **Velocity varies** |

## Running the Code

```bash
python Lesson5_2D_linear_convection.py  # Note: filename mismatch - should be Lesson6!
```

The script will:
1. Display initial condition for `u` (3D surface plot)
2. Solve for 100 timesteps using array operations
3. Display final `u` field
4. Display final `v` field

## Key Takeaways

### Physical Insights

1. **Non-linearity creates complexity**
   - Self-advection causes wave steepening
   - Shocks can form from smooth initial conditions
   - 2D adds richer dynamics than 1D

2. **Coupling is critical**
   - `u` and `v` must be solved together
   - Each component affects the other's evolution
   - Cannot solve independently

3. **Velocity field evolution**
   - Fast regions catch slow regions
   - Gradients sharpen over time
   - Energy concentrates into shocks

### Numerical Insights

1. **Array operations are essential in 2D**
   - 20-100× speedup over loops
   - Minimal code complexity increase
   - Standard approach for production CFD

2. **Stability depends on solution**
   - Maximum velocity can grow
   - CFL must be monitored
   - May need adaptive timestepping

3. **Vectorization syntax**
   - Array slicing replaces explicit indexing
   - Same algorithm, different notation
   - Master this for efficient scientific computing

## Extensions and Experiments

### Add Viscosity → 2D Burgers' Equations

```python
nu = 0.01  # Kinematic viscosity

u[1:, 1:] = (un[1:, 1:] 
             - un[1:, 1:] * dt/dx * (un[1:, 1:] - un[1:, :-1])
             - vn[1:, 1:] * dt/dy * (un[1:, 1:] - un[:-1, 1:])
             + nu * dt/dx**2 * (un[1:, 2:] - 2*un[1:, 1:] + un[1:, :-2])
             + nu * dt/dy**2 * (un[2:, 1:] - 2*un[1:, 1:] + un[:-2, 1:]))
```

This adds diffusion terms, smoothing out shocks.

### Different Initial Conditions

```python
# Circular pulse
for i in range(nx):
    for j in range(ny):
        if (x[i]-1)**2 + (y[j]-1)**2 < 0.25**2:
            u[j, i] = 3
            v[j, i] = 3

# Vortex
u = 1 - np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
v = 1 + np.cos(2*np.pi*X) * np.sin(2*np.pi*Y)
```

### Visualization Improvements

```python
# Velocity magnitude
speed = np.sqrt(u**2 + v**2)
ax.plot_surface(X, Y, speed, cmap='plasma')

# Vector field
plt.quiver(X[::5, ::5], Y[::5, ::5], 
           u[::5, ::5], v[::5, ::5])
plt.title('Velocity Vectors')
```

## Troubleshooting

### Problem: Solution Explodes

**Cause**: CFL condition violated

**Fix**: Reduce `sigma` to 0.1 or smaller

### Problem: No Visible Change

**Cause**: Timestep too small or not enough iterations

**Fix**: Increase `nt` to 200-500

### Problem: Results Look Wrong

**Check**:
1. Are `un` and `vn` computed from correct copied arraysbefore updates?
2. Are boundary conditions applied after each timestep?
3. Is the non-linear term structure correct?

## Mathematical Notes

### Characteristics

For the u-equation:
```
dx/dt = u
dy/dt = v
du/dt = 0
```

**Meaning**: `u` is constant along characteristic curves that move with velocity `(u, v)`. But since `(u, v)` itself changes, characteristics are **curved**, not straight!

### Conservation Form

The equations can be written as:
```
∂u/∂t + ∂(u²/2)/∂x + ∂(uv)/∂y = 0
∂v/∂t + ∂(uv)/∂x + ∂(v²/2)/∂y = 0
```

This conserves momentum flux.

## References

1. [CFD Python: 12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) by Lorena Barba
2. LeVeque, R.J. (2002). "Finite Volume Methods for Hyperbolic Problems"
3. NumPy Documentation: [Array Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
4. Toro, E.F. (2009). "Riemann Solvers and Numerical Methods for Fluid Dynamics"

## License

Based on the CFD Python module by Prof. Lorena A. Barba, shared under Creative Commons Attribution license, CC-BY 4.0.
