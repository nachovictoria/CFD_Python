# Lesson 8: 2D Burgers' Equation

## Overview

This lesson combines all previous concepts—**non-linear convection** (Lesson 6) and **diffusion** (Lesson 7)—into the **2D Burgers' equations**. This represents a major milestone: a coupled system of equations with both non-linear self-advection and viscous dissipation, serving as a **simplified 2D Navier-Stokes** system.

The implementation uses a **function-based approach** (`Burger2D(t)`) that allows easy experimentation with different simulation times.

## Physical Background

### The 2D Burgers' Equations

The governing equations form a coupled system:

```
∂u/∂t + u·∂u/∂x + v·∂u/∂y = ν·(∂²u/∂x² + ∂²u/∂y²)
∂v/∂t + u·∂v/∂x + v·∂v/∂y = ν·(∂²v/∂x² + ∂²v/∂y²)
```

Where:
- `u(x,y,t)` is the velocity component in the x-direction
- `v(x,y,t)` is the velocity component in the y-direction
- `ν` (nu) is the kinematic viscosity
- `t` is time
- `x, y` are spatial coordinates

**Physical Meaning**: A **viscous velocity field** that transports itself. This combines:
1. **Non-linear convection**: Wave steepening and shock formation
2. **Viscous diffusion**: Smoothing and dissipation

### Equation Structure Breakdown

For each component (u and v):

```
∂u/∂t  +  u·∂u/∂x + v·∂u/∂y  =  ν·(∂²u/∂x² + ∂²u/∂y²)
  ↓           ↓                        ↓
Time     Non-linear                 Viscous
rate     Convection               Diffusion
       (self-advection)          (smoothing)
```

**The Competition**:
- **Left side**: Non-linear terms try to steepen gradients → shocks
- **Right side**: Viscous terms try to smooth gradients → diffusion
- **Balance**: Determines whether shocks form or are smoothed away

### Evolution from Previous Lessons

| Lesson | Equation | What's New |
|--------|----------|-----------|
| **Lesson 6** | 2D Non-Linear Convection | Coupled (u,v), self-advection |
| **Lesson 7** | 2D Diffusion | Viscous smoothing |
| **Lesson 8** | **2D Burgers'** | **Both non-linearity + viscosity** |
| Next | 2D Navier-Stokes | + pressure gradient |

### Physical Interpretation

#### 1. **Reynolds Number Regime**

The Reynolds number characterizes the flow:

```
Re = U·L/ν

Where: U = characteristic velocity
       L = characteristic length  
       ν = kinematic viscosity
```

| Re Range | Dominant Physics | Behavior |
|----------|-----------------|----------|
| **Re << 1** | Viscosity dominates | Smooth, diffusion-like |
| **Re ≈ 1** | **Balanced** | Rich dynamics, shocks + smoothing |
| **Re >> 1** | Convection dominates | Shock-like, thin boundary layers |

**Our simulation**: With `u_max ≈ 2`, `L ≈ 0.5`, `ν = 0.01`:
```
Re = 2 × 0.5 / 0.01 = 100
```
**Moderate Reynolds number** → both effects visible!

#### 2. **Shock Structure**

Unlike inviscid Burgers' (discontinuous shocks), **viscous Burgers' has smooth shocks** with finite thickness:

```
Shock thickness ~ ν / U
```

For our parameters: `δ ≈ 0.01 / 2 = 0.005 m` (about 5mm)

#### 3. **Energy Cascade**

- **Non-linearity**: Transfers energy to smaller scales (sharpening)
- **Viscosity**: Dissipates energy at small scales (smoothing)
- **Equilibrium**: Balance creates characteristic structures

#### 4. **Real-World Applications**

- **Fluid dynamics**: Simplified 2D Navier-Stokes (without pressure)
- **Traffic flow**: Viscous traffic models
- **Shock waves**: Supersonic flow with viscosity
- **Turbulence**: Prototype for energy cascade
- **Numerical methods testing**: Standard test case for CFD schemes

## Mathematical Formulation

### Complete Discretization

#### For u-component:

```
u[j,i]ⁿ⁺¹ = u[j,i]ⁿ 
           - Δt/Δx · u[j,i]ⁿ · (u[j,i]ⁿ - u[j,i-1]ⁿ)      [Non-linear x-convection]
           - Δt/Δy · v[j,i]ⁿ · (u[j,i]ⁿ - u[j-1,i]ⁿ)      [Non-linear y-convection]
           + ν·Δt/Δx² · (u[j,i+1]ⁿ - 2u[j,i]ⁿ + u[j,i-1]ⁿ) [x-diffusion]
           + ν·Δt/Δy² · (u[j+1,i]ⁿ - 2u[j,i]ⁿ + u[j-1,i]ⁿ) [y-diffusion]
```

#### For v-component:

```
v[j,i]ⁿ⁺¹ = v[j,i]ⁿ 
           - Δt/Δx · u[j,i]ⁿ · (v[j,i]ⁿ - v[j,i-1]ⁿ)      [Non-linear x-convection]
           - Δt/Δy · v[j,i]ⁿ · (v[j,i]ⁿ - v[j-1,i]ⁿ)      [Non-linear y-convection]
           + ν·Δt/Δx² · (v[j,i+1]ⁿ - 2v[j,i]ⁿ + v[j,i-1]ⁿ) [x-diffusion]
           + ν·Δt/Δy² · (v[j+1,i]ⁿ - 2v[j,i]ⁿ + v[j-1,i]ⁿ) [y-diffusion]
```

**Key features**:
1. **7-point stencil**: center + 4 neighbors (diffusion) + 2 for convection
2. **Coupled equations**: u and v both used in each equation
3. **Explicit time-stepping**: Forward Euler
4. **Backward differences** for convection, **central differences** for diffusion

## Simulation Parameters

### Variable Declarations

```python
nx = 81              # Grid points in x
ny = 81              # Grid points in y
nt = 500             # Time steps (not used in function)
nu = 0.01            # Kinematic viscosity [m²/s]
dx = 2 / (nx - 1)    # Spatial step x ≈ 0.025 m
dy = 2 / (ny - 1)    # Spatial step y ≈ 0.025 m
sigma = 0.0009       # Stability parameter
dt = sigma * dx * dy / nu  # Time step ≈ 0.000056 s
```

### Physical Meaning of Variables

#### **ν (nu): Kinematic Viscosity** [m²/s]

**Physical meaning**: Resistance to shear flow; how "thick" the fluid is
- **Larger ν**: More viscous → more smoothing, shocks less sharp
- **Smaller ν**: Less viscous → sharper shocks, closer to inviscid limit

**Typical values**:
- Air at 20°C: `ν ≈ 1.5×10⁻⁵ m²/s`
- Water at 20°C: `ν ≈ 1.0×10⁻⁶ m²/s`
- Motor oil: `ν ≈ 1×10⁻⁴ m²/s`
- **Our simulation**: `ν = 0.01 m²/s` (moderately viscous)

#### **σ (sigma): Stability Parameter**

**Dimensionless parameter** that controls time step size

For combined convection + diffusion:
```
CFL (convection): u·Δt/Δx ≤ 1
σ (diffusion): ν·Δt/Δx² ≤ 0.5

Combined: must satisfy both!
```

**Current value**: `σ = 0.0009` (very conservative for stability)

#### **Δt: Time Step** [s]

Calculated formula:
```
dt = σ · dx · dy / ν = 0.0009 × 0.025 × 0.025 / 0.01 ≈ 0.000056 s
```

**Very small**! This is because:
1. Fine grid (81×81)
2. Diffusion stability scales as `Δx²`
3. Need to resolve thin shocks

**Consequence**: Many timesteps needed for reasonable simulation time

## Function-Based Implementation

### The `Burger2D(t)` Function

```python
def Burger2D(t):
    """
    Solve 2D Burgers' equations for duration t.
    
    Parameters:
   -----------
    t : float
        Total simulation time [seconds]
        
    Returns:
    --------
    None (displays plots of u and v components)
    
    Physical meaning:
        - Function calculates nt = int(t/dt)
        - Runs simulation for that many timesteps
        - Shows final velocity fields u(x,y) and v(x,y)
    """
    nt = int(t/dt)  # Convert time to number of timesteps
    
    for n in range(nt + 1):
        un = u.copy()
        vn = v.copy()
        
        # Update u (vectorized)
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] 
                         - dt/dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2])
                         - dt/dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1])
                         + nu * dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])
                         + nu * dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
        
        # Update v (vectorized)
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1]
                         - dt/dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2])
                         - dt/dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1])
                         + nu * dt/dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2])
                         + nu * dt/dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1]))
        
        # Boundary conditions
        u[0,:] = 1; u[-1,:] = 1; u[:,0] = 1; u[:,-1] = 1
        v[0,:] = 1; v[-1,:] = 1; v[:,0] = 1; v[:,-1] = 1
    
    # Visualize both components
    # ... plotting code ...
```

### Why Use a Function?

**Advantages**:

1. **Easy time control**: `Burger2D(0.01)` vs `Burger2D(0.1)` 
2. **Automatic timestep calculation**: No need to manually compute `nt`
3. **Reusability**: Call multiple times with different parameters
4. **Clean experimentation**: Quick comparison of different durations

**Usage examples**:
```python
Burger2D(0.01)   # Short time: early shock formation
Burger2D(0.03)   # Medium time: developed shocks
Burger2D(0.1)    # Long time: viscous equilibrium
```

### Array Operations Breakdown

#### U-component update (line 51):

```python
u[1:-1, 1:-1] = (un[1:-1, 1:-1]                                    # Current value
                 - dt/dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2])  # x-convection
                 - dt/dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1])  # y-convection
                 + nu*dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])  # x-diffusion
                 + nu*dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1])) # y-diffusion
```

**Convection terms**:
- `un[1:-1, :-2]`: Left neighbors (i-1) for ∂u/∂x
- `un[:-2, 1:-1]`: Down neighbors (j-1) for ∂u/∂y
- Multiplied by local velocities (`un` and `vn`)

**Diffusion terms**:
- `un[1:-1, 2:]`: Right neighbors (i+1)
- `un[2:, 1:-1]`: Up neighbors (j+1)
- Classic 5-point Laplacian stencil

## Initial Conditions

Both velocity components start with a **square pulse**:

```python
u[int(.5/dy):int(1/dy+1), int(.5/dx):int(1/dx+1)] = 2
v[int(.5/dy):int(1/dy+1), int(.5/dx):int(1/dx+1)] = 2
```

- `u = 2, v = 2` for `0.5 ≤ x ≤ 1.0` and `0.5 ≤ y ≤ 1.0`
- `u = 1, v = 1` everywhere else

**Physical meaning**: 
- A square region of fluid moving at velocity `(2, 2)`
- Embedded in slower-moving background at `(1, 1)`
- Velocity difference of `(1, 1)` drives dynamics

## Stability Analysis

### Combined Stability Requirements

Burgers' equations must satisfy **both** convection and diffusion stability:

#### 1. Convection (CFL) stability:
```
(|u|_max + |v|_max) · Δt / min(Δx, Δy) ≤ 1
```

#### 2. Diffusion stability:
```
σ_diff = ν·Δt·(1/Δx² + 1/Δy²) ≤ 0.25
```

### Current Simulation Stability

With `dt = 0.000056 s`, `Δx = Δy = 0.025 m`, `ν = 0.01 m²/s`:

**Convection check** (u_max ≈ 2):
```
CFL = 2 × 0.000056 / 0.025 = 0.0045 << 1 ✓
```

**Diffusion check**:
```
σ = 0.01 × 0.000056 × (1/0.025² + 1/0.025²)
  = 0.01 × 0.000056 × 3200
  = 0.0018 << 0.25 ✓
```

**Status**: ✓ **Very stable** (highly conservative!)

> [!NOTE]
> The small `σ = 0.0009` leads to very small timesteps. This ensures stability but requires many iterations. Could be increased to ~0.01 for faster simulations while remaining stable.

### Time Step Scaling

**Key insight**: Diffusion dominates the stability requirement!

```
For stability: dt ~ min(Δx/u_max, Δx²/ν)

With our parameters:
- Convection limit: dt ≤ 0.025/2 = 0.0125 s
- Diffusion limit: dt ≤ 0.025²/0.01 = 0.0625 s (less restrictive for this ν)

Actually, combined requirement is more complex, but diffusion typically dominates for fine grids.
```

## Expected Results

### Evolution Timeline

#### **Short Time** (t = 0.015s, ~268 steps):
- Initial sharp edges begin to round
- Slight wave steepening from non-linearity
- Viscosity starts smoothing gradients
- Square pulse still recognizable

#### **Medium Time** (t = 0.03s, ~536 steps):
- Non-linear steepening more pronounced
- Shock-like features develop on leading edges
- Viscosity creates smooth shock profiles
- Pulse elongates and deforms

#### **Long Time** (t = 0.1s, ~1786 steps):
- Complex wave structures
- Mix of steep gradients and smooth regions
- Energy dissipation evident
- Departure from initial symmetry

### Comparison with Other Equations

| Time | No Viscosity (Lesson 6) | With Viscosity (Lesson 8) |
|------|------------------------|---------------------------|
| Early | Sharp discontinuities forming | Rounded edges, smooth |
| Late | Shocks, oscillations | Smooth shocks, no oscillations |
| Energy | Conservative | **Dissipated** |

Viscosity **regularizes** the solution - shocks are smooth rather than discontinuous!

## Running the Code

```bash
python Lesson8_2D_Burger_equation.py
```

The script will:
1. Display initial condition
2. Run `Burger2D(0.03)` → show u and v at t=0.03s
3. Run `Burger2D(0.015)` → show u and v at t=0.015s

### Interactive Usage

```python
# In Python interactive session:
Burger2D(0.01)    # Quick look at early dynamics
Burger2D(0.05)    # See developed structures
Burger2D(0.2)     # Long-time behavior

# Compare different viscosities (modify nu first)
nu = 0.001; dt = sigma*dx*dy/nu   # Lower viscosity
Burger2D(0.03)                    # Sharper shocks

nu = 0.05; dt = sigma*dx*dy/nu    # Higher viscosity  
Burger2D(0.03)                    # Smoother profiles
```

## Key Takeaways

### Physical Insights

1. **Burgers' = Prototype Navier-Stokes**
   - Captures essential physics: non-linearity + dissipation
   - Missing: pressure gradient term
   - Next step toward full fluid dynamics

2. **Competition of mechanisms**
   - Non-linearity → complexity (shocks)
   - Viscosity → simplicity (smoothing)
   - Reynolds number determines balance

3. **Shock structure**
   - Inviscid: discontinuous (infinitely sharp)
   - Viscous: smooth with finite thickness ~ν/U
   - Physically realistic

4. **Energy dissipation**
   - Non-linearity concentrates energy (small scales)
   - Viscosity dissipates energy (heat generation)
   - Total energy decreases over time

### Numerical Insights

1. **Stability is complex**
   - Must satisfy both convection AND diffusion limits
   - Diffusion often more restrictive for fine grids
   - Small timesteps needed → computationally expensive

2. **Array operations essential**
   - 81×81×500 timesteps = ~3.3 million updates
   - Vectorization provides 50-200× speedup
   - Would be impractical with nested loops

3. **Function approach powerful**
   - Time-based interface more intuitive than timestep count
   - Easy to explore different regimes
   - Good software engineering practice

4. **Explicit methods limitations**
   - Very small timesteps required
   - Implicit methods (Crank-Nicolson) could allow larger dt
   - Trade-off: stability vs computational cost per step

## Extensions and Experiments

### Vary Reynolds Number

```python
# Low Re (viscosity-dominated)
nu = 0.1
# Expect: Smooth diffusion-like behavior

# High Re (convection-dominated)  
nu = 0.001
# Expect: Sharp shocks, possible oscillations
```

### Different Initial Conditions

```python
# Vortex pair
u = 1 + np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
v = 1 - np.cos(2*np.pi*X) * np.sin(2*np.pi*Y)

# Gaussian hump
u = 1 + 2*np.exp(-((X-1)**2 + (Y-1)**2)/0.1)
v = 1 + 2*np.exp(-((X-1)**2 + (Y-1)**2)/0.1)

# Shear layer
u[:ny//2, :] = 2
u[ny//2:, :] = 1
v[:, :] = 1
```

### Add Forcing Term

```python
# External forcing (like wind)
F_x = 0.1  # Constant force in x-direction
u[1:-1, 1:-1] += F_x * dt
```

### Measure Quantities

```python
# Total kinetic energy
KE = 0.5 * np.sum(u**2 + v**2) * dx * dy

# Enstrophy (vorticity squared)
omega = (v[1:, :] - v[:-1, :]) / dy - (u[:, 1:] - u[:, :-1]) / dx
enstrophy = 0.5 * np.sum(omega**2) * dx * dy

# Track over time to see dissipation
```

### Visualize Vorticity

```python
# Compute vorticity: ω = ∂v/∂x - ∂u/∂y
omega = np.zeros_like(u)
omega[1:-1, 1:-1] = ((v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx) -
                      (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy))

# Plot
plt.contourf(X, Y, omega, levels=20, cmap='RdBu')
plt.title('Vorticity Field')
```

## Troubleshooting

### Problem: Solution Explodes

**Cause**: Stability violated

**Fixes**:
1. Increase `sigma` cautiously (try 0.001, then 0.005)
2. Increase viscosity `nu`
3. Decrease initial velocity contrast
4. Refine grid (more points)

### Problem: Solution Too Smooth (No Features)

**Cause**: Excessive viscosity or too many timesteps

**Fixes**:
1. Decrease `nu` (try 0.005 or 0.001)
2. Reduce simulation time `t`
3. Increase initial velocity contrast

### Problem: Function Runs Slow

**Cause**: Too many timesteps (small dt, large t)

**Solutions**:
1. Increase `sigma` to use larger `dt`
2. Coarsen grid (reduce `nx`, `ny`)
3. Use shorter simulation time
4. Consider implicit methods (advanced)

## Mathematical Notes

### Characteristics and Shocks

For inviscid Burgers', characteristics can intersect → shocks  
For viscous Burgers', characteristics approach asymptotically → smooth shocks

**Shock speed** (Rankine-Hugoniot):
```
s = (u_L + u_R) / 2
```
Where u_L, u_R are left and right states.

### Cole-Hopf Transformation

1D Burgers' can be linearized via:
```
u = -2ν·∂(ln φ)/∂x
```

For 2D, the transformation becomes more complex but similar ideas apply.

### Similarity Solutions

For certain initial conditions, self-similar solutions exist:
```
u(x,y,t) = f(x/√t, y/√t)
```

## References

1. [CFD Python: 12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) by Lorena Barba
2. Burgers, J.M. (1948). "A mathematical model illustrating the theory of turbulence"
3. Whitham, G.B. (1974). "Linear and Nonlinear Waves"
4. Bec, J. & Khanin, K. (2007). "Burgers turbulence"
5. LeVeque, R.J. (2002). "Finite Volume Methods for Hyperbolic Problems"

## License

Based on the CFD Python module by Prof. Lorena A. Barba, shared under Creative Commons Attribution license, CC-BY 4.0.
