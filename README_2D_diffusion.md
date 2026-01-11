# Lesson 7: 2D Diffusion

## Overview

This lesson extends diffusion from 1D (Lesson 3) to **2 dimensions**, modeling how heat, concentration, or momentum spreads across a 2D plane through molecular diffusion. The implementation uses **array operations** for efficiency and includes a **function-based approach** for easy experimentation with different time durations.

## Physical Background

### The 2D Diffusion Equation

The governing equation is:

```
∂u/∂t = ν·(∂²u/∂x² + ∂²u/∂y²) = ν·∇²u
```

Where:
- `u(x,y,t)` is the diffusing quantity (temperature, concentration, etc.)
- `t` is time
- `x, y` are spatial coordinates  
- `ν` (nu) is the **diffusivity** (or kinematic viscosity, thermal diffusivity)
- `∇²u` is the **Laplacian** (measures curvature in 2D)

**Physical Meaning**: The quantity `u` spreads from regions of high concentration to low concentration in **all directions simultaneously**. The rate of spreading depends on the **local curvature** of the field.

### Physical Interpretation

#### 1. **Isotropic Diffusion**
- Spreads equally in all directions (x and y)
- No preferred direction (unlike convection)
- Driven by **gradients**, not velocity

#### 2. **Smoothing Process**
- **Sharp features are smoothed** (peaks flatten, valleys fill)
- **Gradients decrease** over time
- **Entropy increases** (irreversible process)

#### 3. **The Laplacian: Curvature as Driving Force**

The Laplacian `∇²u = ∂²u/∂x² + ∂²u/∂y²` measures **total curvature**:

```
If ∇²u > 0 at a point: u is locally concave (valley) → u increases
If ∇²u < 0 at a point: u is locally convex (peak) → u decreases  
If ∇²u = 0 at a point: u is at equilibrium → no change
```

**Example**: For a square pulse (high plateau):
- **Center of plateau**: `∇²u ≈ 0` (flat) → little change
- **Edges of plateau**: `∇²u < 0` (peak) → decreases rapidly
- **Outside plateau**: `∇²u > 0` (valley) → increases
- **Result**: Edges smooth out, pulse spreads radially

#### 4. **Real-World Examples**
- **Heat conduction**: Hot spot on a metal plate spreads
- **Chemical diffusion**: Drop of dye in still water
- **Momentum diffusion**: Viscosity in slow-moving fluids
- **Image blurring**: Gaussian blur filters

### Comparison: 1D vs 2D Diffusion

| Aspect | 1D Diffusion | 2D Diffusion |
|--------|-------------|--------------|
| **Equation** | `∂u/∂t = ν·∂²u/∂x²` | `∂u/∂t = ν·(∂²u/∂x² + ∂²u/∂y²)` |
| **Spreading** | Left and right only | **All radial directions** |
| **Stencil** | 3-point (left, center, right) | **5-point cross** |
| **Diffusion length** | `L ~ √(νt)` | `L ~ √(νt)` (same scaling!) |
| **Computational cost** | O(nx·nt) | **O(nx·ny·nt)** |

## Mathematical Formulation

### Discretization

Using **central differences** in both x and y:

#### X-direction second derivative:
```
∂²u/∂x² ≈ (u[j, i+1] - 2u[j, i] + u[j, i-1]) / Δx²
```

#### Y-direction second derivative:
```
∂²u/∂y² ≈ (u[j+1, i] - 2u[j, i] + u[j-1, i]) / Δy²
```

#### Complete discretized equation:

```
u[j,i]ⁿ⁺¹ = u[j,i]ⁿ + ν·Δt/Δx²·(u[j,i+1]ⁿ - 2u[j,i]ⁿ + u[j,i-1]ⁿ)
                      + ν·Δt/Δy²·(u[j+1,i]ⁿ - 2u[j,i]ⁿ + u[j-1,i]ⁿ)
```

### The 5-Point Stencil

Unlike 1D (3 points), 2D diffusion uses a **5-point cross stencil**:

```
         [j-1, i]
            ↑
[j, i-1] ← [j,i] → [j, i+1]
            ↓
         [j+1, i]
```

**All 5 points needed**: The central point plus its 4 neighbors (up, down, left, right).

## Simulation Parameters

### Variable Declarations with Physical Meaning

```python
nx = 31              # Grid points in x-direction
ny = 31              # Grid points in y-direction
nt = 17              # Number of time steps
nu = 0.05            # Diffusivity [m²/s]
dx = 2 / (nx - 1)    # Spatial step in x ≈ 0.0667 m
dy = 2 / (ny - 1)    # Spatial step in y ≈ 0.0667 m  
sigma = 0.25         # Stability parameter
dt = sigma * dx * dy / nu  # Time step ≈ 0.0222 s
```

### Physical Meaning of Each Variable

#### **ν (nu): Diffusivity** [m²/s]
- **Physical meaning**: How fast the quantity spreads
- **Larger ν**: Faster diffusion, more smoothing
- **Smaller ν**: Slower diffusion, sharper features persist longer
- **Typical values**:
  - Thermal diffusivity of air: ~2×10⁻⁵ m²/s
  - Thermal diffusivity of water: ~1.4×10⁻⁷ m²/s
  - Kinematic viscosity of water: ~1×10⁻⁶ m²/s

#### **σ (sigma): Stability Parameter**
- **Physical meaning**: Controls the time step relative to stability limit
- **Dimensionless**: No units
- **Stability requirement**: `σ ≤ 0.25` for 2D diffusion (explicit scheme)
- **Current value**: σ = 0.25 (right at stability limit!)

#### **Δt: Time Step** [s]
- **Calculated from**: `dt = σ · Δx · Δy / ν`
- **Physical meaning**: How much simulated time advances per iteration
- **Key insight**: Smaller grid → much smaller time steps needed (scales as Δx²!)

#### **nx, ny: Grid Points**
- **Physical meaning**: Spatial resolution
- **Trade-off**: 
  - More points → better accuracy, slower computation
  - Fewer points → faster, but less detail

### Time Step Scaling: Why `dt = σ · dx · dy / ν`?

From the stability condition:
```
σ = ν·Δt·(1/Δx² + 1/Δy²) ≤ σ_max

For square grids (Δx = Δy):
σ = ν·Δt·(2/Δx²) ≤ 0.25

Solving for Δt:
Δt ≤ 0.25·Δx² / (2ν) ≈ 0.125·Δx²/ν

Using σ·Δx²/ν generalizes to non-square grids:
Δt = σ·Δx·Δy / ν
```

> [!IMPORTANT]
> **Grid Refinement Impact**: If you halve Δx and Δy (2× finer grid), Δt must be reduced by **4×**, and total computational cost increases by **16×**! This is why diffusion is expensive to simulate.

## Array Operations Implementation

### Complete Inline Solution

```python
# Initial condition: square pulse
u = np.ones((ny, nx))
u[int(.5/dy):int(1/dy+1), int(.5/dx):int(1/dx+1)] = 2

# Time loop
for n in range(nt + 1):
    un = u.copy()
    
    # Update using array operations (vectorized 2D Laplacian)
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] 
                     + nu * dt / dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])
                     + nu * dt / dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
    
    # Enforce boundary conditions
    u[0, :] = 1; u[-1, :] = 1; u[:, 0] = 1; u[:, -1] = 1
```

### Array Slicing Breakdown

Let's break down the update equation:

```python
u[1:-1, 1:-1] = un[1:-1, 1:-1] + ...
```
- `u[1:-1, 1:-1]`: Interior points (excluding boundaries)
- `1:-1` means "from index 1 to second-to-last" (Python slicing)

#### X-direction diffusion:
```python
nu * dt / dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])
                    ↑                    ↑                  ↑
                 right              center              left
```

#### Y-direction diffusion:
```python
nu * dt / dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1])
                    ↑                   ↑                  ↑
                   up               center              down
```

### Visual Stencil Explanation

For a 5×5 grid, updating `u[1:-1, 1:-1]`:

```
Target interior points: u[1:-1, 1:-1]
       j→  0    1    2    3    4
    i↓
    0:     [B]  [B]  [B]  [B]  [B]   ← Boundary
    1:     [B]  [U]  [U]  [U]  [B]
    2:     [B]  [U]  [U]  [U]  [B]
    3:     [B]  [U]  [U]  [U]  [B]
    4:     [B]  [B]  [B]  [B]  [B]   ← Boundary
           ↑                    ↑
       Boundary            Boundary

Right neighbors (un[1:-1, 2:]):
       0    1    2    3    4
    0: [B]  [B]  [B]  [B]  [B]
    1: [B]  [B]  [U]  [U]  [U]
    2: [B]  [B]  [U]  [U]  [U]
    3: [B]  [B]  [U]  [U]  [U]
    4: [B]  [B]  [B]  [B]  [B]
                ← shifted right by 1

Up neighbors (un[2:, 1:-1]):
       0    1    2    3    4
    0: [B]  [B]  [B]  [B]  [B]
    1: [B]  [B]  [B]  [B]  [B]
    2: [B]  [U]  [U]  [U]  [B]
    3: [B]  [U]  [U]  [U]  [B]
    4: [B]  [U]  [U]  [U]  [B]
                ↑ shifted up by 1
```

Each interior point gets the correct 5-point stencil!

## Function-Based Approach for Easy Iteration

### Why Use a Function?

Instead of changing `nt` and re-running the entire script, define a function:

```python
def diffuse(nt):
    """
    Solve 2D diffusion for nt timesteps and visualize result.
    
    Parameters:
    -----------
    nt : int
        Number of time steps to simulate
        
    Physical meaning of nt:
        - Total simulation time = nt × dt
        - Each step: diffusion spreads by ~√(ν·dt)
        - More steps → more spreading and smoothing
    """
    # Reset initial condition
    u = np.ones((ny, nx))
    u[int(.5/dy):int(1/dy+1), int(.5/dx):int(1/dx+1)] = 2
    
    # Time loop
    for n in range(nt + 1):
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] 
                         + nu * dt / dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])
                         + nu * dt / dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
        u[0, :] = 1; u[-1, :] = 1; u[:, 0] = 1; u[:, -1] = 1
    
    # Visualize
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, u, cmap=cm.viridis)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u')
    ax.set_title(f'2D Diffusion at t = {nt*dt:.4f}s ({nt} steps)')
    plt.show()
```

### Usage Examples

```python
# Quick experiments
diffuse(10)    # Short time: t = 10×0.0222 = 0.222s → minimal spreading
diffuse(50)    # Medium time: t = 50×0.0222 = 1.11s → significant spreading
diffuse(100)   # Long time: t = 100×0.0222 = 2.22s → nearly uniform
diffuse(500)   # Very long: complete diffusion across domain
```

### Benefits of Function Approach

1. **Easy experimentation**: Just call `diffuse(n)` with different values
2. **Clean workspace**: Initial condition reset each time
3. **Reusability**: Can call multiple times in one session
4. **Comparison**: Run `diffuse(10)`, then `diffuse(100)` to compare

## Stability Analysis

### 2D Diffusion Stability Condition

For explicit (forward Euler) time-stepping:

```
σ = ν·Δt·(1/Δx² + 1/Δy²) ≤ σ_max

For 2D: σ_max = 0.25
```

**Why more restrictive than 1D?**
- 1D diffusion: `σ_max = 0.5`
- 2D diffusion: `σ_max = 0.25`
- Reason: Diffusion occurs in **two directions simultaneously**

### Current Simulation Stability

With `Δx = Δy = 0.0667 m`, `ν = 0.05 m²/s`, `σ = 0.25`:

```
Δt = σ·Δx² / ν = 0.25 × (0.0667)² / 0.05 = 0.0222 s

Check:
σ = 0.05 × 0.0222 × (1/0.0667² + 1/0.0667²)
  = 0.05 × 0.0222 × (224.7 + 224.7)
  = 0.05 × 0.0222 × 449.4
  = 0.25 ✓
```

**Status**: ✓ **Critical stability** (exactly at the limit!)

> [!CAUTION]
> With `σ = 0.25`, we're at the **stability boundary**. Any numerical round-off could potentially cause instability. For robustness, use `σ = 0.2` in production code.

## Expected Results

### Evolution Sequence

1. **t = 0**: Sharp square pulse (`u=2`) in center, `u=1` elsewhere

2. **Early times** (nt = 5-10, t ≈ 0.1-0.2s):
   - Corners of square begin rounding
   - Edges start smoothing
   - Peak value decreases slightly

3. **Mid times** (nt = 20-50, t ≈ 0.4-1.1s):
   - Square becomes circular (isotropic diffusion!)
   - Significant radial spreading
   - Peak value noticeably lower

4. **Late times** (nt = 100-500, t ≈ 2-11s):
   - Nearly Gaussian profile
   - Widespread distribution
   - Peak approaches background level

### Key Observations

- **Shape evolution**: Square → Circle → Gaussian
- **Maximum value decay**: Decreases as ~1/t
- **Radial symmetry**: Diffusion is isotropic (equal in all directions)
- **Mass conservation**: Total integral of `u` remains constant
- **Spreading rate**: Radius grows as ~√(νt)

## Performance: Array Operations

### Computational Cost

For `nx = ny = 31, nt = 17`:

**Operations per timestep**:
- Interior points: (31-2) × (31-2) = 841 updates
- Each update: ~10 floating-point operations
- Total per step: ~8,410 ops

**Array operations** (vectorized):
- Time: ~0.001-0.005 seconds per timestep
- Total: ~0.02-0.1computation overhead eliminated

**Nested loops** (for comparison):
- Would be ~50-200× slower
- Time: ~0.05-1.0 seconds per timestep
- Not recommended!

### Memory Usage

```
u, un = 2 × 31 × 31 × 8 bytes = 15.5 KB
X, Y (meshgrid) = 2 × 31 × 31 × 8 bytes = 15.5 KB
Temporary slices: ~10 KB
Total: ~40 KB (negligible)
```

## Running the Code

```bash
python Lesson7_2D_diffusion.py
```

The script will:
1. Display initial condition (square pulse)
2. Solve for `nt` time steps using array operations
3. Display final result showing smoothed, spread-out distribution

### Using the Function Approach

```python
# In Python interactive session or Jupyter:
diffuse(10)    # See early spreading
diffuse(50)    # See mid-stage diffusion
diffuse(200)   # See nearly complete diffusion
```

## Key Takeaways

### Physical Insights

1. **Diffusion smooths everything**
   - Sharp edges cannot persist
   - All features eventually become Gaussian-like
   - Irreversible process (entropy increases)

2. **Isotropic spreading**
   - Spreads equally in all directions
   - Square → Circle automatically
   - No preferred direction (unlike convection)

3. **Slow process**
   - Spreads as `√t` (not linearly!)
   - Takes 4× longer to spread 2× farther
   - Why natural systems often use convection for transport

4. **The Laplacian is key**
   - Measures total curvature (all directions)
   - Drives smoothing: peaks fall, valleys rise
   - Fundamental operator in physics

### Numerical Insights

1. **5-point stencil required**
   - Unlike 1D (3-point), need all 4 neighbors
   - Array slicing handles this elegantly
   - Boundary treatment critical

2. **Stability very restrictive**
   - `σ_max = 0.25` (half of 1D!)
   - Grid refinement → quadratic cost increase
   - Implicit methods needed for fine grids

3. **Array operations essential**
   - 50-200× speedup over loops
   - Minimal code complexity
   - Standard for production diffusion codes

4. **Function approach powerful**
   - Easy experimentation
   - Clean repeated runs
   - Educational and practical

## Extensions and Experiments

### Different Diffusivities

```python
nu = 0.01   # Slow diffusion → features persist longer
nu = 0.2    # Fast diffusion → rapid smoothing
```

### Different Initial Conditions

```python
# Circular pulse
for i in range(nx):
    for j in range(ny):
        if (x[i]-1)**2 + (y[j]-1)**2 < 0.3**2:
            u[j, i] = 3

# Gaussian hump
u = 1 + 2*np.exp(-((X-1)**2 + (Y-1)**2)/0.1)

# Multiple hot spots
u = np.ones((ny, nx))
u[10:15, 10:15] = 3
u[20:25, 20:25] = 2
```

### Animate the Diffusion

```python
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    # Perform one diffusion step
    # Update surface plot
    return surf,

ani = FuncAnimation(fig, update, frames=100, interval=50)
plt.show()
```

### Non-Uniform Diffusivity

```python
# Spatially varying diffusivity
nu_field = 0.05 * (1 + 0.5*np.sin(2*np.pi*X/2))

# Update equation becomes more complex
u[1:-1, 1:-1] = un[1:-1, 1:-1] + dt * (...complex expression...)
```

## Troubleshooting

### Problem: Solution Explodes (Oscillates Wildly)

**Cause**: Stability condition violated

**Fixes**:
1. Reduce `sigma` to 0.2 or lower
2. Increase grid resolution (`nx`, `ny`)
3. Decrease diffusivity `nu`

### Problem: Solution Doesn't Change

**Cause**: Time step too small or `nu` too small

**Fixes**:
1. Increase `nt` (more time steps)
2. Increase `nu` (faster diffusion)
3. Check that `dt` is calculated correctly

### Problem: Boundaries Not Respected

**Cause**: Boundary conditions not enforced properly

**Check**: Make sure boundary assignments happen **after** each interior update

## Mathematical Notes

### Analytical Solution

For an initial point source in infinite domain:

```
u(x,y,t) = (M / (4πνt)) · exp(-(x² + y²)/(4νt))
```

Where `M` is the total "mass". This is a **2D Gaussian** that:
- Spreads with standard deviation `σ = √(2νt)`
- Amplitude decreases as `1/t`
- Always remains Gaussian

### Diffusion Length Scale

The characteristic distance diffusion spreads:

```
L_diff(t) ≈ √(4νt)
```

**Example**: With `ν = 0.05 m²/s`, after `t = 1 s`:
```
L_diff = √(4 × 0.05 × 1) = √0.2 ≈ 0.45 m
```

### Maximum Principle

**Important property**: The maximum and minimum values occur:
1. In the initial condition, OR
2. On the boundaries

Interior points **cannot develop new extrema**. This guarantees smoothing!

## References

1. [CFD Python: 12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) by Lorena Barba
2. Crank, J. (1979). "The Mathematics of Diffusion"
3. Carslaw, H.S. & Jaeger, J.C. (1959). "Conduction of Heat in Solids"
4. NumPy Documentation: [Array Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

## License

Based on the CFD Python module by Prof. Lorena A. Barba, shared under Creative Commons Attribution license, CC-BY 4.0.
