# Lesson 5: 2D Linear Convection

## Overview

This lesson extends linear convection from 1D (Lesson 1) to **2 dimensions**, introducing spatial coupling in both x and y directions. The lesson also demonstrates two implementation approaches: **nested loops** (explicit, easy to understand) and **array operations** (vectorized, highly efficient).

## Physical Background

### The 2D Linear Convection Equation

The governing equation is:

```
∂u/∂t + c·∂u/∂x + c·∂u/∂y = 0
```

Where:
- `u(x,y,t)` is the conserved quantity (concentration, temperature, etc.)
- `t` is time
- `x, y` are spatial coordinates
- `c` is the constant wave speed (same in both directions)

**Physical Meaning**: A quantity `u` is transported at constant velocity `c` in both x and y directions simultaneously, resulting in **diagonal motion** at 45° (northeast direction).

### Extension from 1D to 2D

| Aspect | 1D Linear Convection | 2D Linear Convection |
|--------|---------------------|---------------------|
| **Equation** | `∂u/∂t + c·∂u/∂x = 0` | `∂u/∂t + c·∂u/∂x + c·∂u/∂y = 0` |
| **Wave motion** | Left-to-right translation | **Diagonal** translation |
| **Domain** | Line [0, L] | **Plane** [0, Lx] × [0, Ly] |
| **Initial condition** | 1D pulse | **2D pulse** (square, circle, etc.) |
| **Boundary conditions** | 2 points (left, right) | **4 edges** (all sides) |
| **Computational cost** | O(nx·nt) | **O(nx·ny·nt)** - much higher! |

### Physical Interpretation

#### 1. **Transport in 2D**
- Each point moves with velocity `(c, c)` in the (x, y) plane
- The entire field translates diagonally
- **No distortion**: Linear convection preserves shapes (no stretching, rotating, or shearing)

#### 2. **Wavefront Propagation**
- Information propagates along **characteristics**: straight lines at 45°
- Speed of propagation: `|v| = √(c² + c²) = c√2` (diagonal velocity)
- Direction: atan(c_y/c_x) = atan(1) = 45°

#### 3. **Real-World Analogs**
- **Wind transport**: Pollutant carried by constant wind
- **Ocean currents**: Tracer dye in uniform current
- **Conveyor belt**: Object on a moving platform
- **Traffic simulation**: Vehicles moving on a 2D road network (simplified)

### Comparison: 1D vs 2D Behavior

**1D Convection**:
```
Initial: [----[██]----]
After:   [--------[██]]
```
Pulse moves right →

**2D Convection**:
```
Initial:           After:
[----]             [----]
[-██-]      →      [----]
[-██-]             [--██]
[----]             [--██]
```
Pulse moves diagonally ↗

## Mathematical Formulation

### Discretization

Using **backward differences** in both space dimensions:

#### X-direction term:
```
c·∂u/∂x ≈ c·(u[j,i] - u[j,i-1]) / Δx
```

#### Y-direction term:
```
c·∂u/∂y ≈ c·(u[j,i] - u[j-1,i]) / Δy
```

#### Complete discretized equation:

```
u[j,i]ⁿ⁺¹ = u[j,i]ⁿ - c·Δt/Δx·(u[j,i]ⁿ - u[j,i-1]ⁿ) - c·Δt/Δy·(u[j,i]ⁿ - u[j-1,i]ⁿ)
```

### Array Indexing Convention

**IMPORTANT**: In NumPy, arrays are indexed as `u[row, column]` = `u[j, i]`:
- **j (first index)**: row number (y-direction)
- **i (second index)**: column number (x-direction)

```
u[j, i] = u[row, column] = u[y-position, x-position]
```

This can be confusing because we typically think (x, y), but NumPy uses (y, x) or (row, col).

## Numerical Implementation: Two Approaches

### Method 1: Nested For Loops (Explicit)

```python
for n in range(nt + 1):
    un = u.copy()
    for j in range(1, ny-1):      # Loop over rows (y-direction)
        for i in range(1, nx-1):  # Loop over columns (x-direction)
            u[j,i] = (un[j,i] 
                      - c * dt / dx * (un[j,i] - un[j,i-1])  # x-direction
                      - c * dt / dy * (un[j,i] - un[j-1,i])) # y-direction
    
    # Enforce boundary conditions
    u[0, :] = 1   # Bottom
    u[-1, :] = 1  # Top
    u[:, 0] = 1   # Left
    u[:, -1] = 1  # Right
```

**Advantages**:
- ✓ **Clear and intuitive**: Easy to understand what's happening at each point
- ✓ **Direct mapping**: One-to-one correspondence with the mathematical equation
- ✓ **Easy to debug**: Can print values at specific (i, j) locations
- ✓ **Flexible**: Easy to add complex logic or non-uniform grids

**Disadvantages**:
- ✗ **Slow in Python**: Loops are interpreted, not compiled
- ✗ **Verbose**: More lines of code
- ✗ **Cache inefficient**: Random memory access patterns

### Method 2: Array Operations (Vectorized)

```python
for n in range(nt + 1):
    un = u.copy()
    u[1:, 1:] = (un[1:, 1:] 
                 - c * dt / dx * (un[1:, 1:] - un[1:, :-1])   # x-direction
                 - c * dt / dy * (un[1:, 1:] - un[:-1, 1:]))  # y-direction
    
    # Enforce boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
```

**Advantages**:
- ✓ **Fast**: ~10-100× faster (NumPy uses optimized C/Fortran)
- ✓ **Concise**: One line instead of nested loops
- ✓ **Memory efficient**: Contiguous memory operations
- ✓ **Parallelizable**: Can leverage SIMD/GPU acceleration

**Disadvantages**:
- ✗ **Less intuitive**: Slicing syntax can be confusing initially
- ✗ **Harder to debug**: Can't easily inspect individual operations
- ✗ **Memory overhead**: Creates temporary arrays for intermediate results

### Array Slicing Explained

Understanding the slices:

```python
u[1:, 1:]     # Interior points: rows 1→end, cols 1→end
un[1:, :-1]   # Left neighbors:  rows 1→end, cols 0→(end-1)
un[:-1, 1:]   # Down neighbors:  rows 0→(end-1), cols 1→end
```

**Visual Example** (5×5 grid):

```
Target points: u[1:, 1:]
       col: 0    1    2    3    4
row 0:      [.]  [.]  [.]  [.]  [.]
row 1:      [.]  [X]  [X]  [X]  [X]
row 2:      [.]  [X]  [X]  [X]  [X]
row 3:      [.]  [X]  [X]  [X]  [X]
row 4:      [.]  [X]  [X]  [X]  [X]

Left neighbors: un[1:, :-1]
       col: 0    1    2    3    4
row 0:      [.]  [.]  [.]  [.]  [.]
row 1:      [X]  [X]  [X]  [X]  [.]
row 2:      [X]  [X]  [X]  [X]  [.]
row 3:      [X]  [X]  [X]  [X]  [.]
row 4:      [X]  [X]  [X]  [X]  [.]
                ← shifted left by 1

Down neighbors: un[:-1, 1:]
       col: 0    1    2    3    4
row 0:      [.]  [X]  [X]  [X]  [X]
row 1:      [.]  [X]  [X]  [X]  [X]
row 2:      [.]  [X]  [X]  [X]  [X]
row 3:      [.]  [X]  [X]  [X]  [X]
row 4:      [.]  [.]  [.]  [.]  [.]
                ↑ shifted down by 1
```

When you compute `un[1:, 1:] - un[1:, :-1]`, each point is subtracted from its **left neighbor**, implementing the backward difference in x!

## Performance Comparison: Nested Loops vs Array Operations

### Computational Complexity

Both methods have the same **algorithmic complexity**:
```
O(nx · ny · nt)
```

However, **wall-clock time** differs dramatically due to implementation overhead.

### Timing Analysis

For a typical simulation with `nx = 81, ny = 81, nt = 100`:

| Implementation | Approximate Time | Speedup |
|---------------|------------------|---------|
| **Nested Loops** | ~2-5 seconds | 1× (baseline) |
| **Array Operations** | ~0.02-0.1 seconds | **20-100×** |

### Why is Vectorization So Much Faster?

#### 1. **Interpreted vs Compiled**
- **Loops**: Each iteration interpreted by Python (slow)
- **Arrays**: NumPy calls pre-compiled C/Fortran code (fast)

#### 2. **Memory Access Patterns**
- **Loops**: Random access, poor cache utilization
- **Arrays**: Contiguous memory, cache-friendly

#### 3. **Overhead**
- **Loops**: Function call overhead for each operation
- **Arrays**: Single function call for entire operation

#### 4. **SIMD (Single Instruction, Multiple Data)**
- **Loops**: One operation at a time
- **Arrays**: Modern CPUs can process multiple elements simultaneously

### Memory Usage

**Nested Loops**:
```
Memory ≈ 2 × nx × ny × sizeof(float64)
       ≈ 2 × 81 × 81 × 8 bytes
       ≈ 105 KB
```
(Just `u` and `un` arrays)

**Array Operations**:
```
Memory ≈ (2 + temporary arrays) × nx × ny × sizeof(float64)
       ≈ ~3-4 × 81 × 81 × 8 bytes
       ≈ 160-210 KB
```
(Additional memory for intermediate slices like `un[1:, :-1]`)

**Verdict**: Array operations use ~50% more memory, but for small problems this is negligible compared to the massive speed gain.

### Scalability

As grid size increases:

| Grid Size | nx × ny | Nested Loops | Array Ops | Speedup |
|-----------|---------|--------------|-----------|---------|
| Small | 41 × 41 | ~0.5 s | ~0.01 s | 50× |
| Medium | 81 × 81 | ~2 s | ~0.05 s | 40× |
| Large | 161 × 161 | ~8 s | ~0.2 s | 40× |
| Very Large | 321 × 321 | ~30 s | ~0.8 s | 37× |

**Trend**: Speedup remains roughly constant (Python overhead dominates).

### Recommendation

> [!TIP]
> **Always use array operations for production code!** Use nested loops only for:
> - Learning and understanding
> - Debugging specific issues
> - Prototyping complex logic
> - Non-uniform grids or irregular stencils

## Simulation Parameters

```python
nx = 81              # Grid points in x
ny = 81              # Grid points in y
nt = 100             # Time steps
c = 1                # Wave speed [m/s]
dx = 2 / (nx - 1)    # Spatial step x ≈ 0.0247 m
dy = 2 / (ny - 1)    # Spatial step y ≈ 0.0247 m
sigma = 0.2          # CFL-like parameter
dt = sigma * dx      # Time step ≈ 0.00494 s
```

### Initial Condition

A **square pulse** in 2D:
- `u = 2` for `0.5 ≤ x ≤ 1.0` and `0.5 ≤ y ≤ 1.0`
- `u = 1` everywhere else

This creates a "raised platform" in the center of the domain.

### Boundary Conditions

**Dirichlet boundaries** (fixed values):
```
u = 1 on all four edges (left, right, top, bottom)
```

This is enforced by setting:
```python
u[0, :] = 1    # Bottom edge (y = 0)
u[-1, :] = 1   # Top edge (y = 2)
u[:, 0] = 1    # Left edge (x = 0)
u[:, -1] = 1   # Right edge (x = 2)
```

## Stability Analysis

### 2D CFL Condition

For 2D convection, stability requires:

```
CFL_x + CFL_y ≤ 1

Where: CFL_x = c·Δt/Δx
       CFL_y = c·Δt/Δy
```

**More restrictive form**:
```
c·Δt·(1/Δx + 1/Δy) ≤ 1
```

### Current Simulation Stability

With `Δx = Δy = 0.0247` and `Δt = 0.00494`:

```
CFL_x = 1 × 0.00494 / 0.0247 = 0.2
CFL_y = 1 × 0.00494 / 0.0247 = 0.2

CFL_total = 0.2 + 0.2 = 0.4 < 1 ✓
```

**Status**: ✓ **Stable**

> [!IMPORTANT]
> In 2D (and 3D), the CFL condition becomes **more restrictive** because you must satisfy it in all dimensions simultaneously. The effective maximum time step is reduced.

## Expected Results

### Evolution Sequence

1. **t = 0**: Square pulse centered at (x, y) ≈ (0.75, 0.75)
2. **Early times**: Pulse begins moving toward upper-right corner
3. **Mid times**: Pulse travels diagonally, maintaining shape
4. **Late times**: Pulse approaches top-right corner, may exit domain

### Key Observations

- **Shape preservation**: The square pulse maintains its shape (linear equation)
- **Diagonal motion**: Moves at 45° angle (equal x and y velocities)
- **Boundary interaction**: When pulse reaches boundary, it's "absorbed" (Dirichlet BC)
- **Identical results**: Both implementations (loops vs arrays) produce the same answer!

## Running the Code

```bash
python Lesson5_2D_linear_convection.py
```

The script will:
1. Display initial condition (3D surface plot)
2. Solve using nested loops
3. Display result from nested loops
4. Solve using array operations
5. Display result from array operations
6. Both final results should be identical!

## Key Takeaways

### Physical Insights

1. **2D extends transport to planes**
   - Same physics as 1D, just more dimensions
   - Wave propagates along characteristics (straight lines)
   - Velocity vector determines direction

2. **Directionality matters**
   - Different velocities in x vs y → transport at an angle
   - Equal velocities → 45° diagonal motion
   - Shape preservation (linear equation property)

3. **Boundaries become critical**
   - 4 edges instead of 2 points
   - Boundary conditions strongly affect interior solution
   - Information propagates from boundaries along characteristics

### Numerical Insights

1. **2D significantly increases computational cost**
   - O(n) → O(n²) for spatial discretization
   - Memory and time requirements scale quadratically
   - Need efficient implementations for large problems

2. **Vectorization is essential**
   - Python loops are too slow for production
   - NumPy array operations provide 20-100× speedup
   - Minimal change to algorithm, massive performance gain

3. **Stability is more restrictive**
   - Must satisfy CFL in all dimensions
   - Maximum time step smaller than in 1D
   - Trade-off: accuracy vs computational cost

4. **Both methods are pedagogically valuable**
   - Loops: understand the algorithm
   - Arrays: implement efficiently
   - Know both, use arrays in practice!

## Extensions and Experiments

### Different Initial Conditions

Try different shapes:

```python
# Circular pulse
for i in range(nx):
    for j in range(ny):
        if (x[i]-1)**2 + (y[j]-1)**2 < 0.25**2:
            u[j, i] = 2

# Gaussian hump
u = 1 + np.exp(-((X-1)**2 + (Y-1)**2) / 0.1)
```

### Different Velocities

Non-equal velocities in x and y:

```python
cx = 1.0  # Wave speed in x
cy = 0.5  # Wave speed in y (half as fast)

# Update equation
u[j,i] = un[j,i] - cx*dt/dx*(un[j,i]-un[j,i-1]) - cy*dt/dy*(un[j,i]-un[j-1,i])
```

**Result**: Pulse moves at steeper/shallower angle

### Periodic Boundaries

Wrap-around instead of fixed:

```python
# Instead of u[0,:]=1, use:
u[0, :] = u[-2, :]   # Bottom = top
u[-1, :] = u[1, :]   # Top = bottom
u[:, 0] = u[:, -2]   # Left = right
u[:, -1] = u[:, 1]   # Right = left
```

**Result**: Pulse wraps around domain (toroidal topology)

### Performance Benchmarking

Measure actual speedup:

```python
import time

# Method 1: Nested loops
start = time.time()
# ... solver with loops ...
time_loops = time.time() - start

# Method 2: Array operations
start = time.time()
# ... solver with arrays ...
time_arrays = time.time() - start

print(f"Loops: {time_loops:.4f} s")
print(f"Arrays: {time_arrays:.4f} s")
print(f"Speedup: {time_loops/time_arrays:.1f}×")
```

### Visualization Improvements

Animate the solution:

```python
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    # Advance solution one timestep
    # Update surface plot
    return surf,

ani = FuncAnimation(fig, update, frames=nt, interval=50)
plt.show()
```

## Troubleshooting

### Problem: Both Methods Give Different Results

**Cause**: Likely a bug in one implementation

**Check**:
1. Are loop ranges identical? (`range(1, nx-1)` vs `u[1:, 1:]`)
2. Are boundary conditions applied the same way?
3. Is the initial condition set identically?

### Problem: Solution Explodes

**Cause**: CFL condition violated

**Fix**: Reduce `sigma` or increase grid resolution

### Problem: Array Version is Slow

**Cause**: Not actually using vectorization

**Check**: Make sure you're not looping over `u[1:, 1:]` element by element!

## Mathematical Notes

### Characteristics

The equation `∂u/∂t + c·∂u/∂x + c·∂u/∂y = 0` has characteristics:

```
dx/dt = c
dy/dt = c
du/dt = 0
```

**Solution**: `u` is constant along lines `(x - ct, y - ct) = constant`

These are diagonal lines at 45° in the (x, y) plane.

### General Form

For arbitrary velocities:

```
∂u/∂t + cx·∂u/∂x + cy·∂u/∂y = 0
```

**Direction of propagation**: `θ = atan(cy/cx)`  
**Speed of propagation**: `|v| = √(cx² + cy²)`

## References

1. [CFD Python: 12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) by Lorena Barba
2. LeVeque, R.J. (2002). "Finite Volume Methods for Hyperbolic Problems"
3. NumPy Documentation: [Array indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
4. Python Performance Tips: [Performance Python](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)

## License

Based on the CFD Python module by Prof. Lorena A. Barba, shared under Creative Commons Attribution license, CC-BY 4.0.
