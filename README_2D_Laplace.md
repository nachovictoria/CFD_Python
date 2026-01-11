# Lesson 9: 2D Laplace Equation

## Overview

This lesson solves the **2D Laplace equation**, a fundamental **elliptic partial differential equation** that describes steady-state fields with **no sources or sinks**. Unlike previous lessons with time evolution, the Laplace equation represents **equilibrium states** - the final configuration after all transients have died out.

The solution uses an **iterative method** (Jacobi iteration) with **convergence checking** to find the steady-state distribution.

## Physical Background

### The 2D Laplace Equation

The governing equation is:

```
∂²p/∂x² + ∂²p/∂y² = 0

Or equivalently: ∇²p = 0
```

Where:
- `p(x,y)` is a scalar field (pressure, potential, temperature, etc.)
- `∇²` is the **Laplacian operator** (measures curvature)
- The equation states: **total curvature is zero** everywhere

**Physical Meaning**: The field `p` has **zero net curvature** at every point. This means the value at each point is the **average of its neighbors** - a state of perfect balance.

### Physical Interpretation

#### 1. **Equilibrium State**

The Laplace equation describes systems at **steady-state equilibrium**:

- **No time dependence**: ∂p/∂t = 0
- **No creation or destruction**: No source terms
- **Pure conduction/diffusion**: After infinite time

**Example**: Heat distribution in a metal plate after waiting forever (all transients gone).

#### 2. **Harmonic Function**

Solutions to Laplace's equation are called **harmonic functions**. Properties:

- **Mean value property**: Value at any point = average of nearby values
- **No local extrema**: Maximum and minimum only on boundaries
- **Smooth**: Infinitely differentiable
- **Unique**: Given boundary conditions, solution is unique

#### 3. **Physical Analogs**

The Laplace equation appears in many physical contexts:

| Physical System | Variable `p` | Interpretation |
|----------------|--------------|----------------|
| **Electrostatics** | Electric potential | Voltage field (no charges) |
| **Steady heat** | Temperature | Thermal equilibrium |
| **Fluid flow** | Velocity potential | Irrotational, incompressible flow |
| **Elasticity** | Displacement | Membrane under tension |
| **Gravitation** | Gravitational potential | Field in empty space |

#### 4. **The Zero Curvature Condition**

Mathematically, `∇²p = 0` means **zero total curvature**:

```
∂²p/∂x² + ∂²p/∂y² = 0

This means: Curvature in x + Curvature in y = 0

If p curves up in x (∂²p/∂x² > 0), 
it must curve down in y (∂²p/∂y² < 0) by the same amount!
```

**Geometric interpretation**: The surface has **saddle points** everywhere - no peaks or valleys in the interior.

### Comparison with Other Equations

| Equation | Form | Physical Meaning |
|----------|------|------------------|
| **Diffusion** | `∂u/∂t = ν·∇²u` | Time evolution toward equilibrium |
| **Laplace** | `∇²p = 0` | **Already at equilibrium** |
| **Poisson** | `∇²p = f(x,y)` | Equilibrium with sources |
| **Wave** | `∂²u/∂t² = c²·∇²u` | Oscillatory dynamics |

**Key difference**: Laplace is **steady-state** (no time), while diffusion **evolves** toward this state.

## Mathematical Formulation

### Discretization

Using **central differences**:

#### Second derivative in x:
```
∂²p/∂x² ≈ (p[j, i+1] - 2p[j, i] + p[j, i-1]) / Δx²
```

#### Second derivative in y:
```
∂²p/∂y² ≈ (p[j+1, i] - 2p[j, i] + p[j-1, i]) / Δy²
```

#### Discrete Laplace equation:
```
(p[j, i+1] - 2p[j, i] + p[j, i-1]) / Δx² + 
(p[j+1, i] - 2p[j, i] + p[j-1, i]) / Δy² = 0
```

### Jacobi Iteration Formula

Solving for `p[j, i]`:

```
p[j,i] = (Δy²·(p[j,i+1] + p[j,i-1]) + Δx²·(p[j+1,i] + p[j-1,i])) / (2·(Δx² + Δy²))
```

**For square grids** (Δx = Δy):
```
p[j,i] = (p[j,i+1] + p[j,i-1] + p[j+1,i] + p[j-1,i]) / 4
```

**Physical meaning**: Each point's value is the **weighted average** of its 4 neighbors. This is the discrete version of "zero curvature."

### The 5-Point Stencil

```
         p[j-1,i]
             ↑
p[j,i-1] ← p[j,i] → p[j,i+1]
             ↓
         p[j+1,i]
```

The central point depends on its **4 nearest neighbors** (north, south, east, west).

## Boundary Conditions

### Types Implemented

```python
p[:, 0] = 0              # Left (x=0): Dirichlet BC, p = 0
p[:, -1] = y             # Right (x=2): Dirichlet BC, p = y
p[0, :] = p[1, :]        # Bottom (y=0): Neumann BC, ∂p/∂y = 0
p[-1, :] = p[-2, :]      # Top (y=2): Neumann BC, ∂p/∂y = 0
```

### Boundary Condition Types

#### **Dirichlet BC** (Specified value):
- **Left edge**: `p = 0` (ground potential)
- **Right edge**: `p = y` (linear increase from 0 to 2)

**Implementation**: Directly set values
```python
p[:, 0] = 0
p[:, -1] = y
```

#### **Neumann BC** (Specified gradient):
- **Top and bottom**: `∂p/∂y = 0` (insulated, no flux)

**Implementation**: Set edge equal to neighbor
```python
p[0, :] = p[1, :]    # Makes (p[1,:] - p[0,:])/dy ≈ 0
```

### Physical Meaning of BCs

Interpreting as **electrostatic potential**:
- **Left**: Grounded (0 volts)
- **Right**: Voltage increases linearly (0V at bottom to 2V at top)
- **Top/Bottom**: Insulated (no current flow perpendicular to boundary)

## Simulation Parameters

```python
nx = 31              # Grid points in x
ny = 31              # Grid points in y
dx = 2 / (nx - 1)    # Spatial step x ≈ 0.0667 m
dy = 2 / (ny - 1)    # Spatial step y ≈ 0.0667 m
max_iter = 10000     # Maximum iterations
tol = 1e-4           # Convergence tolerance
```

### Physical Meaning

#### **Grid Resolution** (nx, ny):
- **31×31 = 961 points**: Moderate resolution
- **Trade-off**: More points → better accuracy, slower convergence
- **Typical**: 31-101 points per dimension for educational purposes

#### **Convergence Tolerance** (tol):
- **1e-4**: Stop when solution changes by less than 0.0001
- **Smaller tol**: More accurate, more iterations
- **Larger tol**: Faster, less accurate

#### **Maximum Iterations**:
- **Safety limit**: Prevents infinite loops
- **Typical convergence**: 1000-5000 iterations for this problem
- If max reached: Either tolerance too strict or numerical issue

## Iterative Solution Method

### Why Iteration?

Unlike explicit time-stepping, Laplace equation has **no natural time scale**. We use iteration:

1. Start with initial guess (satisfying BCs)
2. Update each point based on neighbors
3. Repeat until solution stops changing
4. **Converged solution** satisfies Laplace equation

### Jacobi Iteration Algorithm

```python
for iter in range(max_iter):
    pn = p.copy()  # Store old values
    
    # Update interior points
    p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) + 
                      dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1])) / 
                     (2 * (dx**2 + dy**2)))
    
    # Enforce boundary conditions
    p[:, 0] = 0
    p[:, -1] = y
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    
    # Check convergence
    if ||p - pn|| < tol:
        break  # Converged!
```

**Key features**:
- **Jacobi**: Uses old values for all updates (parallel, slower convergence)
- **Alternative (Gauss-Seidel)**: Uses new values immediately (serial, faster)

### Convergence Rate

**Typical behavior**:
```
Iteration 100:  diff = 1.5e-2
Iteration 500:  diff = 3.2e-3
Iteration 1000: diff = 8.1e-4
Iteration 2000: diff = 2.0e-4
Iteration 3500: diff = 9.8e-5  ← Converged!
```

Convergence is **exponential** but slow for fine grids.

## Convergence Criteria: L1 vs L2 Norm

### The Convergence Check

After each iteration, check: **Has the solution stopped changing?**

```python
diff = ||p - pn||  # Some norm of difference
if diff < tol:
    break  # Converged
```

### L2 Norm (Euclidean) - Currently Implemented

```python
diff = np.linalg.norm(p - pn)  # L2 norm
```

**Formula**:
```
||Δp||₂ = √(Σᵢⱼ |Δp[i,j]|²)
```

**Physical interpretation**: 
- Measures "energy" of the error field
- **Root mean square** (RMS) error × √(number of points)
- Emphasizes **large errors** (squared)

### L1 Norm (Manhattan)

```python
diff = np.sum(np.abs(p - pn))  # L1 norm
```

**Formula**:
```
||Δp||₁ = Σᵢⱼ |Δp[i,j]|
```

**Physical interpretation**:
- Measures total absolute change
- **Sum** of all pointwise errors
- All errors weighted equally

### Comparison Table

| Aspect | L2 Norm | L1 Norm |
|--------|---------|---------|
| **Formula** | `√(Σ Δp²)` | `Σ |Δp|` |
| **Sensitivity to outliers** | **High** - squares amplify | **Low** - linear |
| **Physical meaning** | Energy/RMS error | Total change |
| **Strictness** | Stricter on large errors | More lenient |
| **Computational cost** | Slightly higher (sqrt) | Lower |
| **Standard for** | Elliptic PDEs, optimization | L1 minimization, robustness |
| **Convergence** | Uniform accuracy | Total accuracy |

### Example: Error Distribution

Suppose after an iteration, errors are:
```
Cell 1: Δp = 0.01
Cells 2-30: Δp = 0.001 each
```

**L2 norm**:
```
||Δp||₂ = √(0.01² + 29×0.001²) 
        = √(0.0001 + 0.000029)
        = √0.000129 ≈ 0.0114
```
The large error (0.01) dominates!

**L1 norm**:
```
||Δp||₁ = 0.01 + 29×0.001
        = 0.01 + 0.029
        = 0.039
```
All errors contribute proportionally.

### When to Use Each

#### **Use L2 Norm** (recommended for Laplace) ✓

**Advantages**:
- ✓ Ensures **uniform accuracy** across domain
- ✓ Standard in scientific computing for elliptic PDEs
- ✓ Penalizes large local errors heavily
- ✓ Well-studied mathematical properties
- ✓ Related to energy minimization

**Use when**:
- Solving elliptic PDEs (Laplace, Poisson)
- Need smooth, uniformly accurate solutions
- Want to avoid localized errors

#### **Use L1 Norm**

**Advantages**:
- ✓ Faster computation (no squaring/square root)
- ✓ More robust to occasional spikes
- ✓ Easier to interpret (direct sum)

**Use when**:
- Computational efficiency critical
- Some local inaccuracy acceptable
- Dealing with sparse or irregular data
- Want robust convergence metric

### L∞ Norm (Maximum Norm)

There's also the **infinity norm**:

```python
diff = np.max(np.abs(p - pn))  # L∞ norm
```

**Formula**: 
```
||Δp||∞ = max |Δp[i,j]|
```

**Physical meaning**: Largest single pointwise error

**Use when**: Need **guaranteed accuracy everywhere** (strictest criterion)

### Tolerance Scaling

For the **same solution quality**, different norms need different tolerances:

```
L∞: tol = 1e-6     (strictest - one point)
L2:  tol = 1e-4     (moderate - RMS)
L1:   tol = 1e-2    (lenient - sum of all)
```

**Rule of thumb** for 31×31 grid:
- L2 tolerance ≈ √(nx × ny) × L∞ tolerance ≈ 31 × L∞
- L1 tolerance ≈ (nx × ny) × L∞ tolerance ≈ 961 × L∞

### Implementation Comparison

```python
# L2 norm (current)
for iter in range(max_iter):
    pn = p.copy()
    # ... update p ...
    diff_L2 = np.linalg.norm(p - pn)
    if diff_L2 < tol:
        print(f"L2 converged: {iter} iterations")
        break

# L1 norm (alternative)
for iter in range(max_iter):
    pn = p.copy()
    # ... update p ...
    diff_L1 = np.sum(np.abs(p - pn))
    if diff_L1 < tol * nx * ny:  # Adjust tolerance!
        print(f"L1 converged: {iter} iterations")
        break

# L∞ norm (strictest)
for iter in range(max_iter):
    pn = p.copy()
    # ... update p ...
    diff_Linf = np.max(np.abs(p - pn))
    if diff_Linf < tol / np.sqrt(nx * ny):  # Adjust tolerance!
        print(f"L∞ converged: {iter} iterations")
        break
```

### Recommendation for Laplace Equation

**Use L2 norm** (as implemented) because:

1. ✅ Standard practice for elliptic PDEs
2. ✅ Ensures smooth, high-quality solutions
3. ✅ Directly related to minimizing discrete energy
4. ✅ Well-understood convergence theory
5. ✅ Balanced between L1 (too lenient) and L∞ (too strict)

**Consider L1** only if:
- Running millions of iterations (optimization matters)
- Have robust tolerance values pre-determined
- Dealing with noisy or irregular problems

## Expected Results

### Solution Characteristics

The converged solution will:

1. **Satisfy boundary conditions exactly**
   - Left edge: p = 0
   - Right edge: p = y (0 to 2)
   - Top/bottom: zero gradient

2. **Be smooth** (no oscillations or discontinuities)

3. **Have no interior extrema**
   - Maximum values only on boundaries
   - Saddle-like surface throughout

4. **Show linear interpolation**
   - With these simple BCs, solution is nearly linear from left to right
   - Contours are approximately vertical lines

### Typical Convergence

```
Iteration 0:    diff = 2.51e+00  (large initial change)
Iteration 500:  diff = 3.45e-03
Iteration 1000: diff = 8.64e-04
Iteration 1500: diff = 2.16e-04
Iteration 2000: diff = 5.40e-05  ← Converged with tol=1e-4
```

**Total iterations**: ~2000-4000 for 31×31 grid with tol=1e-4

## Visualization

### 3D Surface Plot

Shows the field `p(x, y)` as a surface:
- **Height**: Value of p
- **Color**: Also represents p (redundant but helpful)
- **Features**: Smooth interpolation from p=0 (left) to p=y (right)

### Contour Plot

Shows **level curves** (lines of constant p):
- Like topographic map
- Contours perpendicular to gradient
- Spacing shows steepness (close = steep, far = gentle)
- For this problem: approximately vertical lines

## Running the Code

```bash
python Lesson8_2D_Laplace_equation.py
```

The script will:
1. Display initial condition (zeros with BCs)
2. Solve iteratively with convergence reporting
3. Display final solution as 3D surface
4. Display contour plot of solution

## Key Takeaways

### Physical Insights

1. **Laplace = ultimate equilibrium**
   - No time dependence
   - Perfect balance everywhere
   - Found in many natural steady states

2. **Boundary conditions determine everything**
   - Solution is **unique** given BCs
   - Interior automatically adjusts
   - Change BCs → completely different solution

3. **Harmonic functions are special**
   - Mean value property
   - No interior extrema
   - Infinitely smooth
   - Fundamental in physics and mathematics

4. **Connection to other equations**
   - Laplace is the **limit** of diffusion as t→∞
   - Adding sources → Poisson equation
   - Foundation for more complex PDEs

### Numerical Insights

1. **Iteration is necessary**
   - No natural time scale for "marching"
   - Successive approximation converges
   - Slow but reliable for moderate-sized problems

2. **Jacobi vs Gauss-Seidel**
   - Jacobi: slower, parallelizable
   - Gauss-Seidel: faster (~2×), sequential
   - SOR (Successive Over-Relaxation): even faster

3. **Convergence metrics matter**
   - L2 standard for elliptic PDEs
   - Different norms → different accuracy measures
   - Tolerance must be chosen appropriately

4. **Grid refinement is expensive**
   - 2× finer grid → 4× more points → 4× slower per iteration
   - Also more iterations needed
   - Total cost scales roughly as O(n⁴) for grid size n!

## Extensions and Experiments

### Different Boundary Conditions

```python
# All Dirichlet (fixed values)
p[:, 0] = 0
p[:, -1] = 1
p[0, :] = 0
p[-1, :] = 0

# Circular source (Poisson equation hint)
p[ny//2, nx//2] = 10  # Hot spot

# Sinusoidal boundary
p[:, -1] = np.sin(np.pi * y)
```

### Faster Methods

```python
# Gauss-Seidel (don't copy, use new values immediately)
for iter in range(max_iter):
    # NO pn = p.copy()
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            p[j,i] = ... # Uses updated values

# Successive Over-Relaxation (SOR)
omega = 1.5  # Relaxation parameter
p_new = (Jacobi update)
p[j,i] = omega * p_new + (1 - omega) * p[j,i]
```

### Solve Poisson Equation

Add a source term:

```
∇²p = f(x,y)
```

```python
# Add source
f = np.exp(-((X-1)**2 + (Y-1)**2)/0.1)  # Gaussian source

# Modified update
p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) + 
                  dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1]) - 
                  dx**2 * dy**2 * f[1:-1, 1:-1]) /  # Source term added!
                 (2 * (dx**2 + dy**2)))
```

### Analytic Solution Comparison

For simple BCs, analytic solutions exist:

```python
# Analytic for p=0 left, p=y right, Neumann top/bottom
p_analytic = np.outer(y, x/2)  # p = y*x/2

# Compare
error = np.max(np.abs(p - p_analytic))
print(f"Maximum error vs analytic: {error:.2e}")
```

## Troubleshooting

### Problem: Not Converging

**Symptoms**: Reaches max_iter without convergence

**Causes**:
1. Tolerance too strict
2. Inconsistent boundary conditions
3. Numerical instability

**Fixes**:
- Increase tolerance to 1e-3
- Check BC compatibility
- Verify BC enforcement in loop

### Problem: Slow Convergence

**Solution**: Use faster methods
- Gauss-Seidel: ~2× faster
- SOR: 5-10× faster
- Multigrid: 100× faster (advanced)

### Problem: Oscillations in Solution

**Cause**: Likely programming error (not Laplace issue)

**Check**: Boundary conditions enforced correctly every iteration

## Mathematical Notes

### Uniqueness Theorem

**Theorem**: Given Dirichlet BCs on entire boundary, solution to Laplace equation is **unique**.

**Implication**: Your iterative solution will converge to the one and only answer (if it converges).

### Maximum Principle

**Theorem**: Maximum (and minimum) of a harmonic function occur on the boundary.

**Consequence**: No peaks or valleys inside - only saddle points!

### Green's Function

Laplace equation can be solved using Green's functions:

```
p(x,y) = ∫∫ G(x,y;x',y') f(x',y') dx'dy'
```

Where G is the **fundamental solution** (response to point source).

## References

1. [CFD Python: 12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) by Lorena Barba
2. Evans, L.C. (2010). "Partial Differential Equations"
3. Strang, G. (2007). "Computational Science and Engineering"
4. Press, W.H. et al. (2007). "Numerical Recipes"
5. Saad, Y. (2003). "Iterative Methods for Sparse Linear Systems"

## License

Based on the CFD Python module by Prof. Lorena A. Barba, shared under Creative Commons Attribution license, CC-BY 4.0.
