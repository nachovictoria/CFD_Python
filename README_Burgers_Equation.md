# Lesson 4: Burgers' Equation

## Overview

This lesson implements the **Burgers' equation**, which combines **non-linear convection** (Lesson 2) and **diffusion** (Lesson 3) into a single equation. This is a fundamental equation in fluid dynamics that exhibits both wave steepening and viscous smoothing, and it serves as a simplified model for the Navier-Stokes equations.

**Unique Feature**: This lesson includes an **analytical solution** using symbolic mathematics (SymPy), allowing direct comparison between numerical and exact solutions to study discretization errors.

## Physical Background

### Burgers' Equation

The governing equation is:

```
∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
```

Where:
- `u(x,t)` is the velocity field
- `t` is time
- `x` is the spatial coordinate
- `ν` (nu) is the kinematic viscosity

**Structure**: 
```
∂u/∂t  +  u·∂u/∂x  =  ν·∂²u/∂x²
  ↓         ↓              ↓
Time    Non-linear     Diffusion
rate     Convection   (smoothing)
```

### Physical Interpretation

Burgers' equation models the competition between two fundamental mechanisms:

#### 1. **Non-Linear Convection** (`u·∂u/∂x`)
- **Effect**: Wave steepening and shock formation
- **Action**: Sharpens gradients
- **Energy**: Conservative (in inviscid limit)
- **From Lesson 2**: Self-advection causes faster regions to catch slower ones

#### 2. **Viscous Diffusion** (`ν·∂²u/∂x²`)
- **Effect**: Smoothing and spreading
- **Action**: Reduces gradients
- **Energy**: Dissipative
- **From Lesson 3**: Molecular mixing smooths sharp features

#### The Competition:

```
Non-linear convection tries to create shocks
              ⇅
    Viscosity tries to smooth everything
              ⇅
        BALANCE determines behavior
```

### Physical Regimes

The behavior depends on the **Reynolds number**:

```
Re = U·L/ν

Where: U = characteristic velocity
       L = characteristic length
       ν = viscosity
```

| Reynolds Number | Dominant Process | Behavior |
|----------------|------------------|----------|
| **Re << 1** | Diffusion | Smooth, parabolic-like solutions |
| **Re ≈ 1** | **Balanced** | Rich dynamics, both effects visible |
| **Re >> 1** | Convection | Shock-like features with thin layers |

Our simulation: `Re ≈ 10-30` (**balanced regime**)

### Real-World Applications

1. **Shock waves in fluids**
   - Gas dynamics with viscosity
   - Sound waves with finite amplitude
   - Blast wave propagation

2. **Traffic flow**
   - Cars (particles) with finite spacing (viscosity)
   - Shock waves = traffic jams
   - Smooth flow vs. congestion

3. **Simplified Navier-Stokes**
   - 1D model of momentum equation
   - Captures essential physics
   - Used for testing numerical schemes

4. **Turbulence modeling**
   - Energy cascade mechanism
   - Non-linearity + dissipation

## Mathematical Formulation

### Comparison Table

| Equation | Formula | Characteristics |
|----------|---------|----------------|
| **Linear Convection** | `∂u/∂t + c·∂u/∂x = 0` | Shape preservation |
| **Non-linear Convection** | `∂u/∂t + u·∂u/∂x = 0` | Wave steepening, shocks |
| **Diffusion** | `∂u/∂t = ν·∂²u/∂x²` | Smoothing, spreading |
| **Burgers'** | `∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²` | **Both steepening & smoothing** |
| **Navier-Stokes (1D)** | Similar + pressure | Full fluid dynamics |

### Discretization

Combining the schemes from Lessons 2 and 3:

#### Non-Linear Convection Term:
```
u·∂u/∂x ≈ u[i] · (u[i] - u[i-1]) / Δx
```
(Backward difference in space)

#### Diffusion Term:
```
ν·∂²u/∂x² ≈ ν · (u[i+1] - 2u[i] + u[i-1]) / Δx²
```
(Central difference for second derivative)

#### Complete Discretized Equation:

```python
u[i]ⁿ⁺¹ = u[i]ⁿ - u[i]ⁿ · Δt/Δx · (u[i]ⁿ - u[i-1]ⁿ) 
               + ν · Δt/Δx² · (u[i+1]ⁿ - 2u[i]ⁿ + u[i-1]ⁿ)
```

### Periodic Boundary Conditions

This implementation uses **periodic boundaries** (domain is a circle):
- `u[0] = u[nx-1]` (wrap around)
- Physically: what exits on the right re-enters on the left
- Mathematically: domain is `[0, 2π]` with periodicity

## Analytical Solution

### Cole-Hopf Transformation

Burgers' equation has an exact solution using the **Cole-Hopf transformation**:

1. Transform: `u = -2ν · (∂φ/∂x) / φ + C`
2. This maps Burgers' (nonlinear) → Heat equation (linear) for `φ`
3. Solve for `φ`, then back-transform to get `u`

### Implemented Solution

The code uses a specific analytical solution:

```python
φ = exp(-(x - 4t)²/(4ν(t+1))) + exp(-(x - 4t - 2π)²/(4ν(t+1)))
u = -2ν · (∂φ/∂x)/φ + 4
```

This represents:
- **Two Gaussian bumps** that propagate and diffuse
- **Periodic** in space (period = 2π)
- **Exact solution** for comparison with numerical results

## Numerical Implementation

### Code Structure

```python
# Step 1: Define analytical solution symbolically
x, nu, t = sympy.symbols('x nu t')
phi = (exp(-(x - 4*t)²/(4*nu*(t+1))) + 
       exp(-(x - 4*t - 2*π)²/(4*nu*(t+1))))
u_analytical = -2*nu*(sympy.diff(phi,x)/phi) + 4

# Step 2: Convert to numerical function
ufunc = sympy.utilities.lambdify((t, x, nu), u_analytical)

# Step 3: Set initial condition from analytical solution
u = np.asarray([ufunc(0, x0, nu) for x0 in x])

# Step 4: Time evolution loop
for n in range(nt):
    un = u.copy()
    
    # Interior points
    for i in range(1, nx-1):
        u[i] = un[i] - un[i]*dt/dx*(un[i] - un[i-1]) \
                     + nu*dt/dx**2*(un[i+1] - 2*un[i] + un[i-1])
    
    # Periodic boundary
    u[0] = un[0] - un[0]*dt/dx*(un[0] - un[-2]) \
                 + nu*dt/dx**2*(un[1] - 2*un[0] + un[-2])
    u[-1] = u[0]  # Enforce periodicity

# Step 5: Compare with analytical solution at final time
u_analytical = np.asarray([ufunc(nt*dt, xi, nu) for xi in x])
```

> [!IMPORTANT]
> **Periodic Boundary Treatment**: At `i=0`, the "previous" point is `un[-2]` (not `un[-1]`, which is a ghost point set equal to `u[0]`). This ensures the periodic stencil is correct.

### Simulation Parameters

```python
nu = 0.07           # Kinematic viscosity [m²/s]
nx = 101            # Number of grid points
nt = 100            # Number of time steps
dx = 2π/(nx-1)     # Spatial step ≈ 0.0628 [m]
dt = dx*nu         # Time step ≈ 0.0044 [s]
```

**Time step choice**: `dt = dx·ν` 
- This is conservative for both convection and diffusion
- Ensures stability for the coupled system

## Stability Analysis

### Combined Stability Requirements

Burgers' equation must satisfy **both** convection and diffusion stability:

#### 1. Convection Stability (CFL):
```
CFL = u_max · Δt/Δx ≤ 1
```

#### 2. Diffusion Stability:
```
σ = ν · Δt/Δx² ≤ 0.5
```

### Current Simulation Stability

With `dt = dx·ν = 0.0628 × 0.07 ≈ 0.0044 s`:

```
# Diffusion stability
σ = ν·Δt/Δx² = ν·(dx·ν)/dx² = ν²/dx = 0.07²/0.0628 ≈ 0.078 ✓

# Convection stability (assuming u_max ≈ 4-8)
CFL = u_max·Δt/Δx ≈ 8 × 0.0044/0.0628 ≈ 0.56 ≈ OK
```

**Status**: ✓ **Reasonably stable**, though close to CFL limit

> [!TIP]
> For better stability, consider `dt = 0.5·dx·ν` or use adaptive time-stepping based on `max(u)`.

## Discretization Effects and Error Analysis

### Sources of Numerical Error

When comparing numerical vs. analytical solutions, differences arise from:

#### 1. **Truncation Error** (Accuracy)

**Convection term** (backward difference):
- **Order**: O(Δx) - first-order accurate
- **Effect**: Introduces **numerical diffusion**
- **Impact**: Smooths shocks more than physics predicts

**Diffusion term** (central difference):
- **Order**: O(Δx²) - second-order accurate
- **Effect**: More accurate, but still has error
- **Impact**: Smaller than convection error usually

**Time discretization** (forward Euler):
- **Order**: O(Δt) - first-order accurate
- **Effect**: Accumulates error over many steps
- **Impact**: Long simulations → larger errors

#### 2. **Numerical Dissipation**

The backward difference for convection adds **artificial viscosity**:

```
Numerical viscosity ≈ u·Δx/2
```

**Effect**: The numerical solution behaves as if it has:
```
ν_effective = ν_physical + ν_numerical
            = ν + u·Δx/2
```

**Consequence**: 
- Shocks are **more diffused** than in analytical solution
- Gradients are **less steep**
- Maximum values may be **lower**

#### 3. **Numerical Dispersion**

Phase errors accumulate:
- Different wavelengths travel at slightly wrong speeds
- High-frequency components (small scales) are most affected
- Creates **oscillations** near discontinuities

### Observed Differences: Numerical vs. Analytical

When you compare the solutions, expect:

#### At Early Times (small errors):
- ✓ Good agreement in smooth regions
- ✓ Overall shape captured well
- ⚠ Slight amplitude differences

#### At Later Times (accumulated errors):
- ✗ **Peak values lower** in numerical solution (excess diffusion)
- ✗ **Gradients less steep** (numerical smoothing)
- ✗ **Phase shift** possible (dispersion errors)
- ✗ **Small oscillations** near steep gradients (Gibbs phenomenon)

### Quantifying the Error

**L² Norm Error**:
```python
error_L2 = np.sqrt(np.sum((u_numerical - u_analytical)**2) / nx)
```

**Maximum Error**:
```python
error_max = np.max(np.abs(u_numerical - u_analytical))
```

**Typical values** for these parameters:
- `error_L2 ≈ 0.01 - 0.1` (1-10% of signal)
- `error_max ≈ 0.1 - 0.5` (peak differences)

### Why Errors Grow

1. **Error accumulation**: Each timestep adds O(Δt) error
   - Total error after `N` steps ≈ `N·O(Δt) = t·O(Δt)/Δt = t·O(1)`
   - **Error grows linearly with time**

2. **Non-linearity amplifies errors**:
   - Small error in `u` → error in `u·∂u/∂x` is amplified
   - Errors can feedback and grow

3. **Competition of effects**:
   - Convection wants to steepen (amplifies errors)
   - Diffusion wants to smooth (can reduce errors)
   - Balance is delicate

## Improving Numerical Accuracy

### Grid Refinement Study

Test with different `nx`:

```python
nx = [51, 101, 201, 401]  # Refine grid by 2× each time
```

**Expected behavior**:
- Error should decrease as `O(Δx)` or `O(Δx²)`
- Plot `log(error)` vs `log(Δx)` → slope gives order of accuracy

### Higher-Order Schemes

1. **MacCormack** (2nd order):
   - Predictor-corrector approach
   - Reduces numerical diffusion
   - Better shock capture

2. **Lax-Wendroff** (2nd order):
   - Uses Taylor series
   - More dispersive, less dissipative

3. **Flux-limiting schemes**:
   - TVD (Total Variation Diminishing)
   - Prevents oscillations near shocks
   - Essential for realistic simulations

### Implicit Methods

**Crank-Nicolson** for diffusion term:
```python
# Replace explicit diffusion with implicit
# Requires solving tridiagonal system
# Unconditionally stable!
```

Benefits:
- Can use much larger `Δt`
- More stable for stiff problems
- Essential for small viscosity

## Expected Results

### Initial Condition
- Asymmetric sawtooth-like profile
- Peaks around x ≈ π
- Periodic on [0, 2π]

### Evolution
1. **Non-linear steepening**: Gradients sharpen where convection dominates
2. **Viscous smoothing**: Sharp features diffuse
3. **Balance**: Profile evolves to quasi-steady shape
4. **Propagation**: Wave travels to the right (due to +4 offset in solution)

### Comparison
- **Initial condition**: Both match perfectly (same analytical solution)
- **Final time**: Numerical solution is smoother and has lower peaks
- **Difference**: Visible but small if parameters chosen well

## Running the Code

```bash
python Lesson4_Burgers_Equation.py
```

The script will:
1. Compute analytical initial condition using SymPy
2. Evolve numerically for 100 timesteps
3. Compute analytical solution at the same final time
4. Plot all three: initial, numerical final, analytical final
5. Visual comparison shows discretization effects

## Key Takeaways

### Physical Insights

1. **Burgers' = Simplified Navier-Stokes**
   - Captures essential physics: non-linearity + viscosity
   - Used to understand turbulence, shocks, and numerical methods
   - Exact solutions exist (rare for nonlinear PDEs!)

2. **Competition of mechanisms**
   - Convection creates complexity (shocks)
   - Viscosity creates simplicity (smoothing)
   - Reynolds number determines winner

3. **Shock structure**
   - Inviscid shocks are discontinuous
   - Viscosity smooths them into **thin layers**
   - Layer thickness ≈ `ν/U` (viscous length scale)

### Numerical Insights

1. **Discretization introduces errors**
   - Truncation error from finite differences
   - Numerical diffusion from upwinding
   - Dispersion from time-stepping

2. **Having an exact solution is invaluable**
   - Can measure error precisely
   - Validate numerical schemes
   - Study convergence with grid refinement

3. **First-order schemes are very dissipative**
   - Backward difference adds O(Δx) artificial viscosity
   - Smooths shocks more than physics requires
   - Higher-order schemes essential for accuracy

4. **Stability is necessary but not sufficient**
   - Stable ≠ accurate
   - CFL < 1 ensures stability
   - Need CFL ≈ 0.5 for accuracy

## Extensions and Experiments

### Vary Viscosity

```python
nu = 0.01   # Low viscosity → more shock-like
nu = 0.3    # High viscosity → very smooth
```

**Observe**:
- How shock structure changes
- When numerical solution fails to capture thin layers
- Error dependence on Reynolds number

### Grid Refinement Study

```python
for nx in [51, 101, 201, 401]:
    # Run simulation
    # Compute error
    # Plot log(error) vs log(nx)
```

**Expected**: Error ∝ nx⁻¹ (first-order convergence)

### Compare Time-Stepping Schemes

Implement:
- **RK2** (2nd order Runge-Kutta)
- **RK4** (4th order Runge-Kutta)
- **Crank-Nicolson** (implicit)

### Add Error Metrics

```python
# Compute quantitative errors
L1_error = np.sum(np.abs(u - u_analytical)) * dx
L2_error = np.sqrt(np.sum((u - u_analytical)**2) * dx)
Linf_error = np.max(np.abs(u - u_analytical))

print(f"L1 error: {L1_error:.6f}")
print(f"L2 error: {L2_error:.6f}")
print(f"L∞ error: {Linf_error:.6f}")
```

### Visualize Error Distribution

```python
plt.plot(x, u - u_analytical, label='Pointwise Error')
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Numerical - Analytical Solution')
```

**Observation**: Where is error largest? (Usually near steep gradients)

## Troubleshooting

### Problem: Numerical Solution Explodes

**Symptoms**: NaN or infinite values

**Causes**:
1. CFL condition violated (u·Δt/Δx > 1)
2. Diffusion stability violated
3. Boundary condition error

**Fix**: Reduce `dt` or increase `nx`

### Problem: Numerical Solution Too Smooth

**Symptoms**: Peaks much lower than analytical

**Causes**: Excessive numerical diffusion

**Fix**: 
- Use higher-order scheme for convection
- Refine grid (smaller Δx)
- Smaller timestep

### Problem: Oscillations Near Shocks

**Symptoms**: Wiggles around steep gradients

**Causes**: Numerical dispersion

**Fix**:
- Add artificial viscosity
- Use flux-limiting scheme
- Filter high frequencies

## Mathematical Notes

### Cole-Hopf Transformation

Transform `u → φ`:
```
u = -2ν·∂(ln φ)/∂x + C
```

Substituting into Burgers' equation yields the **heat equation** for `φ`:
```
∂φ/∂t = ν·∂²φ/∂x²
```

This is remarkable: **nonlinear → linear** via clever substitution!

### Shock Speed (Rankine-Hugoniot)

For inviscid Burgers', shock speed is:
```
s = (u_left + u_right) / 2
```

(Average of states on either side)

### Similarity Solutions

For inviscid Burgers' with certain initial conditions:
```
u(x,t) = x/t  (rarefaction wave)
u(x,t) = shock propagating at constant speed
```

## References

1. [CFD Python: 12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) by Lorena Barba
2. Burgers, J.M. (1948). "A mathematical model illustrating the theory of turbulence"
3. Cole, J.D. (1951). "On a quasi-linear parabolic equation occurring in aerodynamics"
4. Hopf, E. (1950). "The partial differential equation u_t + uu_x = μu_xx"
5. Whitham, G.B. (1974). "Linear and Nonlinear Waves"

## License

Based on the CFD Python module by Prof. Lorena A. Barba, shared under Creative Commons Attribution license, CC-BY 4.0.
