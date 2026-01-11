# Lesson 3: 1D Diffusion

## Overview

This lesson implements the **1D diffusion equation** (also known as the heat equation), which describes how quantities like heat, concentration, or momentum spread out over time due to molecular motion. This is a fundamentally different process from convection, introducing dissipation and smoothing effects.

## Physical Background

### The Diffusion Equation

The governing equation is:

```
∂u/∂t = ν·∂²u/∂x²
```

Where:
- `u(x,t)` is the diffusing quantity (temperature, concentration, etc.)
- `t` is time
- `x` is the spatial coordinate
- `ν` (nu) is the **diffusivity** (or viscosity, kinematic viscosity)

**Key Physical Insight**: Unlike convection where quantities are **transported** by flow, diffusion causes quantities to **spread** from high to low concentrations. This is a smoothing, dissipative process driven by random molecular motion.

### Physical Interpretation

1. **Fick's Law / Fourier's Law**: Diffusion is driven by gradients
   - Heat flows from hot to cold regions
   - Concentration flows from high to low concentrations
   - Momentum diffuses from fast to slow fluid layers
   - Always acts to **smooth out** variations

2. **Dissipative Process**: Unlike convection which conserves shapes
   - Sharp features become rounded
   - Gradients decrease over time
   - Energy is dissipated (converted to heat)
   - **Irreversible** process (increases entropy)

3. **Real-World Examples**:
   - **Heat conduction**: A hot spot on a metal rod spreads out
   - **Chemical diffusion**: Perfume spreading through a room
   - **Viscous diffusion**: Momentum spreading in viscous fluids
   - **Pollution dispersion**: Contaminants spreading in water/air

### Comparison with Convection

| Aspect | Convection | Diffusion |
|--------|-----------|-----------|
| **Equation** | `∂u/∂t + c·∂u/∂x = 0` | `∂u/∂t = ν·∂²u/∂x²` |
| **Mechanism** | Transport by flow | Molecular mixing |
| **Effect** | Translation | Smoothing/spreading |
| **Derivative** | 1st order in space | **2nd order** in space |
| **Energy** | Conservative | **Dissipative** |
| **Reversibility** | Reversible | Irreversible |
| **Sharp features** | Preserved (linear case) | Always smoothed |

## Mathematical Formulation

### Discretization

Using **central difference in space** and **forward difference in time**:

```
∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / Δx²
```

The discretized diffusion equation becomes:

```
u[i]ⁿ⁺¹ = u[i]ⁿ + ν·(Δt/Δx²)·(u[i+1]ⁿ - 2u[i]ⁿ + u[i-1]ⁿ)
```

**Key features**:
1. Requires values at **i-1, i, and i+1** (three-point stencil)
2. Loop must run from `i=1` to `i=nx-2` to avoid boundary issues
3. The term `(u[i+1] - 2u[i] + u[i-1])` approximates the curvature

### Physical Meaning of the Stencil

The second derivative measures **curvature**:
- If `u[i]` is higher than its neighbors → negative curvature → `u[i]` decreases
- If `u[i]` is lower than its neighbors → positive curvature → `u[i]` increases
- Result: **peaks flatten**, **valleys fill**, gradients smooth

## Numerical Implementation

### Code Structure

```python
# Simulation parameters
nu = 0.3                 # Diffusivity [m²/s]
sigma = 0.2              # Stability parameter
dt = sigma * dx**2 / nu  # Time step (CFL-like condition)

# Time loop
for n in range(nt):
    un = u.copy()  # Store previous timestep
    
    # Space loop: note the range(1, nx-1) to handle the stencil
    for i in range(1, nx-1):
        u[i] = un[i] + nu*dt/dx**2 * (un[i+1] - 2*un[i] + un[i-1])
```

> [!IMPORTANT]
> **Critical Implementation Detail**: The loop must run from `i=1` to `i=nx-2` (i.e., `range(1, nx-1)`) because the stencil requires `un[i+1]`, which would be out of bounds at `i=nx-1`.

### Initial Condition

A **square pulse** (step function):
- `u = 1.0` for `0 ≤ x < 0.5` and `1.0 < x ≤ 2.0`
- `u = 2.0` for `0.5 ≤ x ≤ 1.0`

This initial condition demonstrates diffusion clearly:
- Sharp discontinuities at `x=0.5` and `x=1.0`
- High gradients that will drive rapid diffusion
- Easy to visualize the smoothing process

### Simulation Parameters

```python
nu = 0.3          # Diffusivity [m²/s]
L = 2.0           # Domain length [m]
nx = 81           # Number of spatial points
dx = 0.025        # Spatial step [m]
sigma = 0.2       # Stability parameter
dt = 0.000417     # Time step [s] (calculated from sigma)
t = 10            # Total simulation time [s]
nt = 24001        # Number of time steps
```

> [!NOTE]
> The time step `dt` is **automatically calculated** from the stability parameter `sigma`, diffusivity `nu`, and grid spacing `dx`. This ensures numerical stability.

## Stability Analysis

### Diffusion Stability Condition

For explicit diffusion schemes, stability requires:

```
σ = ν·Δt/Δx² ≤ σ_max
```

For the forward-time, central-space (FTCS) scheme: `σ_max = 0.5`

**Physical interpretation**: 
- `σ` represents how far diffusion "spreads" in one timestep relative to grid spacing
- If `σ > 0.5`, numerical diffusion can spread faster than physically possible
- Results in **unbounded growth** and numerical explosion

### Current Simulation Stability

With our parameters:
- `ν = 0.3 m²/s`
- `Δx = 0.025 m`
- `σ = 0.2` (chosen)

```
Δt = σ·Δx²/ν = 0.2 × (0.025)² / 0.3 ≈ 0.000417 s
```

**Status**: ✓ **Stable** (σ = 0.2 < 0.5)

> [!TIP]
> The stability parameter `sigma = 0.2` is a safe choice, providing good stability with reasonable accuracy. You can increase to `sigma = 0.4` for faster simulations (fewer timesteps) while maintaining stability.

### Comparison: Convection vs Diffusion Stability

| Scheme | Stability Condition | Typical Safe Value |
|--------|--------------------|--------------------|
| **Convection** | `CFL = c·Δt/Δx ≤ 1` | CFL ≈ 0.5 |
| **Diffusion** | `σ = ν·Δt/Δx² ≤ 0.5` | σ ≈ 0.2 |

**Key difference**: Diffusion stability involves `Δx²` (not `Δx`), making it **much more restrictive** for fine grids!

### Time Step Scaling

If you refine the grid by 2× (halve `Δx`):
- **Convection**: `Δt` must be halved (2× more steps)
- **Diffusion**: `Δt` must be reduced by 4× (4× more steps!)

This makes **diffusion very expensive** for fine grids, motivating implicit methods.

## Physical Phenomena Observed

### 1. Smoothing and Spreading

As the simulation progresses:

**Early times (t = 0.25s - 1.25s)**:
- Sharp corners at `x=0.5` and `x=1.0` begin to round
- Gradients immediately start decreasing
- The pulse spreads slightly

**Mid times (t = 2.5s - 5s)**:
- Pulse becomes increasingly smooth and rounded
- Maximum value decreases (mass is conserved, but spreads)
- Profile becomes more Gaussian-like

**Late times (t = 10s)**:
- Nearly smooth, bell-shaped distribution
- Sharp edges completely smoothed
- Continues spreading and flattening indefinitely

### 2. Maximum Value Decay

Unlike convection where the maximum is preserved:
- The maximum value **decreases** over time
- Mass (integral of `u`) is conserved
- Energy (`u²`) decreases (dissipation!)

### 3. Characteristic Diffusion Length

The **diffusion length scale** grows as:

```
L_diff ~ √(ν·t)
```

This shows:
- Diffusion spreads as the **square root** of time (slow!)
- Doubling the distance takes 4× the time
- This is why diffusion is often inefficient for transport

## Expected Results

Running the simulation shows smooth, continuous evolution:

1. **t = 0**: Sharp square pulse
2. **t ≈ 0.25s**: Corners begin rounding
3. **t ≈ 1.25s**: Pulse clearly smoothed, edges rounded
4. **t ≈ 2.5s**: Nearly Gaussian profile
5. **t ≈ 5-10s**: Smooth, spreading distribution

> [!IMPORTANT]
> Unlike convection which translates features, diffusion **spreads and smooths** them. The pulse stays centered (no convection) but becomes wider and flatter over time.

### Visualization Features

The plotting includes:
- **12×6 inch figure** for clear visibility
- **Strategic time snapshots** at important stages
- **Smooth curves** showing continuous smoothing
- **Grid** for easier reading
- **Time in seconds** in legend

## Running the Code

```bash
python Lesson3_1D_Diffusion.py
```

The script will:
1. Display the smooth evolution of the diffusion process
2. Plot snapshots showing progressive smoothing
3. Demonstrate how sharp features are eliminated
4. Show mass conservation with spreading

## Key Takeaways

### Physical Insights

1. **Diffusion smooths everything**
   - Sharp features cannot persist
   - Gradients always decrease
   - Nature "likes" smooth distributions

2. **Irreversible process**
   - Information is lost (entropy increases)
   - Cannot reverse time and "unmix"
   - Fundamental to thermodynamics

3. **Slow process**
   - Spreads as `√t` (not linearly)
   - Inefficient for long-distance transport
   - Why natural systems often use convection for transport

4. **Universal behavior**
   - Same equation describes heat, concentration, momentum
   - Gaussian solutions are "attractors"
   - Fundamental to many physical processes

### Numerical Insights

1. **Second-order derivatives are expensive**
   - Stability condition involves `Δx²`
   - Fine grids → very small timesteps
   - Implicit methods often needed for efficiency

2. **Stencil requirements**
   - Need values at `i-1`, `i`, and `i+1`
   - Boundary conditions become critical
   - Loop indices must be carefully managed

3. **Stability is restrictive**
   - `σ ≤ 0.5` is a hard limit
   - Easy to violate accidentally
   - Always check stability before running

## Extensions and Next Steps

### Parameter Experiments

#### Increase Diffusivity
```python
nu = 1.0  # Faster spreading
```
- **Effect**: Pulse spreads much faster
- **Stability**: Must reduce `sigma` or increase `dt`

#### Decrease Diffusivity
```python
nu = 0.05  # Slower spreading
```
- **Effect**: Pulse spreads very slowly
- **Can run longer**: Features persist longer

#### Different Stability Parameter
```python
sigma = 0.4  # Closer to stability limit
```
- **Effect**: Larger timesteps, faster simulation
- **Warning**: Less margin for error

### Advanced Improvements

1. **Implicit scheme** (Crank-Nicolson)
   ```
   Unconditionally stable (no σ restriction!)
   Requires solving linear system
   Much more efficient for fine grids
   ```

2. **Higher-order in space**
   - 4th or 6th order central differences
   - Better accuracy for same grid
   - More complex stencil

3. **Non-constant diffusivity**
   ```python
   nu[i] = function(x, u)  # Space or solution-dependent
   ```
   - Models turbulent diffusion
   - Nonlinear diffusion equations
   - Porous media flow

4. **2D diffusion**
   ```
   ∂u/∂t = ν·(∂²u/∂x² + ∂²u/∂y²)
   ```
   - Heat spreading on a plate
   - More realistic physics
   - Visualize with contour plots

### Combining with Convection

The **advection-diffusion equation**:

```
∂u/∂t + c·∂u/∂x = ν·∂²u/∂x²
```

Combines:
- **Convection**: Transport at speed `c`
- **Diffusion**: Spreading and smoothing

This describes:
- Heat transfer in moving fluids
- Pollution in rivers
- Most realistic transport phenomena

## Troubleshooting

### Problem: Numerical Instability (Solution Explodes)

**Symptoms**: Values grow unbounded, NaN errors

**Causes**:
1. `sigma > 0.5` (stability violated)
2. Grid too fine for given `dt`
3. Negative diffusivity (check sign!)

**Fix**:
```python
# Ensure sigma <= 0.5
sigma = 0.2  # Safe value
dt = sigma * dx**2 / nu
```

### Problem: Solution Doesn't Change

**Symptoms**: Pulse stays sharp

**Causes**:
1. `nu` too small
2. `dt` too small (not enough time)
3. Range error in loop

**Fix**: Verify loop runs `for i in range(1, nx-1)`

### Problem: Boundary Issues

**Symptoms**: Oscillations at edges

**Causes**: Boundary conditions not properly handled

**Solution**: For this lesson, we use **Dirichlet boundaries** (`u[0] = 1`, `u[nx-1] = 1`), which remain fixed.

## Mathematical Notes

### Analytical Solution

For an initial delta function, the exact solution is:

```
u(x,t) = 1/√(4πνt) · exp(-(x-x₀)²/(4νt))
```

This is a **Gaussian** that:
- Spreads as `√(νt)`
- Amplitude decreases as `1/√t`
- Total mass is conserved

### Similarity Solution

Diffusion has a **self-similar** structure:
```
u(x,t) = f(η)  where  η = x/√(νt)
```

All solutions at different times can be collapsed onto one curve using the similarity variable `η`.

### Maximum Principle

**Important property**: The maximum and minimum values occur either:
1. In the initial condition
2. On the boundaries

The solution **cannot develop new extrema** in the interior. This guarantees smoothing!

## References

1. [CFD Python: 12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) by Lorena Barba
2. Crank, J. (1979). "The Mathematics of Diffusion"
3. Morton, K.W. & Mayers, D.F. (2005). "Numerical Solution of Partial Differential Equations"

## License

Based on the CFD Python module by Prof. Lorena A. Barba, shared under Creative Commons Attribution license, CC-BY 4.0.
