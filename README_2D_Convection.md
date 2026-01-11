# Lesson 2: 1D Non-Linear Convection

## Overview

This lesson implements the 1D **non-linear convection equation** (also known as the inviscid Burgers' equation) and compares it with linear convection to demonstrate the fundamental differences in wave propagation behavior.

## Physical Background

### The Non-Linear Convection Equation

The governing equation is:

```
∂u/∂t + u·∂u/∂x = 0
```

Where:
- `u(x,t)` is the velocity field
- `t` is time
- `x` is the spatial coordinate

**Key Physical Insight**: Unlike linear convection where waves travel at a constant speed `c`, in non-linear convection **the wave speed depends on the solution itself** (`u`). This seemingly small change has profound physical consequences.

### Physical Interpretation

1. **Self-Advection**: Each part of the fluid is transported at its own velocity
   - Regions where `u=2` move twice as fast as regions where `u=1`
   - Faster-moving fluid "catches up" to slower fluid ahead

2. **Wave Steepening**: The characteristic behavior of non-linear convection
   - The trailing edge (higher `u`) travels faster than the leading edge (lower `u`)
   - The wave profile steepens over time
   - Eventually forms a **shock wave** (discontinuity)

3. **Real-World Examples**:
   - **Traffic flow**: Fast cars catch up to slow traffic, creating shocks (traffic jams)
   - **Shallow water waves**: Wave breaking at the beach
   - **Gas dynamics**: Supersonic flow and shock waves
   - **Acoustics**: Finite-amplitude sound waves

## Mathematical Formulation

### Comparison: Linear vs Non-Linear

| Aspect | Linear Convection | Non-Linear Convection |
|--------|------------------|----------------------|
| **Equation** | `∂u/∂t + c·∂u/∂x = 0` | `∂u/∂t + u·∂u/∂x = 0` |
| **Wave Speed** | Constant `c` | Variable `u(x,t)` |
| **Behavior** | Shape preservation | Wave steepening |
| **Solution** | `u(x,t) = u₀(x-ct)` | No simple analytical solution |
| **Linearity** | Linear PDE | **Non-linear** PDE |
| **Energy** | Conservative | Can form shocks |

### Discretization

Using **backward difference in space** and **forward difference in time**:

#### Linear Convection:
```
u[i]ⁿ⁺¹ = u[i]ⁿ - c·(Δt/Δx)·(u[i]ⁿ - u[i-1]ⁿ)
```

#### Non-Linear Convection:
```
u[i]ⁿ⁺¹ = u[i]ⁿ - u[i]ⁿ·(Δt/Δx)·(u[i]ⁿ - u[i-1]ⁿ)
```

**The only difference**: Replace constant `c` with `u[i]ⁿ`

## Numerical Implementation

### Code Structure

```python
# CRITICAL: Both solutions must evolve independently!
for n in range(nt):
    un = u.copy()      # Store previous timestep for non-linear solution
    un2 = u2.copy()    # Store previous timestep for linear solution (MUST copy from u2!)
    
    for i in range(1, nx):
        # Non-linear convection (line 36)
        u[i] = un[i] - un[i]*dt/dx*(un[i] - un[i-1])
        
        # Linear convection for comparison (line 37)
        u2[i] = un2[i] - C*dt/dx*(un2[i] - un2[i-1])  # MUST use un2, not un!
```

> [!CAUTION]
> **Common Bug Alert**: The linear solution MUST use `un2 = u2.copy()` and `un2[i]` in the update equation. Using `un = u.copy()` for both will make both solutions evolve identically, eliminating any visible differences!

### Initial Condition

A **square wave** (step function):
- `u = 1.0` for `0 ≤ x < 0.5` and `1.0 < x ≤ 2.0`
- `u = 5.0` for `0.5 ≤ x ≤ 1.0`

This initial condition is ideal for demonstrating wave steepening because:
1. Different regions have different velocities (5× difference!)
2. The discontinuity provides a clear feature to track
3. Wave steepening is visually dramatic with this large velocity contrast

### Simulation Parameters

```python
L = 2.0           # Domain length [m]
nx = 81           # Number of spatial points
dx = 0.025        # Spatial step [m]
dt = 0.00025      # Time step [s]
t = 10            # Total simulation time [s] (extended for more evolution)
nt = 40001        # Number of time steps
C = 1.0           # Wave speed for linear case [m/s]
```

> [!TIP]
> The velocity contrast of 5:1 (u=5 vs u=1) creates much more dramatic wave steepening than a smaller contrast would. This makes the non-linear effects clearly visible!

## Stability Analysis

### CFL (Courant-Friedrichs-Lewy) Condition

For numerical stability, the **CFL number** must satisfy:

```
CFL = u·Δt/Δx ≤ CFL_max
```

For the first-order upwind scheme used here: `CFL_max ≈ 1.0`

### Current Simulation Stability

With our parameters:
- `max(u) = 5.0 m/s`
- `Δt = 0.00025 s`
- `Δx = 0.025 m`

```
CFL = 5.0 × 0.00025 / 0.025 = 0.05
```

**Status**: ✓ **Very stable** (CFL << 1)

> [!NOTE]
> The CFL number of 0.05 is **20× smaller** than the stability limit. This ensures excellent stability and good accuracy, though it requires more timesteps (40,001 total).

### Critical Insight for Non-Linear Problems

Unlike linear convection where the wave speed is fixed, **in non-linear convection the CFL number depends on the solution**:

```
CFL(t) = max(u(x,t))·Δt/Δx
```

**Implications**:
1. Stability can change during simulation
2. If `u` grows, stability may be lost
3. Adaptive time-stepping may be needed for efficiency
4. The initial maximum `u` determines the worst-case CFL

### Optimal Parameter Selection

**Rule of thumb**: Choose `Δt` such that:

```
Δt ≤ Δx / max(u)
```

For our case:
```
Δt ≤ 0.025 / 5.0 = 0.005 s
```

Our choice of `Δt = 0.00025 s` is **20× smaller** than the stability limit, ensuring:
- ✓ Numerical stability
- ✓ Good accuracy
- ✗ Many time steps required (40,001 steps)

## Physical Phenomena Observed

### 1. Wave Steepening

As the simulation progresses, you'll observe:

**Linear convection (`u2`, dashed lines)**:
- Wave maintains its shape
- Translates uniformly to the right at speed `C = 1.0 m/s`
- No steepening occurs

**Non-linear convection (`u`, solid lines)**:
- Right edge of the square wave steepens dramatically
- The `u=2` region travels faster than `u=1` region
- Wave front becomes increasingly vertical

### 2. Shock Formation

In inviscid (non-dissipative) systems like this one:
- The steepening continues until a **shock** forms
- Mathematically, `∂u/∂x → ∞` at the shock
- Numerically, oscillations appear (Gibbs phenomenon)
- In reality, viscosity would smooth the shock into a thin layer

### 3. Energy Cascade

The non-linear term `u·∂u/∂x`:
- Transfers energy between scales
- Creates smaller-scale features (steepening)
- In 3D turbulence, this leads to the energy cascade

## Expected Results

Running the simulation produces plots at various time steps showing:

1. **t = 0**: Both solutions start identically with the square wave (u=5 in the middle, u=1 elsewhere)
2. **t = 0.25s**: Non-linear wave begins to steepen noticeably on the right edge
3. **t = 1.25s**: Clear difference visible - non-linear shows significant steepening
4. **t = 2.5s - 5s**: 
   - Linear wave maintains shape, just translates
   - Non-linear wave has dramatically steepened front
5. **t = 10s**: 
   - Linear wave: Still maintains square wave shape
   - Non-linear wave: **Extreme steepening** with near-vertical right edge
   - Possible numerical oscillations near the shock

> [!IMPORTANT]
> The **5× velocity difference** (u=5 vs u=1) creates dramatically visible wave steepening. The high-velocity region travels at 5 m/s while the low-velocity region travels at 1 m/s, causing the wave front to compress by up to **40 meters** over 10 seconds!

### Visualization Features

The enhanced plotting includes:
- **12×6 inch figure** for better visibility
- **Solid lines** for non-linear solution
- **Dashed lines** for linear solution
- **Grid** for easier reading
- **Time in seconds** (not timestep numbers)
- **Strategic snapshots** at important evolution stages

## Running the Code

```bash
python Lesson2_Non_Linnear_Convection.py
```

The script will:
1. Display the evolution of both linear and non-linear convection
2. Plot snapshots at t = 0.25s, 1.25s, 2.5s, 5s, and 10s
3. Use solid lines for non-linear and dashed lines for linear solutions
4. Show dramatic wave steepening in the non-linear case

## Troubleshooting & Common Pitfalls

> [!CAUTION]
> **Critical Bug**: Array Copying Errors

### Bug #1: Incorrect Array Copying

**WRONG:**
```python
for n in range(nt):
    un = u.copy()
    un2 = u.copy()  # ❌ WRONG! Copies from u, not u2
```

**CORRECT:**
```python
for n in range(nt):
    un = u.copy()
    un2 = u2.copy()  # ✓ Correctly copies from u2
```

**Symptom**: Both linear and non-linear solutions appear identical.

**Cause**: If you copy the non-linear solution state for both `un` and `un2`, the linear solution will evolve using the non-linear solution's values, making them identical.

### Bug #2: Incorrect Previous Timestep Reference

**WRONG:**
```python
u2[i] = un[i] - C*dt/dx*(un[i] - un[i-1])  # ❌ Uses un instead of un2
```

**CORRECT:**
```python
u2[i] = un2[i] - C*dt/dx*(un2[i] - un2[i-1])  # ✓ Uses un2
```

**Symptom**: Linear solution doesn't evolve correctly; may show artifacts.

**Cause**: The linear solution must use its own previous state (`un2`), not the non-linear solution's state (`un`).

### How to Verify Your Implementation

1. **Check initial conditions**: Both `u` and `u2` should be identical at t=0
2. **Verify independence**: After a few timesteps, `u` and `u2` should differ
3. **Test with extreme parameters**: Use very different values (e.g., u=10 vs C=1) to make differences obvious
4. **Print intermediate values**: Add debug prints to verify `un2` is actually different from `un`

### Debugging Checklist

- [ ] `un2 = u2.copy()` (not `u.copy()`)
- [ ] Linear equation uses `un2[i]` everywhere (not `un[i]`)
- [ ] Non-linear equation uses `un[i]` (with `un[i]` as the wave speed)
- [ ] Both solutions initialized with same initial condition
- [ ] Plotting both solutions on the same axes

## Key Takeaways

### Physical Insights

1. **Non-linearity causes qualitatively different behavior**
   - Not just "a little different" but fundamentally changed physics
   - Shape preservation → wave steepening

2. **Self-interaction is powerful**
   - The solution affects its own evolution
   - Leads to shock formation and complexity

3. **Energy concentration**
   - Non-linear systems can concentrate energy into shocks
   - This is why breaking waves are so powerful

### Numerical Insights

1. **Stability depends on the solution**
   - Must account for maximum possible wave speed
   - CFL condition must be satisfied at all times

2. **Shock capturing is challenging**
   - Simple schemes produce oscillations
   - Advanced schemes (TVD, WENO) needed for clean shocks

3. **Non-linear = More expensive**
   - Need finer resolution to capture steepening
   - Often need smaller time steps

## Extensions and Next Steps

### Parameter Experiments

> [!TIP]
> **Making Differences MORE Visible**

If you want even more dramatic non-linear effects:

#### Option 1: Increase Velocity Contrast
```python
u[int(0.5/dx):int(1/dx+1)] = 10.0  # 10× velocity difference!
```
- **CFL**: 0.1 (still very stable)
- **Effect**: Even more dramatic wave steepening
- **Warning**: May see more numerical oscillations

#### Option 2: Extend Simulation Time
```python
t = 20  # 20 seconds instead of 10
nt = int(t/dt) + 1  # 80,001 timesteps
```
- **Effect**: Watch the wave steepen and potentially form multiple oscillations
- **Cost**: Takes twice as long to run

#### Option 3: Compare Different Wave Speeds
```python
C = 2.5  # Change linear wave speed to match average non-linear speed
```
- **Effect**: See how linear and non-linear waves separate spatially
- **Insight**: Demonstrates that linear waves travel at constant speed

### Making Differences LESS Visible (Educational)

To understand why small velocity contrasts are hard to see:

```python
u[int(0.5/dx):int(1/dx+1)] = 1.5  # Only 1.5× difference
t = 5  # Shorter time
```
- **Result**: Subtle differences, similar to original implementation
- **Lesson**: Non-linearity strength scales with velocity gradient

### Advanced Improvements

1. **Add viscosity** → Burgers' equation
   ```python
   # Add diffusion term: ν·∂²u/∂x²
   nu = 0.01  # Kinematic viscosity
   u[i] = un[i] - un[i]*dt/dx*(un[i]-un[i-1]) + nu*dt/dx**2*(un[i+1]-2*un[i]+un[i-1])
   ```
   - Smooths shocks into boundary layers
   - More realistic physics
   - Requires handling right boundary (i+1)

2. **Higher-order schemes**
   - MacCormack (2nd order)
   - Lax-Wendroff (2nd order)
   - TVD schemes (shock capturing)
   - WENO schemes (high-order shock capturing)
   - Reduces numerical dispersion
   - Better shock capturing

3. **2D non-linear convection**
   ```python
   # Add y-direction
   ∂u/∂t + u·∂u/∂x + v·∂u/∂y = 0
   ∂v/∂t + u·∂v/∂x + v·∂v/∂y = 0
   ```
   - Richer wave interactions
   - More realistic fluid dynamics
   - Much more computationally expensive

4. **Adaptive time-stepping**
   ```python
   dt = CFL_target * dx / np.max(u)  # Adjust dt each step
   ```
   - Maintains constant CFL
   - More efficient for varying wave speeds
   - Requires careful implementation

### Recommended Learning Path

1. ✅ **Start here**: Understand the current implementation
2. **Experiment**: Try different velocity contrasts (u=2, 3, 7, 10)
3. **Compare**: Run with different time durations (t=5, 10, 15, 20)
4. **Add viscosity**: Implement Burgers' equation
5. **Upgrade scheme**: Implement 2nd order method
6. **Go 2D**: Extend to two dimensions

## Mathematical Notes

### Characteristic Method

The equation `∂u/∂t + u·∂u/∂x = 0` can be rewritten as:

```
du/dt = 0  along characteristics  dx/dt = u
```

This means:
- `u` is constant along characteristic curves
- Characteristics are straight lines with slope `1/u`
- **Characteristics can cross** → multi-valued solution → shock!

### Conservation Form

The equation can be written in conservation form:

```
∂u/∂t + ∂(u²/2)/∂x = 0
```

This shows that `u²/2` is the conserved flux, which is crucial for:
- Proper shock jump conditions (Rankine-Hugoniot)
- Conservative numerical schemes

## References

1. [CFD Python: 12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) by Lorena Barba
2. Burgers, J.M. (1948). "A mathematical model illustrating the theory of turbulence"
3. LeVeque, R.J. (2002). "Finite Volume Methods for Hyperbolic Problems"

## License

Based on the CFD Python module by Prof. Lorena A. Barba, shared under Creative Commons Attribution license, CC-BY 4.0.
