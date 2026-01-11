# 1D Linear Convection - README

This repository contains an implementation of the **1D Linear Convection equation**, which is the first lesson of the CFD Python "12 Steps to Navier-Stokes" tutorial by Prof. Lorena Barba.

## Overview

The linear convection equation is a fundamental partial differential equation (PDE) in computational fluid dynamics. It describes the transport of a quantity at a constant wave speed without diffusion or sources/sinks.

### The Governing Equation

The 1D linear convection equation is:

```
∂u/∂t + c ∂u/∂x = 0
```

where:
- `u` is the transported quantity (e.g., velocity, concentration)
- `t` is time
- `x` is the spatial coordinate
- `c` is the wave speed (constant)

## Numerical Method

This code uses the **Finite Difference Method** with:
- **Forward Difference in Time**: `(u[i]^(n+1) - u[i]^n) / Δt`
- **Backward Difference in Space**: `(u[i]^n - u[i-1]^n) / Δx`

The discretized equation becomes:

```
u[i]^(n+1) = u[i]^n - c * (Δt/Δx) * (u[i]^n - u[i-1]^n)
```

## Code Structure

The implementation consists of two main parts:

### 1. Basic Simulation (Lines 11-40)
- Simulates a square wave propagating through a 1D domain
- Uses fixed parameters: `nx=81`, `dt=0.00025`
- Displays the wave evolution at regular intervals

### 2. Parameter Study (Lines 42-61)
- Tests different combinations of spatial resolution (`nx`) and time step (`dt`)
- Creates a grid of subplots showing the effects of these parameters
- Helps visualize numerical stability and accuracy

## Simulation Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Domain length | `L` | 2.0 | Length of the spatial domain |
| Grid points | `nx` | 81 (variable in study) | Number of spatial divisions |
| Grid spacing | `dx` | `L/(nx-1)` | Distance between grid points |
| Time step | `dt` | 0.00025 (variable in study) | Temporal discretization |
| Total time | `t` | 5.0 | Total simulation time |
| Wave speed | `c` | 1.0 | Convection velocity |

## Initial Condition

The simulation uses a **square wave** initial condition:
- `u = 1.0` for most of the domain
- `u = 2.0` for `0.5 ≤ x ≤ 1.0`

This discontinuous profile is useful for testing numerical schemes because it highlights numerical diffusion and dispersion.

## Requirements

```python
numpy
matplotlib
```

## Usage

Simply run the Python script:

```bash
python Lesson1_1D_Linnear_Convection.py
```

The script will:
1. Display a plot showing the wave evolution over time with the basic parameters
2. Display a grid of subplots comparing different `nx` and `dt` combinations

## Results Interpretation

The plots show how the square wave propagates to the right at speed `c=1.0`. You should observe:
- The wave moving from left to right
- Numerical diffusion (smoothing of the sharp corners)
- Grid-dependency effects

## Numerical Stability Considerations

### CFL Condition for Stability

The parameter study section (lines 42-61) explores different combinations of `nx` and `dt`. For numerical stability of this explicit scheme, the **Courant-Friedrichs-Lewy (CFL) condition** must be satisfied:

```
CFL = c * Δt / Δx ≤ 1
```

where:
- `c` is the wave speed (1.0 in this code)
- `Δt` is the time step
- `Δx` is the grid spacing

### Stability Analysis for the Parameter Study

The code tests the following combinations:

| nx | dx = L/(nx-1) | dt | CFL = c·dt/dx | Stable? |
|----|---------------|-----|---------------|---------|
| 41 | 0.05000 | 0.00025 | 0.005 | ✓ Yes |
| 41 | 0.05000 | 0.000125 | 0.0025 | ✓ Yes |
| 41 | 0.05000 | 0.000625 | 0.0125 | ✓ Yes |
| 81 | 0.02500 | 0.00025 | 0.010 | ✓ Yes |
| 81 | 0.02500 | 0.000125 | 0.005 | ✓ Yes |
| 81 | 0.02500 | 0.000625 | 0.025 | ✓ Yes |
| 161 | 0.01250 | 0.00025 | 0.020 | ✓ Yes |
| 161 | 0.01250 | 0.000125 | 0.010 | ✓ Yes |
| 161 | 0.01250 | 0.000625 | 0.050 | ✓ Yes |

**All combinations in the parameter study satisfy CFL ≤ 1** and should produce stable results.

### Recommendations

- **For stability**: Always ensure `CFL ≤ 1`
- **For accuracy**: Smaller `CFL` values (typically 0.1-0.5) reduce numerical diffusion
- **Grid refinement**: Increasing `nx` (finer grid) improves spatial accuracy but requires smaller `dt` to maintain the same CFL number
- **If modifying parameters**: 
  - If you increase `nx`, you may need to decrease `dt` proportionally
  - If you decrease `dx`, you must decrease `dt` by at least the same factor to maintain stability
  - Violating CFL > 1 will cause **exponential growth of errors** and simulation failure

### Warning Signs of Instability

If you modify the code and encounter:
- Oscillations that grow exponentially
- `NaN` or `Inf` values in the solution
- Unrealistic spikes in the wave profile

Then you have likely violated the CFL condition. Reduce `dt` or increase `nx` to restore stability.

## References

- [CFD Python: 12 Steps to Navier-Stokes](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/) by Prof. Lorena A. Barba
- Barba, L. A., & Forsyth, G. F. (2018). CFD Python: the 12 steps to Navier-Stokes equations. Journal of Open Source Education, 1(9), 21.

## License

This code is part of the educational CFD Python tutorial series and follows the original licensing terms.

## Author

Based on the "12 Steps to Navier-Stokes" tutorial by Prof. Lorena A. Barba.
