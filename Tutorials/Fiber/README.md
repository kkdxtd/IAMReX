# Two-dimensional traveling-wave fin (fiber)

This case imports the prescribed fiber implementation from
[kkdxtd/IAMReX-2dIBMdev](https://github.com/kkdxtd/IAMReX-2dIBMdev), commit
`78e0184b`, into the current IAMReX solver. The implementation is in
`Source/DiffusedFiber.H` and `Source/DiffusedFiber.cpp`.

## Configuration

Use `GNUmakefile` with `DIM=2` and `USE_PARTICLES=TRUE`, and the input file
`inputs.2d.flow_past_fiber`. The input prefix is `fiber.input = fiber_inputs`.
AMReX and AMReX-Hydro paths can be supplied through `AMREX_HOME` and
`AMREX_HYDRO_HOME`. Compilation and numerical validation of this integration
are left to the user; neither was performed during the merge.

The original motion is preserved, including its use of the absolute x coordinate:

```text
x_i = x0 + i*h
y_i(t) = y0 + A*sin(2*pi*(t/T - n*x_i/L) + phase)
u_i(t) = 0
v_i(t) = (2*pi*A/T)*cos(2*pi*(t/T - n*x_i/L) + phase)
```

Here `h` is the finest-level Eulerian x spacing. As in the original code,
the actual marker count is `ceil(L/h)+1` and each marker uses area `h*h`.
`num_marker` remains a required legacy input but does not override that count.
Consequently the last marker can extend by less than `h` beyond `x0+L`.
The prescribed positions and target velocities are evaluated at `time+dt`
before coupling to the new fluid state, and reconstructed from the checkpoint
time on restart. The initial velocity also follows the analytic motion.

The supplied case retains the original physical parameters: domain `10 x 5`,
mesh `512 x 256`, origin `(4,2.5)`, amplitude `0.02`, period `0.5`, length `1`,
wave number `1`, and unit inflow speed and density. Unused third components
have been removed from the two-dimensional input vectors.

`inputs.2d.flow_past_fiber.water` also preserves the alternative input formerly
stored at the 2D repository root (`inputs.2d.flow_past_fiber.txt`): density
`1000`, inflow speed `2`, fixed `dt=0.005`, and `max_step=300`. It uses the same
fin geometry; only unused third components and explicit default IBM settings
were adjusted when importing it.

## Outputs and limitations

- `IB_Fiber_1.csv`: `iStep,time,Fx,Fy`. `write_freq` counts finest-level steps.
- The CSV preserves the old convention: integrated acceleration applied to
  the fluid, `sum(a_marker*h*h)`. Multiply by `-ns.fluid_rho` for fluid-on-fin
  force per unit out-of-plane span. This differs from the already dimensional,
  fluid-on-fin loads in the 3D `IB_Fin_1.csv`.
- The standard AMReX plotfile includes a `fibers` marker dataset. Verbose
  marker dumps are named `fiberNNNNN`.
- This tutorial uses one prescribed fiber. The legacy shared container requires
  equal actual marker counts if multiple fibers are specified. No rigid-body
  collision, closed body volume fraction, or 6-DOF update is applied to fibers.
- The imported position-and-velocity penalty formula is retained. This merge
  does not add a flexible structural solver or change marker quadrature.
- Only source, configuration and documentation are versioned; generated loads,
  marker dumps, plotfiles, checkpoints and executables are ignored.
