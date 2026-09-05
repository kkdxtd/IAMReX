# Three-dimensional traveling-wave fin

This tutorial implements a prescribed three-dimensional traveling-wave fin coupled to IAMReX through the multidirect-forcing immersed-boundary method (DIBM). Two local AMR levels provide a broad 4 mm near-fin and wake mesh without storing that resolution over the whole domain.

## Physical model

The fin reference surface is parameterized by chord coordinate `s` and span coordinate `r`:

```text
X(s,r,t) = (x0+s, y0+r*cos(theta), z0+r*sin(theta))
theta(s,t) = A*sin(2*pi*f*t - 2*pi*s/lambda + phase)
```

The analytic marker velocity is used directly as the no-slip target. The surface is prescribed: it does not enter the rigid-particle collision, PVF, or 6-DOF update paths.

The merged input preserves the current local parameters: `L=0.8 m`, `span=0.052 m`, `A=85 deg`, `f=1.0 Hz`, `lambda=0.4 m`, `rho=1000 kg/m3`, and `nu=1e-6 m2/s`. The model was developed from the supplied Palabos fin example; the local frequency has since been changed.

## Discretization

- Domain: `4.0 x 1.6 x 1.6 m`.
- Level 0 mesh: `250 x 100 x 100`, so `dx=16 mm`.
- Two `2:1` refinement levels; Level 2 has `dx=4 mm` (three total grid levels: 16/8/4 mm).
- Two nested physical boxes cover both the fin and its downstream wake. The broad Level-1 box is `(0.35,0.25,0.25)` to `(3.65,1.35,1.35) m`; the finest Level-2 box is `(0.65,0.45,0.45)` to `(3.35,1.15,1.15) m`.
- Per-level blocking factors are `2 4 4`. Level 0 uses 2 because its x extent is 250 cells; refined levels use 4 to satisfy the IAMReX velocity/diffusion FillPatch proper-nesting requirement.
- Lagrangian surface: `201 x 14` markers, approximately 4 mm apart in both parameter directions.
- Four-point regularized Delta function and two multidirect-forcing iterations.
- Smagorinsky LES with `Cs=0.12`.
- Auto subcycling with Level-0 `dt=8.80256437073704e-3 s`; two `2:1` temporal refinements give `dt=2.20064109268426e-3 s` on Level 2.

The marker volume used for force spreading is `dA*dx`; trapezoidal half weights are used on the four parameter-space edges. The reported load is the fluid-on-fin force:

```text
F_fin = -rho * sum(a_marker * dA * dx)
```

The moment is taken about the fin origin `(x0,y0,z0)`.

## Files and outputs

- `inputs.3d.fin`: complete two-level local-refinement (three total levels) run configuration.
- `IB_Fin_1.csv`: step, physical time, phase, three force components, and three moment components. Force units are N and moment units are N m for SI inputs.
- `fin_vtk/fin_XXXXXXXX.vtk`: legacy VTK PolyData time series in a dedicated directory. Files are triggered by the same output hook as the AMReX flow plotfiles, so if `amr.plot_int=12` is selected, `fin_00000012.vtk` corresponds to `plt00012`. Field data contains step, time, total fluid-on-fin force, and total moment. An initial plot, when enabled, also produces `fin_00000000.vtk`. The current input sets `amr.plot_int=-1`, disabling periodic flow and fin geometry output.
- Standard AMReX plotfiles: flow-field state, including `x_velocity`, `y_velocity`, `z_velocity`, the derived scalar `velocity_magnitude=sqrt(u^2+v^2+w^2)`, and the directly evaluated vorticity magnitude `mag_vort`.

The VTK writer reconstructs the prescribed geometry analytically on the I/O rank, so its topology remains ordered even after AMReX redistributes Lagrangian markers among MPI ranks. VTK output is tied to `amr.plot_int`; there is no separate fin VTK frequency. For a single fin the files are named `fin_XXXXXXXX.vtk`; multiple fins include the body ID in the prefix.

In ParaView, `velocity_magnitude` is intended for speed contours and coloring. Vorticity must be calculated from a vector field, not from this scalar. Either use the plotfile's directly written `mag_vort`, or create a Calculator result such as `x_velocity*iHat + y_velocity*jHat + z_velocity*kHat`, name it `velocity`, and apply the Gradient/Vorticity filter to that vector.

## Scope and current limitations

- Refinement uses two static, nested physical boxes. They follow neither vorticity nor a freely translating body; their downstream extent is chosen to retain the principal wake and shed vortices at 4–8 mm resolution.
- Level 0–2 use `1,2,4` relative temporal cycles. This avoids advancing every coarse level at the finest-level frequency.
- The fin is a zero-thickness single Lagrangian surface.
- Only one DIBM body is intended for this tutorial. Mixed fin/rigid-body cases and multiple fins with different marker counts are outside its supported scope.
- Collision and 6-DOF dynamics are disabled for prescribed surfaces.
- The DIBM load is an integrated total; pressure and viscous contributions are not separated.
- The viscosity input is set to the dynamic viscosity of water (`1e-3 Pa s`) consistently with density-weighted IAMReX momentum diffusion. It should be kept in SI units with the other inputs.

The current `write_freq=10` writes the force CSV every ten finest-level steps (approximately `0.0220 s`). Periodic flow/geometry snapshots are disabled by `amr.plot_int=-1`. Changing it to `12` enables both the AMReX plotfile and matching `fin_vtk/fin_XXXXXXXX.vtk` every twelve Level-0 steps (approximately `0.1056 s`). Positive `ns.fixed_dt` selects a fixed step; `ns.cfl` does not adapt this fixed step.

## Integration and build selection

Use `DIM=3`, `USE_PARTICLES=TRUE`, and `PARTICLE_PARALLEL=FALSE` for this fin. `USE_MPI=TRUE` remains supported by this DIBM path. Supply `AMREX_HOME` and `AMREX_HYDRO_HOME` for your dependency locations. The existing RKPM path remains separate and does not implement this prescribed fin.

This merge includes source, inputs, documentation and `force-average.py`; no generated data or executable is included. The postprocessing script requires pandas, NumPy and matplotlib and reads your own `IB_Fin_1.csv`. No compilation, solver run or numerical test was performed for the merge.
