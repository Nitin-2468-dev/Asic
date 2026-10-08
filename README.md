# AquaCell – FLIP fluid-pendant reference model [(ShaderToy)](https://www.shadertoy.com/view/NXyGWd)



Software reference for a small FLIP fluid ASIC, modelled on mitxela's Fluid Simulation
Pendant (which follows Ten Minute Physics "How to write a FLIP Water Simulator").

| | |
|---|---|
| Grid | 16x16 MAC grid, circular container, **216 active cells** (SOLID = outside r²=70 from centre) |
| Particles | 256 (16x16 texture) |
| Time step | fixed, 1/60 s per frame |
| Pressure | red-black Gauss-Seidel, SOR ω=1.6, 30 iterations/frame, warm started |
| Gravity | a vector (the "accelerometer"): mouse tilts it, keys reset / auto-swirl / shake |
| Fixed point | all stored state rounded to Q.12 (`FIXED_POINT` in Common) |

## ShaderToy setup
Create 4 buffer tabs + Common + Image, paste the matching `.glsl` file into each, and set the
channels (all buffer inputs: **Nearest** filter, **Clamp**):

| Tab | iChannel0 | iChannel1 | iChannel2 | iChannel3 |
|---|---|---|---|---|
| Common | – paste `common.glsl` | | | |
| Buffer A | Buffer B | Buffer C | Buffer D | |
| Buffer B | Buffer A | Buffer B | Keyboard | |
| Buffer C | Buffer B | | | |
| Buffer D | Buffer C | Buffer D | Buffer B | |
| Image | Buffer B | Buffer C | Buffer D | Keyboard |

Needs a 60 Hz display (one fixed step per frame).

## Controls
Drag = tilt pendant · S+drag = stir · G = gravity down · A = auto-swirl · Space = shake · R = reset
`0` smooth surface instead of LEDs · `1` particles · `2` cell types · `3` velocity field ·
`4` pressure · `5` divergence (`7` = before solve) · `6` MAC face velocities (`8` = raw P2G velocities)

## Frame pipeline (maps 1:1 to the ASIC stages)
1. **A – G2P:** `v = mix(interp(new), v + interp(new-old), 0.95)`, only faces next to a FLUID/SOLID cell are used; stir/shake; advect.
2. **B – particles:** push-apart (min distance 2r=0.6, needed or the fluid collapses, per the pendant write-up), circular wall collision; accelerometer state.
3. **C – P2G:** tent weights onto u/v faces, density, per-cell particle count → AIR/FLUID; SOLID from mask; faces touching SOLID = 0.
4. **D – solve:** add gravity·dt on the grid, divergence (+ density drift term), pressure iterations, subtract gradient. Pressure is stored as `p' = p·dt/ρ` so `u -= p'_hi - p'_lo` (no multiplies).

## What was verified
Run headless (software OpenGL, desktop GLSL 3.30 harness emulating ShaderToy's buffer order, not the browser):
all tabs compile; block drops, splashes and settles to a flat surface; tilt, stir, shake, reset work;
25 s of continuous swirl + shake: no NaNs, particle count conserved, no escapes from the wall, max cell density ≈1.4×rest.
Not tested in a real browser/WebGL2 – if a tab fails to compile there, tell me the error.

## HDL notes
* **Memory (≈6 KB):** particles 256×4×16 b = 2 KB; grid u,v old+new 4×272×16 b ≈ 2.2 KB; pressure 256×16 b; density/count 256×~8 b; solid mask is a 256-bit ROM.
* **Shader-only redundancy:** C is a gather and D re-solves the whole 16x16 pressure field in every pixel. In hardware C is a *scatter* (each particle adds w·v and w to 4 faces per component, then one divide pass), and D is **one** solver sweeping a 256-entry pressure RAM in place.
* **Push-apart** is O(N²) here; in hardware use the pendant's hash grid / cell list (own + 8 neighbour cells). It is one Jacobi pass/frame with relax 0.6 (two-pass sequential in the original), so pairs can overlap slightly (min NN distance seen ≈0.37 vs 0.6).
* **Wall collision** uses a circle (`|p-c|² > R²`, needs a divide/rsqrt to project back); a cell-lookup push toward the centre avoids that.
* **Divides:** weights normalisation (P2G), 1/s (s∈1..4 → LUT), push-apart 1/d. Everything else is add/multiply/shift.
* Tunables if you want to trade accuracy for area: `SOLVE_ITERS`, `OMEGA`, `PUSH_RELAX`, `DRIFT_K`, `FLIP_RATIO`, `FIX_SCALE`.
