// ============================================================================
//  AquaCell  -  FLIP "fluid pendant" reference model            [ Common tab ]
//  16x16 grid, circular 216-cell container, 256 particles, accelerometer gravity
//  Modelled on mitxela's Fluid Simulation Pendant, which follows Ten Minute
//  Physics "How to write a FLIP Water Simulator".
// ============================================================================
//  Buffers (one fixed time step per frame, executed in this order):
//
//    A : G2P (FLIP/PIC blend) + mouse stir + shake + advect       -> particles
//    B : particle-particle push-apart + circular wall collision   -> particles (final)
//        + "accelerometer" / mouse / key state
//    C : P2G (tent weights) + cell marking (AIR/FLUID/SOLID)      -> old grid velocity,
//                                                                    density, count
//    D : gravity + BCs, divergence, iterative pressure solve,
//        pressure-gradient subtraction                            -> new grid velocity,
//                                                                    pressure, residual
//    Image : LED-matrix view (default) / smooth surface + debug views
//
//  Channel bindings (all buffers: nearest filter, clamp):
//    A : 0=Buffer B  1=Buffer C  2=Buffer D
//    B : 0=Buffer A  1=Buffer B  2=Keyboard
//    C : 0=Buffer B
//    D : 0=Buffer C  1=Buffer D  2=Buffer B
//    Image : 0=Buffer B  1=Buffer C  2=Buffer D  3=Keyboard
//
//  Data layout (bottom-left corner of each buffer):
//    A,B : pixel (i,j), i<PW, j<PH  = particle j*PW+i : (pos.xy, vel.xy)   [cells]
//    B   : pixel (PW  ,0) = (mouse.xy [cells], mouseDown, stirMode)
//          pixel (PW+1,0) = (gravity.xy [cells/s^2], shake, reset)        <- accelerometer
//          pixel (PW+2,0) = (mouseVel.xy, swirlPhase, 0)
//    C   : pixel (i,j), i<=NX, j<=NY
//            .x u at left face of cell (i,j)   (x=i   , y=j+.5)
//            .y v at bottom face of cell (i,j) (x=i+.5, y=j   )
//            .z particle density at cell centre (tent kernel)
//            .w particles in cell (>0 <=> FLUID, 0 <=> AIR)
//    D   : .x u_new  .y v_new  .z pressure  .w residual divergence
//
//  Units: 1 cell = 1 length unit, seconds, FIXED dt.
//  Pressure is stored as p' = p*dt/rho:   u_new = u - (p'_hi - p'_lo)
//  so the projection needs no multiplications by dt, rho or h.
//
//  CONTROLS
//    drag mouse            tilt the pendant (gravity points toward the cursor)
//    S + drag              stir the water instead
//    G                     gravity back to "down"        A  toggle auto-swirl
//    SPACE                 shake                         R  reset
//    0 smooth surface (instead of LEDs)   1 particles   2 cell types
//    3 velocity field     4 pressure      5 divergence (residual)
//    6 MAC grid velocities   7 (with 5) divergence BEFORE solve
//    8 (with 3/6) grid velocity BEFORE solve (raw P2G)
// ============================================================================

#define NX 16
#define NY 16
#define PW 16
#define PH 16
#define NP (PW*PH)                 // 256 particles

#define DT          0.016666667    // fixed step 1/60 s
#define GMAG        60.0           // gravity magnitude, cells/s^2
#define FLIP_RATIO  0.95           // 1 = pure FLIP, 0 = pure PIC
#define VMAX        40.0           // velocity clamp, cells/s  (CFL < 1)

// container: cell is SOLID if its centre is farther than sqrt(MASK_R2) from the
// middle  -> exactly 216 active cells of 256 (same as the pendant)
#define MASK_R2     70.0
#define CEN         vec2(8.0)
#define WALL_R      7.75           // particle collision circle (inscribed in the cell mask)

// particles
#define PRAD        0.3            // particle radius
#define MIN_DIST    (2.0*PRAD)
#define PUSH_RELAX  0.6
#define SPACING     0.65           // initial lattice spacing
#define REST_DENSITY (1.0/(SPACING*SPACING))
#define DRIFT_K     3.0            // grid-side density drift correction

// pressure solve: red-black Gauss-Seidel with over-relaxation, warm started
#define SOLVE_ITERS 30
#define OMEGA       1.6

// Fixed point emulation: all stored state is rounded to Q.12
#define FIXED_POINT 1
#define FIX_SCALE   4096.0
float Q(float x){
#if FIXED_POINT
    return floor(x*FIX_SCALE + 0.5)/FIX_SCALE;
#else
    return x;
#endif
}

const float NOT_FLUID = 1.0e9;

// ---------------------------------------------------------------- view
float viewScale(vec2 res){ return min(res.x/float(NX+2), res.y/float(NY+2)); }
vec2  viewOffset(vec2 res){ return 0.5*(res - viewScale(res)*vec2(float(NX),float(NY))); }
vec2  screenToGrid(vec2 p, vec2 res){ return (p - viewOffset(res))/viewScale(res); }

// ---------------------------------------------------------------- hash
uint ihash(uint x){
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16; return x;
}
float rnd(uint x){ return float(ihash(x)) * (1.0/4294967295.0); }

// ---------------------------------------------------------------- grid / cell types
bool inGrid(ivec2 c){ return c.x>=0 && c.y>=0 && c.x<NX && c.y<NY; }

// SOLID: outside the grid or outside the circle mask
bool isSolid(ivec2 c){
    if (!inGrid(c)) return true;
    vec2 d = vec2(c) + 0.5 - CEN;
    return dot(d,d) > MASK_R2;
}
// faces touching a SOLID cell carry no flow
bool uSolid(ivec2 f){ return isSolid(f) || isSolid(f-ivec2(1,0)); }
bool vSolid(ivec2 f){ return isSolid(f) || isSolid(f-ivec2(0,1)); }

int nonSolidNbrs(ivec2 c){
    return (isSolid(c+ivec2(-1,0))?0:1) + (isSolid(c+ivec2(1,0))?0:1)
         + (isSolid(c+ivec2(0,-1))?0:1) + (isSolid(c+ivec2(0,1))?0:1);
}

// Grid velocity after P2G + gravity, before projection.
// gB = Buffer C (old grid velocity), gdt = gravity*dt. SOLID faces are 0.
float faceUpre(sampler2D gB, ivec2 f, vec2 gdt){
    if (uSolid(f)) return 0.0;
    return texelFetch(gB, f, 0).x + gdt.x;
}
float faceVpre(sampler2D gB, ivec2 f, vec2 gdt){
    if (vSolid(f)) return 0.0;
    return texelFetch(gB, f, 0).y + gdt.y;
}
// divergence of the pre-projection field in cell c (physical, no drift term)
float cellDivRaw(sampler2D gB, ivec2 c, vec2 gdt){
    return faceUpre(gB, c+ivec2(1,0), gdt) - faceUpre(gB, c, gdt)
         + faceVpre(gB, c+ivec2(0,1), gdt) - faceVpre(gB, c, gdt);
}
