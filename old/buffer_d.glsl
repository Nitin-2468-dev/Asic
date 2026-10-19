// ============================================================================
//  Buffer D : gravity + boundary conditions, divergence, pressure solve,
//             pressure-gradient subtraction
//  iChannel0 = Buffer C  (old grid velocity, density, cell counts)
//  iChannel1 = Buffer D  (own previous frame -> pressure warm start in .z)
//  iChannel2 = Buffer B  (accelerometer state: gravity vector)
//
//  The grid is only 16x16, so every pixel redundantly solves the WHOLE pressure
//  field (no window tricks, no approximation). All pixels end up with identical
//  pressures; each then writes out its own face velocities. On the ASIC this is
//  simply ONE solver running in place on a 256-entry pressure RAM.
//
//  Solver: red-black Gauss-Seidel with over-relaxation (OMEGA), warm started.
//    AIR   cell : p' = 0                          (free surface, Dirichlet)
//    SOLID cell : excluded                        (zero normal velocity, Neumann)
//    FLUID cell : p' <- p' + OMEGA*( (sum_nbr p' - div)/s - p' )
//                 s = number of non-SOLID neighbours
//  Divergence has a density drift term: compressed regions (density > rest) are
//  asked to expand a little, which fights the slow volume loss of FLIP.
// ============================================================================
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    ivec2 px = ivec2(fragCoord);
    if (px.x > NX || px.y > NY){ fragColor = vec4(0.0); return; }

    vec2 gdt = DT * texelFetch(iChannel2, ivec2(PW+1,0), 0).xy;     // gravity * dt

    float P   [NX*NY];
    float Dv  [NX*NY];
    float Sinv[NX*NY];

    // ---- set up: pressure (warm start), divergence, 1/s
    for (int k=0; k<NX*NY; k++){
        ivec2 c = ivec2(k%NX, k/NX);
        P[k] = 0.0;  Dv[k] = NOT_FLUID;  Sinv[k] = 0.0;
        if (isSolid(c)) continue;
        vec4 b = texelFetch(iChannel0, c, 0);
        if (b.w < 0.5) continue;                                    // AIR
        P[k]    = texelFetch(iChannel1, c, 0).z;
        Dv[k]   = cellDivRaw(iChannel0, c, gdt)
                - DRIFT_K*max(b.z/REST_DENSITY - 1.0, 0.0);
        Sinv[k] = 1.0/float(nonSolidNbrs(c));
    }

    // ---- red / black sweeps
    for (int it=0; it<SOLVE_ITERS; it++){
        for (int color=0; color<2; color++){
            for (int k=0; k<NX*NY; k++){
                if (Dv[k] > 0.5*NOT_FLUID) continue;
                ivec2 c = ivec2(k%NX, k/NX);
                if (((c.x+c.y)&1) != color) continue;
                float sum = (c.x>0    ? P[k-1]  : 0.0) + (c.x<NX-1 ? P[k+1]  : 0.0)
                          + (c.y>0    ? P[k-NX] : 0.0) + (c.y<NY-1 ? P[k+NX] : 0.0);
                float pn = (sum - Dv[k]) * Sinv[k];
                P[k] += OMEGA*(pn - P[k]);
            }
        }
    }

    // ---- pressure at this pixel's cell and the lower neighbours
    float pC = inGrid(px)            ? P[px.y*NX + px.x]       : 0.0;
    float pL = inGrid(px-ivec2(1,0)) ? P[px.y*NX + px.x-1]     : 0.0;
    float pB = inGrid(px-ivec2(0,1)) ? P[(px.y-1)*NX + px.x]   : 0.0;

    // ---- projection: subtract pressure gradient from the face velocities
    float u = 0.0, v = 0.0;
    if (!uSolid(px)) u = faceUpre(iChannel0, px, gdt) - (pC - pL);
    if (!vSolid(px)) v = faceVpre(iChannel0, px, gdt) - (pC - pB);

    // ---- diagnostics: divergence left over after the solve (FLUID cells)
    float res = 0.0;
    if (inGrid(px) && !isSolid(px) && texelFetch(iChannel0, px, 0).w > 0.5){
        float pR = inGrid(px+ivec2(1,0)) ? P[px.y*NX + px.x+1]     : 0.0;
        float pU = inGrid(px+ivec2(0,1)) ? P[(px.y+1)*NX + px.x]   : 0.0;
        res = cellDivRaw(iChannel0, px, gdt) - (pL + pR + pB + pU - float(nonSolidNbrs(px))*pC);
    }
    fragColor = vec4(Q(u), Q(v), Q(pC), res);
}
