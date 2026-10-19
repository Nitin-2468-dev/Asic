// ============================================================================
//  Buffer C : Particle -> Grid on the staggered MAC grid + cell marking
//  iChannel0 = Buffer B (final particles)
//
//  Written as a gather (every face loops over all particles). The ASIC does the
//  scatter: each particle adds w*v and w to its 4 surrounding faces per component.
//  Output (for pixel (i,j), 0<=i<=NX, 0<=j<=NY):
//    .x u (face at x=i,   y=j+.5)  weighted average of particle vx
//    .y v (face at x=i+.5,y=j  )  weighted average of particle vy
//    .z particle density at the cell centre (tent weights)
//    .w particle count  ( >0  <=>  FLUID,  0  <=>  AIR; SOLID decided by the mask )
// ============================================================================
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    ivec2 px = ivec2(fragCoord);
    if (px.x > NX || px.y > NY){ fragColor = vec4(0.0); return; }

    vec2 fu = vec2(px) + vec2(0.0, 0.5);          // u sample point
    vec2 fv = vec2(px) + vec2(0.5, 0.0);          // v sample point
    vec2 fc = vec2(px) + 0.5;                     // cell centre
    vec2  su = vec2(0.0), sv = vec2(0.0);         // (sum w*v, sum w)
    float sd = 0.0, cnt = 0.0;

    for (int id=0; id<NP; id++){
        vec4 s = texelFetch(iChannel0, ivec2(id%PW, id/PW), 0);
        vec2 d = abs(s.xy - fu);
        float w = max(1.0-d.x, 0.0)*max(1.0-d.y, 0.0);
        su += w*vec2(s.z, 1.0);
        d = abs(s.xy - fv);
        w = max(1.0-d.x, 0.0)*max(1.0-d.y, 0.0);
        sv += w*vec2(s.w, 1.0);
        d = abs(s.xy - fc);
        sd += max(1.0-d.x, 0.0)*max(1.0-d.y, 0.0);
        if (ivec2(floor(s.xy)) == px) cnt += 1.0;
    }
    float u = su.y > 1e-6 ? su.x/su.y : 0.0;
    float v = sv.y > 1e-6 ? sv.x/sv.y : 0.0;
    if (uSolid(px)) u = 0.0;                      // boundary condition: no flow into walls
    if (vSolid(px)) v = 0.0;
    if (isSolid(px)){ sd = 0.0; cnt = 0.0; }
    fragColor = vec4(Q(u), Q(v), sd, cnt);
}
