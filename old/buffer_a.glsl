// ============================================================================
//  Buffer A : Grid -> Particle (FLIP/PIC) + mouse stir + shake + advect
//  iChannel0 = Buffer B   (final particles + state of previous frame)
//  iChannel1 = Buffer C   (OLD grid velocity  = raw P2G, previous frame)
//  iChannel2 = Buffer D   (NEW grid velocity  = after gravity + projection)
//
//  FLIP update:   v_p = (1-r)*interp(new) + r*( v_p + interp(new - old) )
// ============================================================================

// a face sample is only trusted if one of its two cells is FLUID or SOLID
bool cellActive(ivec2 c){
    if (isSolid(c)) return true;
    return texelFetch(iChannel1, c, 0).w > 0.5;
}

// comp 0: u (faces at (i, j+.5))   comp 1: v (faces at (i+.5, j))
// returns (interp new, interp (new-old), total valid weight)
vec3 g2pComp(vec2 pos, int comp){
    vec2 g, lim;
    if (comp==0){ g = vec2(pos.x, pos.y-0.5); lim = vec2(float(NX),   float(NY-1)); }
    else        { g = vec2(pos.x-0.5, pos.y); lim = vec2(float(NX-1), float(NY));   }
    g = clamp(g, vec2(0.0), lim - 0.001);
    ivec2 i0 = ivec2(floor(g));
    vec2  t  = g - vec2(i0);
    float sw = 0.0, sn = 0.0, sd = 0.0;
    for (int k=0; k<4; k++){
        int dx = k&1, dy = k>>1;
        ivec2 f = i0 + ivec2(dx,dy);
        float w = (dx==0 ? 1.0-t.x : t.x) * (dy==0 ? 1.0-t.y : t.y);
        ivec2 other = (comp==0) ? f-ivec2(1,0) : f-ivec2(0,1);
        if (!(cellActive(f) || cellActive(other))) continue;
        float o = texelFetch(iChannel1, f, 0)[comp];
        float n = texelFetch(iChannel2, f, 0)[comp];
        sw += w;  sn += w*n;  sd += w*(n-o);
    }
    return sw > 1e-5 ? vec3(sn/sw, sd/sw, sw) : vec3(0.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    ivec2 px = ivec2(fragCoord);
    if (px.x >= PW || px.y >= PH){ fragColor = vec4(0.0); return; }
    int id = px.y*PW + px.x;

    vec4 S0 = texelFetch(iChannel0, ivec2(PW  ,0), 0);   // mouse
    vec4 S1 = texelFetch(iChannel0, ivec2(PW+1,0), 0);   // gravity, shake, reset
    vec4 S2 = texelFetch(iChannel0, ivec2(PW+2,0), 0);   // mouse velocity

    // ---------------- (re)initialise: block of water on a jittered lattice
    if (iFrame == 0 || S1.w > 0.5){
        vec2 jit = vec2(rnd(uint(id)*2u+1u), rnd(uint(id)*2u+2u)) - 0.5;
        vec2 pos = vec2(8.0 - 0.5*float(PW)*SPACING, 2.5)
                 + (vec2(px)+0.5)*SPACING + jit*0.2*SPACING;
        fragColor = vec4(Q(pos.x), Q(pos.y), 0.0, 0.0);
        return;
    }

    vec4 s   = texelFetch(iChannel0, px, 0);
    vec2 pos = s.xy;
    vec2 vel = s.zw;

    // ---------------- G2P
    vec3 gu = g2pComp(pos, 0);
    vec3 gv = g2pComp(pos, 1);
    vec2 vFlip = vel + vec2(gu.y, gv.y);                       // v += d(grid velocity)
    vec2 vPic  = vec2(gu.z>0.0 ? gu.x : vel.x, gv.z>0.0 ? gv.x : vel.y);
    vel = mix(vPic, vFlip, FLIP_RATIO);

    // ---------------- interaction
    if (S0.z > 0.5 && S0.w > 0.5){                             // S + drag : stir
        vec2  d = pos - S0.xy;
        float f = exp(-dot(d,d)/(2.0*2.0*2.0));                // sigma = 2 cells
        float sp = length(S2.xy);
        vel += (S2.xy - vel) * f * 0.5 * smoothstep(0.0, 4.0, sp);
    }
    if (S1.z > 0.5){                                           // shake
        uint h = uint(iFrame)*977u + uint(id);
        vel += (vec2(rnd(h), rnd(h^0x9E3779B9u)) - 0.5) * 30.0;
    }

    vel = clamp(vel, -VMAX, VMAX);
    pos += vel * DT;                                           // advect
    fragColor = vec4(Q(pos.x), Q(pos.y), Q(vel.x), Q(vel.y));
}
