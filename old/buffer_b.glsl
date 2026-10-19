// ============================================================================
//  Buffer B : particle push-apart + circular wall collision + input state
//  iChannel0 = Buffer A   (particles after G2P + advect, this frame)
//  iChannel1 = Buffer B   (own previous frame: input state)
//  iChannel2 = Keyboard
//
//  Push-apart is REQUIRED (the pendant author found the fluid collapses into an
//  overlapping mess without it). One Jacobi pass per frame, every pair closer
//  than 2r moves apart by half the overlap each (x PUSH_RELAX).
//  On the ASIC this is the hash-grid / cell-list step: each particle only has to
//  look at the particles in its own and the 8 neighbouring cells.
// ============================================================================
#define KEY_SPACE 32
#define KEY_A 65
#define KEY_G 71
#define KEY_R 82
#define KEY_S 83

bool keyDown(int k){ return texelFetch(iChannel2, ivec2(k,0), 0).x > 0.5; }
bool keyTog (int k){ return texelFetch(iChannel2, ivec2(k,2), 0).x > 0.5; }

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    ivec2 px = ivec2(fragCoord);

    // ------------------------------------------------ input / accelerometer state
    if (px.y == 0 && px.x >= PW && px.x <= PW+2){
        vec2  mouse = screenToGrid(iMouse.xy, iResolution.xy);
        bool  down  = iMouse.z > 0.0;
        bool  stir  = keyDown(KEY_S);
        vec4  P0 = texelFetch(iChannel1, ivec2(PW  ,0), 0);
        vec4  P1 = texelFetch(iChannel1, ivec2(PW+1,0), 0);
        vec4  P2 = texelFetch(iChannel1, ivec2(PW+2,0), 0);

        // mouse velocity (cells/s), only valid while the button stays down
        vec2 mv = (down && P0.z > 0.5) ? (mouse - P0.xy)/DT : vec2(0.0);
        float ml = length(mv);  if (ml > 60.0) mv *= 60.0/ml;

        // accelerometer: gravity vector (cells/s^2)
        vec2  g     = (iFrame==0 || dot(P1.xy,P1.xy) < 1e-4) ? vec2(0.0,-GMAG) : P1.xy;
        float phase = P2.z;
        vec2  tgt   = g;
        if (keyTog(KEY_A)){ phase += 0.025; tgt = GMAG*vec2(sin(phase), -cos(phase)); }
        vec2  dm = mouse - CEN;
        if (down && !stir && length(dm) > 0.5) tgt = GMAG*normalize(dm);   // tilt
        if (keyDown(KEY_G)) tgt = vec2(0.0,-GMAG);
        g += (tgt - g)*0.25;                                     // sensor filtering

        float shake = keyDown(KEY_SPACE) ? 1.0 : 0.0;
        float reset = (keyDown(KEY_R) || iFrame==0) ? 1.0 : 0.0;

        if (px.x == PW)   fragColor = vec4(mouse, down?1.0:0.0, stir?1.0:0.0);
        if (px.x == PW+1) fragColor = vec4(g, shake, reset);
        if (px.x == PW+2) fragColor = vec4(mv, phase, 0.0);
        return;
    }
    if (px.x >= PW || px.y >= PH){ fragColor = vec4(0.0); return; }

    // ------------------------------------------------ particle push-apart
    int  id = px.y*PW + px.x;
    vec4 s  = texelFetch(iChannel0, px, 0);
    vec2 pos = s.xy,  vel = s.zw;

    vec2 disp = vec2(0.0);
    for (int j=0; j<NP; j++){
        if (j == id) continue;
        vec2  q  = texelFetch(iChannel0, ivec2(j%PW, j/PW), 0).xy;
        vec2  d  = pos - q;
        float d2 = dot(d,d);
        if (d2 < MIN_DIST*MIN_DIST){
            if (d2 < 1e-8){                                   // coincident: random direction
                float a = 6.2831853*rnd(uint(id*NP+j));
                d = 1e-3*vec2(cos(a), sin(a));  d2 = dot(d,d);
            }
            float dl = sqrt(d2);
            disp += d * (0.5*(MIN_DIST - dl)/dl);
        }
    }
    pos += PUSH_RELAX * disp;

    // ------------------------------------------------ circular container
    vec2  r  = pos - CEN;
    float rl = length(r);
    if (rl > WALL_R){
        vec2 n = r/rl;
        pos = CEN + n*WALL_R;
        float vn = dot(vel, n);
        if (vn > 0.0) vel -= n*vn;                            // kill outward velocity
    }
    fragColor = vec4(Q(pos.x), Q(pos.y), Q(vel.x), Q(vel.y));
}
