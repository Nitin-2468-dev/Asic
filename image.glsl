// ============================================================================
//  Image : pendant LED view (default), smooth surface, debug views
//  iChannel0 = Buffer B (particles + input state)   iChannel1 = Buffer C (old grid)
//  iChannel2 = Buffer D (new grid, pressure)        iChannel3 = Keyboard
// ============================================================================
#define K_SMOOTH 48   // 0
#define K_PART   49   // 1
#define K_CELLS  50   // 2
#define K_VEL    51   // 3
#define K_PRESS  52   // 4
#define K_DIV    53   // 5
#define K_MAC    54   // 6
#define K_PREDIV 55   // 7
#define K_OLD    56   // 8

#define PRESS_SCALE 12.0
#define DIV_SCALE   3.0

bool  tog(int k){ return texelFetch(iChannel3, ivec2(k,2), 0).x > 0.5; }
vec4  gP(ivec2 p){ return texelFetch(iChannel0, p, 0); }      // particles / state
vec4  gC(ivec2 p){ return texelFetch(iChannel1, p, 0); }      // old grid
vec4  gD(ivec2 p){ return texelFetch(iChannel2, p, 0); }      // new grid

bool isFluid(ivec2 c){ return !isSolid(c) && gC(c).w > 0.5; }

vec3 heat(float t){
    t = clamp(t, 0.0, 1.0);
    vec3 a = vec3(0.02,0.03,0.18), b = vec3(0.75,0.15,0.45), c = vec3(1.0,0.9,0.35);
    return t < 0.5 ? mix(a,b,t*2.0) : mix(b,c,t*2.0-1.0);
}
vec3 diverge(float x){
    x = clamp(x, -1.0, 1.0);
    return x < 0.0 ? mix(vec3(1.0), vec3(0.1,0.3,1.0), -x) : mix(vec3(1.0), vec3(1.0,0.2,0.1), x);
}
float sdSeg(vec2 p, vec2 a, vec2 b){
    vec2 pa = p-a, ba = b-a;
    float h = clamp(dot(pa,ba)/max(dot(ba,ba),1e-8), 0.0, 1.0);
    return length(pa - ba*h);
}
float arrow(vec2 g, vec2 c, vec2 vs){                // vs = displacement vector
    float l = length(vs);
    if (l > 1.3){ vs *= 1.3/l; l = 1.3; }
    vec2 e = c + vs;
    float d = sdSeg(g, c, e);
    if (l > 0.12){
        vec2 dir = vs/l, n = vec2(-dir.y, dir.x);
        float hl = min(0.25, l*0.5);
        d = min(d, sdSeg(g, e, e - dir*hl + n*hl*0.5));
        d = min(d, sdSeg(g, e, e - dir*hl - n*hl*0.5));
    }
    return d;
}
vec2 cellVel(ivec2 c, bool old){                      // cell-centred velocity from its faces
    vec4 a = old ? gC(c)            : gD(c);
    vec4 r = old ? gC(c+ivec2(1,0)) : gD(c+ivec2(1,0));
    vec4 t = old ? gC(c+ivec2(0,1)) : gD(c+ivec2(0,1));
    return vec2(0.5*(a.x+r.x), 0.5*(a.y+t.y));
}
// density with SOLID cells borrowing from their fluid neighbours (surface touches walls)
float dens(ivec2 c){
    c = clamp(c, ivec2(0), ivec2(NX-1,NY-1));
    if (isSolid(c)){
        float m = 0.0;
        for (int k=0; k<4; k++){
            ivec2 n = c + ((k==0)?ivec2(1,0):(k==1)?ivec2(-1,0):(k==2)?ivec2(0,1):ivec2(0,-1));
            if (inGrid(n) && !isSolid(n)) m = max(m, gC(n).z);
        }
        return m;
    }
    return gC(c).z;
}
float densField(vec2 g){
    vec2 q = g - 0.5;  vec2 f = floor(q);  vec2 t = q - f;  ivec2 i0 = ivec2(f);
    return mix(mix(dens(i0), dens(i0+ivec2(1,0)), t.x),
               mix(dens(i0+ivec2(0,1)), dens(i0+ivec2(1,1)), t.x), t.y);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2  res = iResolution.xy;
    float pxw = 1.0/viewScale(res);                  // one screen pixel in grid units
    vec2  g   = screenToGrid(fragCoord, res);
    ivec2 c   = ivec2(floor(g));
    float rr  = length(g - CEN);

    vec3 col = vec3(0.012,0.013,0.016);
    // gold case ring + black display
    if (rr < 8.95) col = vec3(0.72,0.55,0.20) * (0.75 + 0.25*smoothstep(8.5,8.95,rr));
    bool inDisp = rr < 8.5;
    if (inDisp) col = vec3(0.01,0.012,0.016);

    bool dbg = tog(K_CELLS)||tog(K_PRESS)||tog(K_DIV)||tog(K_VEL)||tog(K_MAC);

    if (inDisp && inGrid(c) && !isSolid(c)){
        vec4 cc = gC(c), dd = gD(c);
        bool fluid = cc.w > 0.5;
        bool old = tog(K_OLD);

        if (!tog(K_SMOOTH)){
            // ---------------- LED matrix: brightness = particle density in the cell
            float b = smoothstep(0.1, 1.0, cc.z/REST_DENSITY);
            float d = length(g - (vec2(c)+0.5));
            float led = 1.0 - smoothstep(0.34, 0.34+pxw*1.5, d);
            vec3 lc = vec3(0.25,0.75,1.0);
            col = vec3(0.03,0.04,0.05)*led + lc*b*led + lc*0.18*b*exp(-d*d*14.0);
            if (dbg) col *= 0.45;
        } else {
            // ---------------- smooth surface from the grid density
            float f  = densField(g);
            float th = 0.45*REST_DENSITY;
            float w  = max(fwidth(f), 1e-3);
            float m  = smoothstep(th-w, th+w, f);
            vec3 deep = vec3(0.02,0.20,0.48), shallow = vec3(0.20,0.62,0.92);
            vec3 wc = mix(shallow, deep, clamp((f-th)/(0.8*REST_DENSITY), 0.0, 1.0));
            float rim = exp(-pow((f-th)/(REST_DENSITY*0.12), 2.0));
            wc += vec3(0.5,0.75,1.0)*0.45*rim;
            col = mix(col, wc, m);
        }

        if (tog(K_DIV)){
            float x = tog(K_PREDIV) ? cellDivRaw(iChannel1, c, DT*gP(ivec2(PW+1,0)).xy) : dd.w;
            if (fluid) col = mix(col, diverge(x/DIV_SCALE), 0.85);
        } else if (tog(K_PRESS)){
            if (fluid) col = mix(col, heat(dd.z/PRESS_SCALE), 0.85);
        } else if (tog(K_CELLS)){
            col = mix(col, fluid ? vec3(0.15,0.45,1.0) : vec3(0.07,0.07,0.09), 0.6);
        } else if (tog(K_VEL)){
            if (fluid) col = mix(col, heat(length(cellVel(c,old))/30.0), 0.55);
        }
    }
    if (tog(K_CELLS) && inGrid(c) && isSolid(c)) col = mix(col, vec3(0.45,0.46,0.5), 0.7);

    // grid lines
    if (dbg && g.x>0.0 && g.y>0.0 && g.x<float(NX) && g.y<float(NY)){
        vec2 gf = fract(g), dl = min(gf, 1.0-gf);
        col = mix(col, vec3(1.0), 0.10*(1.0 - smoothstep(0.0, pxw*1.2, min(dl.x,dl.y))));
    }

    // velocity arrows (cell centred)
    if (tog(K_VEL) && inDisp){
        bool old = tog(K_OLD);
        float best = 1e9;
        for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++){
            ivec2 n = c + ivec2(dx,dy);
            if (!isFluid(n)) continue;
            best = min(best, arrow(g, vec2(n)+0.5, cellVel(n,old)*0.05));
        }
        col = mix(col, vec3(1.0), 1.0 - smoothstep(0.03, 0.03+pxw*1.5, best));
    }

    // MAC grid velocities: u bars on vertical faces (red), v bars on horizontal faces (green)
    if (tog(K_MAC) && inDisp){
        bool old = tog(K_OLD);
        float du = 1e9, dv = 1e9, pu = 1e9, pv = 1e9;
        for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=2; dx++){
            ivec2 f = c + ivec2(dx,dy);                          // u face at (f.x, f.y+.5)
            if (f.x<0 || f.x>NX || f.y<0 || f.y>=NY) continue;
            if (!(isFluid(f) || isFluid(f-ivec2(1,0)))) continue;
            float u = old ? gC(f).x : gD(f).x;
            vec2 p0 = vec2(f) + vec2(0.0,0.5);
            du = min(du, sdSeg(g, p0, p0 + vec2(clamp(u*0.04,-1.0,1.0), 0.0)));
            pu = min(pu, length(g-p0));
        }
        for (int dy=-1; dy<=2; dy++) for (int dx=-1; dx<=1; dx++){
            ivec2 f = c + ivec2(dx,dy);                          // v face at (f.x+.5, f.y)
            if (f.x<0 || f.x>=NX || f.y<0 || f.y>NY) continue;
            if (!(isFluid(f) || isFluid(f-ivec2(0,1)))) continue;
            float v = old ? gC(f).y : gD(f).y;
            vec2 p0 = vec2(f) + vec2(0.5,0.0);
            dv = min(dv, sdSeg(g, p0, p0 + vec2(0.0, clamp(v*0.04,-1.0,1.0))));
            pv = min(pv, length(g-p0));
        }
        float aa = pxw*1.5;
        col = mix(col, vec3(1.0,0.35,0.30), 1.0 - smoothstep(0.035, 0.035+aa, du));
        col = mix(col, vec3(0.30,1.00,0.45), 1.0 - smoothstep(0.035, 0.035+aa, dv));
        col = mix(col, vec3(1.0,0.8,0.7), 1.0 - smoothstep(0.07, 0.07+aa, pu));
        col = mix(col, vec3(0.8,1.0,0.8), 1.0 - smoothstep(0.07, 0.07+aa, pv));
    }

    // particles
    if (tog(K_PART) && inDisp){
        float cov = 0.0;  vec3 pc = vec3(1.0);
        for (int id=0; id<NP; id++){
            vec4 s = gP(ivec2(id%PW, id/PW));
            float d = length(g - s.xy);
            if (d < PRAD + 0.05){
                float a = 1.0 - smoothstep(PRAD*0.8 - pxw, PRAD*0.8 + pxw, d);
                if (a > cov){ cov = a; pc = mix(vec3(0.6,0.95,1.0), vec3(1.0,0.5,0.2), length(s.zw)/25.0); }
            }
        }
        col = mix(col, pc, cov*0.9);
    }

    // accelerometer indicator: orange dot on the case ring toward gravity
    {
        vec2 gv = gP(ivec2(PW+1,0)).xy;
        vec2 gd = normalize(gv + vec2(0.0,-1e-4));
        vec2 mk = CEN + gd*8.72;
        col = mix(col, vec3(1.0,0.45,0.1), 1.0 - smoothstep(0.2, 0.2+pxw*1.5, length(g-mk)));
    }
    // cursor ring (stir radius)
    if (iMouse.z > 0.0 && texelFetch(iChannel3, ivec2(83,0), 0).x > 0.5){
        vec2 mg = screenToGrid(iMouse.xy, res);
        col = mix(col, vec3(1.0), 0.5*(1.0 - smoothstep(pxw, 2.5*pxw, abs(length(g-mg)-2.5))));
    }
    fragColor = vec4(col, 1.0);
}
