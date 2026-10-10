from pyray import *
import  pyray as pr
import numpy as np

width = 800
height = 800

AIR   = 0
WATER = 1
SOLID = 2

DARKERGRAY = Color(50, 50, 50, 255)
WATER_COLOR = get_color(0x8BCCEFFF)

class Main:
    def __init__(self):
        self.title = "Flip Example"
        init_window(width, height, self.title)
        set_window_state(ConfigFlags.FLAG_WINDOW_UNDECORATED)
        set_config_flags(ConfigFlags.FLAG_VSYNC_HINT)   
        set_target_fps(60)
        self.f = Flip()
        
    def loop(self):
        while not window_should_close():
            self.update()
            begin_drawing()
            clear_background(BLACK)
            self.draw()
            end_drawing()
        
        close_window()
        
    def update(self):
        if is_key_pressed(KeyboardKey.KEY_R):
            self.f.reset()
        self.f.grab()
        
        
        self.f.update()
    
    def draw(self):
        self.f.draw_grid()
        self.f.draw()
        draw_fps(10, 10)
        
    
    
    
class Flip:
    def __init__(self):
        self.grid_size = 16 # 16x16 grid
        self.amount = 13 ** 2 # number of particles (Perfect square for grid initialization)
        self.gravity_mag = 40.0
        self.gravity = np.array([0, -self.gravity_mag]) # gravity vector
        self.base_gravity = np.array([0, -self.gravity_mag]) # gravity vector
        self.win_v = np.zeros(2)
        self.push = np.zeros(2)
        self.dt = 1/60 # time step (seconds)
        self.flip_ratio = 0.7 # flip ratio (0 < flip_ratio < 1) 1 pure flip, 0 pure pic
        
        self.k = 1.0 # pressure / density drift Stiffness coefficient
        self.vmax = 40.0 # maximum velocity for particles
        self.O = 1.9 # Overrelaxtions (1 < O < 2)
        self.itterations = 35 # number of iterations for pressure solver
        
        self.spacing = 0.65 # spacing between particles
        self.rest_density = 1 / self.spacing**2 # divergence coefficient
        self.min_distance = 0.6 # minimum distance between particles for collision detection
        self.push_relax = 0.5 
        self.push_itters = 1
        
        self.tablesize  =  self.grid_size * self.grid_size # size of the hash table for particle collision detection
        self.debug_check = True
        
        self._build_contanter()
        self.reset()
        
        self.last_x  = get_mouse_x()
        self.last_y = get_mouse_y()

        # # Partical 
        # self.positions= np.zeros((self.amount, 2), dtype=np.float32) # x , y 
        # self.velocity= np.zeros((self.amount, 2), dtype=np.float32) #  u , v 
        # self.radius = 0.01 # particle radius
        
        
        # x = np.linspace(0.2, 0.8, 16)
        # y = np.linspace(0.2, 0.8, 16)

        # xx, yy = np.meshgrid(x, y)

        # self.positions[:, 0] = xx.ravel()
        # self.positions[:, 1] = yy.ravel()
        
        # # GRID 
        # self.u = np.zeros((self.grid_size + 1, self.grid_size ), dtype=np.float32) # u (horizontal  velocity)
        # self.v = np.zeros((self.grid_size , self.grid_size + 1), dtype=np.float32) # v (vertical  velocity)
        # self.u_sum = np.zeros((self.grid_size + 1, self.grid_size), dtype=np.float32) #  accumulated (horizontal velocity)
        # self.v_sum = np.zeros((self.grid_size , self.grid_size + 1), dtype=np.float32) #  accumulated (vertical velocity)
        # self.u_weight = np.zeros((self.grid_size + 1, self.grid_size), dtype=np.float32) #  weight (horizontal velocity)
        # self.v_weight = np.zeros((self.grid_size , self.grid_size +1 ), dtype=np.float32) #  weight (vertical velocity)
        
        # self.Pressure = np.zeros((self.grid_size, self.grid_size), dtype=np.float32) # pressure
        # self.density = np.zeros((self.grid_size, self.grid_size), dtype=np.float32) # density
        # self.divergence = np.zeros((self.grid_size, self.grid_size), dtype=np.float32) # divergence
        # #self.s = np.zeros((self.grid_size, self.grid_size), dtype=np.int32) # solid cells (0: Air, 1: Water, 2: Solid)
        # self.cell_type = np.zeros((self.grid_size, self.grid_size), dtype=np.int32) # cell type (0: Air, 1: Water, 2: Solid)
        
        # self.cell_start = np.zeros(self.grid_size * self.grid_size + 1, dtype=np.int32) # cell start index
        
        
        # x ,y = np.indices((self.grid_size, self.grid_size))
        # cx = (self.grid_size-1)  / 2
        # cy = (self.grid_size - 1) / 2 
        # r = self.grid_size / 2
        # mask = (x - cx) ** 2 + (y - cy) **  2 < r ** 2
        # self.solid_mask = mask
        # self.cell_type[~self.solid_mask] = SOLID
        # # self.cell_start[1:] = np.cumsum(np.sum(self.cell_type == 0, axis=1))
        # #self.s[self.solid_mask] = 1 # 0 for differentiate solid cells from fluid cells
        # self.cell_type[self.solid_mask & ~(y < self.grid_size / 2)] = WATER

    def grab(self):
            self.mx = get_mouse_x()
            self.my = get_mouse_y()
    
            if is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT):
                dx = self.mx - self.last_x
                dy = self.my - self.last_y
                v = np.array([dx, -dy])
                a = (v - self.win_v) * 3600 / self.dt
                self.win_v = v
                self.push += 0.3 * (a - self.push)
                
                g = self.base_gravity - 0.5 * self.push
                n = np.hypot(*g)
                if n > 60:
                    g = g / n * 60
                self.gravity = g
                
                if dx != 0 or dy != 0:  # only move if there's actual movement
                    self.gravity_mag = np.hypot(dx, dy) * 0.5
                    self.gravity = np.array([dx, -dy]) * 0.5
                    
                    set_window_position(int(get_window_position().x + dx),int(get_window_position().y + dy))
            else:
                self.last_x = self.mx
                self.last_y = self.my   
            
    def _build_contanter(self):
        N = self.grid_size
        ix , iy = np.indices((N, N))
        c = N /2 # center
        
        self.inside = ((ix + 0.5 - c) ** 2 + (iy + 0.5 - c) ** 2 ) < 70.0
        self.center = np.array([c, c])
        
        # 1 for Open non-solid cells, 0 for solid cells
        self.s = np.where(self.inside, 1, 0).astype(np.float64)
        
        sp = np.pad(self.s ,1)
        self.sx0 = sp[0:N , 1:N + 1] # s[i-1, j]
        self.sx1 = sp[2:N + 2, 1:N + 1] # s[i+1, j]
        self.sy0 = sp[1:N + 1, 0:N] # s[i, j-1]
        self.sy1 = sp[1:N + 1, 2:N + 2] # s[i, j+1]
        
        self.u_open = sp[0:N+1, 1:N+1] * sp[1:N+2, 1:N+1] # (N+1,N)
        self.v_open = sp[1:N+1, 0:N+1] * sp[1:N+1, 1:N+2] # (N,N+1)
        
        # Compute the distance to the nearest wall for each cell
        # si ,sj = np.where(not self.inside.all(), 1 , 0)
        si , sj = np.where(~self.inside)
        dx = np.maximum.reduce([si - c , c - (si + 1), np.zeros(len(si))])
        dy = np.maximum.reduce([sj - c , c - (sj + 1), np.zeros(len(sj))])
        self.wall_r = np.hypot(dx, dy).min() - 0.05
        
    def reset(self):
        N = self.grid_size
        rng = np.random.default_rng(1)
        
        self.positions = np.zeros((self.amount, 2), dtype=np.float32)
        self.velocity = np.zeros((self.amount, 2), dtype=np.float32)
        
        side = int(round(np.sqrt(self.amount)))
        assert side * side == self.amount, "Amount must be a perfect square"
        xs = self.center[0] + (np.arange(side) - side / 2 + 0.5) * self.spacing
        ys = 2.5 + (np.arange(side) + 0.5) * self.spacing
        xx , yy = np.meshgrid(xs, ys)
        self.positions[:, 0] = xx.ravel()
        self.positions[:, 1] = yy.ravel()
        self.positions += (rng.random((self.amount, 2)) - 0.5) * self.spacing * 0.1
        
        # GRID (index order [x,y])
        self.u = np.zeros((N + 1, N), dtype=np.float32) # u (horizontal velocity)
        self.u_sum = np.zeros((N + 1, N), dtype=np.float32) # accumulated (horizontal velocity)
        self.u_weight = np.zeros((N + 1, N), dtype=np.float32) # weight (horizontal velocity)
        self.u_old = np.zeros((N + 1, N), dtype=np.float32) # previous (horizontal velocity)
        self.u_valid = np.zeros((N + 1, N), dtype=np.bool_) # valid (horizontal velocity)
        
        self.v = np.zeros((N, N + 1), dtype=np.float32) # v (vertical velocity)
        self.v_sum = np.zeros((N, N + 1), dtype=np.float32) # accumulated (vertical velocity)
        self.v_weight = np.zeros((N, N + 1), dtype=np.float32) # weight (vertical velocity)
        self.v_old = np.zeros((N, N + 1), dtype=np.float32) # previous (vertical velocity)
        self.v_valid = np.zeros((N, N + 1), dtype=np.bool_) # valid (vertical velocity)
        
        self.Pressure = np.zeros((N, N), dtype=np.float32) # pressure
        self.density = np.zeros((N, N), dtype=np.float32) # density
        self.divergence = np.zeros((N, N), dtype=np.float32) # divergence
        self.cell_type = np.zeros((N, N), dtype=np.int32) # cell type (0: Air, 1: Water, 2: Solid)
        
    # Helper functions 
    def _check(self, where):
        if self.debug_check:
            for name in ("positions", "velocity" , "u" , "v"):
                assert not np.isnan(getattr(self, name)).any(), f"Nan in {name} after {where}"
                
    def _corners(self, gx , gy , shape):
        """The 4 grid points around (gx, gy) and their bilinear weights.
        Clamped using the array's OWN shape, so it is right for u, v and cells alike."""
        
        gx = np.clip(gx , 0 , shape[0] - 1 - 1e-6)
        gy = np.clip(gy , 0 , shape[1] - 1 - 1e-6)
        i0 = np.floor(gx).astype(np.int32)
        j0 = np.floor(gy).astype(np.int32)
        dx = gx - i0
        dy = gy - j0
        idx = ((i0 , j0) , (i0 + 1, j0) , (i0 + 1, j0 + 1) , (i0, j0 + 1))
        w = ((1 - dx) * (1 - dy) , (dx) * (1 - dy) , (dx) * (dy) , (1 - dx) * (dy))
        return idx , w
    def _scatter(self, total , weight , gx , gy , values=None):
        idx , w = self._corners(gx , gy , weight.shape)
        for (i , j), wk in zip(idx , w):
            np.add.at(weight , (i , j) , wk) # np.add.at: duplicates ADD UP
            if values is not None:
                np.add.at(total , (i , j) , values * wk)
    
    def _gather(self , new ,old , weight , gx , gy , valid):
        """Bilinear sample of `new` and of (new - old), ignoring invalid (all-AIR) faces."""
        idx , w = self._corners(gx , gy , weight.shape)
        wsum = 0.0
        vnew = 0.0
        vdelta = 0.0
        for (i , j), wk in zip(idx , w):
            wv = wk * valid[i, j]
            wsum += wv
            vnew += new[i, j] * wv
            vdelta += (new[i, j] - old[i, j]) * wv
        ok = wsum > 1e-6
        safe = np.where(ok, wsum, 1.0)
        
        return vnew/ safe, vdelta /safe, ok

    def update(self):
        self.apply_gravity();                       self._check("gravity")
        self.positions += self.velocity * self.dt   # advect
        self.push_apart();                          self._check("push_apart")
        self.collide_walls();                       self._check("walls")
        self.Particle_to_Grid();                    self._check("particle_to_grid")
        self.update_cell_type();                    
        self.Divergence();                          self._check("pressure_solve")
        self.Grid_to_Particle();                    self._check("grid_to_particle")
        
    def apply_gravity(self):
        self.velocity += self.gravity * self.dt
        self.velocity = np.clip(self.velocity, -self.vmax, self.vmax)
        
    def push_apart(self):
        md = self.min_distance
        for _ in range(self.push_itters):
            p = self.positions
            d = p[:, None, :] - p[None, :, :] # d[i,j] = p[i] - p[j]    (N.N,2)
            d2 = (d * d).sum(axis=-1)
            near = (d2 < md * md) & (d2 > 1e-10) # avoid self-collision
            dist = np.sqrt(np.where(near, d2, 1.0)) # avoid division by zero
            push = np.where(near, 0.5 * (md - dist) / dist, 0.0) # push amount
            self.positions = p + self.push_relax * (push[:,:,None] * d).sum(axis=1)

    def collide_walls(self):
        r = self.positions - self.center
        dist = np.hypot(r[:, 0], r[:, 1])
        out = dist > self.wall_r
        
        if out.any():
            n = r[out] / dist[out][:, None]
            self.positions[out] = self.center + n * self.wall_r
            vn = (self.velocity[out] * n).sum(axis=1)
            vn = np.maximum(vn, 0.0) # only remove Outward velocity
            self.velocity[out] -= n * vn[:, None]
    
    def update_cell_type(self):    
        N = self.grid_size
        self.cell_type.fill(AIR)
        self.cell_type[~self.inside] = SOLID
        # count = np.zeros((N, N), dtype=np.int32)
        # i = np.clip(np.floor(self.positions[: , 0]).astype(np.int32), 0, N - 1)
        # j = np.clip(np.floor(self.positions[: , 1]).astype(np.int32), 0, N - 1)
        # np.add.at(count, (i, j), 1)
        
        water = (self.density > 0.1  * self.rest_density) & self.inside
        self.cell_type[water] = WATER
        
        # which faces may be trusted when going from grid to particle (only faces with at least one water cell)
        ct = np.pad(self.cell_type, 1, constant_values=SOLID)
        self.u_valid = ((ct[0:N + 1, 1:N + 1] != AIR) | (ct[1:N + 2, 1:N + 1] != AIR)).astype(np.float32) # (N+1,N)
        self.v_valid = ((ct[1:N + 1, 0:N + 1] != AIR) | (ct[1:N + 1, 1:N + 2] != AIR)).astype(np.float32) # (N,N+1)
    
    # partical --> grid
    def Grid_to_Particle(self):        
        # # u
        # gx = np.clip(self.positions[:, 0] * self.grid_size,0, self.grid_size - 1 - 1e-6)
        # gy = np.clip(self.positions[:, 1] * self.grid_size - 0.5,0, self.grid_size - 1 - 1e-6)
        
        # i0 = np.floor(gx).astype(np.int32)
        # j0 = np.floor(gy).astype(np.int32)
        
        # dx = gx - i0
        # dy = gy - j0
        
        # w1 = (1 - dx) * (1 - dy)
        # w2 = (dx) * (1 - dy)
        # w3 = (dx) * (dy)
        # w4 = (1 - dx) * (dy)
        
        # u_vel = (self.u[i0, j0] * w1 + self.u[i0 + 1, j0] * w2 + self.u[i0 + 1, j0 + 1] * w3 + self.u[i0, j0 + 1] * w4)
        
        # # v
        # gx = np.clip(self.positions[:, 0] * self.grid_size - 0.5, 0, self.grid_size - 1 - 1e-6)
        # gy = np.clip(self.positions[:, 1] * self.grid_size, 0 , self.grid_size - 1 - 1e-6)
        
        # i0 = np.floor(gx).astype(np.int32)
        # j0 = np.floor(gy).astype(np.int32)
        
        # dx = gx - i0
        # dy = gy - j0
        
        # w1 = (1 - dx) * (1 - dy)
        # w2 = (dx) * (1 - dy)
        # w3 = (dx) * (dy)
        # w4 = (1 - dx) * (dy)
        
        # v_vel = (self.v[i0, j0] * w1 + self.v[i0 + 1, j0] * w2 + self.v[i0 + 1, j0 + 1] * w3 + self.v[i0, j0 + 1] * w4)
        
        # self.velocity[:, 0] = u_vel
        # self.velocity[:, 1] = v_vel
        p = self.positions
        r = self.flip_ratio
        for comp , (new , old , valid , gx , gy) in enumerate((
                (self.u, self.u_old, self.u_valid, p[:, 0], p[:, 1] - 0.5),
                    (self.v, self.v_old, self.v_valid, p[:, 0] - 0.5, p[:, 1])
                )):
            vnew , vdelta , ok = self._gather(new , old , valid , gx, gy, valid)
            v = self.velocity[:, comp]
            v_pic = np.where(ok, vnew, v)
            v_flip = np.where(ok, vdelta, 0.0)
            self.velocity[:, comp] = (1 -r) * v_pic + r * (v + v_flip)
                
    def Particle_to_Grid(self):       
        # # U
        # gx = np.clip(self.positions[:, 0] * self.grid_size ,0, self.grid_size - 1 - 1e-6)
        # gy = np.clip(self.positions[:, 1] * self.grid_size - 0.5, 0 , self.grid_size - 1 - 1e-6)
        
        # i0 = np.floor(gx).astype(np.int32)
        # j0 = np.floor(gy).astype(np.int32)
        
        # dx = gx - i0
        # dy = gy - j0
        
        # # clear
        # self.u_weight.fill(0)
        # self.u_sum.fill(0)
        # self.v_weight.fill(0)
        # self.v_sum.fill(0)
        
        
        # w1 = (1 - dx) * (1 - dy)
        # w2 = (dx) * (1 - dy)
        # w3 = (dx) * (dy)
        # w4 = (1 - dx) * (dy)
        
        # # self.u_weight[i0,     j0]       += w1
        # # self.u_weight[i0 + 1, j0]       += w2
        # # self.u_weight[i0 + 1, j0 + 1]   += w3
        # # self.u_weight[i0,     j0 + 1]   += w4
        
        # np.add.at(self.u_weight, (i0, j0), w1)
        # np.add.at(self.u_weight, (i0 + 1, j0), w2)
        # np.add.at(self.u_weight, (i0 + 1,   j0 + 1), w3)
        # np.add.at(self.u_weight, (i0,     j0 + 1), w4)  
        
        # # self.u_sum[i0,      j0]      += (self.velocity[:, 0] * w1)
        # # self.u_sum[i0 + 1,  j0]      += (self.velocity[:, 0] * w2)
        # # self.u_sum[i0 + 1,  j0 + 1]  += (self.velocity[:, 0] * w3)
        # # self.u_sum[i0,      j0 + 1]  += (self.velocity[:, 0] * w4)
        
        # np.add.at(self.u_sum, (i0, j0), self.velocity[:, 0] * w1)
        # np.add.at(self.u_sum, (i0 + 1, j0), self.velocity[:, 0] * w2)
        # np.add.at(self.u_sum, (i0 + 1, j0 + 1), self.velocity[:, 0] * w3)
        # np.add.at(self.u_sum, (i0, j0 + 1), self.velocity[:, 0] * w4)
        
        # mask = self.u_weight > 0
        # self.u[mask] = self.u_sum[mask] / self.u_weight[mask]
        # # self.u[i0,      j0]     = self.u_sum[i0, j0] / self.u_weight[i0, j0]
        # # self.u[i0 + 1,  j0]     = self.u_sum[i0 + 1, j0] / self.u_weight[i0 + 1, j0]
        # # self.u[i0 + 1,  j0 + 1] = self.u_sum[i0 + 1, j0 + 1] / self.u_weight[i0 + 1, j0 + 1]
        # # self.u[i0,      j0 + 1] = self.u_sum[i0, j0 + 1] / self.u_weight[i0, j0 + 1]
        
        # # V
        # gx = np.clip(self.positions[:, 0] * self.grid_size - 0.5, 0 , self.grid_size - 1 - 1e-6)
        # gy = np.clip(self.positions[:, 1] * self.grid_size, 0 , self.grid_size - 1 - 1e-6)

        # i0 = np.floor(gx).astype(np.int32)
        # j0 = np.floor(gy).astype(np.int32)
                
        # dx = gx - i0
        # dy = gy - j0
        
        # w1 = (1 - dx) * (1 - dy)
        # w2 = (dx) * (1 - dy)
        # w3 = (dx) * (dy)
        # w4 = (1 - dx) * (dy)
        
        # # self.v_weight[i0,     j0]           += w1
        # # self.v_weight[i0 + 1, j0]           += w2
        # # self.v_weight[i0 + 1, j0 + 1]       += w3
        # # self.v_weight[i0,     j0 + 1]       += w4
        
        # np.add.at(self.v_weight, (i0,     j0), w1)
        # np.add.at(self.v_weight, (i0 + 1, j0), w2)
        # np.add.at(self.v_weight, (i0 + 1, j0 + 1), w3)
        # np.add.at(self.v_weight, (i0,     j0 + 1), w4)
        
        # # self.v_sum[i0,      j0]     += (self.velocity[:, 1] * w1) 
        # # self.v_sum[i0 + 1,  j0]     += (self.velocity[:, 1] * w2) 
        # # self.v_sum[i0 + 1,  j0 + 1] += (self.velocity[:, 1] * w3) 
        # # self.v_sum[i0,      j0 + 1] += (self.velocity[:, 1] * w4) 
        
        # np.add.at(self.v_sum, (i0,      j0), self.velocity[:, 1] * w1) 
        # np.add.at(self.v_sum, (i0 + 1,  j0), self.velocity[:, 1] * w2) 
        # np.add.at(self.v_sum, (i0 + 1,  j0 + 1), self.velocity[:, 1] * w3) 
        # np.add.at(self.v_sum, (i0,      j0 + 1), self.velocity[:, 1] * w4) 
        
        # mask = self.v_weight > 0
        # self.v[mask] = self.v_sum[mask] / self.v_weight[mask]
        # # self.v[i0,      j0]     = self.v_sum[i0, j0] / self.v_weight[i0, j0]
        # # self.v[i0 + 1,  j0]     = self.v_sum[i0 + 1, j0] / self.v_weight[i0 + 1, j0]
        # # self.v[i0 + 1,  j0 + 1] = self.v_sum[i0 + 1, j0 + 1] / self.v_weight[i0 + 1, j0 + 1]
        # # self.v[i0,      j0 + 1] = self.v_sum[i0, j0 + 1] / self.v_weight[i0, j0 + 1]
        p = self.positions
        for a in (self.u_sum, self.u_weight, self.v_sum, self.v_weight, self.density):
            a.fill(0)
        
        self._scatter(self.u_sum, self.u_weight, p[:, 0], p[:, 1] - 0.5, self.velocity[:, 0])
        self._scatter(self.v_sum, self.v_weight, p[:, 0] - 0.5, p[:, 1], self.velocity[:, 1])
        self._scatter(None, self.density, p[:, 0] - 0.5, p[:, 1] - 0.5) # density at cell centers
        
        
        
        self.u.fill(0) ; self.v.fill(0)
        m = self.u_weight > 0 ; self.u[m] = self.u_sum[m] / self.u_weight[m]
        m = self.v_weight > 0 ; self.v[m] = self.v_sum[m] / self.v_weight[m]
        
        self.u *= self.u_open # no flow through solid faces
        self.v *= self.v_open
        self.u_old = self.u.copy() # FLIP needs to remember the previous grid velocity to compute the change in velocity
        self.v_old = self.v.copy()
               
    # pressure solve (Gauss-Seidel iteration. red-black ordering)         [10-minute-physics metho]
    def Divergence(self):        
        # for _ in range(self.itterations):
        #     i,j = np.where(self.cell_type == WATER)
        #     s = (self.s[i + 1, j] - self.s[i, j]) + (self.s[i, j + 1] - self.s[i, j]) 
        #     d = self.O*(self.u[i + 1, j] - self.u[i, j]) + (self.v[i, j + 1] - self.v[i, j]) - self.k * (self.density[i, j] - self.density_o)
        #     self.u[i,j] = (self.u[i,j] + d*self.s[i-1,j]/s)
        #     self.u[i+1,j] = (self.u[i,j] + d*self.s[i+1,j]/s)
        #     self.u[i,j] = (self.u[i,j] + d*self.s[i,j-1]/s)
        #     self.u[i,j+1] = (self.u[i,j] + d*self.s[i,j+1]/s)
            
        #     self.dense_to_grid()
        self.Pressure.fill(0)
        wi , wj = np.where(self.cell_type == WATER)
        red = (wi + wj) % 2 == 0
        sets = [(wi[red], wj[red]), (wi[~red], wj[~red])]
        
        for _ in range(self.itterations):
            for i , j in sets:
                
                s = (self.sx0[i, j] + self.sx1[i, j] + self.sy0[i, j] + self.sy1[i, j])
                s = np.maximum(s, 1.0)  # Avoid division by zero
                d = (self.u[i + 1, j] - self.u[i, j]) + (self.v[i, j + 1] - self.v[i, j]) 
                
                compression = self.density[i , j] - self.rest_density
                d = d - self.k * np.maximum(compression, 0)  # Only apply pressure if density exceeds rest density
                p = -d /s * self.O
                self.Pressure[i, j] += p
                
                self.u[i, j] -= p * self.sx0[i, j]
                self.u[i + 1, j] += p * self.sx1[i, j]
                self.v[i, j] -= p * self.sy0[i, j]
                self.v[i, j + 1] += p * self.sy1[i, j]
        
        self.divergence.fill(0)
        i , j = wi , wj
        self.divergence[i, j] = (self.u[i + 1, j] - self.u[i, j]) + (self.v[i, j + 1] - self.v[i, j])
            
    def dense_to_grid(self):    
        self.density.fill(0)
        i = np.floor(self.positions[:, 0] * self.grid_size).astype(np.int32)
        j = np.floor(self.positions[:, 1] * self.grid_size).astype(np.int32)
        valid = (i >=0 ) & (i < self.grid_size) & (j >= 0) & (j < self.grid_size)
        np.add.at(self.density, (i[valid], j[valid]), 1)
        
    def draw(self):
        cell = width / self.grid_size
        for i in range(self.amount):
            x = int(self.positions[i, 0] * cell)
            y = int((self.grid_size - self.positions[i, 1]) * cell)
            draw_circle(x, y, 3, RED)   
            
    def draw_grid(self):
        # cell_width = int(width / self.grid_size)
        # cell_height = int(height / self.grid_size)
        # for y in range(self.grid_size):
        #     for x in range(self.grid_size):
        #         cell_x = int(x * (width)/ self.grid_size) 
        #         cell_y = int(y * height / self.grid_size)
        
        N = self.grid_size
        cw = width // N
        ch = height // N
        for y in range(N):
            for x in range(N):
                cx = int(x * cw)
                cy = int((N - y - 1) * ch)
            
                if self.cell_type[x, y] == SOLID:
                    draw_rectangle(cx, cy, cw, ch, DARKERGRAY)
                elif self.cell_type[x, y] == WATER:
                    draw_rectangle(cx, cy, cw, ch, fade(WATER_COLOR, (self.density[x, y] / self.rest_density)))
                
                draw_rectangle_lines(cx, cy, cw, ch, BLACK)
                    
    def collistion(self):
        xi = np.floor(self.positions[:, 0] * self.grid_size).astype(np.int32)
        yi = np.floor(self.positions[:, 1] * self.grid_size).astype(np.int32)

        h = (xi * 92837111) ^ (yi * 88928748)
        self.bucket = np.abs(h) % self.tablesize

        counts = np.bincount(
            self.bucket,
            minlength=self.tablesize
        )

        offset = np.zeros(
            self.tablesize + 1,
            dtype=np.int32
        )

        offset[1:] = np.cumsum(counts)

        qorder = np.argsort(self.bucket)

        # Process each bucket
        for bucket in range(self.tablesize):

            start = offset[bucket]
            end = offset[bucket + 1]

            particles = qorder[start:end]

            # Nothing or only one particle
            if len(particles) < 2:
                continue

            # All unique pairs inside this bucket
            i, j = np.triu_indices(len(particles), k=1)

            a = particles[i]
            b = particles[j]

            delta = self.positions[b] - self.positions[a]

            dist_sq = np.sum(delta * delta, axis=1)

            # Only overlapping pairs
            collision = dist_sq < self.min_distance ** 2

            a = a[collision]
            b = b[collision]
            delta = delta[collision]
            dist_sq = dist_sq[collision]

            # Avoid division by zero
            valid = dist_sq > 1e-8

            a = a[valid]
            b = b[valid]
            delta = delta[valid]
            dist_sq = dist_sq[valid]

            if len(a) == 0:
                continue

            dist = np.sqrt(dist_sq)

            normal = delta / dist[:, None]

            penetration = self.min_distance - dist

            correction = normal * (penetration[:, None] * 0.5)

            self.positions[a] -= correction
            self.positions[b] += correction
            self.update_cell_type()
        
        
if __name__ == "__main__":
    main = Main()
    main.loop() 