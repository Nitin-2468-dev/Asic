import dis
from pyray import *
import numpy as np

width = 800
height = 800

AIR   = 0
WATER = 1
SOLID = 2

DARKERGRAY = Color(50, 50, 50, 255)
WTAER_COLOR = get_color(0x8BCCEFFF)

class Main:
    def __init__(self):
        self.title = "Flip Example"
        init_window(width, height, self.title)
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
        self.f.update_cell_type()
        self.f.apply_gravity()
        self.f.collistion()
        self.f.Particle_to_Grid()
        self.f.Divergence()
        self.f.Grid_to_Particle()
        self.f.positions += self.f.velocity * self.f.dt
        self.f.positions = np.clip(self.f.positions, 0.001, 0.999)
    
    def draw(self):
        self.f.draw_grid()
        self.f.draw()
    
    
class Flip:
    def __init__(self):
        self.grid_size = 16 # 16x16 grid
        self.grid_scale = 2
        self.amount = 256 # number of particles
        self.gravity = np.array([0, -9.81]) # gravity vector
        self.dt = 1/60 # time step (seconds)
        self.k = 1.0 # pressure / Stiffness coefficient
        self.O = 1.5 # Overrelaxtions (1 < O < 2)
        self.itterations = 10 # number of iterations for pressure solver
        self.density_o = 1.0 # divergence coefficient
        self.tablesize  =  self.grid_size * self.grid_size # size of the hash table for particle collision detection
        
        
        # Partical 
        self.positions= np.zeros((self.amount, 2), dtype=np.float32) # x , y 
        self.velocity= np.zeros((self.amount, 2), dtype=np.float32) #  u , v 
        self.radius = 0.01 # particle radius
        self.min_distance = self.radius * 2 # minimum distance between particles for collision detection
        
        x = np.linspace(0.2, 0.8, 16)
        y = np.linspace(0.2, 0.8, 16)

        xx, yy = np.meshgrid(x, y)

        self.positions[:, 0] = xx.ravel()
        self.positions[:, 1] = yy.ravel()
        
        # GRID 
        self.u = np.zeros((self.grid_size + 1, self.grid_size ), dtype=np.float32) # u (horizontal  velocity)
        self.v = np.zeros((self.grid_size , self.grid_size + 1), dtype=np.float32) # v (vertical  velocity)
        self.u_sum = np.zeros((self.grid_size + 1, self.grid_size), dtype=np.float32) #  accumulated (horizontal velocity)
        self.v_sum = np.zeros((self.grid_size , self.grid_size + 1), dtype=np.float32) #  accumulated (vertical velocity)
        self.u_weight = np.zeros((self.grid_size + 1, self.grid_size), dtype=np.float32) #  weight (horizontal velocity)
        self.v_weight = np.zeros((self.grid_size , self.grid_size +1 ), dtype=np.float32) #  weight (vertical velocity)
        
        self.Pressure = np.zeros((self.grid_size, self.grid_size), dtype=np.float32) # pressure
        self.density = np.zeros((self.grid_size, self.grid_size), dtype=np.float32) # density
        self.divergence = np.zeros((self.grid_size, self.grid_size), dtype=np.float32) # divergence
        self.s = np.zeros((self.grid_size, self.grid_size), dtype=np.int32) # solid cells (0: Air, 1: Water, 2: Solid)
        self.cell_type = np.zeros((self.grid_size, self.grid_size), dtype=np.int32) # cell type (0: Air, 1: Water, 2: Solid)
        
        self.cell_start = np.zeros(self.grid_size * self.grid_size + 1, dtype=np.int32) # cell start index
        
        
        x ,y = np.indices((self.grid_size, self.grid_size))
        cx = (self.grid_size-1)  / 2
        cy = (self.grid_size - 1) / 2 
        r = self.grid_size / 2
        mask = (x - cx) ** 2 + (y - cy) **  2 < r ** 2
        self.solid_mask = mask
        self.cell_type[~self.solid_mask] = SOLID
        # self.cell_start[1:] = np.cumsum(np.sum(self.cell_type == 0, axis=1))
        self.s[self.solid_mask] = 1 # 0 for differentiate solid cells from fluid cells
        self.cell_type[self.solid_mask & ~(y < self.grid_size / 2)] = WATER
        
        
        
    def apply_gravity(self):
        self.velocity += self.gravity * self.dt
        # Todo : ADD Collistions and push apart.  
    def Grid_to_Particle(self):        
        # u
        gx = np.clip(self.positions[:, 0] * self.grid_size,0, self.grid_size - 1 - 1e-6)
        gy = np.clip(self.positions[:, 1] * self.grid_size - 0.5,0, self.grid_size - 1 - 1e-6)
        
        i0 = np.floor(gx).astype(np.int32)
        j0 = np.floor(gy).astype(np.int32)
        
        dx = gx - i0
        dy = gy - j0
        
        w1 = (1 - dx) * (1 - dy)
        w2 = (dx) * (1 - dy)
        w3 = (dx) * (dy)
        w4 = (1 - dx) * (dy)
        
        u_vel = (self.u[i0, j0] * w1 + self.u[i0 + 1, j0] * w2 + self.u[i0 + 1, j0 + 1] * w3 + self.u[i0, j0 + 1] * w4)
        
        # v
        gx = np.clip(self.positions[:, 0] * self.grid_size - 0.5, 0, self.grid_size - 1 - 1e-6)
        gy = np.clip(self.positions[:, 1] * self.grid_size, 0 , self.grid_size - 1 - 1e-6)
        
        i0 = np.floor(gx).astype(np.int32)
        j0 = np.floor(gy).astype(np.int32)
        
        dx = gx - i0
        dy = gy - j0
        
        w1 = (1 - dx) * (1 - dy)
        w2 = (dx) * (1 - dy)
        w3 = (dx) * (dy)
        w4 = (1 - dx) * (dy)
        
        v_vel = (self.v[i0, j0] * w1 + self.v[i0 + 1, j0] * w2 + self.v[i0 + 1, j0 + 1] * w3 + self.v[i0, j0 + 1] * w4)
        
        self.velocity[:, 0] = u_vel
        self.velocity[:, 1] = v_vel
        
    def Particle_to_Grid(self):       
        # U
        gx = np.clip(self.positions[:, 0] * self.grid_size ,0, self.grid_size - 1 - 1e-6)
        gy = np.clip(self.positions[:, 1] * self.grid_size - 0.5, 0 , self.grid_size - 1 - 1e-6)
        
        i0 = np.floor(gx).astype(np.int32)
        j0 = np.floor(gy).astype(np.int32)
        
        dx = gx - i0
        dy = gy - j0
        
        # clear
        self.u_weight.fill(0)
        self.u_sum.fill(0)
        self.v_weight.fill(0)
        self.v_sum.fill(0)
        
        
        w1 = (1 - dx) * (1 - dy)
        w2 = (dx) * (1 - dy)
        w3 = (dx) * (dy)
        w4 = (1 - dx) * (dy)
        
        # self.u_weight[i0,     j0]       += w1
        # self.u_weight[i0 + 1, j0]       += w2
        # self.u_weight[i0 + 1, j0 + 1]   += w3
        # self.u_weight[i0,     j0 + 1]   += w4
        
        np.add.at(self.u_weight, (i0, j0), w1)
        np.add.at(self.u_weight, (i0 + 1, j0), w2)
        np.add.at(self.u_weight, (i0 + 1,   j0 + 1), w3)
        np.add.at(self.u_weight, (i0,     j0 + 1), w4)  
        
        # self.u_sum[i0,      j0]      += (self.velocity[:, 0] * w1)
        # self.u_sum[i0 + 1,  j0]      += (self.velocity[:, 0] * w2)
        # self.u_sum[i0 + 1,  j0 + 1]  += (self.velocity[:, 0] * w3)
        # self.u_sum[i0,      j0 + 1]  += (self.velocity[:, 0] * w4)
        
        np.add.at(self.u_sum, (i0, j0), self.velocity[:, 0] * w1)
        np.add.at(self.u_sum, (i0 + 1, j0), self.velocity[:, 0] * w2)
        np.add.at(self.u_sum, (i0 + 1, j0 + 1), self.velocity[:, 0] * w3)
        np.add.at(self.u_sum, (i0, j0 + 1), self.velocity[:, 0] * w4)
        
        mask = self.u_weight > 0
        self.u[mask] = self.u_sum[mask] / self.u_weight[mask]
        # self.u[i0,      j0]     = self.u_sum[i0, j0] / self.u_weight[i0, j0]
        # self.u[i0 + 1,  j0]     = self.u_sum[i0 + 1, j0] / self.u_weight[i0 + 1, j0]
        # self.u[i0 + 1,  j0 + 1] = self.u_sum[i0 + 1, j0 + 1] / self.u_weight[i0 + 1, j0 + 1]
        # self.u[i0,      j0 + 1] = self.u_sum[i0, j0 + 1] / self.u_weight[i0, j0 + 1]
        
        # V
        gx = np.clip(self.positions[:, 0] * self.grid_size - 0.5, 0 , self.grid_size - 1 - 1e-6)
        gy = np.clip(self.positions[:, 1] * self.grid_size, 0 , self.grid_size - 1 - 1e-6)

        i0 = np.floor(gx).astype(np.int32)
        j0 = np.floor(gy).astype(np.int32)
                
        dx = gx - i0
        dy = gy - j0
        
        w1 = (1 - dx) * (1 - dy)
        w2 = (dx) * (1 - dy)
        w3 = (dx) * (dy)
        w4 = (1 - dx) * (dy)
        
        # self.v_weight[i0,     j0]           += w1
        # self.v_weight[i0 + 1, j0]           += w2
        # self.v_weight[i0 + 1, j0 + 1]       += w3
        # self.v_weight[i0,     j0 + 1]       += w4
        
        np.add.at(self.v_weight, (i0,     j0), w1)
        np.add.at(self.v_weight, (i0 + 1, j0), w2)
        np.add.at(self.v_weight, (i0 + 1, j0 + 1), w3)
        np.add.at(self.v_weight, (i0,     j0 + 1), w4)
        
        # self.v_sum[i0,      j0]     += (self.velocity[:, 1] * w1) 
        # self.v_sum[i0 + 1,  j0]     += (self.velocity[:, 1] * w2) 
        # self.v_sum[i0 + 1,  j0 + 1] += (self.velocity[:, 1] * w3) 
        # self.v_sum[i0,      j0 + 1] += (self.velocity[:, 1] * w4) 
        
        np.add.at(self.v_sum, (i0,      j0), self.velocity[:, 1] * w1) 
        np.add.at(self.v_sum, (i0 + 1,  j0), self.velocity[:, 1] * w2) 
        np.add.at(self.v_sum, (i0 + 1,  j0 + 1), self.velocity[:, 1] * w3) 
        np.add.at(self.v_sum, (i0,      j0 + 1), self.velocity[:, 1] * w4) 
        
        mask = self.v_weight > 0
        self.v[mask] = self.v_sum[mask] / self.v_weight[mask]
        # self.v[i0,      j0]     = self.v_sum[i0, j0] / self.v_weight[i0, j0]
        # self.v[i0 + 1,  j0]     = self.v_sum[i0 + 1, j0] / self.v_weight[i0 + 1, j0]
        # self.v[i0 + 1,  j0 + 1] = self.v_sum[i0 + 1, j0 + 1] / self.v_weight[i0 + 1, j0 + 1]
        # self.v[i0,      j0 + 1] = self.v_sum[i0, j0 + 1] / self.v_weight[i0, j0 + 1]
        
    def Divergence(self):
        # Gauss-Seidel Iteration for Pressure Solve
        for _ in range(self.itterations):
            i,j = np.where(self.cell_type == WATER)
            s = (self.s[i + 1, j] - self.s[i, j]) + (self.s[i, j + 1] - self.s[i, j]) 
            d = self.O*(self.u[i + 1, j] - self.u[i, j]) + (self.v[i, j + 1] - self.v[i, j]) - self.k * (self.density[i, j] - self.density_o)
            self.u[i,j] = (self.u[i,j] + d*self.s[i-1,j]/s)
            self.u[i+1,j] = (self.u[i,j] + d*self.s[i+1,j]/s)
            self.u[i,j] = (self.u[i,j] + d*self.s[i,j-1]/s)
            self.u[i,j+1] = (self.u[i,j] + d*self.s[i,j+1]/s)
            
            self.dense_to_grid()
            
            
        
    def dense_to_grid(self):    
        self.density.fill(0)
        i = np.floor(self.positions[:, 0] * self.grid_size).astype(np.int32)
        j = np.floor(self.positions[:, 1] * self.grid_size).astype(np.int32)
        valid = (i >=0 ) & (i < self.grid_size) & (j >= 0) & (j < self.grid_size)
        np.add.at(self.density, (i[valid], j[valid]), 1)
        
    def update_cell_type(self):    
        self.cell_type.fill(AIR)
        self.cell_type[~self.solid_mask] = SOLID
        
        i = np.floor(self.positions[:, 0] * self.grid_size).astype(np.int32)
        j = np.floor(self.positions[:, 1] * self.grid_size).astype(np.int32)
        
        valid = (i >=0 ) & (i < self.grid_size) & (j >= 0) & (j < self.grid_size)
        self.cell_type[i[valid], j[valid]] = WATER
    
    def draw(self):
        for i in range(self.amount):
            x = int(self.positions[i, 0] * width)
            y = int((1 - self.positions[i, 1]) * height)
            draw_circle(x, y, 2, RED)   
            
    def draw_grid(self):
        cell_width = int(width / self.grid_size)
        cell_height = int(height / self.grid_size)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell_x = int(x * (width)/ self.grid_size) 
                cell_y = int(y * height / self.grid_size)
            
                if self.cell_type[x, y] == SOLID:
                    draw_rectangle(cell_x, cell_y, cell_width, cell_height, DARKERGRAY)
                elif self.cell_type[x, y] == WATER:
                    draw_rectangle(cell_x, cell_y, cell_width, cell_height, WTAER_COLOR)
                
                draw_rectangle_lines(cell_x, cell_y, cell_width, cell_height, BLACK)
                    
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
        
        
if __name__ == "__main__":
    main = Main()
    main.loop() 