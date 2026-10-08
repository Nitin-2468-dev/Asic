from pyray import *
import numpy as np

width = 800
height = 800

AIR   = 0
WATER = 1
SOLID = 2

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
        pass
    
    def draw(self):
        self.f.draw_grid()
    
    
class Flip:
    def __init__(self):
        self.grid_size = 16 # 16x16 grid
        self.grid_scale = 2
        self.amount = 256 # number of particles
        self.gravity = np.array([0, -9.81]) # gravity vector
        self.dt = 1/60 # time step (seconds)
        self.k = 1.0 # pressure / Stiffness coefficient
        self.O = 1.5 # Overrelaxtions (1 < O < 2)
        
        # Partical 
        self.positions= np.zeros((self.amount, 2), dtype=np.float32) # x , y 
        self.velocity= np.zeros((self.amount, 2), dtype=np.float32) #  u , v 
        
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
        # self.s = np.zeros((self.grid_size, self.grid_size), dtype=np.int32) # solid cells (0: Air, 1: Water, 2: Solid)
        self.cell_type = np.zeros((self.grid_size, self.grid_size), dtype=np.int32) # cell type (0: Air, 1: Water, 2: Solid)
        
        
        
        x ,y = np.indices((self.grid_size, self.grid_size))
        cx = self.grid_size / 2
        cy = self.grid_size / 2
        r = self.grid_size / 4
        mask = (x - cx) ** 2 + (y - cy) **  2 < r ** 2
        self.cell_type[~mask] = SOLID
        
        
        
        
    def apply_gravity(self):
        self.velocity += self.gravity * self.dt
        # Todo : ADD Collistions and push apart.  
    def Grid_to_Particle(self):        
        # u
        gx = self.positions[:, 0] * self.grid_size
        gy = self.positions[:, 1] * self.grid_size - 0.5
        
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
        gx = self.positions[:, 0] * self.grid_size - 0.5
        gy = self.positions[:, 1] * self.grid_size 
        
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
        gx = self.positions[:, 0] * self.grid_size
        gy = max(0, self.positions[:, 1] * self.grid_size - 0.5)
        
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
        gx = max(0, self.positions[:, 0] * self.grid_size - 0.5)
        gy = self.positions[:, 1] * self.grid_size 

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
        i,j = np.where(self.cell_type == AIR)
        
        self.divergence[i ,j] = (self.u[i + 1, j] - self.u[i, j]) + (self.v[i, j + 1] - self.v[i, j]) 
        
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
                cell_x = int(x * width / self.grid_size)
                cell_y = int(y * height / self.grid_size)
            
                if self.cell_type[x, y] == SOLID:
                    draw_rectangle(cell_x, cell_y, cell_width, cell_height, BLACK)
                elif self.cell_type[x, y] == WATER:
                    draw_rectangle(cell_x, cell_y, cell_width, cell_height, BLUE)
                else:
                    draw_rectangle_lines(cell_x, cell_y, cell_width, cell_height, GRAY)
                


if __name__ == "__main__":
    main = Main()
    main.loop() 