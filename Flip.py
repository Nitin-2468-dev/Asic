import raylib
from pyray import *
import numpy as np

class Main:
    def __init__(self):
        
        raylib.SetTargetFPS(60)
        self.dt = 1/60
        self.Flip_Raito = 0.95
        self.Vmax = 40
        self.width = 700
        self.height = 700
        self.s = simulation(100, self.width, self.height, self.dt)
        init_window(self.width, self.height, "Flip")
        
            
    def draw_grid(self):
        N = 16 
        cell = 40
        offset_x = (self.width - N * cell) // 2
        offset_y = (self.height - N * cell) // 2
        
        h = N / 2
        k = N / 2
        r = N / 2
        
        # (x - h)^2 + (y - k)^2 = r^2
        # c(h, k, r)
        
        
        
        for x  in  range(cell):
            for y in range(cell):
                inside = (x + 0.5 - h)**2 + (y + 0.5 - k)**2 < r**2
                color = DARKGRAY if inside else BLACK
                draw_rectangle(offset_x + x * cell, offset_y + y * cell, cell -1, cell -1, color)
    
    def update(self):
        #pass
        self.s.update(self.dt)
        
    def draw(self):
        while not window_should_close():
            self.update()
            begin_drawing()
            clear_background(BLACK)
            self.draw_grid()
            self.s.draw()
            end_drawing()
        close_window()
        
    
    
class simulation:
    def __init__(self, h, width, height , dt):
       self.h = h
       self.gravity = np.array([0, 30.00])
       self.dt = dt
       self.center = np.array([width // 2,height // 2]) 
       self.spread = 100
       self.inital_veclocity = np.array([0, 5])
       
       self.particles = np.array([patical(self.center + np.random.rand(2) * self.spread - self.spread /2 , self.inital_veclocity)])
       
    def update(self , dt):
        # Applying gracity to the particles
        for patical in self.particles:
            patical.vel += self.gravity * dt
            patical.pos += patical.vel * dt
            self.particles = np.array([patical for patical in self.particles if patical.pos[1] > 0])
            
        # collitions
        
    def patical_to_cell(self, cell_size , particles):        
        self.cell = np.floor(particles.pos[:,1] / cell_size).astype(int)
        self.offest = particles.pos - cell_size
        
        
    
    def Grid_to_partical(self, cell_size , grid):
        # grid = np.array([cell], [{q1},{q2},{q3},{q4}])])
        # cell <- patical.partical_to_cell(cell_size, patical.pos)
        # 
        # qp = w1 * grid[cell][0] + w2 * grid[cell][1] + w3 * grid[cell][2] + w4 * grid[cell][3] / (w1 + w2 + w3 + w4)
        qp = np.zeros((len(self.particles), 2))
        for patical in self.particles:
            qp_1 = np.array(
                (
                    ((1 - patical.offest[0] / cell_size) * (1 - patical.offest[1] / cell_size) * grid[patical.partical_to_cell(cell_size, patical.pos)][0] ) + # w1 * q1
                    ((patical.offest[0] / cell_size) * (1 - patical.offest[1] / cell_size) * grid[patical.partical_to_cell(cell_size, patical.pos)][1] ) + # w2 * q2
                    ((patical.offest[0] / cell_size) * (patical.offest[1] / cell_size) * grid[patical.partical_to_cell(cell_size, patical.pos)][2] ) + # w3 * q3 
                    ((1 - patical.offest[0] / cell_size) * (patical.offest[1] / cell_size) * grid[patical.partical_to_cell(cell_size, patical.pos)][3] ) # w4 * q4
                )
            )
            qp[patical] = qp_1
        print(qp)
        
    def draw(self):
        for patical in self.particles:
            patical.draw()
                
class patical:
    def __init__(self, postions, velocity):
        self.pos = np.array(postions, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        self.offest = np.array([0,0])
        
    def draw(self):
        draw_circle(int(self.pos[0]), int(self.pos[1]), 5, RED)

    
    
if __name__ == "__main__":
    main = Main()
    main.draw()