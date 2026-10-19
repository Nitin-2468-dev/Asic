from pyray import (
    BLACK,
    RED,
    Color,
    begin_drawing,
    clear_background,
    draw_circle,
    draw_fps,
    draw_rectangle,
    draw_rectangle_lines,
    end_drawing,
    fade,
    get_color,
)
from simulation import SOLID, WATER

WIDTH = 800
HEIGHT = 800

DARKERGRAY = Color(50, 50, 50, 255)


class Renderer:
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        self.water_color = get_color(0x8BCCEFFF)

    def draw_particles(self, sim):
        """Draws all fluid particles from read-only simulation state."""
        cell = self.width / sim.grid_size
        for i in range(sim.amount):
            x = int(sim.positions[i, 0] * cell)
            y = int((sim.grid_size - sim.positions[i, 1]) * cell)
            draw_circle(x, y, 3, RED)

    def draw_grid(self, sim):
        """Draws solid and water cells along with grid lines."""
        N = sim.grid_size
        cw = self.width // N
        ch = self.height // N
        for y in range(N):
            for x in range(N):
                cx = int(x * cw)
                cy = int((N - y - 1) * ch)

                if sim.cell_type[x, y] == SOLID:
                    draw_rectangle(cx, cy, cw, ch, DARKERGRAY)
                elif sim.cell_type[x, y] == WATER:
                    alpha = sim.density[x, y] / sim.rest_density
                    draw_rectangle(cx, cy, cw, ch, fade(self.water_color, alpha))
        for y in range(N):
            for x in range(N):
                draw_rectangle_lines(x * cw, (N - y - 1) * cw, cw, cw, BLACK)

    def render(self, sim):
        """Executes a complete rendering pass for the current frame."""
        begin_drawing()
        clear_background(BLACK)
        self.draw_grid(sim)
        self.draw_particles(sim)
        draw_fps(10, 10)
        end_drawing()