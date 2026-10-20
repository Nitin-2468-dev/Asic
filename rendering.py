import numpy as np
from pyray import (
    BLACK,
    WHITE,
    Color,
    Rectangle,
    Vector2,
    begin_drawing,
    clear_background,
    draw_circle,
    draw_circle_gradient,
    draw_circle_lines,
    draw_fps,
    draw_rectangle_pro,
    draw_ring,
    draw_text,
    end_drawing,
)
from sim_hardware_model import HardwareFlip
from simulation_fixed import FixedFlip

WIDTH = 800
HEIGHT = 800


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = float(np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


class PendantRenderer:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT):
        self.width = width
        self.height = height
        self.show_particles = False   # Hidden by default (press '1' to toggle)
        self.green_theme = True       # True = Mitxela Emerald Green, False = AquaCell Cyan
        self.angled_smd = True        # True = 45-deg SMD 0402 LEDs, False = Round Shader LEDs

    def grid_to_screen(self, gx: float, gy: float, grid_size: int = 16) -> tuple[float, float, float]:
        scale = min(self.width / (grid_size + 2.5), self.height / (grid_size + 2.5))
        off_x = 0.5 * (self.width - scale * grid_size)
        off_y = 0.5 * (self.height - scale * grid_size)
        sx = off_x + gx * scale
        sy = self.height - (off_y + gy * scale)
        return sx, sy, scale

    def screen_to_grid(self, sx: float, sy: float, grid_size: int = 16) -> np.ndarray:
        scale = min(self.width / (grid_size + 2.5), self.height / (grid_size + 2.5))
        off_x = 0.5 * (self.width - scale * grid_size)
        off_y = 0.5 * (self.height - scale * grid_size)
        gx = (sx - off_x) / scale
        gy = ((self.height - sy) - off_y) / scale
        return np.array([gx, gy], dtype=np.float32)

    def draw_pendant_matrix(self, sim: FixedFlip | HardwareFlip, stir_pos: np.ndarray | None = None):
        N = sim.grid_size
        cx_scr, cy_scr, scale = self.grid_to_screen(N / 2.0, N / 2.0, N)
        center_vec = Vector2(cx_scr, cy_scr)

        # 1. Pendant Gold Case & Bezel Ring
        bezel_inner = 8.45 * scale
        bezel_outer = 9.05 * scale
        loop_center = Vector2(cx_scr + 6.7 * scale, cy_scr + 6.7 * scale)
        draw_ring(loop_center, 0.85 * scale, 1.35 * scale, 0, 360, 48, Color(195, 150, 55, 255))
        draw_circle(int(cx_scr), int(cy_scr), bezel_inner + 2, Color(6, 8, 11, 255))
        draw_ring(center_vec, bezel_inner, bezel_outer, 0, 360, 96, Color(205, 160, 60, 255))
        draw_ring(center_vec, bezel_outer - 3, bezel_outer + 2, 0, 360, 96, Color(135, 100, 30, 255))

        # 2. Gravity Direction Indicator Dot on Gold Bezel
        gnorm = float(np.hypot(sim.gravity[0], sim.gravity[1]))
        if gnorm > 1e-4:
            gx_dir = float(sim.gravity[0]) / gnorm
            gy_dir = float(sim.gravity[1]) / gnorm
        else:
            gx_dir, gy_dir = 0.0, -1.0
        ind_x, ind_y, _ = self.grid_to_screen(8.0 + gx_dir * 8.75, 8.0 + gy_dir * 8.75, N)
        draw_circle(int(ind_x), int(ind_y), int(0.22 * scale), Color(255, 115, 25, 255))

        # 3. 216-LED Matrix Rendering (smoothstep brightness + optical bloom + hot core)
        rest_d = sim.rest_density_f
        led_r = int(0.34 * scale)
        bloom_r = int(0.85 * scale)

        for y in range(N):
            for x in range(N):
                if not sim.inside[x, y]:
                    continue

                lx, ly, _ = self.grid_to_screen(x + 0.5, y + 0.5, N)
                ix, iy = int(lx), int(ly)

                d_ratio = float(sim.density[x, y]) / rest_d
                b = _smoothstep(0.08, 1.0, d_ratio)
                over_b = _smoothstep(0.85, 1.45, d_ratio)

                if self.angled_smd:
                    rect = Rectangle(lx, ly, 0.52 * scale, 0.30 * scale)
                    origin = Vector2(0.26 * scale, 0.15 * scale)
                    draw_rectangle_pro(rect, origin, -45.0, Color(28, 32, 34, 255))
                else:
                    draw_circle(ix, iy, led_r, Color(22, 26, 30, 255))

                if b > 0.005:
                    if self.green_theme:
                        glow_col = Color(10, int(210 * b), int(45 * b), int(95 * b))
                        core_r = int(40 * b + 185 * over_b)
                        core_g = int(min(255, 70 + 185 * (b ** 0.7)))
                        core_b = int(35 * b + 165 * over_b)
                    else:
                        glow_col = Color(int(45 * b), int(160 * b), int(245 * b), int(95 * b))
                        core_r = int(min(255, 64 * b + 160 * over_b))
                        core_g = int(min(255, 192 * (b ** 0.75) + 63 * over_b))
                        core_b = int(min(255, 255 * (b ** 0.65)))

                    draw_circle_gradient(Vector2(ix, iy), bloom_r, glow_col, Color(0, 0, 0, 0))

                    core_color = Color(core_r, core_g, core_b, 255)
                    if self.angled_smd:
                        rect = Rectangle(lx, ly, 0.52 * scale, 0.30 * scale)
                        origin = Vector2(0.26 * scale, 0.15 * scale)
                        draw_rectangle_pro(rect, origin, -45.0, core_color)
                        if b > 0.35:
                            hot_col = Color(
                                min(255, core_r + 70),
                                255,
                                min(255, core_b + 70),
                                int(220 * b),
                            )
                            draw_circle(ix, iy, int(0.13 * scale), hot_col)
                    else:
                        draw_circle(ix, iy, led_r, core_color)

        # 4. Optional Particle Overlay (toggled with '1', hidden by default)
        if self.show_particles:
            p_rad = max(2, int(0.24 * scale))
            for i in range(sim.amount):
                px, py, _ = self.grid_to_screen(float(sim.positions[i, 0]), float(sim.positions[i, 1]), N)
                speed = float(np.hypot(sim.velocity[i, 0], sim.velocity[i, 1]))
                t_vel = min(speed / 25.0, 1.0)
                pr = int(153 * (1.0 - t_vel) + 255 * t_vel)
                pg = int(242 * (1.0 - t_vel) + 128 * t_vel)
                pb = int(255 * (1.0 - t_vel) + 51 * t_vel)
                draw_circle(int(px), int(py), p_rad, Color(pr, pg, pb, 230))

        # 5. Stir Cursor Ring
        if stir_pos is not None:
            sx, sy, _ = self.grid_to_screen(float(stir_pos[0]), float(stir_pos[1]), N)
            draw_circle_lines(int(sx), int(sy), 2.5 * scale, Color(255, 255, 255, 140))

    def render(
        self,
        sim: FixedFlip | HardwareFlip,
        auto_swirl: bool = False,
        stir_pos: np.ndarray | None = None,
    ):
        begin_drawing()
        clear_background(Color(8, 9, 12, 255))
        self.draw_pendant_matrix(sim, stir_pos=stir_pos)

        draw_fps(12, 10)
        if isinstance(sim, HardwareFlip):
            pwr = "SLEEP (CLK GATED)" if sim.clock_gated_sleep else "ACTIVE"
            hw_info = f"[H] Mode: HW-Q4.12  |  Hash Pairs: {sim.last_pair_checks} / 32640  |  ASIC: {pwr}"
        else:
            hw_info = "[H] Mode: Float32 Reference  |  Pairs: 32640 (O(N^2))"
        draw_text(hw_info, 120, 12, 15, Color(120, 230, 150, 255))

        status = (
            f"[Drag] Tilt  [S+Drag] Stir  [A] Swirl:{'ON' if auto_swirl else 'OFF'}  "
            f"[Space] Shake  [1] Particles:{'ON' if self.show_particles else 'OFF'}  "
            f"[C] Color  [L] LED Style  [R] Reset"
        )
        draw_text(status, 12, self.height - 24, 14, Color(150, 160, 175, 255))
        end_drawing()