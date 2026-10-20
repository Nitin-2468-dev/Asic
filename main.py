import numpy as np
from pyray import (
     Color,
    ConfigFlags,
    KeyboardKey,
    MouseButton,
    close_window,
    draw_text,
    get_frame_time,
    get_mouse_x,
    get_mouse_y,
    get_window_position,
    init_window,
    is_key_down,
    is_key_pressed,
    is_mouse_button_down,
    set_config_flags,
    set_target_fps,
    set_window_position,
    set_window_state,
    window_should_close,
)
from rendering import HEIGHT, WIDTH, PendantRenderer
from sim_hardware_model import HardwareFlip
from simulation_fixed import FixedFlip


class MainApp:
    def __init__(self):
        self.title = "Fluid Pendant ASIC Model (Q4.12 Integer + Spatial Hash + LUTs)"
        init_window(WIDTH, HEIGHT, self.title)
        set_window_state(ConfigFlags.FLAG_WINDOW_UNDECORATED)
        set_config_flags(ConfigFlags.FLAG_VSYNC_HINT)
        set_target_fps(60)

        self.hw_sim = HardwareFlip()
        self.fp_sim = FixedFlip()
        self.use_hw_model = True
        self.sim = self.hw_sim

        self.renderer = PendantRenderer(WIDTH, HEIGHT)
        self.acc = 0.0
        self.frame_idx = 0

        self.gravity = np.array([0.0, -self.sim.gravity_mag], dtype=np.float32)
        self.auto_swirl = False
        self.swirl_phase = 0.0
        self.prev_mouse_grid = np.array([8.0, 8.0], dtype=np.float32)
        self.was_mouse_down = False

        self.last_win_mx = get_mouse_x()
        self.last_win_my = get_mouse_y()

    def loop(self):
        while not window_should_close():
            stir_pos = self.update()
            self.renderer.render(self.sim, auto_swirl=self.auto_swirl, stir_pos=stir_pos)
        close_window()

    def update(self):
        if is_key_pressed(KeyboardKey.KEY_R):
            self.sim.reset()
        if is_key_pressed(KeyboardKey.KEY_H):
            self.use_hw_model = not self.use_hw_model
            self.sim = self.hw_sim if self.use_hw_model else self.fp_sim
        if is_key_pressed(KeyboardKey.KEY_ONE):
            self.renderer.show_particles = not self.renderer.show_particles
        if is_key_pressed(KeyboardKey.KEY_A):
            self.auto_swirl = not self.auto_swirl
        if is_key_pressed(KeyboardKey.KEY_C):
            self.renderer.green_theme = not self.renderer.green_theme
        if is_key_pressed(KeyboardKey.KEY_L):
            self.renderer.angled_smd = not self.renderer.angled_smd

        mx = get_mouse_x()
        my = get_mouse_y()
        down = is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT)
        stir_mode = is_key_down(KeyboardKey.KEY_S)
        win_drag_mode = is_key_down(KeyboardKey.KEY_W)

        if win_drag_mode and down:
            dx = mx - self.last_win_mx
            dy = my - self.last_win_my
            if dx or dy:
                p = get_window_position()
                set_window_position(int(p.x + dx), int(p.y + dy))
        else:
            self.last_win_mx = mx
            self.last_win_my = my

        mouse_grid = self.renderer.screen_to_grid(mx, my, self.sim.grid_size)
        if down and self.was_mouse_down:
            mouse_vel = (mouse_grid - self.prev_mouse_grid) / self.sim.dt
            ml = float(np.hypot(mouse_vel[0], mouse_vel[1]))
            if ml > 60.0:
                mouse_vel *= 60.0 / ml
        else:
            mouse_vel = np.zeros(2, dtype=np.float32)

        self.prev_mouse_grid = mouse_grid
        self.was_mouse_down = down

        tgt = self.gravity.copy()
        if self.auto_swirl:
            self.swirl_phase += 0.025
            tgt = self.sim.gravity_mag * np.array(
                [np.sin(self.swirl_phase), -np.cos(self.swirl_phase)], dtype=np.float32
            )

        dm = mouse_grid - self.sim.center
        dm_len = float(np.hypot(dm[0], dm[1]))
        if down and not stir_mode and not win_drag_mode and dm_len > 0.5:
            tgt = self.sim.gravity_mag * (dm / dm_len)

        if is_key_down(KeyboardKey.KEY_G):
            tgt = np.array([0.0, -self.sim.gravity_mag], dtype=np.float32)

        self.gravity += (tgt - self.gravity) * 0.25
        gx_8bit = int(np.clip(round(self.gravity[0]), -127, 127))
        gy_8bit = int(np.clip(round(self.gravity[1]), -127, 127))

        stir_pos = mouse_grid if (down and stir_mode) else None
        shake = is_key_down(KeyboardKey.KEY_SPACE)

        self.acc = min(self.acc + get_frame_time(), 3.0 * self.sim.dt)
        while self.acc >= self.sim.dt:
            self.frame_idx += 1
            self.sim.update(
                gx_8bit=gx_8bit,
                gy_8bit=gy_8bit,
                stir_pos=stir_pos,
                stir_vel=mouse_vel if stir_pos is not None else None,
                shake=shake,
                frame_idx=self.frame_idx,
            )
            self.acc -= self.sim.dt

        return stir_pos


if __name__ == "__main__":
    app = MainApp()
    app.loop()