from pyray import (
    ConfigFlags,
    KeyboardKey,
    MouseButton,
    close_window,
    get_frame_time,
    get_mouse_x,
    get_mouse_y,
    get_window_position,
    init_window,
    is_key_pressed,
    is_mouse_button_down,
    set_config_flags,
    set_target_fps,
    set_window_position,
    set_window_state,
    window_should_close,
)
from rendering import HEIGHT, WIDTH, Renderer
from simulation import Flip


class Main:
    def __init__(self):
        self.title = "Flip Example"
        init_window(WIDTH, HEIGHT, self.title)
        set_window_state(ConfigFlags.FLAG_WINDOW_UNDECORATED)
        set_config_flags(ConfigFlags.FLAG_VSYNC_HINT)
        set_target_fps(60)

        self.f = Flip()
        self.renderer = Renderer(WIDTH, HEIGHT)
        self.acc = 0.0

        self.last_x = get_mouse_x()
        self.last_y = get_mouse_y()

    def loop(self):
        while not window_should_close():
            self.update()
            self.draw()

        close_window()

    def grab(self):
        mx = get_mouse_x()
        my = get_mouse_y()

        dx = dy = 0
        if is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT):
            dx = mx - self.last_x
            dy = my - self.last_y
            if dx or dy:
                p = get_window_position()
                set_window_position(int(p.x + dx), int(p.y + dy))
        else:
            self.last_x = mx
            self.last_y = my

        ft = get_frame_time()
        self.f.apply_window_motion(dx, dy, ft)

    def update(self):
        if is_key_pressed(KeyboardKey.KEY_R):
            self.f.reset()
        self.grab()
        self.acc = min(self.acc + get_frame_time(), 3 * self.f.dt)
        while self.acc >= self.f.dt:
            self.f.update()
            self.acc -= self.f.dt

    def draw(self):
        self.renderer.render(self.f)


if __name__ == "__main__":
    main = Main()
    main.loop()