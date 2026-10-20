import numpy as np

AIR = 0
WATER = 1
SOLID = 2


class ExternalSPIRAM:
    """Emulates a 64KB SPI SRAM/FRAM chip (e.g., 23LC512 / MB85RS256B) for the ASIC."""
    def __init__(self, size=65536):
        self.memory = np.zeros(size, dtype=np.uint8)

    def write_block(self, addr, data_bytes):
        data = np.frombuffer(data_bytes, dtype=np.uint8)
        end = addr + len(data)
        self.memory[addr:end] = data

    def read_block(self, addr, length):
        return self.memory[addr:addr + length].tobytes()


class FixedFlip:
    def __init__(self):
        # Matches AquaCell (common.glsl) & Mitxela 16x16 216-LED pendant parameters
        self.grid_size = 16
        self.amount = 16 ** 2          # 256 particles (PW=16, PH=16)
        self.gravity_mag = 60.0        # GMAG = 60.0 cells/s^2
        self.dt = 1.0 / 60.0           # DT = 1/60 s
        self.flip_ratio = 0.95         # FLIP_RATIO = 0.95
        self.wall_friction = 1.0       # Free-slip tangential wall (matches buffer_b.glsl)
        self.k = 3.0                   # DRIFT_K = 3.0 density drift correction
        self.vmax = 40.0               # VMAX = 40.0
        self.O = 1.6                   # OMEGA = 1.6 SOR over-relaxation
        self.itterations = 30          # SOLVE_ITERS = 30
        self.spacing = 0.65            # SPACING = 0.65
        self.rest_density_f = 1.0 / (self.spacing ** 2)
        self.rest_density = int(round(self.rest_density_f * 1024))
        self.min_distance = 0.6        # MIN_DIST = 2 * PRAD = 0.6
        self.push_relax = 0.6          # PUSH_RELAX = 0.6
        self.push_itters = 2

        # Fixed-point quantization scale (Q.12 = 4096.0, matching common.glsl)
        self.fix_scale = 4096.0

        # Allocate External SPI RAM (64KB)
        self.ram = ExternalSPIRAM(65536)

        self._build_container()
        self.reset()

    def _quantize(self, arr):
        """Rounds array values to Q.12 fixed-point precision (matching Q(x) in common.glsl)."""
        return np.floor(arr * self.fix_scale + 0.5).astype(np.float32) / self.fix_scale

    def _build_container(self):
        N = self.grid_size
        ix, iy = np.indices((N, N))
        c = N / 2.0
        # MASK_R2 = 70.0 -> exactly 216 active cells inside the 16x16 grid
        self.inside = ((ix + 0.5 - c) ** 2 + (iy + 0.5 - c) ** 2) < 70.0
        self.center = np.array([c, c], dtype=np.float32)
        self.s = np.where(self.inside, 1.0, 0.0).astype(np.float32)

        sp = np.pad(self.s, 1)
        self.sx0 = sp[0:N, 1:N + 1]
        self.sx1 = sp[2:N + 2, 1:N + 1]
        self.sy0 = sp[1:N + 1, 0:N]
        self.sy1 = sp[1:N + 1, 2:N + 2]
        self.non_solid_nbrs = np.maximum(self.sx0 + self.sx1 + self.sy0 + self.sy1, 1.0)

        self.u_open = sp[0:N + 1, 1:N + 1] * sp[1:N + 2, 1:N + 1]
        self.v_open = sp[1:N + 1, 0:N + 1] * sp[1:N + 1, 1:N + 2]

        # Inscribed collision circle matching WALL_R = 7.75 in common.glsl
        self.wall_r = 7.75

    def reset(self, seed=1):
        rng = np.random.default_rng(seed)
        self.positions = np.zeros((self.amount, 2), dtype=np.float32)
        self.velocity = np.zeros((self.amount, 2), dtype=np.float32)

        side = int(round(np.sqrt(self.amount)))
        xs = self.center[0] + (np.arange(side) - side / 2.0 + 0.5) * self.spacing
        ys = 2.5 + (np.arange(side) + 0.5) * self.spacing
        xx, yy = np.meshgrid(xs, ys)
        self.positions[:, 0] = xx.ravel()
        self.positions[:, 1] = yy.ravel()
        self.positions += (rng.random((self.amount, 2)) - 0.5) * self.spacing * 0.2
        self.positions = self._quantize(self.positions)

        N = self.grid_size
        self.u = np.zeros((N + 1, N), dtype=np.float32)
        self.u_sum = np.zeros((N + 1, N), dtype=np.float32)
        self.u_weight = np.zeros((N + 1, N), dtype=np.float32)
        self.u_old = np.zeros((N + 1, N), dtype=np.float32)
        self.u_valid = np.zeros((N + 1, N), dtype=np.float32)

        self.v = np.zeros((N, N + 1), dtype=np.float32)
        self.v_sum = np.zeros((N, N + 1), dtype=np.float32)
        self.v_weight = np.zeros((N, N + 1), dtype=np.float32)
        self.v_old = np.zeros((N, N + 1), dtype=np.float32)
        self.v_valid = np.zeros((N, N + 1), dtype=np.float32)

        self.Pressure = np.zeros((N, N), dtype=np.float32)
        self.density = np.zeros((N, N), dtype=np.float32)
        self.particle_count = np.zeros((N, N), dtype=np.int32)
        self.divergence = np.zeros((N, N), dtype=np.float32)
        self.cell_type = np.zeros((N, N), dtype=np.int32)

        self.gravity = np.array([0.0, -self.gravity_mag], dtype=np.float32)
        self.swirl_phase = 0.0

        # Populate initial grid density so frame 0 renders immediately
        self.Particle_to_Grid()
        self.update_cell_type()
        self._flush_to_ram()

    def _flush_to_ram(self):
        """Serializes current simulation state into the External SPI RAM emulator."""
        self.ram.write_block(0x0000, self.positions.tobytes())
        self.ram.write_block(0x0800, self.velocity.tobytes())
        self.ram.write_block(0x1000, self.u.tobytes())
        self.ram.write_block(0x1440, self.v.tobytes())
        self.ram.write_block(0x1880, self.density.tobytes())

    def _pull_from_ram(self):
        """Restores state from External SPI RAM emulator."""
        n_bytes = self.amount * 8
        self.positions = np.frombuffer(
            self.ram.read_block(0x0000, n_bytes), dtype=np.float32
        ).reshape(self.amount, 2).copy()
        self.velocity = np.frombuffer(
            self.ram.read_block(0x0800, n_bytes), dtype=np.float32
        ).reshape(self.amount, 2).copy()

    def apply_gravity(self, gx_8bit=0, gy_8bit=-60):
        g = np.array([gx_8bit, gy_8bit], dtype=np.float32) * (self.gravity_mag / 60.0)
        self.gravity = g
        self.velocity += g * self.dt
        self.velocity = np.clip(self.velocity, -self.vmax, self.vmax)

    def apply_stir(self, mouse_xy, mouse_vel):
        """Applies local Gaussian stir impulse around mouse_xy (matching buffer_a.glsl)."""
        d = self.positions - mouse_xy[None, :]
        d2 = (d * d).sum(axis=1)
        f = np.exp(-d2 / 8.0)  # sigma = 2 cells
        sp = float(np.hypot(mouse_vel[0], mouse_vel[1]))
        t = np.clip(sp / 4.0, 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        self.velocity += (mouse_vel[None, :] - self.velocity) * (f * 0.5 * smooth)[:, None]

    def apply_shake(self, frame_idx=1):
        """Adds random shake impulse to particles (matching Spacebar in buffer_a.glsl)."""
        rng = np.random.default_rng(frame_idx * 977 + 1)
        self.velocity += (rng.random((self.amount, 2), dtype=np.float32) - 0.5) * 30.0

    def push_apart(self):
        md = self.min_distance
        p = self.positions
        for _ in range(self.push_itters):
            d = p[:, None, :] - p[None, :, :]
            d2 = (d * d).sum(axis=-1)
            I, J = np.nonzero(np.triu((d2 < md * md) & (d2 > 1e-10), 1))
            if len(I) == 0:
                break
            dv = p[I] - p[J]
            dist = np.sqrt((dv * dv).sum(axis=1))
            corr = dv * (self.push_relax * 0.5 * (md - dist) / dist)[:, None]
            np.add.at(p, I, corr)
            np.add.at(p, J, -corr)
        self.positions = p.astype(np.float32)

    def collide_walls(self):
        r = self.positions - self.center
        dist = np.hypot(r[:, 0], r[:, 1])
        out = dist > self.wall_r
        if out.any():
            n = r[out] / dist[out][:, None]
            self.positions[out] = self.center + n * self.wall_r
            vn = (self.velocity[out] * n).sum(axis=1)
            vn = np.maximum(vn, 0.0)
            self.velocity[out] -= n * vn[:, None]
            if self.wall_friction < 1.0:
                vv = self.velocity[out]
                vnn = (vv * n).sum(axis=1)[:, None] * n
                self.velocity[out] = vnn + (vv - vnn) * self.wall_friction

    def update_cell_type(self):
        N = self.grid_size
        self.cell_type.fill(AIR)
        self.cell_type[~self.inside] = SOLID

        # Mark cells containing at least one particle as WATER (matching buffer_c.glsl cnt > 0)
        self.particle_count.fill(0)
        ci = np.clip(np.floor(self.positions[:, 0]).astype(np.int32), 0, N - 1)
        cj = np.clip(np.floor(self.positions[:, 1]).astype(np.int32), 0, N - 1)
        np.add.at(self.particle_count, (ci, cj), 1)

        water = ((self.particle_count > 0) | (self.density > 0.1 * self.rest_density_f)) & self.inside
        self.cell_type[water] = WATER

        ct = np.pad(self.cell_type, 1, constant_values=SOLID)
        self.u_valid = ((ct[0:N + 1, 1:N + 1] != AIR) | (ct[1:N + 2, 1:N + 1] != AIR)).astype(np.float32)
        self.v_valid = ((ct[1:N + 1, 0:N + 1] != AIR) | (ct[1:N + 1, 1:N + 2] != AIR)).astype(np.float32)

    def _corners(self, gx, gy, shape):
        gx = np.clip(gx, 0.0, shape[0] - 1.0 - 1e-4)
        gy = np.clip(gy, 0.0, shape[1] - 1.0 - 1e-4)
        i0 = np.floor(gx).astype(np.int32)
        j0 = np.floor(gy).astype(np.int32)
        dx = gx - i0
        dy = gy - j0
        idx = ((i0, j0), (i0 + 1, j0), (i0 + 1, j0 + 1), (i0, j0 + 1))
        w = ((1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), dx * dy, (1.0 - dx) * dy)
        return idx, w

    def _scatter(self, total, weight, gx, gy, values=None):
        idx, w = self._corners(gx, gy, weight.shape)
        for (i, j), wk in zip(idx, w):
            np.add.at(weight, (i, j), wk)
            if values is not None:
                np.add.at(total, (i, j), values * wk)

    def _gather(self, new, old, weight, gx, gy, valid):
        idx, w = self._corners(gx, gy, weight.shape)
        wsum = 0.0
        vnew = 0.0
        vdelta = 0.0
        for (i, j), wk in zip(idx, w):
            wv = wk * valid[i, j]
            wsum += wv
            vnew += new[i, j] * wv
            vdelta += (new[i, j] - old[i, j]) * wv
        ok = wsum > 1e-5
        safe = np.where(ok, wsum, 1.0)
        return vnew / safe, vdelta / safe, ok

    def Particle_to_Grid(self):
        p = self.positions
        for a in (self.u_sum, self.u_weight, self.v_sum, self.v_weight, self.density):
            a.fill(0.0)
        self._scatter(self.u_sum, self.u_weight, p[:, 0], p[:, 1] - 0.5, self.velocity[:, 0])
        self._scatter(self.v_sum, self.v_weight, p[:, 0] - 0.5, p[:, 1], self.velocity[:, 1])
        self._scatter(None, self.density, p[:, 0] - 0.5, p[:, 1] - 0.5)

        self.u.fill(0.0)
        self.v.fill(0.0)
        m = self.u_weight > 1e-6
        self.u[m] = self.u_sum[m] / self.u_weight[m]
        m = self.v_weight > 1e-6
        self.v[m] = self.v_sum[m] / self.v_weight[m]

        self.u *= self.u_open
        self.v *= self.v_open
        self.density[~self.inside] = 0.0

        self.u = self._quantize(self.u)
        self.v = self._quantize(self.v)
        self.u_old = self.u.copy()
        self.v_old = self.v.copy()

    def Divergence(self):
        wi, wj = np.where(self.cell_type == WATER)
        if len(wi) == 0:
            self.Pressure.fill(0.0)
            self.divergence.fill(0.0)
            return

        # Clear pressure on non-water cells while keeping warm-start pressure on WATER cells (buffer_d.glsl)
        self.Pressure[self.cell_type != WATER] = 0.0

        red = (wi + wj) % 2 == 0
        sets = [(wi[red], wj[red]), (wi[~red], wj[~red])]

        for _ in range(self.itterations):
            for i, j in sets:
                s = self.non_solid_nbrs[i, j]
                d = (self.u[i + 1, j] - self.u[i, j]) + (self.v[i, j + 1] - self.v[i, j])
                compression = np.maximum(self.density[i, j] / self.rest_density_f - 1.0, 0.0)
                d = d - self.k * compression
                p = -d / s * self.O
                self.Pressure[i, j] += p
                self.u[i, j] -= p * self.sx0[i, j]
                self.u[i + 1, j] += p * self.sx1[i, j]
                self.v[i, j] -= p * self.sy0[i, j]
                self.v[i, j + 1] += p * self.sy1[i, j]

        self.u = self._quantize(self.u)
        self.v = self._quantize(self.v)
        self.Pressure = self._quantize(self.Pressure)

        self.divergence.fill(0.0)
        self.divergence[wi, wj] = (self.u[wi + 1, wj] - self.u[wi, wj]) + (self.v[wi, wj + 1] - self.v[wi, wj])

    def Grid_to_Particle(self):
        p = self.positions
        r = self.flip_ratio
        for comp, (new, old, valid, gx, gy) in enumerate((
            (self.u, self.u_old, self.u_valid, p[:, 0], p[:, 1] - 0.5),
            (self.v, self.v_old, self.v_valid, p[:, 0] - 0.5, p[:, 1])
        )):
            vnew, vdelta, ok = self._gather(new, old, valid, gx, gy, valid)
            v = self.velocity[:, comp]
            v_pic = np.where(ok, vnew, v)
            v_flip = np.where(ok, v + vdelta, v)
            self.velocity[:, comp] = (1.0 - r) * v_pic + r * v_flip

        self.velocity = np.clip(self.velocity, -self.vmax, self.vmax)

    def update(self, gx_8bit=0, gy_8bit=-60, stir_pos=None, stir_vel=None, shake=False, frame_idx=1):
        self._pull_from_ram()

        # 1. Particle forces & advection
        self.apply_gravity(gx_8bit, gy_8bit)
        if stir_pos is not None and stir_vel is not None:
            self.apply_stir(stir_pos, stir_vel)
        if shake:
            self.apply_shake(frame_idx)

        self.positions += self.velocity * self.dt
        self.positions = self._quantize(self.positions)

        # 2. Particle push-apart & circular wall collision (Buffer B)
        self.push_apart()
        self.collide_walls()
        self.positions = self._quantize(self.positions)
        self.velocity = self._quantize(self.velocity)

        # 3. Particle-to-Grid & cell classification (Buffer C)
        self.Particle_to_Grid()
        self.update_cell_type()

        # 4. Pressure solve & Grid-to-Particle transfer (Buffer D -> Buffer A)
        self.Divergence()
        self.Grid_to_Particle()

        self._flush_to_ram()