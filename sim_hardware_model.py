import math
import numpy as np

from simulation_fixed import AIR, SOLID, WATER, ExternalSPIRAM

# =====================================================================
# Hardware Fixed-Point Constants (Q4.12 format: 1.0 cell = 4096 LSBs)
# =====================================================================
SHIFT = 12
SCALE = 1 << SHIFT                    # 4096
HALF_CELL_Q12 = 1 << (SHIFT - 1)      # 2048 (0.5 cells)

NX = 16
NY = 16
NP = 256

CEN_Q12 = 8 * SCALE                   # 32768 (8.0)
WALL_R_Q12 = int(round(7.75 * SCALE)) # 31744 (7.75)
WALL_R2_Q8 = int(round(7.75 * 7.75 * 256))  # 15376 (r^2 in Q8.8)

MIN_DIST_Q12 = int(round(0.6 * SCALE))      # 2458 (0.6)
MIN_DIST_SQ_Q12 = (MIN_DIST_Q12 * MIN_DIST_Q12) >> SHIFT  # 1474 (0.36 * 4096)

VMAX_Q12 = 40 * SCALE                 # 163840 (40.0 cells/s)
DT_Q12 = 68                           # 68 / 4096 = 1 / 60.235 s (~1/60 s)

FLIP_R_Q12 = int(round(0.95 * SCALE)) # 3891
PIC_R_Q12 = SCALE - FLIP_R_Q12        # 205 (3891 + 205 = 4096)

SPACING_Q12 = int(round(0.65 * SCALE))
REST_DENSITY_F = 1.0 / (0.65 * 0.65)  # 2.3668639
REST_DENSITY_Q12 = int(round(REST_DENSITY_F * SCALE))     # 9695
INV_REST_DENSITY_Q12 = int(round(SCALE / REST_DENSITY_F)) # 1731
WATER_DENS_THS_Q12 = int(round(0.1 * REST_DENSITY_F * SCALE))  # 969
DRIFT_K_INT = 3                       # Density drift stiffness = 3

# =====================================================================
# Precomputed Hardware Look-Up Tables (Synthesize as small ROMs in HDL)
# =====================================================================

# 1. Pressure solver 1/s * OMEGA table for s in {0, 1, 2, 3, 4} (OMEGA = 1.6)
OMEGA_INV_S_LUT = np.array([
    0,
    int(round(1.6 * SCALE / 1)),  # 6554
    int(round(1.6 * SCALE / 2)),  # 3277
    int(round(1.6 * SCALE / 3)),  # 2185
    int(round(1.6 * SCALE / 4)),  # 1638
], dtype=np.int32)

# 2. Push-apart repulsion LUT (64 entries, indexed by d2_q12 >> 5)
# Computes round(4096 * PUSH_RELAX * 0.5 * (0.6 - d) / d) with PUSH_RELAX = 0.6
PUSH_LUT = np.zeros(64, dtype=np.int32)
for _idx in range(64):
    _d2 = max((_idx + 0.5) * 32.0 / SCALE, 1e-4)
    _d = math.sqrt(_d2)
    if _d < 0.6:
        PUSH_LUT[_idx] = min(int(round(SCALE * 0.6 * 0.5 * (0.6 - _d) / _d)), 8 * SCALE)

# 3. Circular wall inverse-radius LUT (64 entries, indexed by (r2_q8 - WALL_R2_Q8) >> 6)
# Computes round(4096 / sqrt(r^2)) for particles outside r = 7.75
WALL_INV_R_LUT = np.zeros(64, dtype=np.int32)
for _idx in range(64):
    _r2 = (WALL_R2_Q8 + (_idx + 0.5) * 64.0) / 256.0
    WALL_INV_R_LUT[_idx] = int(round(SCALE / math.sqrt(_r2)))


class HardwareFlip:
    """
    100% Float-Free Integer Q4.12 Hardware Reference Model of the FLIP Fluid ASIC.
    All physics updates use integer arithmetic, bit shifts, LUT ROMs, and a Spatial Hash Grid.
    """
    def __init__(self):
        self.grid_size = NX
        self.amount = NP
        self.gravity_mag = 60.0
        self.dt = 1.0 / 60.0
        self.spacing = 0.65
        self.rest_density_f = REST_DENSITY_F
        self.wall_r = 7.75
        self.itterations = 30
        self.center = np.array([8.0, 8.0], dtype=np.float32)

        # External 64KB SPI SRAM Emulator
        self.ram = ExternalSPIRAM(65536)

        # Hardware profiling & power management telemetry
        self.last_pair_checks = 0
        self.naive_pair_checks = (NP * (NP - 1)) // 2  # 32,640
        self.clock_gated_sleep = False
        self.sleep_Hold_frames = 0
        self._last_gx = 0
        self._last_gy = -60

        self._build_container_rom()
        self.reset()

    def _build_container_rom(self):
        """Builds the 256-bit container mask ROM using exact integer math: (2x-15)^2 + (2y-15)^2 < 280."""
        ix, iy = np.indices((NX, NY), dtype=np.int32)
        dist_sq_x4 = (2 * ix - 15) ** 2 + (2 * iy - 15) ** 2
        self.inside = dist_sq_x4 < 280  # 216 active cells

        s = np.where(self.inside, 1, 0).astype(np.int32)
        sp = np.pad(s, 1)
        self.sx0 = sp[0:NX, 1:NY + 1]
        self.sx1 = sp[2:NX + 2, 1:NY + 1]
        self.sy0 = sp[1:NX + 1, 0:NY]
        self.sy1 = sp[1:NX + 1, 2:NY + 2]
        self.s_count = self.sx0 + self.sx1 + self.sy0 + self.sy1

        self.u_open = (sp[0:NX + 1, 1:NY + 1] & sp[1:NX + 2, 1:NY + 1]).astype(np.int32)
        self.v_open = (sp[1:NX + 1, 0:NY + 1] & sp[1:NX + 1, 1:NY + 2]).astype(np.int32)

    def reset(self, seed=1):
        rng = np.random.default_rng(seed)
        side = 16
        xs = 8.0 + (np.arange(side) - side / 2.0 + 0.5) * self.spacing
        ys = 2.5 + (np.arange(side) + 0.5) * self.spacing
        xx, yy = np.meshgrid(xs, ys)
        pos_f = np.zeros((NP, 2), dtype=np.float32)
        pos_f[:, 0] = xx.ravel()
        pos_f[:, 1] = yy.ravel()
        pos_f += (rng.random((NP, 2)) - 0.5) * self.spacing * 0.2

        # Primary Q4.12 Integer Hardware State
        self.pos_q12 = np.floor(pos_f * SCALE + 0.5).astype(np.int32)
        self.vel_q12 = np.zeros((NP, 2), dtype=np.int32)

        self.u_q12 = np.zeros((NX + 1, NY), dtype=np.int32)
        self.u_old_q12 = np.zeros((NX + 1, NY), dtype=np.int32)
        self.u_sum_q12 = np.zeros((NX + 1, NY), dtype=np.int32)
        self.u_weight_q12 = np.zeros((NX + 1, NY), dtype=np.int32)
        self.u_valid = np.zeros((NX + 1, NY), dtype=np.int32)

        self.v_q12 = np.zeros((NX, NY + 1), dtype=np.int32)
        self.v_old_q12 = np.zeros((NX, NY + 1), dtype=np.int32)
        self.v_sum_q12 = np.zeros((NX, NY + 1), dtype=np.int32)
        self.v_weight_q12 = np.zeros((NX, NY + 1), dtype=np.int32)
        self.v_valid = np.zeros((NX, NY + 1), dtype=np.int32)

        self.density_q12 = np.zeros((NX, NY), dtype=np.int32)
        self.pressure_q12 = np.zeros((NX, NY), dtype=np.int32)
        self.divergence_q12 = np.zeros((NX, NY), dtype=np.int32)
        self.particle_count = np.zeros((NX, NY), dtype=np.int32)
        self.cell_type = np.zeros((NX, NY), dtype=np.int32)

        # Spatial Hash Table in SPI RAM (256-entry head + 256-entry linked list)
        self.cell_head = np.full(NX * NY, -1, dtype=np.int16)
        self.particle_next = np.full(NP, -1, dtype=np.int16)

        self.gravity = np.array([0.0, -self.gravity_mag], dtype=np.float32)
        self.clock_gated_sleep = False
        self.sleep_Hold_frames = 0

        self.p2g_q12()
        self.update_cell_type_q12()
        self._sync_views_and_ram()

    def _sync_views_and_ram(self):
        """Packs 16-bit integer words into SPI RAM and updates read-only float views for the renderer."""
        self.ram.write_block(0x0000, self.pos_q12.astype(np.uint16).tobytes())
        self.ram.write_block(0x0400, (self.vel_q12 >> 2).astype(np.int16).tobytes())
        self.ram.write_block(0x0800, (self.u_q12 >> 2).astype(np.int16).tobytes())
        self.ram.write_block(0x0A20, (self.v_q12 >> 2).astype(np.int16).tobytes())
        self.ram.write_block(0x0C40, self.density_q12.astype(np.uint16).tobytes())
        self.ram.write_block(0x0E40, self.cell_head.tobytes())
        self.ram.write_block(0x1040, self.particle_next.tobytes())

        # Float views for compatibility with PendantRenderer and tests
        self.positions = self.pos_q12.astype(np.float32) / SCALE
        self.velocity = self.vel_q12.astype(np.float32) / SCALE
        self.density = self.density_q12.astype(np.float32) / SCALE
        self.u = self.u_q12.astype(np.float32) / SCALE
        self.v = self.v_q12.astype(np.float32) / SCALE
        self.Pressure = self.pressure_q12.astype(np.float32) / SCALE
        self.divergence = self.divergence_q12.astype(np.float32) / SCALE

    # -----------------------------------------------------------------
    # Stage 1: Integer Gravity & Advection
    # -----------------------------------------------------------------
    def apply_gravity_and_advect_q12(self, gx_8bit=0, gy_8bit=-60):
        self.gravity[0] = float(gx_8bit)
        self.gravity[1] = float(gy_8bit)

        # dv = (g_q12 * DT_Q12) >> 12 = (g_8bit * 4096 * 68) >> 12 = g_8bit * 68
        dv_x = int(gx_8bit) * DT_Q12
        dv_y = int(gy_8bit) * DT_Q12

        self.vel_q12[:, 0] = np.clip(self.vel_q12[:, 0] + dv_x, -VMAX_Q12, VMAX_Q12)
        self.vel_q12[:, 1] = np.clip(self.vel_q12[:, 1] + dv_y, -VMAX_Q12, VMAX_Q12)

        # dx = (vel_q12 * DT_Q12) >> 12
        self.pos_q12[:, 0] += (self.vel_q12[:, 0] * DT_Q12) >> SHIFT
        self.pos_q12[:, 1] += (self.vel_q12[:, 1] * DT_Q12) >> SHIFT

    # -----------------------------------------------------------------
    # Stage 2A: O(N) Spatial Hash Grid + LUT Push-Apart (Zero sqrt/div)
    # -----------------------------------------------------------------
    def push_apart_spatial_hash_q12(self):
        pos = self.pos_q12
        head = self.cell_head
        nxt = self.particle_next
        pair_checks = 0

        # Half-stencil of neighbor offsets so each unique cell pair is visited once
        nbr_offsets = ((1, 0), (-1, 1), (0, 1), (1, 1))

        for _ in range(2):
            # 1. Build Spatial Hash Linked List (256 cycles in hardware)
            head.fill(-1)
            for p_id in range(NP):
                cx = int(pos[p_id, 0]) >> SHIFT
                cy = int(pos[p_id, 1]) >> SHIFT
                cx = 0 if cx < 0 else (15 if cx > 15 else cx)
                cy = 0 if cy < 0 else (15 if cy > 15 else cy)
                c_idx = (cy << 4) | cx
                nxt[p_id] = head[c_idx]
                head[c_idx] = p_id

            # 2. Traverse 3x3 neighborhood via half-stencil
            for cy in range(NY):
                for cx in range(NX):
                    c_idx = (cy << 4) | cx
                    p_i = int(head[c_idx])
                    while p_i != -1:
                        px_i = int(pos[p_i, 0])
                        py_i = int(pos[p_i, 1])

                        # A. Pairs within the same cell (p_j follows p_i in linked list)
                        p_j = int(nxt[p_i])
                        while p_j != -1:
                            pair_checks += 1
                            dx = px_i - int(pos[p_j, 0])
                            dy = py_i - int(pos[p_j, 1])
                            if -MIN_DIST_Q12 < dx < MIN_DIST_Q12 and -MIN_DIST_Q12 < dy < MIN_DIST_Q12:
                                d2_q12 = (dx * dx + dy * dy) >> SHIFT
                                if 0 < d2_q12 < MIN_DIST_SQ_Q12:
                                    factor = int(PUSH_LUT[d2_q12 >> 5])
                                    corr_x = (dx * factor) >> SHIFT
                                    corr_y = (dy * factor) >> SHIFT
                                    px_i += corr_x
                                    py_i += corr_y
                                    pos[p_j, 0] -= corr_x
                                    pos[p_j, 1] -= corr_y
                            p_j = int(nxt[p_j])

                        # B. Pairs in the 4 forward neighbor cells
                        for ox, oy in nbr_offsets:
                            nx_c = cx + ox
                            ny_c = cy + oy
                            if 0 <= nx_c < NX and 0 <= ny_c < NY:
                                p_j = int(head[(ny_c << 4) | nx_c])
                                while p_j != -1:
                                    pair_checks += 1
                                    dx = px_i - int(pos[p_j, 0])
                                    dy = py_i - int(pos[p_j, 1])
                                    if -MIN_DIST_Q12 < dx < MIN_DIST_Q12 and -MIN_DIST_Q12 < dy < MIN_DIST_Q12:
                                        d2_q12 = (dx * dx + dy * dy) >> SHIFT
                                        if 0 < d2_q12 < MIN_DIST_SQ_Q12:
                                            factor = int(PUSH_LUT[d2_q12 >> 5])
                                            corr_x = (dx * factor) >> SHIFT
                                            corr_y = (dy * factor) >> SHIFT
                                            px_i += corr_x
                                            py_i += corr_y
                                            pos[p_j, 0] -= corr_x
                                            pos[p_j, 1] -= corr_y
                                    p_j = int(nxt[p_j])

                        pos[p_i, 0] = px_i
                        pos[p_i, 1] = py_i
                        p_i = int(nxt[p_i])

        self.last_pair_checks = pair_checks // 2

    # -----------------------------------------------------------------
    # Stage 2B: LUT Circular Wall Collision (Zero sqrt, Zero division)
    # -----------------------------------------------------------------
    def collide_walls_lut_q12(self):
        rx = (self.pos_q12[:, 0] - CEN_Q12).astype(np.int64)
        ry = (self.pos_q12[:, 1] - CEN_Q12).astype(np.int64)
        r2_q8 = (rx * rx + ry * ry) >> 16

        out = np.nonzero(r2_q8 > WALL_R2_Q8)[0]
        if len(out) > 0:
            lut_idx = np.clip((r2_q8[out] - WALL_R2_Q8) >> 6, 0, 63).astype(np.int32)
            inv_r_q12 = WALL_INV_R_LUT[lut_idx].astype(np.int64)

            # Unit normal in Q4.12: n = (r * inv_r) >> 12
            nx_q12 = (rx[out] * inv_r_q12) >> SHIFT
            ny_q12 = (ry[out] * inv_r_q12) >> SHIFT

            # Project position onto WALL_R_Q12
            self.pos_q12[out, 0] = (CEN_Q12 + ((nx_q12 * WALL_R_Q12) >> SHIFT)).astype(np.int32)
            self.pos_q12[out, 1] = (CEN_Q12 + ((ny_q12 * WALL_R_Q12) >> SHIFT)).astype(np.int32)

            # Remove outward normal velocity: vn = (v . n) >> 12
            vx = self.vel_q12[out, 0].astype(np.int64)
            vy = self.vel_q12[out, 1].astype(np.int64)
            vn_q12 = np.maximum((vx * nx_q12 + vy * ny_q12) >> SHIFT, 0)

            self.vel_q12[out, 0] = (vx - ((vn_q12 * nx_q12) >> SHIFT)).astype(np.int32)
            self.vel_q12[out, 1] = (vy - ((vn_q12 * ny_q12) >> SHIFT)).astype(np.int32)

    # -----------------------------------------------------------------
    # Wire-Sliced Bilinear Weights & P2G / G2P Transfers
    # -----------------------------------------------------------------
    def _corners_q12(self, gx_q12, gy_q12, shape):
        max_x = ((shape[0] - 1) << SHIFT) - 1
        max_y = ((shape[1] - 1) << SHIFT) - 1
        gx = np.clip(gx_q12, 0, max_x)
        gy = np.clip(gy_q12, 0, max_y)

        # In hardware, i0 is just the upper bits [15:12] and dx is the lower bits [11:0]
        i0 = gx >> SHIFT
        j0 = gy >> SHIFT
        dx = gx & (SCALE - 1)
        dy = gy & (SCALE - 1)
        om_dx = SCALE - dx
        om_dy = SCALE - dy

        idx = ((i0, j0), (i0 + 1, j0), (i0 + 1, j0 + 1), (i0, j0 + 1))
        w = (
            (om_dx * om_dy) >> SHIFT,
            (dx * om_dy) >> SHIFT,
            (dx * dy) >> SHIFT,
            (om_dx * dy) >> SHIFT,
        )
        return idx, w

    def _scatter_q12(self, total, weight, gx_q12, gy_q12, values_q12=None):
        idx, w = self._corners_q12(gx_q12, gy_q12, weight.shape)
        for (i, j), wk in zip(idx, w):
            np.add.at(weight, (i, j), wk)
            if values_q12 is not None:
                weighted_val = (values_q12.astype(np.int64) * wk) >> SHIFT
                np.add.at(total, (i, j), weighted_val.astype(np.int32))

    def _gather_q12(self, new_q12, old_q12, gx_q12, gy_q12, valid):
        idx, w = self._corners_q12(gx_q12, gy_q12, new_q12.shape)
        wsum = np.zeros(NP, dtype=np.int64)
        vnew_acc = np.zeros(NP, dtype=np.int64)
        vdelta_acc = np.zeros(NP, dtype=np.int64)

        for (i, j), wk in zip(idx, w):
            wv = (wk * valid[i, j]).astype(np.int64)
            wsum += wv
            vnew_acc += new_q12[i, j].astype(np.int64) * wv
            vdelta_acc += (new_q12[i, j] - old_q12[i, j]).astype(np.int64) * wv

        ok = wsum > 0
        safe_w = np.where(ok, wsum, 1)
        vnew_out = np.where(ok, vnew_acc // safe_w, 0).astype(np.int32)
        vdelta_out = np.where(ok, vdelta_acc // safe_w, 0).astype(np.int32)
        return vnew_out, vdelta_out, ok

    def p2g_q12(self):
        for arr in (self.u_sum_q12, self.u_weight_q12, self.v_sum_q12, self.v_weight_q12, self.density_q12):
            arr.fill(0)

        px = self.pos_q12[:, 0]
        py = self.pos_q12[:, 1]
        self._scatter_q12(self.u_sum_q12, self.u_weight_q12, px, py - HALF_CELL_Q12, self.vel_q12[:, 0])
        self._scatter_q12(self.v_sum_q12, self.v_weight_q12, px - HALF_CELL_Q12, py, self.vel_q12[:, 1])
        self._scatter_q12(None, self.density_q12, px - HALF_CELL_Q12, py - HALF_CELL_Q12)

        self.u_q12.fill(0)
        self.v_q12.fill(0)

        mu = self.u_weight_q12 > 0
        self.u_q12[mu] = ((self.u_sum_q12[mu].astype(np.int64) << SHIFT) // self.u_weight_q12[mu]).astype(np.int32)

        mv = self.v_weight_q12 > 0
        self.v_q12[mv] = ((self.v_sum_q12[mv].astype(np.int64) << SHIFT) // self.v_weight_q12[mv]).astype(np.int32)

        self.u_q12 *= self.u_open
        self.v_q12 *= self.v_open
        self.density_q12[~self.inside] = 0

        self.u_old_q12[:] = self.u_q12
        self.v_old_q12[:] = self.v_q12

    def update_cell_type_q12(self):
        self.cell_type.fill(AIR)
        self.cell_type[~self.inside] = SOLID

        self.particle_count.fill(0)
        ci = np.clip(self.pos_q12[:, 0] >> SHIFT, 0, NX - 1)
        cj = np.clip(self.pos_q12[:, 1] >> SHIFT, 0, NY - 1)
        np.add.at(self.particle_count, (ci, cj), 1)

        water = ((self.particle_count > 0) | (self.density_q12 > WATER_DENS_THS_Q12)) & self.inside
        self.cell_type[water] = WATER

        ct = np.pad(self.cell_type, 1, constant_values=SOLID)
        self.u_valid = ((ct[0:NX + 1, 1:NY + 1] != AIR) | (ct[1:NX + 2, 1:NY + 1] != AIR)).astype(np.int32)
        self.v_valid = ((ct[1:NX + 1, 0:NY + 1] != AIR) | (ct[1:NX + 1, 1:NY + 2] != AIR)).astype(np.int32)

    # -----------------------------------------------------------------
    # Stage 4: Integer Red-Black Pressure Solver (Uses OMEGA_INV_S_LUT)
    # -----------------------------------------------------------------
    def divergence_solve_q12(self):
        wi, wj = np.where(self.cell_type == WATER)
        if len(wi) == 0:
            self.pressure_q12.fill(0)
            self.divergence_q12.fill(0)
            return

        self.pressure_q12[self.cell_type != WATER] = 0
        red = (wi + wj) % 2 == 0
        sets = [(wi[red], wj[red]), (wi[~red], wj[~red])]

        for _ in range(self.itterations):
            for i, j in sets:
                s_idx = self.s_count[i, j]
                omega_inv_s = OMEGA_INV_S_LUT[s_idx].astype(np.int64)

                div_q12 = (
                    (self.u_q12[i + 1, j] - self.u_q12[i, j])
                    + (self.v_q12[i, j + 1] - self.v_q12[i, j])
                ).astype(np.int64)

                # Density drift compensation in Q4.12
                comp_q12 = np.maximum(
                    ((self.density_q12[i, j].astype(np.int64) * INV_REST_DENSITY_Q12) >> SHIFT) - SCALE,
                    0,
                )
                div_q12 -= DRIFT_K_INT * comp_q12

                # Single LUT multiply replaces float division (-d / s * O)
                p_q12 = ((-div_q12 * omega_inv_s) >> SHIFT).astype(np.int32)
                self.pressure_q12[i, j] += p_q12

                self.u_q12[i, j] -= p_q12 * self.sx0[i, j]
                self.u_q12[i + 1, j] += p_q12 * self.sx1[i, j]
                self.v_q12[i, j] -= p_q12 * self.sy0[i, j]
                self.v_q12[i, j + 1] += p_q12 * self.sy1[i, j]

        self.divergence_q12.fill(0)
        self.divergence_q12[wi, wj] = (
            (self.u_q12[wi + 1, wj] - self.u_q12[wi, wj])
            + (self.v_q12[wi, wj + 1] - self.v_q12[wi, wj])
        )

    def g2p_q12(self):
        px = self.pos_q12[:, 0]
        py = self.pos_q12[:, 1]

        for comp, (new, old, valid, gx, gy) in enumerate((
            (self.u_q12, self.u_old_q12, self.u_valid, px, py - HALF_CELL_Q12),
            (self.v_q12, self.v_old_q12, self.v_valid, px - HALF_CELL_Q12, py),
        )):
            vnew, vdelta, ok = self._gather_q12(new, old, gx, gy, valid)
            v_cur = self.vel_q12[:, comp].astype(np.int64)
            v_pic = np.where(ok, vnew, v_cur).astype(np.int64)
            v_flip = np.where(ok, v_cur + vdelta, v_cur).astype(np.int64)

            blended = (PIC_R_Q12 * v_pic + FLIP_R_Q12 * v_flip) >> SHIFT
            self.vel_q12[:, comp] = np.clip(blended, -VMAX_Q12, VMAX_Q12).astype(np.int32)

    def apply_stir(self, mouse_xy, mouse_vel):
        mx_q12 = int(round(mouse_xy[0] * SCALE))
        my_q12 = int(round(mouse_xy[1] * SCALE))
        mvx_q12 = int(round(mouse_vel[0] * SCALE))
        mvy_q12 = int(round(mouse_vel[1] * SCALE))

        dx = (self.pos_q12[:, 0] - mx_q12).astype(np.int64)
        dy = (self.pos_q12[:, 1] - my_q12).astype(np.int64)
        d2_q12 = (dx * dx + dy * dy) >> SHIFT
        near = d2_q12 < (6 * SCALE)
        if np.any(near):
            # Linear falloff approximation for interactive stir
            w_q12 = np.maximum(SCALE - (d2_q12[near] // 6), 0)
            self.vel_q12[near, 0] += (((mvx_q12 - self.vel_q12[near, 0]) * w_q12) >> (SHIFT + 1)).astype(np.int32)
            self.vel_q12[near, 1] += (((mvy_q12 - self.vel_q12[near, 1]) * w_q12) >> (SHIFT + 1)).astype(np.int32)

    def apply_shake(self, frame_idx=1):
        rng = np.random.default_rng(frame_idx * 977 + 1)
        jitter = rng.integers(-15 * SCALE, 15 * SCALE, size=(NP, 2), dtype=np.int32)
        self.vel_q12 = np.clip(self.vel_q12 + jitter, -VMAX_Q12, VMAX_Q12)

    # -----------------------------------------------------------------
    # Master Frame Update (with Autonomous Clock-Gating Sleep Check)
    # -----------------------------------------------------------------
    def update(
        self,
        gx_8bit=0,
        gy_8bit=-60,
        stir_pos=None,
        stir_vel=None,
        shake=False,
        frame_idx=1,
        lis_sleep_state=False,
    ):
        accel_changed = (
            abs(int(gx_8bit) - self._last_gx) > 1
            or abs(int(gy_8bit) - self._last_gy) > 1
        )
        self._last_gx = int(gx_8bit)
        self._last_gy = int(gy_8bit)

        if accel_changed or stir_pos is not None or shake:
            self.clock_gated_sleep = False
            self.sleep_Hold_frames = 0

        if self.clock_gated_sleep:
            self.last_pair_checks = 0
            return

        prev_pos = self.pos_q12.copy()

        self.apply_gravity_and_advect_q12(gx_8bit, gy_8bit)
        if stir_pos is not None and stir_vel is not None:
            self.apply_stir(stir_pos, stir_vel)
        if shake:
            self.apply_shake(frame_idx)

        self.push_apart_spatial_hash_q12()
        self.collide_walls_lut_q12()
        self.p2g_q12()
        self.update_cell_type_q12()
        self.divergence_solve_q12()
        self.g2p_q12()

        # Check if fluid has settled or LIS2DW12 INT2 (SLEEP_STATE) is asserted for 90 frames
        mean_pos_delta = int(np.abs(self.pos_q12 - prev_pos).mean())
        if (mean_pos_delta < 32 or lis_sleep_state) and not accel_changed and stir_pos is None and not shake:
            self.sleep_Hold_frames += 1
            if self.sleep_Hold_frames >= 90:  # 1.5 seconds stationary -> gate ASIC clock
                self.clock_gated_sleep = True
        else:
            self.sleep_Hold_frames = 0

        self._sync_views_and_ram()