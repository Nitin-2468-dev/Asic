import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim_hardware_model import (
    CEN_Q12,
    OMEGA_INV_S_LUT,
    PUSH_LUT,
    SCALE,
    WALL_INV_R_LUT,
    WALL_R_Q12,
    HardwareFlip,
)


class TestHardwareIntegerModel(unittest.TestCase):
    def test_pure_integer_state_types(self):
        """All internal physics arrays use integer dtypes (no float32 in core state)."""
        sim = HardwareFlip()
        for name in ("pos_q12", "vel_q12", "u_q12", "v_q12", "density_q12", "pressure_q12"):
            arr = getattr(sim, name)
            self.assertTrue(np.issubdtype(arr.dtype, np.integer), f"{name} is not an integer array")

    def test_spatial_hash_reduces_pair_checks_dramatically(self):
        """Spatial hash linked list checks ~2,100 pairs instead of 32,640 (upper-tri) / 65,536 (N^2) pairs."""
        sim = HardwareFlip()
        for _ in range(15):
            sim.update(gx_8bit=15, gy_8bit=-55)

        self.assertGreater(sim.last_pair_checks, 0)
        # Verify >12x reduction vs upper-triangle (32,640) and >24x vs full N^2 (65,536)
        self.assertLess(sim.last_pair_checks, sim.naive_pair_checks // 12)

    def test_wall_lut_keeps_particles_inside_circle(self):
        """64-entry WALL_INV_R_LUT projects out-of-bounds particles inside WALL_R without sqrt."""
        sim = HardwareFlip()
        for _ in range(45):
            sim.update(gx_8bit=35, gy_8bit=-45)

        r = sim.positions - sim.center
        max_dist = float(np.hypot(r[:, 0], r[:, 1]).max())
        # Allow ~0.03 cell quantization tolerance from 64-entry LUT
        self.assertLessEqual(max_dist, sim.wall_r + 0.04)

    def test_pressure_solver_lut_reduces_divergence(self):
        """Integer Red-Black Gauss-Seidel solver with OMEGA_INV_S_LUT reduces velocity divergence."""
        sim = HardwareFlip()
        centroid = sim.pos_q12.mean(axis=0).astype(np.int32)
        sim.vel_q12 = -((sim.pos_q12 - centroid) * 4)

        sim.p2g_q12()
        sim.update_cell_type_q12()

        wi, wj = np.where(sim.cell_type == 1)
        raw_div = (sim.u_q12[wi + 1, wj] - sim.u_q12[wi, wj]) + (sim.v_q12[wi, wj + 1] - sim.v_q12[wi, wj])
        pre_norm = float(np.linalg.norm(raw_div))

        sim.divergence_solve_q12()
        post_norm = float(np.linalg.norm(sim.divergence_q12[wi, wj]))
        self.assertLess(post_norm, pre_norm * 0.25)

    def test_autonomous_clock_gated_sleep_and_wake(self):
        """Engine enters clock_gated_sleep when LIS2DW12 INT2 (SLEEP_STATE) is active for 90 frames and wakes on tilt."""
        sim = HardwareFlip()
        for _ in range(95):
            sim.update(gx_8bit=0, gy_8bit=-60, lis_sleep_state=True)
            if sim.clock_gated_sleep:
                break

        self.assertTrue(sim.clock_gated_sleep)
        sim.update(gx_8bit=0, gy_8bit=-60, lis_sleep_state=True)
        self.assertEqual(sim.last_pair_checks, 0)  # Zero work performed while clock-gated

        # Tilt the pendant -> must wake up immediately
        sim.update(gx_8bit=25, gy_8bit=-50, lis_sleep_state=False)
        self.assertFalse(sim.clock_gated_sleep)
        self.assertGreater(sim.last_pair_checks, 0)


if __name__ == "__main__":
    unittest.main()