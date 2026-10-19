import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation import AIR, SOLID, WATER, Flip


class TestSimulationLifecycle(unittest.TestCase):
    def test_headless_import_and_initialization(self):
        """Simulation initializes without importing pyray or opening a window."""
        sim = Flip()
        self.assertNotIn("pyray", sys.modules)
        self.assertEqual(sim.positions.shape, (sim.amount, 2))
        self.assertEqual(sim.velocity.shape, (sim.amount, 2))
        self.assertEqual(sim.u.shape, (sim.grid_size + 1, sim.grid_size))
        self.assertEqual(sim.v.shape, (sim.grid_size, sim.grid_size + 1))
        self.assertEqual(sim.cell_type.shape, (sim.grid_size, sim.grid_size))
        self.assertTrue(np.all(sim.velocity == 0.0))

    def test_reset_restores_initial_state(self):
        """Calling reset() after stepping restores exact initial particle positions and zeroed grids."""
        sim = Flip()
        initial_positions = sim.positions.copy()

        for _ in range(10):
            sim.update()

        self.assertFalse(np.allclose(sim.positions, initial_positions))
        self.assertFalse(np.all(sim.velocity == 0.0))

        sim.reset()
        np.testing.assert_array_equal(sim.positions, initial_positions)
        np.testing.assert_array_equal(sim.velocity, np.zeros_like(sim.velocity))
        np.testing.assert_array_equal(sim.u, np.zeros_like(sim.u))
        np.testing.assert_array_equal(sim.v, np.zeros_like(sim.v))
        np.testing.assert_array_equal(sim.Pressure, np.zeros_like(sim.Pressure))

    def test_determinism_across_runs(self):
        """Two simulations initialized with the same seed produce identical trajectories."""
        sim_a = Flip()
        sim_b = Flip()

        for _ in range(25):
            sim_a.update()
            sim_b.update()

        np.testing.assert_array_equal(sim_a.positions, sim_b.positions)
        np.testing.assert_array_equal(sim_a.velocity, sim_b.velocity)
        np.testing.assert_array_equal(sim_a.u, sim_b.u)
        np.testing.assert_array_equal(sim_a.v, sim_b.v)
        np.testing.assert_array_equal(sim_a.Pressure, sim_b.Pressure)

    def test_finite_positions_and_velocities_over_multiple_steps(self):
        """Particle positions, velocities, and grid fields remain finite (no NaN or Inf) over 60 steps."""
        sim = Flip()
        for _ in range(60):
            sim.update()

        self.assertTrue(np.all(np.isfinite(sim.positions)))
        self.assertTrue(np.all(np.isfinite(sim.velocity)))
        self.assertTrue(np.all(np.isfinite(sim.u)))
        self.assertTrue(np.all(np.isfinite(sim.v)))
        self.assertTrue(np.all(np.isfinite(sim.Pressure)))
        self.assertTrue(np.all(np.isfinite(sim.density)))

    def test_window_motion_updates_gravity_within_bounds(self):
        """External window motion updates effective gravity and clamps magnitude to 1.5 * gravity_mag."""
        sim = Flip()
        base_g = sim.base_gravity.copy()

        sim.apply_window_motion(dx=150, dy=-80, frame_time=1 / 60)
        self.assertFalse(np.allclose(sim.gravity, base_g))

        mag = np.hypot(*sim.gravity)
        self.assertLessEqual(mag, 1.5 * sim.gravity_mag + 1e-5)


if __name__ == "__main__":
    unittest.main()