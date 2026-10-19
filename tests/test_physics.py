import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation import AIR, SOLID, WATER, Flip


class TestFluidPhysics(unittest.TestCase):
    def test_particles_remain_inside_container_after_wall_collisions(self):
        """collide_walls clamps out-of-bounds particles to wall_r and removes outward normal velocity."""
        sim = Flip()
        angles = np.linspace(0, 2 * np.pi, sim.amount, endpoint=False)
        far_radius = sim.wall_r + 5.0
        sim.positions[:, 0] = sim.center[0] + far_radius * np.cos(angles)
        sim.positions[:, 1] = sim.center[1] + far_radius * np.sin(angles)
        sim.velocity[:, 0] = 20.0 * np.cos(angles)
        sim.velocity[:, 1] = 20.0 * np.sin(angles)

        sim.collide_walls()

        r = sim.positions - sim.center
        distances = np.hypot(r[:, 0], r[:, 1])
        self.assertTrue(np.all(distances <= sim.wall_r + 1e-5))

        normals = r / distances[:, None]
        vn = (sim.velocity * normals).sum(axis=1)
        self.assertTrue(np.all(vn <= 1e-5))

    def test_particles_stay_within_container_during_simulation(self):
        """Over 60 full simulation updates, particles remain confined within wall_r after wall collision."""
        sim = Flip()
        for _ in range(60):
            sim.update()
            r = sim.positions - sim.center
            distances = np.hypot(r[:, 0], r[:, 1])
            self.assertTrue(np.all(distances <= sim.wall_r + 1e-5))

    def test_push_apart_separates_overlapping_particles(self):
        """push_apart increases the distance between two close particles."""
        sim = Flip()
        sim.positions[0] = sim.center.copy()
        sim.positions[1] = sim.center + np.array([0.1, 0.0], dtype=np.float32)

        initial_dist = np.linalg.norm(sim.positions[1] - sim.positions[0])
        self.assertLess(initial_dist, sim.min_distance)

        sim.push_apart()

        final_dist = np.linalg.norm(sim.positions[1] - sim.positions[0])
        self.assertGreater(final_dist, initial_dist)

    def test_particle_to_grid_and_grid_to_particle_transfer(self):
        """Uniform particle velocities transfer cleanly to the MAC grid and back to particles."""
        sim = Flip()
        uniform_vel = np.array([3.5, -2.25], dtype=np.float32)
        sim.velocity[:] = uniform_vel

        sim.Particle_to_Grid()
        sim.update_cell_type()

        self.assertAlmostEqual(float(sim.density.sum()), float(sim.amount), places=3)

        self.assertTrue(np.all(sim.u[sim.u_open == 0] == 0.0))
        self.assertTrue(np.all(sim.v[sim.v_open == 0] == 0.0))

        sim.Grid_to_Particle()
        np.testing.assert_allclose(sim.velocity, np.broadcast_to(uniform_vel, sim.velocity.shape), atol=1e-4)

    def test_pressure_solver_reduces_divergence_on_water_cells(self):
        """Divergence() significantly reduces net velocity divergence in water cells."""
        sim = Flip()
        # Impart an inward-converging velocity field around the fluid centroid to create strong negative divergence
        centroid = sim.positions.mean(axis=0)
        sim.velocity = -(sim.positions - centroid).astype(np.float32) * 5.0

        sim.Particle_to_Grid()
        sim.update_cell_type()

        wi, wj = np.where(sim.cell_type == WATER)
        self.assertGreater(len(wi), 0)

        # Measure the solver's target residual (divergence - compression compensation) before and after Divergence()
        raw_div = (sim.u[wi + 1, wj] - sim.u[wi, wj]) + (sim.v[wi, wj + 1] - sim.v[wi, wj])
        compression_target = sim.k * np.maximum(sim.density[wi, wj] - sim.rest_density, 0.0)
        pre_norm = float(np.linalg.norm(raw_div - compression_target))
        self.assertGreater(pre_norm, 1.0)

        sim.Divergence()

        post_div = sim.divergence[wi, wj]
        post_norm = float(np.linalg.norm(post_div - compression_target))
        self.assertLess(post_norm, pre_norm * 0.2)


if __name__ == "__main__":
    unittest.main()