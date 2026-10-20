import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation_fixed import ExternalSPIRAM, FixedFlip


class TestFixedPointSimulation(unittest.TestCase):
    def test_external_spi_ram_operations(self):
        """External SPI RAM emulator correctly stores and retrieves blocks."""
        ram = ExternalSPIRAM(1024)
        test_data = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        ram.write_block(0x100, test_data)
        read_back = ram.read_block(0x100, 8)
        self.assertEqual(test_data, read_back)

    def test_fixed_simulation_initialization_and_ram_sync(self):
        """Simulation initializes 256 particles on 216 active circular cells and syncs to SPI RAM."""
        sim = FixedFlip()
        self.assertEqual(sim.positions.shape, (256, 2))
        self.assertEqual(sim.velocity.shape, (256, 2))
        self.assertEqual(int(sim.inside.sum()), 216)
        pos_bytes = sim.ram.read_block(0x0000, 8)
        self.assertNotEqual(pos_bytes, b"\x00" * 8)

    def test_fixed_simulation_runs_finite_steps(self):
        """Simulation runs 30 steps with accelerometer input with finite positions inside wall_r."""
        sim = FixedFlip()
        for _ in range(30):
            sim.update(gx_8bit=20, gy_8bit=-50)

        self.assertTrue(np.all(np.isfinite(sim.positions)))
        self.assertTrue(np.all(np.isfinite(sim.velocity)))
        r = sim.positions - sim.center
        self.assertTrue(np.all(np.hypot(r[:, 0], r[:, 1]) <= sim.wall_r + 1e-3))

    def test_reset_restores_initial_state(self):
        """Calling reset() restores initial particle layout."""
        sim = FixedFlip()
        init_pos = sim.positions.copy()
        for _ in range(10):
            sim.update()
        sim.reset()
        np.testing.assert_array_equal(sim.positions, init_pos)


if __name__ == "__main__":
    unittest.main()