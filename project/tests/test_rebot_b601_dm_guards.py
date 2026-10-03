"""Fault injection tests for the empty-scene bring-up stop guards."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebot_integration.guarded_empty_env import BringupStop, GuardedReBotB601DMEmpty


class BringupGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = GuardedReBotB601DMEmpty()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()

    def test_nan_state_and_action_stop_before_integration(self):
        for target in ("qpos", "qvel", "ctrl"):
            self.env.reset()
            getattr(self.env.sim.data, target)[0] = np.nan
            with self.assertRaises(BringupStop):
                self.env.step(np.zeros(7))
            self.assertEqual(self.env.sim.data.time, 0)
        self.env.reset()
        with self.assertRaisesRegex(BringupStop, "policy action"):
            self.env.step(np.full(7, np.nan))

    def test_speed_and_excursion_guards(self):
        for index, speed in ((0, 1.01), (6, 0.151)):
            self.env.reset()
            self.env.sim.data.qvel[index] = speed
            with self.assertRaisesRegex(BringupStop, "speed"):
                self.env.step(np.zeros(7))
            self.assertEqual(self.env.sim.data.time, 0)
        self.env.reset()
        self.env.sim.data.qpos[0] += 0.101
        with self.assertRaisesRegex(BringupStop, "excursion"):
            self.env.step(np.zeros(7))

    def test_nonfinite_torque_stops_before_integration(self):
        controller = self.env.robots[0].part_controllers["right"]

        def bad_output():
            controller.torques = np.full(6, np.nan)
            return controller.torques

        with patch.object(controller, "run_controller", side_effect=bad_output):
            with self.assertRaisesRegex(BringupStop, "controller output torque"):
                self.env.step(np.zeros(7))
        self.assertEqual(self.env.sim.data.time, 0)

    def test_saturation_timer_is_per_joint_and_clears(self):
        controller = self.env.robots[0].part_controllers["right"]
        controller.torques = np.array([27, 0, 0, 0, 0, 0.0])
        for _ in range(40):
            self.env.check_torques()
        controller.torques = np.zeros(6)
        self.env.check_torques()
        np.testing.assert_array_equal(self.env.saturated_for, 0)
        controller.torques = np.array([0, 0, 0, -7, 0, 0.0])
        with self.assertRaisesRegex(BringupStop, "joints \\[4\\]"):
            for _ in range(126):
                self.env.check_torques()

    def test_normal_hold(self):
        for _ in range(10):
            self.env.step(np.zeros(7))
        self.assertFalse(self.env.saturated.any())


if __name__ == "__main__":
    unittest.main()
