"""Fixed expert reset and contact-free Cartesian approach regression checks."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demos.scripted_lift_expert import (
    READY_GRASP_WORLD, TOP_DOWN_BASE_ROT, TCP_TO_GRASP,
    APPROACH_WAYPOINT_SPACING, MIN_APPROACH_JOINT_MARGIN,
    solve_ready_pose, plan_ready_approach,
)
from rebot_integration.lift_setup import make_rebot_lift, CUBE_X_RANGE, CUBE_Y_RANGE
from rebot_integration.lift_validation import LiftDiagnostics, ReBotLiftValidation


class ExpertApproachTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_rebot_lift(env_class=ReBotLiftValidation)
        cls.env.reset()
        cls.ready, cls.report = solve_ready_pose(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_ready_independent_of_cube_and_global_default(self):
        model_default = self.env.robots[0].robot_model.init_qpos.copy()
        for _ in range(2):
            self.env.reset()
            before = self.env.sim.data.qpos.copy()
            ready, report = solve_ready_pose(self.env)
            np.testing.assert_allclose(ready, self.ready, atol=1e-10)
            np.testing.assert_array_equal(self.env.sim.data.qpos, before)
            self.assertGreater(report["minimum_joint_margin_rad"], MIN_APPROACH_JOINT_MARGIN)
            self.assertGreater(report["gripper_table_clearance_m"], 0.13)
            self.assertEqual(report["robot_contacts"], [])
        np.testing.assert_array_equal(model_default, [0, -0.75, -0.55, 0, 0, 0])

    def test_workspace_corner_paths_keep_orientation_and_do_not_mutate_live_data(self):
        workspace = self.env.placement_initializer.workspace
        for x in CUBE_X_RANGE:
            for y in CUBE_Y_RANGE:
                for z in (0.820, 0.822):
                    before = self.env.sim.data.qpos.copy()
                    path, report = plan_ready_approach(self.env, self.ready, [x, y, z])
                    np.testing.assert_array_equal(self.env.sim.data.qpos, before)
                    previous = READY_GRASP_WORLD.copy()
                    for step in path:
                        target = np.asarray(step["physical_grasp_target"])
                        self.assertLessEqual(np.linalg.norm(target - previous), APPROACH_WAYPOINT_SPACING + 1e-9)
                        if step["phase"] == "ABOVE_CUBE":
                            self.assertAlmostEqual(target[2], READY_GRASP_WORLD[2])
                        else:
                            np.testing.assert_allclose(target[:2], [x, y])
                        workspace._forward(step["q"])
                        actual_R = workspace.data.site_xmat[workspace.site].reshape(3, 3)
                        np.testing.assert_allclose(actual_R, TOP_DOWN_BASE_ROT, atol=1e-5)
                        actual = workspace.data.site_xpos[workspace.site] + actual_R @ TCP_TO_GRASP
                        np.testing.assert_allclose(actual + [-.30, 0, .8], target, atol=1e-5)
                        previous = target
                    self.assertEqual(report["robot_contacts"], [])
                    self.assertGreater(report["minimum_joint_margin_rad"], MIN_APPROACH_JOINT_MARGIN)
                    self.assertLess(report["max_adjacent_delta_q_rad"], 0.05)
                    np.testing.assert_allclose(path[-1]["physical_grasp_target"], [x, y, z + .1])

    def test_instance_ready_reset_and_controller_hold(self):
        robot = self.env.robots[0]
        original = robot.init_qpos.copy()
        try:
            robot.init_qpos = self.ready.copy()
            self.env.reset()
            self.env.sim.forward()
            diag = LiftDiagnostics(self.env, arm_excursion_limit=2.5)
            self.env.diagnostics = diag
            np.testing.assert_allclose(self.env.sim.data.qpos[diag.arm_qpos], self.ready)
            for _ in range(20):
                self.env.step(np.zeros(7))
                diag.check()
                self.assertEqual(diag.contacts()["robot_contacts"], [])
            np.testing.assert_allclose(self.env.sim.data.qpos[diag.arm_qpos], self.ready, atol=1e-4)
            np.testing.assert_array_equal(robot.robot_model.init_qpos, [0, -.75, -.55, 0, 0, 0])
        finally:
            robot.init_qpos = original


if __name__ == "__main__":
    unittest.main()
