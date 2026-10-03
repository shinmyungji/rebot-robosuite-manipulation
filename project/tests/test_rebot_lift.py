"""Stock Lift integration, position-workspace, and table-collision validation."""

from pathlib import Path
import sys
import unittest

import numpy as np
from robosuite.environments.manipulation.lift import Lift

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebot_integration.guarded_empty_env import BringupStop
from rebot_integration.lift_setup import BASE_POSITION, CUBE_X_RANGE, CUBE_Y_RANGE, make_rebot_lift
from rebot_integration.lift_validation import LiftDiagnostics, ReBotLiftValidation, snapshot


class ReBotLiftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_rebot_lift(env_class=ReBotLiftValidation)
        cls.env.reset()
        cls.diag = LiftDiagnostics(cls.env)
        cls.env.diagnostics = cls.diag

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()
        self.env.sim.forward()

    def test_task_logic_and_controller_remain_stock(self):
        for name in ("_load_model", "_reset_internal", "reward", "_check_success", "_setup_observables"):
            self.assertIs(getattr(ReBotLiftValidation, name), getattr(Lift, name))
        controller = self.env.robots[0].part_controllers["right"]
        self.assertEqual(controller.name, "JOINT_POSITION")
        self.assertEqual(controller.impedance_mode, "fixed")
        self.assertEqual(self.env.action_dim, 7)

    def test_base_table_and_collision_geometry(self):
        info = snapshot(self.env, self.diag)
        np.testing.assert_allclose(info["base_pos"], BASE_POSITION)
        np.testing.assert_allclose(self.env.table_full_size, [0.8, 0.8, 0.05])
        self.assertEqual(len(self.diag.arm_geoms), 7)
        self.assertEqual(len(self.env.robots[0].gripper["right"].contact_geoms), 11)
        model = self.env.sim.model._model
        self.assertTrue(np.all(model.geom_contype[self.diag.arm_geoms] == 1))
        self.assertTrue(np.all(model.geom_conaffinity[self.diag.arm_geoms] == 1))
        self.assertTrue(np.all(model.geom_group[self.diag.arm_geoms] == 0))
        clearance = self.diag.arm_table_clearance()
        self.assertAlmostEqual(clearance["robot0_base_link_collision"]["lowest_above_table_m"], 0, places=6)
        self.assertFalse(any(v["below_tabletop"] for v in clearance.values()))
        self.diag.check()

    def test_cube_region_and_reachability(self):
        sampler = self.env.placement_initializer
        for x in np.linspace(*CUBE_X_RANGE, 3):
            for y in np.linspace(*CUBE_Y_RANGE, 3):
                for z in (0.820, 0.832):
                    result = sampler.workspace.check(np.array([x, y, z]) - BASE_POSITION)
                    self.assertTrue(result.reachable, (x, y, z, result.error_m))
        self.assertFalse(sampler.workspace.check([3, 0, 0]).reachable)
        for _ in range(8):
            self.env.reset()
            self.env.sim.forward()
            cube = self.env.sim.data.body_xpos[self.env.cube_body_id]
            self.assertTrue(CUBE_X_RANGE[0] <= cube[0] <= CUBE_X_RANGE[1])
            self.assertTrue(CUBE_Y_RANGE[0] <= cube[1] <= CUBE_Y_RANGE[1])
            self.assertFalse(self.env._check_success())
            self.assertTrue(all(r.reachable for r in sampler.last_checks))
            before = self.env.sim.data.qpos.copy()
            self.assertTrue(sampler.workspace.check(cube - BASE_POSITION).reachable)
            np.testing.assert_array_equal(self.env.sim.data.qpos, before)

    def test_stock_success_height_threshold(self):
        # Synthetic cube states test task semantics, not autonomous manipulation.
        qpos = self.env.sim.data.get_joint_qpos(self.env.cube.joints[0]).copy()
        for height, expected in ((0.839, False), (0.840, False), (0.841, True)):
            qpos[2] = height
            self.env.sim.data.set_joint_qpos(self.env.cube.joints[0], qpos)
            self.env.sim.forward()
            self.assertEqual(bool(self.env._check_success()), expected)
            self.assertAlmostEqual(self.env.reward(), float(expected))

    def test_penetration_detector_and_physical_table_contacts(self):
        model = self.env.sim.model._model
        base = model.body(self.env.robots[0].robot_model.root_body).id
        original = model.body_pos[base].copy()
        try:
            # Fault injection only: lower robot through table to prove both
            # enabled contact masks and geometry-based penetration reporting.
            model.body_pos[base, 2] -= 0.10
            self.env.sim.forward()
            self.assertTrue(any(v["below_tabletop"] for v in self.diag.arm_table_clearance().values()))
            contacts = self.diag.contacts()["table_contacts"]
            self.assertTrue(any(any(n.startswith("robot0_") for n in c["geoms"]) for c in contacts), contacts)
            with self.assertRaisesRegex(BringupStop, "below tabletop"):
                self.diag.check()
            # A deep displacement still gets detected if contacts disappear.
            model.body_pos[base, 2] -= 1.0
            self.env.sim.forward()
            self.assertTrue(any(v["below_tabletop"] for v in self.diag.arm_table_clearance().values()))
        finally:
            model.body_pos[base] = original
            self.env.sim.forward()

    def test_small_motion_gripper_and_cube_table_contact(self):
        initial = self.env.robots[0].robot_model.init_qpos
        action = np.zeros(7)

        def step(target, aperture=0):
            action[:6] = np.clip((target - self.env.sim.data.qpos[:6]) / 0.005, -1, 1)
            action[-1] = aperture
            self.env.step(action)
            self.assertTrue(np.isfinite(self.env.sim.data.qpos).all())
            self.assertTrue(np.isfinite(self.env.sim.data.ctrl).all())
            self.assertFalse(self.env._check_success())
            self.assertFalse(any(v["below_tabletop"] for v in self.diag.arm_table_clearance().values()))

        for _ in range(20):
            step(initial)
        self.assertTrue(any("cube" in " ".join(c["geoms"]) for c in self.diag.contacts()["table_contacts"]))
        for joint in range(6):
            for sign in (1, -1):
                target = initial.copy()
                target[joint] += sign * 0.003
                for _ in range(10):
                    step(target)
                for _ in range(10):
                    step(initial)
        for start, stop in ((0, -1), (-1, 1)):
            for aperture in np.linspace(start, stop, 80):
                step(initial, aperture)
            for _ in range(20):
                step(initial, stop)
        self.assertLess(self.env.sim.data.qpos[6], 0.005)
        np.testing.assert_allclose(self.env.sim.data.qfrc_applied, 0)


if __name__ == "__main__":
    unittest.main()
