"""Physics-step contact monitoring and stop-before-close validation helpers."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demos.scripted_lift_expert import solve_ready_pose, TCP_TO_GRASP, TOP_DOWN_BASE_ROT
from rebot_integration.descent_validation import DescentMonitor, ReBotDescentValidation, contact_category
from rebot_integration.guarded_empty_env import BringupStop
from rebot_integration.lift_setup import make_rebot_lift
from rebot_integration.lift_validation import LiftDiagnostics


class DescentValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_rebot_lift(env_class=ReBotDescentValidation)
        q, _ = solve_ready_pose(cls.env)
        cls.env.robots[0].init_qpos = q

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.descent_monitor = None
        self.env.reset()
        gripper = self.env.robots[0].gripper["right"]
        fingers = [self.env.sim.model.get_joint_qpos_addr(n) for n in gripper.joints]
        self.env.sim.data.qpos[fingers] = [.05, -.05]
        self.env.sim.forward()
        self.diag = LiftDiagnostics(self.env, arm_excursion_limit=2.5)
        self.env.diagnostics = self.diag
        self.directory = tempfile.TemporaryDirectory()
        self.monitor = DescentMonitor(self.env, self.diag, TCP_TO_GRASP,
                                      TOP_DOWN_BASE_ROT, self.directory.name)
        self.env.descent_monitor = self.monitor

    def tearDown(self):
        self.env.descent_monitor = None
        self.monitor.close()
        self.directory.cleanup()

    def test_every_physics_step_is_logged(self):
        action = np.zeros(7)
        action[-1] = -1
        for _ in range(2):
            self.env.step(action)
        self.monitor.stream.flush()
        rows = [json.loads(line) for line in (Path(self.directory.name) / 'descent_steps.jsonl').read_text().splitlines()]
        expected = 2 * int(self.env.control_timestep / self.env.model_timestep)
        self.assertEqual(len(rows), expected)
        np.testing.assert_allclose(np.diff([r['time_s'] for r in rows]), self.env.model_timestep, atol=1e-10)
        self.assertTrue(all(r['solver_completed'] for r in rows))

    def test_first_palm_contact_survives_stop(self):
        data, model = self.env.sim.data, self.env.sim.model
        palm = model.geom_name2id(self.env.robots[0].gripper['right'].naming_prefix + 'gripper_palm_collision')
        q = data.get_joint_qpos(self.env.cube.joints[0]).copy()
        # Contact the lower palm surface; its mesh origin overlaps the wrist.
        q[:3] = data.geom_xpos[palm] + [0, 0, -0.03]
        data.set_joint_qpos(self.env.cube.joints[0], q)
        self.env.sim.forward()
        state = self.monitor.capture()
        with self.assertRaisesRegex(BringupStop, 'premature descent contact'):
            self.monitor.record(state)
        history = self.monitor.summary()['contact_history']['palm_cube']
        self.assertTrue(history)
        self.assertEqual(history[0]['first_physical_grasp_xyz'], state['physical_grasp_xyz'])
        self.monitor.stream.flush()
        self.assertIn('palm_cube', (Path(self.directory.name) / 'descent_steps.jsonl').read_text())

    def test_contact_categories_are_order_independent(self):
        pairs = {
            'left_finger_cube': ('gripper0_right_finger_left_front_collision', 'cube_g0'),
            'right_finger_cube': ('gripper0_right_finger_right_mid_collision', 'cube_g0'),
            'palm_cube': ('gripper0_right_gripper_palm_collision', 'cube_g0'),
            'gripper_table': ('gripper0_right_finger_left_front_collision', 'table_collision'),
            'arm_table': ('robot0_link3_collision', 'table_collision'),
            'cube_table': ('cube_g0', 'table_collision'),
        }
        for category, pair in pairs.items():
            self.assertEqual(contact_category(pair), category)
            self.assertEqual(contact_category(pair[::-1]), category)


if __name__ == '__main__':
    unittest.main()
