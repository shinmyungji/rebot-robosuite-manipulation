"""Model and controller unit tests; no task environment or demo is run."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rebot_integration import ReBotB601DM, ReBotB601DMGripper  # noqa: E402
from rebot_integration.empty_env import ReBotB601DMEmpty  # noqa: E402
from robosuite.models.bases import robot_base_factory  # noqa: E402
from robosuite.models.grippers import GRIPPER_MAPPING  # noqa: E402
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS  # noqa: E402
from robosuite.robots import FixedBaseRobot, ROBOT_CLASS_MAPPING  # noqa: E402


class ReBotModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = ReBotB601DM()
        cls.gripper = ReBotB601DMGripper(idn="0_right")
        cls.robot.add_base(robot_base_factory("NullMount", idn=0))
        cls.robot.add_gripper(cls.gripper, cls.robot.eef_name["right"])
        cls.model = cls.robot.get_model()

    def test_registration_and_joint_order(self):
        self.assertIs(REGISTERED_ROBOTS["ReBotB601DM"], ReBotB601DM)
        self.assertIs(ROBOT_CLASS_MAPPING["ReBotB601DM"], FixedBaseRobot)
        self.assertIs(GRIPPER_MAPPING["ReBotB601DMGripper"], ReBotB601DMGripper)
        expected = [f"robot0_joint{i}" for i in range(1, 7)]
        self.assertEqual(self.robot.joints, expected)
        self.assertEqual(self.model.nq, 8)
        self.assertEqual(self.model.nv, 8)
        self.assertEqual([self.model.joint(i).name for i in range(6)], expected)
        self.assertEqual(self.robot.default_base, "NullMount")
        self.assertFalse(np.any(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))

    def test_joint_limits_axes_and_source_inertials(self):
        source_path = PROJECT.parent / "third_party/reBot-DevArm/Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf"
        source = ET.parse(source_path).getroot()
        expected_limits = np.array([
            [-2.8, 2.8], [-3.14, 0], [-3.14, 0],
            [-1.87, 1.57], [-1.57, 1.57], [-3.14, 3.14],
        ])
        np.testing.assert_allclose(self.model.jnt_range[:6], expected_limits)
        for i in range(1, 7):
            source_joint = source.find(f"joint[@name='joint{i}']")
            limit = source_joint.find("limit")
            joint = self.model.joint(f"robot0_joint{i}")
            np.testing.assert_allclose(joint.range, [float(limit.get("lower")), float(limit.get("upper"))])
            np.testing.assert_allclose(joint.axis, np.fromstring(source_joint.find("axis").get("xyz"), sep=" "))
            source_mass = float(source.find(f"link[@name='link{i}']/inertial/mass").get("value"))
            self.assertAlmostEqual(float(self.model.body(f"robot0_link{i}").mass[0]), source_mass)
        np.testing.assert_array_equal(self.model.joint("robot0_joint2").axis, [0, 0, -1])
        np.testing.assert_allclose(self.model.dof_damping, 0.8)
        np.testing.assert_allclose(self.model.dof_armature, 0.01)

    def test_actuator_semantics(self):
        m = self.model
        self.assertEqual(m.nu, 7)
        self.assertEqual(self.robot.actuators, [f"robot0_torque_joint{i}" for i in range(1, 7)])
        np.testing.assert_allclose(m.actuator_ctrlrange[:6], [[-27, 27]] * 3 + [[-7, 7]] * 3)
        np.testing.assert_array_equal(m.actuator_trnid[:6, 0], np.arange(6))
        np.testing.assert_allclose(m.actuator_gear[:6], [[1, 0, 0, 0, 0, 0]] * 6)
        np.testing.assert_allclose(m.actuator_gainprm[:6, 0], 1)
        np.testing.assert_allclose(m.actuator_biasprm[:6], 0)
        self.assertTrue(np.all(m.actuator_biastype[:6] == mujoco.mjtBias.mjBIAS_NONE))
        self.assertTrue(np.all(m.actuator_ctrllimited))
        self.assertTrue(np.all(m.actuator_forcelimited))
        gid = m.actuator(self.gripper.actuators[0]).id
        np.testing.assert_allclose(m.actuator_ctrlrange[gid], [0, 0.05])
        np.testing.assert_allclose(m.actuator_forcerange[gid], [-8, 8])
        self.assertEqual(m.actuator_gainprm[gid, 0], 100)
        self.assertEqual(m.actuator_biasprm[gid, 1], -100)

    def test_gripper_coupling_frames_and_contacts(self):
        m, g = self.model, self.gripper
        self.assertEqual(g.joints, ["gripper0_right_finger_left", "gripper0_right_finger_right"])
        np.testing.assert_allclose(m.jnt_range[6:], [[0, 0.05], [-0.05, 0]])
        self.assertEqual(m.neq, 1)
        self.assertEqual(m.eq_type[0], mujoco.mjtEq.mjEQ_JOINT)
        self.assertEqual(m.eq_obj1id[0], m.joint(g.joints[1]).id)
        self.assertEqual(m.eq_obj2id[0], m.joint(g.joints[0]).id)
        np.testing.assert_allclose(m.eq_data[0, :5], [0, -1, 0, 0, 0])
        for name in g.important_sites.values():
            self.assertGreaterEqual(m.site(name).id, 0)
        for name in g.important_sensors.values():
            self.assertGreaterEqual(m.sensor(name).id, 0)
        for names in g.important_geoms.values():
            for name in names:
                geom = m.geom(name)
                self.assertEqual(int(geom.group[0]), 0)
                self.assertEqual(int(geom.contype[0]), 1)
        for i in range(m.ngeom):
            if m.geom(i).name.endswith("_visual"):
                self.assertEqual(m.geom_group[i], 1)
                self.assertEqual(m.geom_contype[i], 0)
                self.assertEqual(m.geom_conaffinity[i], 0)
        data = mujoco.MjData(m)
        data.qpos[:6] = self.robot.init_qpos
        data.qpos[6:] = g.init_qpos
        mujoco.mj_forward(m, data)
        tcp = data.site_xpos[m.site(g.naming_prefix + "tcp").id]
        grasp = data.site_xpos[m.site(g.important_sites["grip_site"]).id]
        np.testing.assert_allclose(grasp, tcp, atol=1e-10)
        # Source FK checkpoint catches a duplicated end_link transform.
        np.testing.assert_allclose(tcp, [0.228411874, 0.000000408, 0.291296125], atol=1e-6)

    def test_gripper_action_mapping(self):
        for policy, expected in [(-1, 1), (0, 0), (1, -1), (2, -1)]:
            np.testing.assert_allclose(self.gripper.format_action([policy]), [expected])
        for invalid in ([float("nan")], [0, 0]):
            with self.assertRaises(ValueError):
                self.gripper.format_action(invalid)

    def test_standalone_assets_compile_without_upstream_paths(self):
        root = PROJECT / "rebot_integration/assets"
        for relative in ("robots/rebot_b601_dm/robot.xml", "grippers/rebot_b601_dm/gripper.xml"):
            path = root / relative
            xml = ET.parse(path).getroot()
            self.assertIsNone(xml.find("default"))
            self.assertEqual(len(xml.findall("./worldbody/body")), 1)
            self.assertFalse(xml.findall(".//camera"))
            for mesh in xml.findall("./asset/mesh"):
                resolved = (path.parent / mesh.get("file")).resolve()
                self.assertTrue(resolved.is_relative_to(root.resolve()))
                self.assertTrue(resolved.is_file())
            mujoco.MjModel.from_xml_path(str(path))

    def test_robosuite_controller_steps_and_reset(self):
        # Exercise the actual BASIC -> JOINT_POSITION / GRIP -> data.ctrl path.
        env = ReBotB601DMEmpty()
        try:
            env.reset()
            robot = env.robots[0]
            self.assertIsInstance(robot, FixedBaseRobot)
            self.assertEqual(env.action_dim, 7)
            controller = robot.part_controllers["right"]
            self.assertEqual(controller.name, "JOINT_POSITION")
            self.assertEqual(controller.impedance_mode, "fixed")
            initial = env.sim.data.qpos[:6].copy()
            action = np.zeros(7)

            def checked_step():
                previous_time = env.sim.data.time
                env.step(action)
                data = env.sim.data
                for values in (data.qpos, data.qvel, data.ctrl):
                    self.assertTrue(np.isfinite(values).all())
                self.assertGreater(data.time, previous_time)
                self.assertLess(np.max(np.abs(data.qvel)), 10)
                self.assertLess(np.max(np.abs(data.qpos[:6] - initial)), 0.1)
                np.testing.assert_allclose(data.qfrc_applied, 0)
                self.assertTrue(np.all(data.ctrl >= env.sim.model.actuator_ctrlrange[:, 0] - 1e-9))
                self.assertTrue(np.all(data.ctrl <= env.sim.model.actuator_ctrlrange[:, 1] + 1e-9))
                self.assertLess(abs(data.qpos[6] + data.qpos[7]), 1e-3)

            for _ in range(20):
                checked_step()
            np.testing.assert_allclose(env.sim.data.qpos[:6], initial, atol=0.01)
            # Gravity support must be supplied through the torque motors.
            self.assertGreater(np.max(np.abs(env.sim.data.ctrl[:6])), 1)
            for joint in range(6):
                before = env.sim.data.qpos[joint]
                action[joint] = 0.2
                for _ in range(6):
                    checked_step()
                self.assertGreater(env.sim.data.qpos[joint], before)
                action[joint] = 0
            for aperture in np.linspace(0, -1, 20):
                action[-1] = aperture
                checked_step()
            for _ in range(10):
                checked_step()
            self.assertGreater(env.sim.data.qpos[6], 0.04)
            for aperture in np.linspace(-1, 1, 40):
                action[-1] = aperture
                checked_step()
            for _ in range(10):
                checked_step()
            self.assertLess(env.sim.data.qpos[6], 0.01)
            env.reset()
            np.testing.assert_allclose(env.sim.data.qpos[:6], initial)
            np.testing.assert_allclose(env.sim.data.qpos[6:], [0.025, -0.025])
            action[:] = 0
            checked_step()
            self.assertAlmostEqual(env.sim.data.ctrl[-1], 0.025)
            np.testing.assert_array_equal(env.sim.data._data.warning.number, 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
