"""Seeed reBot B601-DM, using robosuite's torque-output arm controllers."""

from pathlib import Path

import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import register_robot_class


@register_robot_class("FixedBaseRobot")
class ReBotB601DM(ManipulatorModel):
    arms = ["right"]

    def __init__(self, idn=0):
        path = Path(__file__).parent / "assets/robots/rebot_b601_dm/robot.xml"
        super().__init__(str(path), idn=idn)

    @property
    def default_base(self):
        return "NullMount"

    @property
    def default_gripper(self):
        return {"right": "ReBotB601DMGripper"}

    @property
    def default_controller_config(self):
        # Informational model metadata. Robosuite 1.5's runtime must receive
        # the external composite config explicitly; empty_env.py does this.
        return {"right": "rebot_b601_dm_joint_position"}

    @property
    def init_qpos(self):
        # DM pick_place.yaml ready_point, in joint1 ... joint6 order.
        return np.array([0.0, -0.75, -0.55, 0.0, 0.0, 0.0])

    @property
    def base_xpos_offset(self):
        # Stock Lift uses a 0.8 m tabletop. The base collision mesh spans
        # x=+/-0.07, y=+/-0.10 and starts at z=0 (roundoff < 1e-10).
        # A 0.10 m inset leaves a 0.03 m margin at the table edge.
        return {
            "empty": (0.0, 0.0, 0.0),
            "table": lambda length: (-length / 2 + 0.10, 0.0, 0.8),
            "bins": (-0.18, 0.06, 0.8),
        }

    @property
    def top_offset(self):
        return np.array([0.0, 0.0, 0.8])

    @property
    def _horizontal_radius(self):
        return 0.75

    @property
    def arm_type(self):
        return "single"
