"""One policy action and actuator for the DM's coupled slide fingers."""

from pathlib import Path

import numpy as np
from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.gripper_model import GripperModel


@register_gripper
class ReBotB601DMGripper(GripperModel):
    def __init__(self, idn=0):
        path = Path(__file__).parent / "assets/grippers/rebot_b601_dm/gripper.xml"
        super().__init__(str(path), idn=idn)

    @property
    def dof(self):
        return 1

    @property
    def init_qpos(self):
        # Zero policy action maps to this half-open position, including reset.
        return np.array([0.025, -0.025])

    def format_action(self, action):
        """Absolute aperture: -1 opens, +1 closes; zero is half open.

        GRIP scales this normalized output to the position actuator's 0..0.05 m
        range. No integration state means soft resets cannot retain old goals.
        """
        action = np.asarray(action, dtype=float)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("Expected one finite gripper action")
        self.current_action = -np.clip(action, -1.0, 1.0)
        return self.current_action.copy()

    @property
    def _important_geoms(self):
        return {
            "left_finger": [f"finger_left_{part}_collision" for part in
                            ("front", "mid", "rear", "carriage", "travel_stop")],
            "right_finger": [f"finger_right_{part}_collision" for part in
                             ("front", "mid", "rear", "carriage", "travel_stop")],
            "left_fingerpad": ["finger_left_front_collision", "finger_left_mid_collision"],
            "right_fingerpad": ["finger_right_front_collision", "finger_right_mid_collision"],
        }
