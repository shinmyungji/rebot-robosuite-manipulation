"""Minimal robosuite environment: one fixed robot and an empty floor."""

from pathlib import Path

import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask


def load_controller_config():
    path = Path(__file__).resolve().parents[1] / "configs/rebot_b601_dm_joint_position.json"
    return load_composite_controller_config(controller=str(path))


class ReBotB601DMEmpty(ManipulationEnv):
    """No objects, task reward, camera observations, or table integration."""

    def __init__(self, *, has_renderer=False):
        super().__init__(
            robots="ReBotB601DM",
            controller_configs=load_controller_config(),
            initialization_noise=None,
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            render_camera=None,
            control_freq=20,
            hard_reset=False,
            ignore_done=True,
        )

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0].robot_model
        robot.set_base_xpos(np.array(robot.base_xpos_offset["empty"]))
        self.model = ManipulationTask(
            mujoco_arena=EmptyArena(), mujoco_robots=[robot], mujoco_objects=[]
        )
        # Retain the source contact solver settings. Robosuite sets timestep
        # from its own simulation macro (normally 0.002 s).
        self.model.root.find("option").set("iterations", "100")
        self.model.root.find("option").set("noslip_iterations", "20")

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False
