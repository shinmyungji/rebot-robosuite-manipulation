"""Project configuration for stock robosuite PickPlaceCan.

No PickPlace task logic is modified here.
The ReBot model's existing `bins` base offset is used by stock PickPlace.
"""

from robosuite.environments.manipulation.pick_place import PickPlaceCan

from .empty_env import load_controller_config


def make_rebot_pickplace_can(
    *,
    has_renderer=False,
    seed=7,
    env_class=PickPlaceCan,
):
    return env_class(
        robots="ReBotB601DM",
        controller_configs=load_controller_config(),

        # Keep reset deterministic while bringing up PickPlace.
        initialization_noise=None,

        # No policy cameras yet. First validate geometry / reachability.
        use_camera_obs=False,
        has_offscreen_renderer=False,

        has_renderer=has_renderer,
        render_camera=None,

        # Same bring-up convention as Lift.
        hard_reset=False,
        control_freq=20,
        ignore_done=True,
        reward_shaping=False,
        seed=seed,
    )
