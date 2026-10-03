"""Project configuration for stock robosuite Lift; no task logic changes."""

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.utils.errors import RandomizationError
from robosuite.utils.placement_samplers import UniformRandomSampler

from .empty_env import load_controller_config
from .reachability import PositionWorkspace


TABLE_SIZE = (0.8, 0.8, 0.05)
TABLE_TOP = 0.8
BASE_POSITION = np.array([-0.30, 0.0, TABLE_TOP])
# Cube CENTER bounds, in world coordinates, for the stock table only.
CUBE_X_RANGE = (-0.02, 0.04)
CUBE_Y_RANGE = (-0.03, 0.03)


class ReachableCubeSampler(UniformRandomSampler):
    """Bounded placement, rejecting samples without a TCP-position witness."""

    def __init__(self, seed=None):
        super().__init__(
            name="ReBotCubeSampler", x_range=CUBE_X_RANGE, y_range=CUBE_Y_RANGE,
            rotation=0.0, ensure_object_boundary_in_range=False,
            ensure_valid_placement=True, reference_pos=(0, 0, TABLE_TOP),
            z_offset=0.01, rng=np.random.default_rng(seed),
        )
        self.workspace = PositionWorkspace()
        self.last_checks = None

    def sample(self, fixtures=None, reference=None, on_top=True):
        for _ in range(30):
            placements = super().sample(fixtures=fixtures, reference=reference, on_top=on_top)
            pos, _, _ = placements["cube"]
            target = np.asarray(pos) - BASE_POSITION
            # Check both the stock 1 cm spawn drop and the settled center height.
            initial = self.workspace.check(target)
            settled = self.workspace.check(target - [0, 0, self.z_offset])
            if initial.reachable and settled.reachable:
                self.last_checks = (initial, settled)
                return placements
        raise RandomizationError("No kinematically reachable ReBot cube placement found")


def make_rebot_lift(*, has_renderer=False, seed=7, env_class=Lift):
    """Use Lift directly, or a safety-only subclass with unchanged task methods."""
    return env_class(
        robots="ReBotB601DM", controller_configs=load_controller_config(),
        table_full_size=TABLE_SIZE, placement_initializer=ReachableCubeSampler(seed),
        initialization_noise=None, use_camera_obs=False, has_offscreen_renderer=False,
        has_renderer=has_renderer, render_camera=None, hard_reset=False,
        control_freq=20, ignore_done=True, reward_shaping=False, seed=seed,
    )
