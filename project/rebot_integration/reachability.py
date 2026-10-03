"""Scratch FK / IK utilities for reBot B601-DM.

This module never controls the live task robot.

It builds an independent MuJoCo model and provides:

1. Position-only IK
   - kept for workspace / reachability tests

2. 6D pose IK
   - TCP position + orientation
   - used by the scripted manipulation expert

Pose IK supports deterministic multi-start so that failure from one
initial joint configuration is not mistaken for true unreachability.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import ReBotB601DM, ReBotB601DMGripper


@dataclass
class ReachabilityResult:
    reachable: bool
    error_m: float
    qpos: np.ndarray


@dataclass
class PoseReachabilityResult:
    reachable: bool
    position_error_m: float
    orientation_error_rad: float
    qpos: np.ndarray


class PositionWorkspace:
    """Independent scratch IK model in the robot base frame."""

    def __init__(self):
        robot = ReBotB601DM()

        gripper = ReBotB601DMGripper(
            idn="workspace"
        )

        robot.add_gripper(
            gripper,
            robot.eef_name["right"],
        )

        self.model = robot.get_model()
        self.data = mujoco.MjData(self.model)

        # Arm qpos occupies the first six positions.
        # Remaining positions are the gripper.
        self.data.qpos[6:] = gripper.init_qpos

        self.site = self.model.site(
            gripper.important_sites["grip_site"]
        ).id

        self.initial = robot.init_qpos.copy()

        self.lower, self.upper = (
            self.model.jnt_range[:6].T
        )

        self.position_jac = np.zeros(
            (3, self.model.nv)
        )

    # ---------------------------------------------------------
    # Basic scratch-model helpers
    # ---------------------------------------------------------

    def _forward(self, q):
        q = np.asarray(
            q,
            dtype=float,
        )

        self.data.qpos[:6] = q

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    def _seeds(
        self,
        seed=None,
        random_count=0,
    ):
        """Return deterministic IK starting configurations.

        Order:
        1. caller supplied q
        2. known hand-written configurations
        3. deterministic random configurations

        Random seeds are deterministic so debugging is reproducible.
        """

        seeds = []

        if seed is not None:
            seed = np.asarray(
                seed,
                dtype=float,
            )

            if (
                seed.shape == (6,)
                and np.isfinite(seed).all()
            ):
                seeds.append(
                    seed.copy()
                )

        seeds.extend(
            [
                self.initial.copy(),

                np.array(
                    [
                        0.0,
                        -1.3,
                        -1.3,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dtype=float,
                ),

                np.array(
                    [
                        0.0,
                        -0.3,
                        -2.3,
                        0.8,
                        0.5,
                        0.0,
                    ],
                    dtype=float,
                ),

                np.array(
                    [
                        0.0,
                        -2.0,
                        -0.7,
                        -0.5,
                        0.0,
                        0.0,
                    ],
                    dtype=float,
                ),
            ]
        )

        random_count = int(
            max(0, random_count)
        )

        if random_count > 0:
            rng = np.random.default_rng(
                12345
            )

            lo = (
                self.lower
                + 1e-4
            )

            hi = (
                self.upper
                - 1e-4
            )

            for _ in range(
                random_count
            ):
                seeds.append(
                    rng.uniform(
                        lo,
                        hi,
                    )
                )

        return seeds

    # ---------------------------------------------------------
    # Position-only IK
    # ---------------------------------------------------------

    def check(
        self,
        position,
        seed=None,
    ):
        """Find q that places TCP at target xyz.

        This is position-only reachability.

        It intentionally does not constrain orientation and should not
        be used for actual grasp execution.
        """

        target = np.asarray(
            position,
            dtype=float,
        )

        if (
            target.shape != (3,)
            or not np.isfinite(target).all()
            or np.linalg.norm(target) > 0.8
        ):
            return ReachabilityResult(
                False,
                float("inf"),
                self.initial.copy(),
            )

        def residual(q):
            self._forward(q)

            return (
                self.data.site_xpos[
                    self.site
                ]
                - target
            )

        def jacobian(q):
            residual(q)

            mujoco.mj_jacSite(
                self.model,
                self.data,
                self.position_jac,
                None,
                self.site,
            )

            return (
                self.position_jac[
                    :, :6
                ]
                .copy()
            )

        best = ReachabilityResult(
            False,
            float("inf"),
            self.initial.copy(),
        )

        for candidate_seed in (
            self._seeds(seed)
        ):
            candidate_seed = np.clip(
                np.asarray(
                    candidate_seed,
                    dtype=float,
                ),
                self.lower + 1e-5,
                self.upper - 1e-5,
            )

            result = least_squares(
                residual,
                candidate_seed,
                jac=jacobian,
                bounds=(
                    self.lower,
                    self.upper,
                ),
                max_nfev=200,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )

            error = float(
                np.linalg.norm(
                    residual(result.x)
                )
            )

            candidate = (
                ReachabilityResult(
                    reachable=(
                        error < 0.0005
                    ),
                    error_m=error,
                    qpos=(
                        result.x.copy()
                    ),
                )
            )

            if (
                candidate.error_m
                < best.error_m
            ):
                best = candidate

            # Caller supplied seed is first.
            # Prefer a nearby valid branch.
            if candidate.reachable:
                return candidate

        return best

    # ---------------------------------------------------------
    # Position + orientation IK
    # ---------------------------------------------------------

    def check_pose(
        self,
        position,
        target_rotation,
        seed=None,
        random_count=0,
    ):
        """Solve a full TCP pose.

        Parameters
        ----------
        position:
            Desired TCP xyz in robot-base coordinates.

        target_rotation:
            Desired 3x3 TCP rotation matrix in robot-base coordinates.

        seed:
            Preferred starting joint configuration.
            For trajectory waypoints this should normally be the
            previous waypoint q.

        random_count:
            Number of additional deterministic random IK starts.
        """

        target_position = np.asarray(
            position,
            dtype=float,
        )

        target_rotation = np.asarray(
            target_rotation,
            dtype=float,
        )

        if (
            target_position.shape != (3,)
            or target_rotation.shape
            != (3, 3)
            or not np.isfinite(
                target_position
            ).all()
            or not np.isfinite(
                target_rotation
            ).all()
            or np.linalg.norm(
                target_position
            ) > 0.8
        ):
            return PoseReachabilityResult(
                False,
                float("inf"),
                float("inf"),
                self.initial.copy(),
            )

        def pose_error(q):
            self._forward(q)

            current_position = (
                self.data.site_xpos[
                    self.site
                ]
                .copy()
            )

            current_rotation = (
                self.data.site_xmat[
                    self.site
                ]
                .reshape(3, 3)
                .copy()
            )

            position_error = (
                current_position
                - target_position
            )

            # Relative rotation from current TCP frame
            # to desired TCP frame.
            rotation_delta = (
                target_rotation
                @ current_rotation.T
            )

            orientation_error = (
                Rotation.from_matrix(
                    rotation_delta
                ).as_rotvec()
            )

            return (
                position_error,
                orientation_error,
            )

        def residual(q):
            (
                position_error,
                orientation_error,
            ) = pose_error(q)

            # Position and orientation have different units.
            #
            # 5 mm position error contributes roughly the same
            # residual magnitude as 5 degrees orientation error.
            return np.concatenate(
                [
                    position_error
                    / 0.005,

                    orientation_error
                    / np.deg2rad(5.0),
                ]
            )

        best = PoseReachabilityResult(
            False,
            float("inf"),
            float("inf"),
            self.initial.copy(),
        )

        best_score = float("inf")

        for candidate_seed in (
            self._seeds(
                seed=seed,
                random_count=random_count,
            )
        ):
            candidate_seed = np.clip(
                np.asarray(
                    candidate_seed,
                    dtype=float,
                ),
                self.lower + 1e-5,
                self.upper - 1e-5,
            )

            result = least_squares(
                residual,
                candidate_seed,
                bounds=(
                    self.lower,
                    self.upper,
                ),
                max_nfev=500,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )

            (
                position_error,
                orientation_error,
            ) = pose_error(
                result.x
            )

            position_error_m = float(
                np.linalg.norm(
                    position_error
                )
            )

            orientation_error_rad = (
                float(
                    np.linalg.norm(
                        orientation_error
                    )
                )
            )

            reachable = (
                position_error_m
                < 0.001
                and
                orientation_error_rad
                < np.deg2rad(3.0)
            )

            # Used only to retain the best failed attempt.
            score = (
                position_error_m
                + 0.05
                * orientation_error_rad
            )

            candidate = (
                PoseReachabilityResult(
                    reachable=reachable,
                    position_error_m=(
                        position_error_m
                    ),
                    orientation_error_rad=(
                        orientation_error_rad
                    ),
                    qpos=(
                        result.x.copy()
                    ),
                )
            )

            if score < best_score:
                best_score = score
                best = candidate

            # Seeds are ordered with caller-provided q first.
            # Return the first genuinely valid solution.
            if reachable:
                return candidate

        return best