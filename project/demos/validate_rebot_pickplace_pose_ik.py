"""Milestone 4-B: ReBot PickPlaceCan 6D pose-IK validation.

No robot motion is executed.

Checks the same fixed top-down grasp orientation already validated in Lift:

READY
-> SOURCE PREGRASP
-> SOURCE GRASP
-> SOURCE LIFT
-> TARGET ABOVE
-> TARGET PLACE

The physical grasp point is converted to the legacy TCP using the
same audited TCP_TO_GRASP used by the successful Lift expert.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parents[1]
    ),
)


from rebot_integration.guarded_empty_env import BringupStop
from rebot_integration.pickplace_setup import make_rebot_pickplace_can
from rebot_integration.reachability import PositionWorkspace


# -------------------------------------------------------------
# Keep these identical to the successful Lift expert.
# -------------------------------------------------------------

TCP_TO_GRASP = np.array(
    [0.090, 0.0, 0.0],
    dtype=float,
)

TOP_DOWN_BASE_ROT = (
    Rotation.from_euler(
        "y",
        np.pi / 2.0,
        degrees=False,
    )
    .as_matrix()
)

GRASP_Z_OFFSET = 0.012

PREGRASP_HEIGHT = 0.10

LIFT_HEIGHT = 0.08

READY_GRASP_WORLD = np.array(
    [0.01, 0.0, 0.95],
    dtype=float,
)

MIN_JOINT_MARGIN = 0.10

MAX_ENDPOINT_JUMP = 1.0


def grasp_point_to_tcp_target(
    grasp_point_world,
    target_rotation,
):
    grasp_point_world = np.asarray(
        grasp_point_world,
        dtype=float,
    )

    target_rotation = np.asarray(
        target_rotation,
        dtype=float,
    )

    return (
        grasp_point_world
        - target_rotation @ TCP_TO_GRASP
    )


def joint_margin(
    q,
    workspace,
):
    q = np.asarray(
        q,
        dtype=float,
    )

    return float(
        np.min(
            np.minimum(
                q - workspace.lower,
                workspace.upper - q,
            )
        )
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--ik-random-starts",
        type=int,
        default=24,
    )

    args = parser.parse_args()


    env = make_rebot_pickplace_can(
        has_renderer=False,
        seed=args.seed,
    )

    try:

        env.reset()
        env.sim.forward()

        robot = env.robots[0]

        base_position = np.asarray(
            robot.robot_model
            .base_xpos_offset["bins"],
            dtype=float,
        )


        # -----------------------------------------------------
        # Active Can
        # -----------------------------------------------------

        object_id = int(
            env.object_id
        )

        object_name = (
            env.obj_names[
                object_id
            ]
        )

        object_body_id = (
            env.obj_body_id[
                object_name
            ]
        )

        source_center = (
            env.sim.data
            .body_xpos[
                object_body_id
            ]
            .copy()
        )


        # -----------------------------------------------------
        # Stock target XY
        # -----------------------------------------------------

        stock_target = (
            env.target_bin_placements[
                object_id
            ]
            .copy()
        )


        # The reset Can center tells us its standing center
        # height above the bin/table plane.
        #
        # We preserve that same center height at placement.
        # PickPlace target_bin_placements is the CENTER of the
        # object's assigned goal quadrant, not the only successful point.
        #
        # Can is object_id=3, whose valid quadrant begins at bin2_pos.
        # Choose a nearer point safely inside that quadrant.
        target_center = np.array(
            [
                env.bin2_pos[0] + 0.04,
                env.bin2_pos[1] + 0.04,
                source_center[2],
            ],
            dtype=float,
        )


        # -----------------------------------------------------
        # Physical grasp-point targets
        # -----------------------------------------------------

        source_pregrasp = (
            source_center
            + np.array(
                [0.0, 0.0, PREGRASP_HEIGHT]
            )
        )

        source_grasp = (
            source_center
            + np.array(
                [0.0, 0.0, GRASP_Z_OFFSET]
            )
        )

        source_lift = (
            source_grasp
            + np.array(
                [0.0, 0.0, LIFT_HEIGHT]
            )
        )

        target_place = (
            target_center
            + np.array(
                [0.0, 0.0, GRASP_Z_OFFSET]
            )
        )

        target_above = (
            target_place
            + np.array(
                [0.0, 0.0, LIFT_HEIGHT]
            )
        )


        print(
            "\n"
            "========================================\n"
            "REBOT PICKPLACE 6D POSE-IK VALIDATION\n"
            "========================================",
            flush=True,
        )

        print(
            "base xyz:",
            base_position,
            flush=True,
        )

        print(
            "object:",
            object_name,
            flush=True,
        )

        print(
            "source center:",
            source_center,
            flush=True,
        )

        print(
            "target center:",
            target_center,
            flush=True,
        )


        workspace = PositionWorkspace()


        def solve_pose(
            label,
            physical_grasp_point,
            seed=None,
            random_count=0,
        ):

            physical_grasp_point = (
                np.asarray(
                    physical_grasp_point,
                    dtype=float,
                )
            )

            tcp_world = (
                grasp_point_to_tcp_target(
                    physical_grasp_point,
                    TOP_DOWN_BASE_ROT,
                )
            )

            tcp_base = (
                tcp_world
                - base_position
            )

            result = (
                workspace.check_pose(
                    tcp_base,
                    TOP_DOWN_BASE_ROT,
                    seed=seed,
                    random_count=random_count,
                )
            )

            print(
                f"\n=== {label} ===",
                flush=True,
            )

            print(
                "physical grasp xyz:",
                np.round(
                    physical_grasp_point,
                    6,
                ),
                flush=True,
            )

            print(
                "TCP world xyz:",
                np.round(
                    tcp_world,
                    6,
                ),
                flush=True,
            )

            print(
                "TCP base xyz:",
                np.round(
                    tcp_base,
                    6,
                ),
                flush=True,
            )

            print(
                "reachable:",
                bool(
                    result.reachable
                ),
                flush=True,
            )

            print(
                "position error mm:",
                float(
                    result.position_error_m
                    * 1000.0
                ),
                flush=True,
            )

            print(
                "orientation error deg:",
                float(
                    np.rad2deg(
                        result.orientation_error_rad
                    )
                ),
                flush=True,
            )

            print(
                "q:",
                np.round(
                    result.qpos,
                    6,
                ),
                flush=True,
            )

            if not result.reachable:
                raise BringupStop(
                    f"{label}: 6D pose IK unreachable"
                )

            margin = joint_margin(
                result.qpos,
                workspace,
            )

            print(
                "minimum joint-limit margin rad:",
                margin,
                flush=True,
            )

            if margin < MIN_JOINT_MARGIN:
                print(
                    f"[WARN] {label}: joint-limit margin "
                    f"{margin:.6f} < "
                    f"{MIN_JOINT_MARGIN:.3f} rad",
                    flush=True,
                )

            if seed is not None:

                jump = float(
                    np.max(
                        np.abs(
                            result.qpos
                            - np.asarray(
                                seed,
                                dtype=float,
                            )
                        )
                    )
                )

                print(
                    "max dq from seed:",
                    jump,
                    "rad",
                    flush=True,
                )

            return (
                result.qpos.copy()
            )


        # -----------------------------------------------------
        # READY
        #
        # Use random fallback only here to establish the
        # known top-down branch.
        # -----------------------------------------------------

        q_ready = solve_pose(
            "READY",
            READY_GRASP_WORLD,
            seed=workspace.initial,
            random_count=args.ik_random_starts,
        )


        # -----------------------------------------------------
        # Source side
        # -----------------------------------------------------

        q_source_pregrasp = solve_pose(
            "SOURCE PREGRASP",
            source_pregrasp,
            seed=q_ready,
            random_count=4,
        )

        q_source_grasp = solve_pose(
            "SOURCE GRASP",
            source_grasp,
            seed=q_source_pregrasp,
            random_count=4,
        )

        q_source_lift = solve_pose(
            "SOURCE LIFT",
            source_lift,
            seed=q_source_grasp,
            random_count=4,
        )


        # -----------------------------------------------------
        # Target side
        #
        # Endpoint solve only.
        # We are NOT assuming a direct joint interpolation from
        # source lift to target above.
        # The real expert will later solve Cartesian carry
        # waypoints continuously.
        # -----------------------------------------------------

        q_target_above = solve_pose(
            "TARGET ABOVE",
            target_above,
            seed=q_source_lift,
            random_count=args.ik_random_starts,
        )

        q_target_place = solve_pose(
            "TARGET PLACE",
            target_place,
            seed=q_target_above,
            random_count=4,
        )


        print(
            "\n"
            "========================================\n"
            "6D POSE-IK SUMMARY\n"
            "========================================",
            flush=True,
        )

        print(
            "READY             : PASS",
            flush=True,
        )

        print(
            "SOURCE PREGRASP   : PASS",
            flush=True,
        )

        print(
            "SOURCE GRASP      : PASS",
            flush=True,
        )

        print(
            "SOURCE LIFT       : PASS",
            flush=True,
        )

        print(
            "TARGET ABOVE      : PASS",
            flush=True,
        )

        print(
            "TARGET PLACE      : PASS",
            flush=True,
        )

        print(
            "\nNo robot motion was executed.",
            flush=True,
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
