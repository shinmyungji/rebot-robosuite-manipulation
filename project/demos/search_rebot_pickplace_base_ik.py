"""Search a PickPlace ReBot base XY that supports all fixed top-down 6D poses.

No robot motion.
No controller changes.
No joint-limit changes.
"""

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


from rebot_integration.pickplace_setup import (
    make_rebot_pickplace_can,
)
from rebot_integration.reachability import (
    PositionWorkspace,
)


# -------------------------------------------------------------
# Keep successful Lift manipulation geometry unchanged
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

PREGRASP_HEIGHT = 0.10
GRASP_Z_OFFSET = 0.012
LIFT_HEIGHT = 0.12

BASE_Z = 0.8
MIN_MARGIN = 0.10

# ReBot base collision footprint already audited earlier.
BASE_HALF_X = 0.07
BASE_HALF_Y = 0.10

# Conservative XY gap for the initial search.
CLEARANCE_MARGIN = 0.015


def grasp_to_tcp(
    grasp_world,
):
    return (
        np.asarray(
            grasp_world,
            dtype=float,
        )
        - TOP_DOWN_BASE_ROT
        @ TCP_TO_GRASP
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

    env = make_rebot_pickplace_can(
        has_renderer=False,
        seed=7,
    )

    try:

        env.reset()
        env.sim.forward()

        object_id = int(
            env.object_id
        )

        object_name = (
            env.obj_names[
                object_id
            ]
        )

        body_id = (
            env.obj_body_id[
                object_name
            ]
        )

        source = (
            env.sim.data
            .body_xpos[
                body_id
            ]
            .copy()
        )

        stock_target = (
            env.target_bin_placements[
                object_id
            ]
            .copy()
        )

        target_center = np.array(
            [
                stock_target[0],
                stock_target[1],
                source[2],
            ],
            dtype=float,
        )


        poses = {
            "SOURCE_PREGRASP":
                source
                + [0, 0, PREGRASP_HEIGHT],

            "SOURCE_GRASP":
                source
                + [0, 0, GRASP_Z_OFFSET],

            "SOURCE_LIFT":
                source
                + [
                    0,
                    0,
                    GRASP_Z_OFFSET
                    + LIFT_HEIGHT,
                ],

            "TARGET_ABOVE":
                target_center
                + [
                    0,
                    0,
                    GRASP_Z_OFFSET
                    + LIFT_HEIGHT,
                ],

            "TARGET_PLACE":
                target_center
                + [0, 0, GRASP_Z_OFFSET],
        }


        print(
            "source center:",
            np.round(source, 6),
        )

        print(
            "target center:",
            np.round(
                target_center,
                6,
            ),
        )

        print(
            "bin1:",
            env.bin1_pos,
        )

        print(
            "bin2:",
            env.bin2_pos,
        )

        print(
            "bin size:",
            env.bin_size,
        )


        # -----------------------------------------------------
        # Conservative bin / base XY overlap check
        # -----------------------------------------------------

        bin_half_x = (
            float(env.bin_size[0])
            / 2.0
        )

        bin_half_y = (
            float(env.bin_size[1])
            / 2.0
        )


        def base_xy_clear(
            base_xy,
        ):

            for center in (
                np.asarray(
                    env.bin1_pos[:2],
                    dtype=float,
                ),
                np.asarray(
                    env.bin2_pos[:2],
                    dtype=float,
                ),
            ):

                dx = abs(
                    float(
                        base_xy[0]
                        - center[0]
                    )
                )

                dy = abs(
                    float(
                        base_xy[1]
                        - center[1]
                    )
                )

                separated_x = (
                    dx
                    >
                    BASE_HALF_X
                    + bin_half_x
                    + CLEARANCE_MARGIN
                )

                separated_y = (
                    dy
                    >
                    BASE_HALF_Y
                    + bin_half_y
                    + CLEARANCE_MARGIN
                )

                if not (
                    separated_x
                    or separated_y
                ):
                    return False

            return True


        # -----------------------------------------------------
        # Candidate region
        #
        # Includes current base and positions closer to the
        # midpoint between source / destination.
        # -----------------------------------------------------

        # Search farther forward (+X).
        # GUI inspection suggests the arm should be pulled
        # closer to the PickPlace workspace.
        x_candidates = np.arange(
            -0.30,
            0.121,
            0.02,
        )

        # Center Y roughly between source and target.
        y_candidates = np.arange(
            0.00,
            0.181,
            0.02,
        )


        workspace = (
            PositionWorkspace()
        )

        successes = []
        all_kinematic_successes = []


        print(
            "\n=== SEARCH START ==="
        )


        for base_x in x_candidates:

            for base_y in y_candidates:

                base = np.array(
                    [
                        base_x,
                        base_y,
                        BASE_Z,
                    ],
                    dtype=float,
                )

                clear = base_xy_clear(
                    base[:2]
                )

                pose_results = {}
                margins = []

                all_ok = True


                for (
                    label,
                    grasp_world,
                ) in poses.items():

                    tcp_world = (
                        grasp_to_tcp(
                            grasp_world
                        )
                    )

                    tcp_base = (
                        tcp_world
                        - base
                    )

                    result = (
                        workspace.check_pose(
                            tcp_base,
                            TOP_DOWN_BASE_ROT,
                            seed=workspace.initial,
                            random_count=8,
                        )
                    )

                    margin = (
                        joint_margin(
                            result.qpos,
                            workspace,
                        )
                        if result.reachable
                        else -np.inf
                    )

                    pose_results[
                        label
                    ] = {
                        "reachable":
                            bool(
                                result.reachable
                            ),
                        "margin":
                            margin,
                        "pos_mm":
                            float(
                                result.position_error_m
                                * 1000.0
                            ),
                        "rot_deg":
                            float(
                                np.rad2deg(
                                    result.orientation_error_rad
                                )
                            ),
                    }

                    if (
                        not result.reachable
                        or margin
                        < MIN_MARGIN
                    ):
                        all_ok = False

                    margins.append(
                        margin
                    )


                if all_ok:

                    min_margin = float(
                        min(margins)
                    )

                    row = {
                        "base":
                            base.copy(),
                        "clear":
                            clear,
                        "min_margin":
                            min_margin,
                        "poses":
                            pose_results,
                    }

                    all_kinematic_successes.append(
                        row
                    )

                    if clear:
                        successes.append(
                            row
                        )

                    print(
                        "[PASS]",
                        "base=",
                        np.round(
                            base,
                            3,
                        ),
                        "clear=",
                        clear,
                        "min_margin=",
                        f"{min_margin:.3f}",
                    )


        print(
            "\n"
            "========================================"
        )

        print(
            "KINEMATIC PASS COUNT:",
            len(
                all_kinematic_successes
            ),
        )

        print(
            "KINEMATIC + CONSERVATIVE "
            "XY CLEAR COUNT:",
            len(successes),
        )


        if successes:

            successes.sort(
                key=lambda x:
                    x["min_margin"],
                reverse=True,
            )

            print(
                "\n=== BEST CLEAR CANDIDATES ==="
            )

            for row in successes[:10]:

                print(
                    "base =",
                    np.round(
                        row["base"],
                        3,
                    ),
                    "min_margin =",
                    f"{row['min_margin']:.4f}",
                )

                for (
                    label,
                    result,
                ) in row[
                    "poses"
                ].items():

                    print(
                        " ",
                        label,
                        "margin=",
                        f"{result['margin']:.3f}",
                        "pos_mm=",
                        f"{result['pos_mm']:.3f}",
                        "rot_deg=",
                        f"{result['rot_deg']:.3f}",
                    )


        elif all_kinematic_successes:

            all_kinematic_successes.sort(
                key=lambda x:
                    x["min_margin"],
                reverse=True,
            )

            print(
                "\n"
                "6D IK solutions exist, "
                "but current conservative "
                "bin/base AABB check rejects them."
            )

            print(
                "\n=== BEST KINEMATIC CANDIDATES ==="
            )

            for row in (
                all_kinematic_successes[
                    :10
                ]
            ):

                print(
                    "base =",
                    np.round(
                        row["base"],
                        3,
                    ),
                    "clear=",
                    row["clear"],
                    "min_margin=",
                    f"{row['min_margin']:.4f}",
                )


        else:

            print(
                "\n"
                "NO BASE CANDIDATE passed "
                "all five fixed top-down poses."
            )

            print(
                "If this occurs, the next move "
                "is to bring the PickPlace bins "
                "closer together, not to alter "
                "robot joint limits."
            )


    finally:

        env.close()


if __name__ == "__main__":
    main()
