"""Validate continuous ReBot PickPlace carry path.

No dynamics / controller motion is executed.

Checks:
SOURCE LIFT -> TARGET ABOVE

- fixed +90 deg top-down orientation
- sequential 6D IK
- joint-limit margin >= 0.10 rad
- adjacent IK jump <= 0.25 rad
- static robot collision check at each solved configuration
"""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from rebot_integration.guarded_empty_env import BringupStop
from rebot_integration.pickplace_setup import make_rebot_pickplace_can
from rebot_integration.reachability import PositionWorkspace


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
LIFT_HEIGHT = 0.08

MIN_JOINT_MARGIN = 0.10
MAX_WAYPOINT_JUMP = 0.25

# Around 2 cm per Cartesian carry waypoint
CARRY_WAYPOINTS = 26


def grasp_to_tcp(grasp_world):
    return (
        np.asarray(grasp_world, dtype=float)
        - TOP_DOWN_BASE_ROT @ TCP_TO_GRASP
    )


def joint_margin(q, ws):
    q = np.asarray(q, dtype=float)

    return float(
        np.min(
            np.minimum(
                q - ws.lower,
                ws.upper - q,
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

        robot = env.robots[0]

        base = np.asarray(
            robot.robot_model.base_xpos_offset["bins"],
            dtype=float,
        )

        oid = int(env.object_id)
        name = env.obj_names[oid]

        body_id = env.obj_body_id[name]

        source_center = (
            env.sim.data.body_xpos[body_id].copy()
        )

        # Same valid near-side target used by the successful
        # endpoint 6D IK validation.
        target_center = np.array(
            [
                env.bin2_pos[0] + 0.04,
                env.bin2_pos[1] + 0.04,
                source_center[2],
            ],
            dtype=float,
        )

        source_lift = (
            source_center
            + np.array(
                [
                    0.0,
                    0.0,
                    GRASP_Z_OFFSET + LIFT_HEIGHT,
                ]
            )
        )

        target_above = (
            target_center
            + np.array(
                [
                    0.0,
                    0.0,
                    GRASP_Z_OFFSET + LIFT_HEIGHT,
                ]
            )
        )

        ws = PositionWorkspace()

        print(
            "\n========================================"
        )
        print(
            "REBOT PICKPLACE CARRY PATH VALIDATION"
        )
        print(
            "========================================"
        )

        print("base         :", base)
        print("source lift  :", source_lift)
        print("target above :", target_above)


        # --------------------------------------------------
        # Solve source lift to establish the correct branch
        # --------------------------------------------------

        source_tcp = grasp_to_tcp(source_lift)

        source_result = ws.check_pose(
            source_tcp - base,
            TOP_DOWN_BASE_ROT,
            seed=ws.initial,
            random_count=24,
        )

        if not source_result.reachable:
            raise BringupStop(
                "SOURCE LIFT endpoint unreachable"
            )

        previous_q = source_result.qpos.copy()

        source_margin = joint_margin(
            previous_q,
            ws,
        )

        print(
            "\nSOURCE LIFT q =",
            np.round(previous_q, 6),
        )

        print(
            "SOURCE LIFT margin =",
            source_margin,
        )

        if source_margin < MIN_JOINT_MARGIN:
            raise BringupStop(
                "SOURCE LIFT margin below 0.10 rad"
            )


        # --------------------------------------------------
        # Robot geom IDs for static collision checking
        # --------------------------------------------------

        raw_model = env.sim.model._model

        robot_geom_names = (
            list(robot.robot_model.contact_geoms)
            + list(
                robot.gripper["right"].contact_geoms
            )
        )

        robot_geom_ids = set()

        for geom_name in robot_geom_names:
            try:
                robot_geom_ids.add(
                    int(
                        raw_model.geom(geom_name).id
                    )
                )
            except Exception:
                pass

        arm_qpos_ids = np.array(
            [
                env.sim.model.get_joint_qpos_addr(j)
                for j in robot.robot_model.joints
            ],
            dtype=int,
        )

        original_qpos = env.sim.data.qpos.copy()


        def robot_contacts_at(q):

            env.sim.data.qpos[
                arm_qpos_ids
            ] = q

            env.sim.forward()

            contacts = []

            for i in range(
                int(env.sim.data.ncon)
            ):

                c = env.sim.data.contact[i]

                g1 = int(c.geom1)
                g2 = int(c.geom2)

                if (
                    g1 not in robot_geom_ids
                    and g2 not in robot_geom_ids
                ):
                    continue

                try:
                    n1 = raw_model.geom(g1).name
                except Exception:
                    n1 = str(g1)

                try:
                    n2 = raw_model.geom(g2).name
                except Exception:
                    n2 = str(g2)

                b1 = int(raw_model.geom_bodyid[g1])
                b2 = int(raw_model.geom_bodyid[g2])

                try:
                    body1 = raw_model.body(b1).name
                except Exception:
                    body1 = str(b1)

                try:
                    body2 = raw_model.body(b2).name
                except Exception:
                    body2 = str(b2)

                contacts.append(
                    {
                        "geom1_id": g1,
                        "geom1_name": n1,
                        "geom1_body_id": b1,
                        "geom1_body_name": body1,
                        "geom1_xpos": (
                            env.sim.data.geom_xpos[g1]
                            .copy()
                            .tolist()
                        ),
                        "geom1_size": (
                            raw_model.geom_size[g1]
                            .copy()
                            .tolist()
                        ),

                        "geom2_id": g2,
                        "geom2_name": n2,
                        "geom2_body_id": b2,
                        "geom2_body_name": body2,
                        "geom2_xpos": (
                            env.sim.data.geom_xpos[g2]
                            .copy()
                            .tolist()
                        ),
                        "geom2_size": (
                            raw_model.geom_size[g2]
                            .copy()
                            .tolist()
                        ),

                        "contact_pos": (
                            np.asarray(c.pos)
                            .copy()
                            .tolist()
                        ),
                        "distance_m": float(c.dist),
                    }
                )

            return contacts


        # --------------------------------------------------
        # Continuous Cartesian carry
        # --------------------------------------------------

        min_margin_seen = np.inf
        max_jump_seen = 0.0

        path = []

        print(
            "\n=== SOLVE CARRY WAYPOINTS ==="
        )

        for i, alpha in enumerate(
            np.linspace(
                0.0,
                1.0,
                CARRY_WAYPOINTS + 1,
            )[1:],
            start=1,
        ):

            grasp_xyz = (
                (1.0 - alpha) * source_lift
                + alpha * target_above
            )

            tcp_world = grasp_to_tcp(
                grasp_xyz
            )

            result = ws.check_pose(
                tcp_world - base,
                TOP_DOWN_BASE_ROT,
                seed=previous_q,
                random_count=4,
            )

            if not result.reachable:

                raise BringupStop(
                    f"CARRY_{i:02d}: "
                    "6D pose IK unreachable"
                )

            q = result.qpos.copy()

            margin = joint_margin(
                q,
                ws,
            )

            jump = float(
                np.max(
                    np.abs(
                        q - previous_q
                    )
                )
            )

            contacts = robot_contacts_at(q)

            print(
                f"CARRY_{i:02d}: "
                f"xyz={np.round(grasp_xyz, 4)} "
                f"margin={margin:.4f} "
                f"max_dq={jump:.4f} "
                f"contacts={len(contacts)}"
            )

            if margin < MIN_JOINT_MARGIN:
                raise BringupStop(
                    f"CARRY_{i:02d}: "
                    f"joint margin "
                    f"{margin:.6f} < 0.10"
                )

            if jump > MAX_WAYPOINT_JUMP:
                raise BringupStop(
                    f"CARRY_{i:02d}: "
                    f"IK branch jump "
                    f"{jump:.6f} rad"
                )

            if contacts:
                print(
                    "  CONTACTS:",
                    contacts,
                )

                raise BringupStop(
                    f"CARRY_{i:02d}: "
                    "robot collision detected"
                )

            min_margin_seen = min(
                min_margin_seen,
                margin,
            )

            max_jump_seen = max(
                max_jump_seen,
                jump,
            )

            path.append(q)

            previous_q = q


        env.sim.data.qpos[:] = original_qpos
        env.sim.forward()


        print(
            "\n========================================"
        )
        print(
            "CARRY PATH RESULT: PASS"
        )
        print(
            "========================================"
        )

        print(
            "waypoints:",
            len(path),
        )

        print(
            "minimum joint margin:",
            min_margin_seen,
        )

        print(
            "maximum adjacent dq:",
            max_jump_seen,
        )

        print(
            "No controller motion was executed."
        )


    finally:
        env.close()


if __name__ == "__main__":
    main()
