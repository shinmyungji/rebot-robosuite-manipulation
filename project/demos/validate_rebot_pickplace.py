"""Milestone 4-A: static ReBot PickPlaceCan validation.

This does NOT execute a pick-and-place trajectory.

Checks:
- stock PickPlaceCan loads with ReBot
- ReBot `bins` base placement
- active can location
- stock target-bin location
- basic position workspace witnesses
- unexpected robot contacts at reset
- optional static GUI visualization
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np


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


def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--view-seconds",
        type=float,
        default=12.0,
    )

    args = parser.parse_args()


    env = make_rebot_pickplace_can(
        has_renderer=args.render,
        seed=args.seed,
    )

    try:

        # ------------------------------------------------------
        # Reset
        # ------------------------------------------------------

        env.reset()
        env.sim.forward()

        robot = env.robots[0]

        print(
            "\n"
            "========================================\n"
            "REBOT PICKPLACE STATIC VALIDATION\n"
            "========================================",
            flush=True,
        )


        # ------------------------------------------------------
        # Robot base
        # ------------------------------------------------------

        base_position = np.asarray(
            robot.robot_model
            .base_xpos_offset["bins"],
            dtype=float,
        )

        print(
            "\n=== ROBOT ===",
            flush=True,
        )

        print(
            "configured bins base xyz:",
            base_position,
            flush=True,
        )

        print(
            "init qpos:",
            robot.robot_model.init_qpos,
            flush=True,
        )


        # ------------------------------------------------------
        # PickPlace arena
        # ------------------------------------------------------

        print(
            "\n=== BINS ===",
            flush=True,
        )

        print(
            "bin1_pos:",
            np.asarray(env.bin1_pos),
            flush=True,
        )

        print(
            "bin2_pos:",
            np.asarray(env.bin2_pos),
            flush=True,
        )

        print(
            "bin_size:",
            np.asarray(env.bin_size),
            flush=True,
        )


        # ------------------------------------------------------
        # Active object
        #
        # PickPlaceCan uses single_object_mode=2.
        # object_id therefore selects the fixed Can entry.
        # ------------------------------------------------------

        object_id = int(
            env.object_id
        )

        active_object_name = (
            env.obj_names[
                object_id
            ]
        )

        active_body_id = (
            env.obj_body_id[
                active_object_name
            ]
        )

        object_position = (
            env.sim.data
            .body_xpos[
                active_body_id
            ]
            .copy()
        )

        target_position = (
            env.target_bin_placements[
                object_id
            ]
            .copy()
        )

        print(
            "\n=== ACTIVE OBJECT ===",
            flush=True,
        )

        print(
            "object_id:",
            object_id,
            flush=True,
        )

        print(
            "object_name:",
            active_object_name,
            flush=True,
        )

        print(
            "object world xyz:",
            object_position,
            flush=True,
        )

        print(
            "stock target bin xyz:",
            target_position,
            flush=True,
        )


        # ------------------------------------------------------
        # Position-only reachability witnesses.
        #
        # This is NOT yet the final grasp IK.
        # We only test whether the basic workspace covers
        # source and destination neighborhoods.
        # ------------------------------------------------------

        workspace = (
            PositionWorkspace()
        )

        source_center = (
            object_position.copy()
        )

        source_above = (
            object_position.copy()
        )

        source_above[2] += 0.10


        # Use the actual source object's settled center height
        # as a reasonable destination-center witness.
        target_center = np.array(
            [
                target_position[0],
                target_position[1],
                object_position[2],
            ],
            dtype=float,
        )

        target_above = (
            target_center.copy()
        )

        target_above[2] += 0.10


        def report_workspace(
            name,
            world_xyz,
        ):

            relative_xyz = (
                np.asarray(
                    world_xyz,
                    dtype=float,
                )
                - base_position
            )

            result = (
                workspace.check(
                    relative_xyz
                )
            )

            radial_xy = float(
                np.linalg.norm(
                    relative_xyz[:2]
                )
            )

            print(
                f"{name}:",
                flush=True,
            )

            print(
                "  world xyz    =",
                np.round(
                    world_xyz,
                    6,
                ),
                flush=True,
            )

            print(
                "  relative xyz =",
                np.round(
                    relative_xyz,
                    6,
                ),
                flush=True,
            )

            print(
                "  radial xy    =",
                f"{radial_xy:.4f} m",
                flush=True,
            )

            print(
                "  reachable    =",
                bool(
                    result.reachable
                ),
                flush=True,
            )

            print(
                "  workspace    =",
                result,
                flush=True,
            )

            return bool(
                result.reachable
            )


        print(
            "\n=== POSITION WORKSPACE ===",
            flush=True,
        )

        source_center_ok = (
            report_workspace(
                "SOURCE CENTER",
                source_center,
            )
        )

        source_above_ok = (
            report_workspace(
                "SOURCE ABOVE +10cm",
                source_above,
            )
        )

        target_center_ok = (
            report_workspace(
                "TARGET CENTER WITNESS",
                target_center,
            )
        )

        target_above_ok = (
            report_workspace(
                "TARGET ABOVE +10cm",
                target_above,
            )
        )


        # ------------------------------------------------------
        # Reset contacts involving robot / gripper
        # ------------------------------------------------------

        raw_model = (
            env.sim.model._model
        )

        robot_geom_names = (
            list(
                robot.robot_model
                .contact_geoms
            )
            + list(
                robot.gripper["right"]
                .contact_geoms
            )
        )

        robot_geom_ids = set()

        for name in robot_geom_names:

            try:
                robot_geom_ids.add(
                    int(
                        raw_model
                        .geom(name)
                        .id
                    )
                )
            except Exception:
                pass


        contacts = []

        for i in range(
            int(
                env.sim.data.ncon
            )
        ):

            contact = (
                env.sim.data
                .contact[i]
            )

            g1 = int(
                contact.geom1
            )

            g2 = int(
                contact.geom2
            )

            if (
                g1 not in robot_geom_ids
                and
                g2 not in robot_geom_ids
            ):
                continue

            try:
                n1 = (
                    raw_model
                    .geom(g1)
                    .name
                )
            except Exception:
                n1 = str(g1)

            try:
                n2 = (
                    raw_model
                    .geom(g2)
                    .name
                )
            except Exception:
                n2 = str(g2)

            contacts.append(
                {
                    "geom1": n1,
                    "geom2": n2,
                    "distance_m": float(
                        contact.dist
                    ),
                }
            )


        print(
            "\n=== ROBOT CONTACTS AT RESET ===",
            flush=True,
        )

        if contacts:
            for contact in contacts:
                print(
                    contact,
                    flush=True,
                )
        else:
            print(
                "none",
                flush=True,
            )


        # ------------------------------------------------------
        # Summary
        # ------------------------------------------------------

        print(
            "\n=== STATIC VALIDATION SUMMARY ===",
            flush=True,
        )

        print(
            "source_center_reachable:",
            source_center_ok,
            flush=True,
        )

        print(
            "source_above_reachable:",
            source_above_ok,
            flush=True,
        )

        print(
            "target_center_reachable:",
            target_center_ok,
            flush=True,
        )

        print(
            "target_above_reachable:",
            target_above_ok,
            flush=True,
        )

        print(
            "robot_contacts_at_reset:",
            len(contacts),
            flush=True,
        )

        print(
            "stock_success_at_reset:",
            bool(
                env._check_success()
            ),
            flush=True,
        )


        # ------------------------------------------------------
        # Static visualization only.
        #
        # Do not step the controller yet. We only want to inspect
        # the stock PickPlace geometry.
        # ------------------------------------------------------

        if args.render:

            env.viewer.update()

            with (
                env.viewer
                .viewer
                .lock()
            ):

                env.viewer.viewer.cam.lookat[:] = [
                    0.0,
                    0.0,
                    0.85,
                ]

                env.viewer.viewer.cam.distance = (
                    2.2
                )

                env.viewer.viewer.cam.azimuth = (
                    135
                )

                env.viewer.viewer.cam.elevation = (
                    -30
                )

            start = time.monotonic()

            print(
                "\n"
                "GUI open for "
                f"{args.view_seconds:.1f}s...",
                flush=True,
            )

            while (
                time.monotonic()
                - start
                < args.view_seconds
            ):

                if not (
                    env.viewer
                    .viewer
                    .is_running()
                ):
                    break

                env.viewer.viewer.sync()

                time.sleep(
                    1.0 / 60.0
                )

    finally:

        env.close()


if __name__ == "__main__":
    main()
