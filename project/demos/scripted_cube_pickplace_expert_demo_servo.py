# ============================================================
# AUTO-GENERATED DEMO-SERVO VARIANT
#
# Original scripted_lift_expert.py is NOT modified here.
#
# Arm:
#   absolute position servo
#   60-Hz command trajectory
#   max command speed 0.75 rad/s
#
# Gains:
#   joints 1-3: kp=300 kv=5
#   joints 4-6: kp=150 kv=2
#
# IMPORTANT:
#   ReBot original actuator force limits are preserved.
# ============================================================

"""Milestone 3: scripted reBot simple cube pick-and-place expert.

Uses simulator GT cube position only for scripted expert trajectory
generation. The learned imitation policy will NOT receive GT cube
position.

Pipeline:
1. Validate a fixed top-down READY pose, then reset this expert instance there.
2. Open gripper.
3. Move horizontally at READY height using fixed-orientation Cartesian waypoints.
4. Descend vertically to pregrasp using the same orientation.
5. Descend while keeping the selected orientation fixed.
6. Close gripper.
7. Lift while keeping the selected orientation fixed.
8. Evaluate stock robosuite Lift success.

Important:
- Grasp execution uses 6D TCP pose IK.
- Position-only IK is not used for manipulation.
- Previous waypoint q is used as the next IK seed.
- Physical grasp targets are converted to the legacy TCP before pose IK.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import PchipInterpolator
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from robosuite.utils.binding_utils import MjSim

sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parents[1]
    ),
)

from rebot_integration.descent_validation import DescentMonitor, ReBotDescentValidation

from rebot_integration.guarded_empty_env import (
    BringupStop,
)

from rebot_integration.lift_setup import (
    BASE_POSITION,
    make_rebot_lift,
)

from rebot_integration.lift_validation import (
    LiftDiagnostics,
    ReBotLiftValidation,
    snapshot,
)


# -------------------------------------------------------------
# Controller / gripper constants
# -------------------------------------------------------------

# Must match controller output range.
JOINT_ACTION_SCALE = 0.08

# ============================================================
# DEMO SERVO MODE
# Discoverse-style command trajectory
# ============================================================

DEMO_SERVO_HZ = 60.0
DEMO_SERVO_DT = 1.0 / DEMO_SERVO_HZ

# Same commanded joint trajectory speed used in the
# previous Discoverse Piper demonstration generator.
DEMO_MOVE_SPEED_RAD_S = 0.75

# Piper-like position-servo gains.
# ReBot torque / force caps remain the ORIGINAL ReBot values.
DEMO_SERVO_KP = np.array(
    [300.0, 300.0, 300.0, 150.0, 150.0, 150.0],
    dtype=float,
)

DEMO_SERVO_KV = np.array(
    [5.0, 5.0, 5.0, 2.0, 2.0, 2.0],
    dtype=float,
)

# GUI update rate ~= 30 Hz while servo command runs at 60 Hz.
DEMO_RENDER_EVERY = 1

# Demo-only gripper velocity damping.
# Existing kp=100 and force limit ±8 N stay unchanged.
DEMO_GRIPPER_KV = 3.0

DEMO_GRIPPER_CLOSE_SECONDS = 2.5


OPEN_GRIPPER = -1.0
CLOSE_GRIPPER = 1.0


# -------------------------------------------------------------
# Task geometry
# -------------------------------------------------------------

PREGRASP_HEIGHT = 0.10

# Physical grasp-point target relative to cube center.
# Do not tune this until pre-grasp orientation has been verified.
GRASP_Z_OFFSET = 0.012

LIFT_HEIGHT = 0.12

# Physical grasp reference in the legacy TCP / end_link-aligned frame.
TCP_TO_GRASP = np.array([0.090, 0.0, 0.0], dtype=float)


# -------------------------------------------------------------
# Cartesian path
# -------------------------------------------------------------

# Fixed world-frame target, independent of cube RNG. Global robot reset is unchanged.
READY_GRASP_WORLD = np.array([0.01, 0.0, 0.95], dtype=float)
APPROACH_WAYPOINT_SPACING = 0.015
MIN_APPROACH_JOINT_MARGIN = 0.10

DESCENT_MAX_STEP_M = 0.006
LIFT_WAYPOINTS = 10

MAX_WAYPOINT_JOINT_JUMP = 0.25


# -------------------------------------------------------------
# Grasp orientation
# -------------------------------------------------------------

# The fingers extend along local +X; top-down maps +X to world -Z.
# Keep zero yaw and the geometry-audited +90-degree pitch.
TOP_DOWN_BASE_ROT = (
    Rotation.from_euler(
        "y",
        np.pi / 2.0,
        degrees=False,
    )
    .as_matrix()
)

def grasp_point_to_tcp_target(grasp_point_world, target_rotation):
    """Convert a physical grasp-point target to the unchanged legacy TCP."""
    return (
        np.asarray(grasp_point_world, dtype=float)
        - np.asarray(target_rotation, dtype=float) @ TCP_TO_GRASP
    )


class ApproachConfigurationCheck:
    """Check candidate q in separate data with the actual table/collision model.

    Sampled configuration checks are supplemented by the live safety checks;
    they do not claim continuous swept-volume collision certification.
    """

    def __init__(self, env):
        self.sim = MjSim(env.sim.model._model)
        self.sim.data.qpos[:] = env.sim.data.qpos
        scratch_env = SimpleNamespace(sim=self.sim, robots=env.robots)
        self.diagnostic = LiftDiagnostics(scratch_env, arm_excursion_limit=2.5)
        self.robot = env.robots[0]
        self.workspace = env.placement_initializer.workspace
        self.fingers = [self.sim.model.get_joint_qpos_addr(n)
                        for n in self.robot.gripper["right"].joints]
        self.finger_geoms = [self.sim.model.geom_name2id(n)
                             for n in self.robot.gripper["right"].contact_geoms]
        self.finger_hulls = {}
        model = self.sim.model._model
        for gid in self.finger_geoms:
            mid = model.geom_dataid[gid]
            start = model.mesh_vertadr[mid]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mid]].astype(float)
            self.finger_hulls[gid] = vertices[ConvexHull(vertices).vertices]

    def check(self, q, opening=0.05):
        q = np.asarray(q, dtype=float)
        margin = float(np.min(np.minimum(q - self.workspace.lower, self.workspace.upper - q)))
        if not np.isfinite(q).all() or margin < MIN_APPROACH_JOINT_MARGIN:
            raise BringupStop(f"approach joint-limit margin {margin:.6f} rad is too small")
        self.sim.data.qpos[self.diagnostic.arm_qpos] = q
        self.sim.data.qpos[self.fingers] = [opening, -opening]
        self.sim.forward()
        self.diagnostic.check()
        contacts = self.diagnostic.contacts()["robot_contacts"]
        if contacts:
            raise BringupStop(f"approach configuration has robot contacts: {contacts}")
        arm = self.diagnostic.arm_table_clearance()
        # The fixed base intentionally sits on the tabletop; report it separately.
        arm_min = min(arm[self.sim.model.geom_id2name(gid)]["lowest_above_table_m"]
                      for gid in self.diagnostic.arm_geoms[1:])
        table = self.diagnostic.table
        data = self.sim.data
        table_R = data.geom_xmat[table].reshape(3, 3)
        half_height = self.sim.model._model.geom_size[table, 2]
        finger_min = min(float(np.min(((vertices @ data.geom_xmat[gid].reshape(3, 3).T
                          + data.geom_xpos[gid] - data.geom_xpos[table]) @ table_R)[:, 2])
                          - half_height) for gid, vertices in self.finger_hulls.items())
        if finger_min < 0.02:
            raise BringupStop(f"approach gripper clearance {finger_min:.6f} m is too small")
        return {"minimum_joint_margin_rad": margin,
                "moving_arm_table_clearance_m": float(arm_min),
                "gripper_table_clearance_m": finger_min, "robot_contacts": contacts}


def solve_ready_pose(env, random_count=0):
    """Use a fixed seed and physical target, never the cube position."""
    workspace = env.placement_initializer.workspace
    tcp = grasp_point_to_tcp_target(READY_GRASP_WORLD, TOP_DOWN_BASE_ROT)
    result = workspace.check_pose(tcp - BASE_POSITION, TOP_DOWN_BASE_ROT,
                                  seed=workspace.initial, random_count=random_count)
    if not result.reachable:
        raise BringupStop("fixed READY pose IK failed")
    checker = ApproachConfigurationCheck(env)
    reports = [checker.check(result.qpos, opening) for opening in np.linspace(0.025, 0.05, 6)]
    report = {key: min(r[key] for r in reports) for key in reports[0] if key != "robot_contacts"}
    report.update(physical_grasp_xyz=READY_GRASP_WORLD.tolist(), tcp_target_xyz=tcp.tolist(),
                  q=result.qpos.tolist(), robot_contacts=[])
    return result.qpos.copy(), report


def plan_ready_approach(env, q_ready, cube_xyz):
    """Plan READY -> above cube -> pregrasp; all poses use the fixed rotation."""
    cube = np.asarray(cube_xyz, dtype=float)
    above = np.array([cube[0], cube[1], READY_GRASP_WORLD[2]])
    pregrasp = cube + [0, 0, PREGRASP_HEIGHT]
    if pregrasp[2] >= above[2]:
        raise BringupStop("pregrasp is not below the fixed READY approach height")
    checker = ApproachConfigurationCheck(env)
    workspace = env.placement_initializer.workspace
    previous_q = np.asarray(q_ready).copy()
    previous_xyz = READY_GRASP_WORLD.copy()
    path, reports = [], [checker.check(previous_q)]
    max_jump = 0.0
    for phase, end in (("ABOVE_CUBE", above), ("PREGRASP", pregrasp)):
        count = max(1, int(np.ceil(np.linalg.norm(end - previous_xyz) / APPROACH_WAYPOINT_SPACING)))
        start = previous_xyz.copy()
        for alpha in np.linspace(0, 1, count + 1)[1:]:
            point = start + alpha * (end - start)
            tcp = grasp_point_to_tcp_target(point, TOP_DOWN_BASE_ROT)
            result = workspace.check_pose(tcp - BASE_POSITION, TOP_DOWN_BASE_ROT,
                                          seed=previous_q, random_count=0)
            if not result.reachable:
                raise BringupStop(f"{phase}: approach waypoint pose IK failed")
            jump = float(np.max(np.abs(result.qpos - previous_q)))
            if jump > MAX_WAYPOINT_JOINT_JUMP:
                raise BringupStop(f"{phase}: IK branch jump {jump:.6f} rad")
            # Also inspect intermediate q, at most 0.002 rad apart per joint.
            subdivisions = max(1, int(np.ceil(jump / 0.002)))
            for beta in np.linspace(0, 1, subdivisions + 1)[1:]:
                reports.append(checker.check(previous_q + beta * (result.qpos - previous_q)))
            max_jump = max(max_jump, jump)
            path.append({"phase": phase, "physical_grasp_target": point.tolist(),
                         "tcp_target": tcp.tolist(), "q": result.qpos.tolist()})
            previous_q = result.qpos.copy()
        previous_xyz = end
    summary = {key: min(r[key] for r in reports) for key in reports[0] if key != "robot_contacts"}
    summary.update(waypoint_count=len(path), max_adjacent_delta_q_rad=max_jump,
                   checked_configurations=len(reports), robot_contacts=[])
    return path, summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Rendered playback speed multiplier.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--pregrasp-only", "--approach-only",
        action="store_true",
    )

    parser.add_argument(
        "--ik-random-starts",
        type=int,
        default=24,
        help=(
            "Number of deterministic random IK seeds "
            "for the fixed top-down pose."
        ),
    )

    parser.add_argument("--descent-only", action="store_true",
                        help="Keep fingers open and stop after grasp alignment; never close or lift.")
    parser.add_argument(
        "--close-only",
        action="store_true",
        help=(
            "Close the gripper at the validated grasp pose "
            "and stop before lift."
        ),
    )

    parser.add_argument("--descent-output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "demo_output/rebot_b601_dm/descent")
    parser.add_argument("--trajectory-log", type=Path,
                        help="Write measured startup/approach poses and planned waypoints as JSON.")
    args = parser.parse_args()
    if args.playback_speed <= 0:
        parser.error("--playback-speed must be > 0")
    stop_mode_count = sum([
        bool(args.pregrasp_only),
        bool(args.descent_only),
        bool(args.close_only),
    ])

    if stop_mode_count > 1:
        parser.error(
            "choose only one stop mode: "
            "--pregrasp-only / --approach-only, "
            "--descent-only, or --close-only"
        )
    descent_monitor = None

    env = make_rebot_lift(
        has_renderer=args.render,
        seed=args.seed,
        env_class=ReBotDescentValidation if args.descent_only else ReBotLiftValidation,
    )

    try:
        # -----------------------------------------------------
        # Environment initialization
        # -----------------------------------------------------

        q_ready, ready_report = solve_ready_pose(env, args.ik_random_starts)
        print("READY VALIDATION", json.dumps(ready_report), flush=True)
        # This runtime instance only. Normal reset rebuilds controller goals at READY.
        # No simulated folded-to-READY motion and no change to robot_model.init_qpos.
        env.robots[0].init_qpos = q_ready.copy()
        env.reset()
        if args.descent_only:
            # Validation starts fully open; no change to the gripper model/default.
            gripper = env.robots[0].gripper["right"]
            fingers = [env.sim.model.get_joint_qpos_addr(n) for n in gripper.joints]
            env.sim.data.qpos[fingers] = [0.05, -0.05]
            env.sim.data.ctrl[env.sim.model.actuator_name2id(gripper.actuators[0])] = 0.05
        env.sim.forward()
        approach_active = True
        approach_samples = []
        max_orientation_error_deg = 0.0

        diagnostic = LiftDiagnostics(
            env,
            arm_excursion_limit=2.5,
        )

        # Observe robot contacts at every existing safety-check invocation,
        # including simulation substeps, without changing any safety limits.
        robot_contacts_seen = {}
        safety_check = diagnostic.check

        def check_and_record_contacts():
            nonlocal max_orientation_error_deg
            safety_check()
            if approach_active:
                site = env.robots[0].eef_site_id["right"]
                actual_rotation = env.sim.data.site_xmat[site].reshape(3, 3)
                error_deg = float(np.rad2deg(Rotation.from_matrix(
                    TOP_DOWN_BASE_ROT @ actual_rotation.T).magnitude()))
                max_orientation_error_deg = max(max_orientation_error_deg, error_deg)
                if error_deg > 3.0:
                    raise BringupStop(f"approach orientation drift {error_deg:.6f} deg")
            for contact in diagnostic.contacts()["robot_contacts"]:
                if approach_active:
                    raise BringupStop(f"unexpected approach robot contact: {contact}")
                key = tuple(sorted(contact["geoms"]))
                if key not in robot_contacts_seen:
                    robot_contacts_seen[key] = dict(contact)
                else:
                    entry = robot_contacts_seen[key]
                    entry["distance_m"] = min(entry["distance_m"], contact["distance_m"])
                    entry["normal_force_N"] = max(entry["normal_force_N"], contact["normal_force_N"])

        diagnostic.check = check_and_record_contacts
        env.diagnostics = diagnostic
        diagnostic.check()
        if args.descent_only:
            descent_monitor = DescentMonitor(env, diagnostic, TCP_TO_GRASP,
                                             TOP_DOWN_BASE_ROT, args.descent_output)
            env.descent_monitor = descent_monitor

        if args.render:
            env.viewer.update()

            with (
                env.viewer
                .viewer
                .lock()
            ):
                env.viewer.viewer.cam.lookat[:] = [
                    -0.1,
                    0.0,
                    0.95,
                ]

                env.viewer.viewer.cam.distance = (
                    1.6
                )

                env.viewer.viewer.cam.azimuth = (
                    135
                )

                env.viewer.viewer.cam.elevation = (
                    -25
                )

            env.viewer.viewer.sync()

        arm_ids = (
            diagnostic.arm_qpos
        )

        initial_q = q_ready.copy()

        eef_site_id = (
            env.robots[0]
            .eef_site_id["right"]
        )

        # -----------------------------------------------------
        # State helpers
        # -----------------------------------------------------

        def current_q():
            return (
                env.sim.data
                .qpos[arm_ids]
                .copy()
            )

        def current_tcp():
            return (
                env.sim.data
                .site_xpos[eef_site_id]
                .copy()
            )

        def current_tcp_rotation():
            return env.sim.data.site_xmat[eef_site_id].reshape(3, 3).copy()

        def current_grasp_point():
            return current_tcp() + current_tcp_rotation() @ TCP_TO_GRASP

        def current_cube():
            return (
                env.sim.data
                .body_xpos[
                    env.cube_body_id
                ]
                .copy()
            )

        # -----------------------------------------------------
        # Low-level joint control
        # -----------------------------------------------------

        # =================================================
        # DEMO MOTOR SERVO
        #
        # Discoverse-style q trajectory:
        #   60 Hz
        #   max 0.75 rad/s
        #
        # ReBot actuation:
        #   original torque motors
        #   original force limits
        #
        # Direct servo torque:
        #
        # tau =
        #     kp * (q_cmd - q)
        #   - kv * qvel
        #   + qfrc_bias
        #
        # qfrc_bias supplies gravity / Coriolis compensation.
        # =================================================

        demo_servo_initialized = False

        demo_servo_cmd_q = None

        demo_servo_next_time = None

        demo_render_counter = 0

        demo_servo_sat_time = np.zeros(
            6,
            dtype=float,
        )


        # -------------------------------------------------
        # Arm actuator / DOF IDs
        # -------------------------------------------------

        arm_actuator_ids = np.asarray(
            [
                env.sim.model.actuator_name2id(name)
                for name in (
                    env.robots[0]
                    .robot_model
                    .actuators
                )
            ],
            dtype=int,
        )

        if arm_actuator_ids.shape != (6,):
            raise BringupStop(
                "expected exactly 6 arm actuators, "
                f"got {arm_actuator_ids.shape}"
            )


        # qvel addresses == MuJoCo DOF addresses
        # for these scalar hinge joints.
        arm_dof_ids = np.asarray(
            diagnostic.arm_qvel,
            dtype=int,
        )


        model_raw = env.sim.model._model


        # ReBot's actual existing actuator limits.
        arm_force_limits = (
            model_raw
            .actuator_forcerange[
                arm_actuator_ids
            ]
            .copy()
        )

        arm_ctrl_limits = (
            model_raw
            .actuator_ctrlrange[
                arm_actuator_ids
            ]
            .copy()
        )


        # We respect BOTH ctrl and force limits.
        arm_torque_low = np.maximum(
            arm_force_limits[:, 0],
            arm_ctrl_limits[:, 0],
        )

        arm_torque_high = np.minimum(
            arm_force_limits[:, 1],
            arm_ctrl_limits[:, 1],
        )


        gripper_actuator_id = (
            env.sim.model.actuator_name2id(
                env.robots[0]
                .gripper["right"]
                .actuators[0]
            )
        )


        # -------------------------------------------------
        # Restore original MOTOR semantics
        #
        # The previous experiment converted these actuators
        # into position actuators at runtime.
        #
        # Here we explicitly restore a MuJoCo motor:
        #
        # force = ctrl
        # -------------------------------------------------

        def restore_demo_motor_semantics():

            for actuator_id in arm_actuator_ids:

                actuator_id = int(
                    actuator_id
                )

                model_raw.actuator_gaintype[
                    actuator_id
                ] = int(
                    mujoco.mjtGain.mjGAIN_FIXED
                )

                model_raw.actuator_biastype[
                    actuator_id
                ] = int(
                    mujoco.mjtBias.mjBIAS_NONE
                )

                model_raw.actuator_gainprm[
                    actuator_id,
                    :
                ] = 0.0

                model_raw.actuator_gainprm[
                    actuator_id,
                    0
                ] = 1.0

                model_raw.actuator_biasprm[
                    actuator_id,
                    :
                ] = 0.0


        # -------------------------------------------------
        # Direct torque servo
        # -------------------------------------------------

        def apply_demo_motor_servo():

            q = current_q()

            qvel = (
                env.sim.data
                .qvel[
                    arm_dof_ids
                ]
                .copy()
            )

            # MuJoCo bias generalized force:
            # gravity + Coriolis / centrifugal terms.
            bias = (
                env.sim.data
                .qfrc_bias[
                    arm_dof_ids
                ]
                .copy()
            )

            position_term = (
                DEMO_SERVO_KP
                * (
                    demo_servo_cmd_q
                    - q
                )
            )

            damping_term = (
                -DEMO_SERVO_KV
                * qvel
            )

            raw_tau = (
                position_term
                + damping_term
                + bias
            )

            tau = np.clip(
                raw_tau,
                arm_torque_low,
                arm_torque_high,
            )

            env.sim.data.ctrl[
                arm_actuator_ids
            ] = tau

            return (
                raw_tau,
                tau,
                bias,
            )


        # -------------------------------------------------
        # Initialize
        # -------------------------------------------------

        def initialize_demo_servo():

            nonlocal demo_servo_initialized
            nonlocal demo_servo_cmd_q
            nonlocal demo_servo_next_time

            if demo_servo_initialized:
                return

            print(
                "\n"
                "=== INITIALIZE DEMO MOTOR SERVO ===",
                flush=True,
            )

            restore_demo_motor_semantics()

            # ---------------------------------------------
            # Demo-only gripper damping
            #
            # Keep:
            #   kp = existing 100
            #   force range = existing +/-8 N
            #
            # Add only velocity damping:
            #   force -= kv * qvel
            # ---------------------------------------------

            model_raw.actuator_biasprm[
                gripper_actuator_id,
                2
            ] = -DEMO_GRIPPER_KV

            print(
                "gripper actuator =",
                mujoco.mj_id2name(
                    model_raw,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    gripper_actuator_id,
                ),
                flush=True,
            )

            print(
                "gripper gainprm =",
                model_raw.actuator_gainprm[
                    gripper_actuator_id
                ].copy(),
                flush=True,
            )

            print(
                "gripper biasprm =",
                model_raw.actuator_biasprm[
                    gripper_actuator_id
                ].copy(),
                flush=True,
            )

            print(
                "gripper force range =",
                model_raw.actuator_forcerange[
                    gripper_actuator_id
                ].copy(),
                flush=True,
            )

            # Recompute dynamics at the current READY pose.
            env.sim.forward()

            demo_servo_cmd_q = (
                current_q().copy()
            )

            demo_servo_next_time = float(
                env.sim.data.time
            )

            raw_tau, tau, bias = (
                apply_demo_motor_servo()
            )

            print(
                "command_hz =",
                DEMO_SERVO_HZ,
                flush=True,
            )

            print(
                "move_speed_rad_s =",
                DEMO_MOVE_SPEED_RAD_S,
                flush=True,
            )

            print(
                "physics_timestep =",
                model_raw.opt.timestep,
                flush=True,
            )

            print(
                "initial q =",
                np.round(
                    demo_servo_cmd_q,
                    6,
                ),
                flush=True,
            )

            print(
                "initial gravity/bias torque =",
                np.round(
                    bias,
                    4,
                ),
                flush=True,
            )

            print(
                "initial commanded torque =",
                np.round(
                    tau,
                    4,
                ),
                flush=True,
            )

            print(
                "torque low =",
                arm_torque_low,
                flush=True,
            )

            print(
                "torque high =",
                arm_torque_high,
                flush=True,
            )

            for i, actuator_id in enumerate(
                arm_actuator_ids
            ):
                actuator_id = int(
                    actuator_id
                )

                name = mujoco.mj_id2name(
                    model_raw,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    actuator_id,
                )

                print(
                    f"motor[{i + 1}] "
                    f"name={name} "
                    f"kp={DEMO_SERVO_KP[i]:.1f} "
                    f"kv={DEMO_SERVO_KV[i]:.1f} "
                    f"limit=["
                    f"{arm_torque_low[i]:.1f}, "
                    f"{arm_torque_high[i]:.1f}]",
                    flush=True,
                )

            demo_servo_initialized = True

            print(
                "=== DEMO MOTOR SERVO READY ===\n",
                flush=True,
            )


        # -------------------------------------------------
        # One 60-Hz command tick
        # -------------------------------------------------

        def step_to_joint_target(
            q_target,
            gripper_cmd,
            phase,
        ):

            nonlocal demo_servo_cmd_q
            nonlocal demo_servo_next_time
            nonlocal demo_render_counter

            initialize_demo_servo()

            if (
                args.render
                and not (
                    env.viewer
                    .viewer
                    .is_running()
                )
            ):
                raise BringupStop(
                    "viewer closed"
                )

            q_target = np.asarray(
                q_target,
                dtype=float,
            )


            # ---------------------------------------------
            # Discoverse-style synchronized interpolation
            #
            # joint with largest remaining error:
            #     0.75 rad/s
            #
            # other joints:
            #     proportional speed
            #
            # This makes all joints arrive together.
            # ---------------------------------------------

            command_error = (
                q_target
                - demo_servo_cmd_q
            )

            max_error = float(
                np.max(
                    np.abs(
                        command_error
                    )
                )
            )

            if max_error > 1e-12:

                ratio = (
                    np.abs(
                        command_error
                    )
                    / max_error
                )

                max_delta = (
                    DEMO_MOVE_SPEED_RAD_S
                    * ratio
                    * DEMO_SERVO_DT
                )

                command_delta = (
                    np.sign(
                        command_error
                    )
                    * np.minimum(
                        np.abs(
                            command_error
                        ),
                        max_delta,
                    )
                )

                demo_servo_cmd_q = (
                    demo_servo_cmd_q
                    + command_delta
                )


            # ---------------------------------------------
            # Existing gripper position actuator
            #
            # gripper_cmd:
            # -1 -> open  = 0.05
            # +1 -> close = 0.00
            # ---------------------------------------------

            g = float(
                np.clip(
                    gripper_cmd,
                    -1.0,
                    1.0,
                )
            )

            gripper_target = (
                0.025
                * (
                    1.0
                    - g
                )
            )

            env.sim.data.ctrl[
                gripper_actuator_id
            ] = gripper_target


            if descent_monitor is not None:

                descent_monitor.phase = (
                    phase
                )

                if (
                    gripper_cmd
                    != OPEN_GRIPPER
                ):
                    raise BringupStop(
                        "descent-only requires "
                        "a fully open gripper command"
                    )


            # ---------------------------------------------
            # Advance physics until next 60-Hz servo tick.
            #
            # Physics timestep itself stays 0.002 s.
            # Gravity compensation is refreshed before
            # EVERY physics step.
            # ---------------------------------------------

            demo_servo_next_time += (
                DEMO_SERVO_DT
            )

            while (
                env.sim.data.time
                + 1e-12
                < demo_servo_next_time
            ):

                raw_tau, tau, bias = (
                    apply_demo_motor_servo()
                )


                # -----------------------------------------
                # Sustained saturation monitor
                # Same 0.25-s concept used by the
                # original validation controller.
                # -----------------------------------------

                saturated = (
                    np.abs(
                        raw_tau - tau
                    )
                    > 1e-8
                )

                demo_servo_sat_time[:] = (
                    np.where(
                        saturated,
                        (
                            demo_servo_sat_time
                            + model_raw.opt.timestep
                        ),
                        0.0,
                    )
                )

                if np.any(
                    demo_servo_sat_time
                    >= 0.25
                ):
                    joints = (
                        np.flatnonzero(
                            demo_servo_sat_time
                            >= 0.25
                        )
                        + 1
                    ).tolist()

                    raise BringupStop(
                        "demo servo sustained "
                        "torque saturation on "
                        f"joints {joints}"
                    )


                env.sim.step()

                # Existing finite-state, speed,
                # collision and orientation checks.
                diagnostic.check()


            # ---------------------------------------------
            # Existing trajectory diagnostics
            # ---------------------------------------------

            if approach_active:

                approach_samples.append(
                    {
                        "time_s": float(
                            env.sim.data.time
                        ),
                        "phase": phase,
                        "q": (
                            current_q()
                            .tolist()
                        ),
                        "tcp_xyz": (
                            current_tcp()
                            .tolist()
                        ),
                        "physical_grasp_xyz": (
                            current_grasp_point()
                            .tolist()
                        ),
                        "local_positive_x_world": (
                            current_tcp_rotation()[
                                :,
                                0
                            ]
                            .tolist()
                        ),
                    }
                )


            # ---------------------------------------------
            # Viewer ~= 30 Hz.
            #
            # No wall-clock sleep.
            # ---------------------------------------------

            if args.render:

                demo_render_counter += 1

                if (
                    demo_render_counter
                    % DEMO_RENDER_EVERY
                    == 0
                ):
                    env.viewer.update()


            actual_error = (
                q_target
                - current_q()
            )

            return float(
                np.max(
                    np.abs(
                        actual_error
                    )
                )
            )


        def hold(
            q_target,
            seconds,
            gripper_cmd,
            phase,
        ):
            n_steps = max(
                1,
                round(
                    seconds / DEMO_SERVO_DT
                ),
            )

            for _ in range(
                n_steps
            ):
                step_to_joint_target(
                    q_target,
                    gripper_cmd,
                    phase,
                )

        def move_to_joint_target(
            q_target,
            gripper_cmd,
            phase,
            tolerance=0.010,
            max_steps=500,
        ):
            print(
                f"\n=== {phase} ===",
                flush=True,
            )

            stable = 0

            for step_idx in range(
                max_steps
            ):
                err = (
                    step_to_joint_target(
                        q_target,
                        gripper_cmd,
                        phase,
                    )
                )

                if (
                    step_idx % 20 == 0
                ):
                    print(
                        f"{phase}: "
                        f"step={step_idx}, "
                        f"max_joint_error="
                        f"{err:.5f}, "
                        f"tcp="
                        f"{np.round(current_tcp(), 5)}",
                        flush=True,
                    )

                if err < tolerance:
                    stable += 1
                else:
                    stable = 0

                if stable >= 2:
                    print(
                        f"{phase}: "
                        f"target reached "
                        f"(max joint error "
                        f"{err:.6f})",
                        flush=True,
                    )
                    return

            raise BringupStop(
                f"{phase}: "
                f"joint target not reached"
            )

        # -----------------------------------------------------
        # Pose IK wrapper
        # -----------------------------------------------------

        def solve_tcp_pose(
            world_xyz,
            target_rotation,
            label,
            seed=None,
            random_count=0,
        ):
            """Solve legacy TCP pose IK for a physical grasp-point target."""

            world_xyz = np.asarray(
                world_xyz,
                dtype=float,
            )

            tcp_target = grasp_point_to_tcp_target(world_xyz, target_rotation)
            print(
                f"{label}: physical grasp target xyz={world_xyz}, "
                f"converted legacy TCP target xyz={tcp_target}",
                flush=True,
            )
            target_xyz_base = (
                tcp_target
                - np.asarray(
                    BASE_POSITION,
                    dtype=float,
                )
            )

            result = (
                env.placement_initializer
                .workspace
                .check_pose(
                    target_xyz_base,
                    target_rotation,
                    seed=seed,
                    random_count=(
                        random_count
                    ),
                )
            )

            print(
                f"{label} POSE IK: "
                f"reachable="
                f"{result.reachable}, "
                f"pos_error="
                f"{result.position_error_m * 1000:.3f} mm, "
                f"rot_error="
                f"{np.rad2deg(result.orientation_error_rad):.3f} deg, "
                f"q="
                f"{np.round(result.qpos, 5)}",
                flush=True,
            )

            if not result.reachable:
                raise BringupStop(
                    f"{label}: "
                    f"pose IK unreachable "
                    f"(pos="
                    f"{result.position_error_m * 1000:.2f} mm, "
                    f"rot="
                    f"{np.rad2deg(result.orientation_error_rad):.2f} deg)"
                )

            return (
                result.qpos.copy()
            )


        # -----------------------------------------------------
        # Cartesian waypoint IK
        # -----------------------------------------------------

        def solve_cartesian_path(
            start_xyz,
            end_xyz,
            q_seed,
            target_rotation,
            label,
            num_waypoints,
        ):
            """Interpolate physical grasp points, then solve each converted TCP pose."""

            start_xyz = np.asarray(
                start_xyz,
                dtype=float,
            )

            end_xyz = np.asarray(
                end_xyz,
                dtype=float,
            )

            previous_q = (
                np.asarray(
                    q_seed,
                    dtype=float,
                )
                .copy()
            )

            q_path = []

            alphas = np.linspace(
                0.0,
                1.0,
                num_waypoints + 1,
            )[1:]

            for i, alpha in enumerate(
                alphas
            ):
                xyz = (
                    (1.0 - alpha)
                    * start_xyz
                    + alpha
                    * end_xyz
                )

                waypoint_label = (
                    f"{label}_"
                    f"{i + 1:02d}"
                )

                # Normally previous_q should be enough.
                # Four fallback random starts help if the
                # local solve becomes difficult.
                q = solve_tcp_pose(
                    xyz,
                    target_rotation,
                    waypoint_label,
                    seed=previous_q,
                    random_count=4,
                )

                jump = float(
                    np.max(
                        np.abs(
                            q
                            - previous_q
                        )
                    )
                )

                print(
                    f"{waypoint_label}: "
                    f"max dq="
                    f"{jump:.4f} rad",
                    flush=True,
                )

                if (
                    jump
                    > MAX_WAYPOINT_JOINT_JUMP
                ):
                    raise BringupStop(
                        f"{label}: "
                        f"IK branch jump "
                        f"{jump:.3f} rad"
                    )

                q_path.append(
                    q.copy()
                )

                previous_q = (
                    q.copy()
                )

            return q_path

        def execute_joint_path(
            q_path,
            gripper_cmd,
            label,
            final_tolerance=0.010,
            tolerance=None,
            **kwargs,
        ):
            """
            Execute the full IK waypoint sequence as one smooth
            time-parameterized joint trajectory.

            Spatial interpolation:
                PCHIP through the existing IK configurations.

            Temporal interpolation:
                minimum-jerk scalar profile.

            This removes the velocity-direction discontinuity that
            previously occurred at every waypoint boundary.
            """

            if tolerance is not None:
                final_tolerance = float(
                    tolerance
                )

            if not q_path:
                return

            # -------------------------------------------------
            # Build knot list:
            #
            # actual current q
            # + existing IK path
            # -------------------------------------------------

            raw_points = [
                current_q().copy()
            ]

            raw_points.extend(
                np.asarray(
                    q,
                    dtype=float,
                ).copy()
                for q in q_path
            )


            # Remove numerically duplicate consecutive knots.
            points = [
                raw_points[0]
            ]

            for q in raw_points[1:]:

                if (
                    np.max(
                        np.abs(
                            q
                            - points[-1]
                        )
                    )
                    > 1e-9
                ):
                    points.append(q)


            if len(points) == 1:
                return


            points = np.asarray(
                points,
                dtype=float,
            )


            # -------------------------------------------------
            # L-infinity joint-space arc coordinate.
            #
            # A segment length is determined by the joint
            # moving the most in that segment.
            # -------------------------------------------------

            segment_lengths = np.max(
                np.abs(
                    np.diff(
                        points,
                        axis=0,
                    )
                ),
                axis=1,
            )

            s_knots = np.concatenate(
                [
                    np.array(
                        [0.0],
                        dtype=float,
                    ),
                    np.cumsum(
                        segment_lengths
                    ),
                ]
            )

            total_s = float(
                s_knots[-1]
            )

            if total_s <= 1e-12:
                return


            # -------------------------------------------------
            # Shape-preserving cubic joint interpolation.
            #
            # Unlike a generic cubic spline, PCHIP strongly
            # reduces overshoot between neighboring q knots.
            # -------------------------------------------------

            spline = PchipInterpolator(
                s_knots,
                points,
                axis=0,
            )


            # -------------------------------------------------
            # Estimate maximum |dq/ds|.
            #
            # We use this to guarantee that the time-scaled
            # spline does not exceed DEMO_MOVE_SPEED_RAD_S.
            # -------------------------------------------------

            probe_count = max(
                300,
                80 * len(points),
            )

            s_probe = np.linspace(
                0.0,
                total_s,
                probe_count,
            )

            dq_ds = (
                spline.derivative()(
                    s_probe
                )
            )

            max_dq_ds = float(
                np.max(
                    np.abs(
                        dq_ds
                    )
                )
            )

            max_dq_ds = max(
                max_dq_ds,
                1e-6,
            )


            # -------------------------------------------------
            # Minimum-jerk:
            #
            # h(u) =
            #   10u^3 - 15u^4 + 6u^5
            #
            # max dh/du = 1.875
            #
            # Pick duration so:
            #
            # max |dq/dt|
            # <= DEMO_MOVE_SPEED_RAD_S
            # -------------------------------------------------

            min_duration = (
                1.875
                * total_s
                * max_dq_ds
                / DEMO_MOVE_SPEED_RAD_S
            )

            duration = max(
                2.0 * DEMO_SERVO_DT,
                min_duration,
            )

            n_steps = max(
                2,
                int(
                    np.ceil(
                        duration
                        / DEMO_SERVO_DT
                    )
                ),
            )


            print(
                f"\n=== {label} SMOOTH TRAJECTORY ===",
                flush=True,
            )

            print(
                f"{label}: "
                f"knots={len(points)}, "
                f"duration={duration:.3f}s, "
                f"servo_steps={n_steps}, "
                f"speed_limit="
                f"{DEMO_MOVE_SPEED_RAD_S:.3f} rad/s",
                flush=True,
            )


            # -------------------------------------------------
            # Execute at the existing 60-Hz servo rate.
            # -------------------------------------------------

            report_steps = {
                1,
                max(1, n_steps // 4),
                max(1, n_steps // 2),
                max(1, 3 * n_steps // 4),
                n_steps,
            }

            for step_index in range(
                1,
                n_steps + 1,
            ):

                u = (
                    step_index
                    / float(
                        n_steps
                    )
                )

                smooth = (
                    10.0 * u**3
                    - 15.0 * u**4
                    + 6.0 * u**5
                )

                s_ref = (
                    total_s
                    * smooth
                )

                q_ref = np.asarray(
                    spline(
                        s_ref
                    ),
                    dtype=float,
                )

                step_to_joint_target(
                    q_ref,
                    gripper_cmd,
                    label,
                )

                if (
                    step_index
                    in report_steps
                ):
                    print(
                        f"{label}: "
                        f"{step_index}/{n_steps}, "
                        f"tcp="
                        f"{np.round(current_tcp(), 5)}",
                        flush=True,
                    )


            # -------------------------------------------------
            # Exact final IK configuration.
            #
            # The minimum-jerk trajectory already ends at
            # zero reference velocity, so this should only be
            # a small settling correction.
            # -------------------------------------------------

            final_q = np.asarray(
                q_path[-1],
                dtype=float,
            )

            move_to_joint_target(
                final_q,
                gripper_cmd,
                f"{label} FINAL",
                tolerance=final_tolerance,
                max_steps=150,
            )


        def print_gripper_debug():
            print(
                "\n=== GRIPPER FRAME DEBUG ===",
                flush=True,
            )

            tcp_pos = current_tcp()

            tcp_rot = (
                env.sim.data
                .site_xmat[
                    eef_site_id
                ]
                .reshape(3, 3)
                .copy()
            )

            print(
                "CUBE:",
                np.round(
                    current_cube(),
                    6,
                ),
            )

            print(
                "CONTROLLED TCP:",
                np.round(
                    tcp_pos,
                    6,
                ),
            )

            print(
                "TCP local +X in world:",
                np.round(
                    tcp_rot[:, 0],
                    6,
                ),
            )

            print(
                "TCP local -X in world:",
                np.round(
                    -tcp_rot[:, 0],
                    6,
                ),
            )

            print(
                "TCP local +Y in world:",
                np.round(
                    tcp_rot[:, 1],
                    6,
                ),
            )

            print(
                "TCP local +Z in world:",
                np.round(
                    tcp_rot[:, 2],
                    6,
                ),
            )

            model = env.sim.model
            data = env.sim.data

            print(
                "\n--- SITES ---"
            )

            for i in range(
                model.nsite
            ):
                name = (
                    model.site_id2name(i)
                )

                if (
                    name
                    and any(
                        key
                        in name.lower()
                        for key in [
                            "grip",
                            "tcp",
                            "ee",
                        ]
                    )
                ):
                    print(
                        "SITE",
                        name,
                        np.round(
                            data.site_xpos[i],
                            6,
                        ),
                    )

            print(
                "\n--- BODIES ---"
            )

            for i in range(
                model.nbody
            ):
                name = (
                    model.body_id2name(i)
                )

                if (
                    name
                    and any(
                        key
                        in name.lower()
                        for key in [
                            "eef",
                            "finger",
                            "gripper",
                            "hand",
                        ]
                    )
                ):
                    print(
                        "BODY",
                        name,
                        np.round(
                            data.body_xpos[i],
                            6,
                        ),
                    )

            print(
                "\n--- GRIPPER GEOMS ---"
            )

            for i in range(
                model.ngeom
            ):
                name = (
                    model.geom_id2name(i)
                )

                if (
                    name
                    and any(
                        key
                        in name.lower()
                        for key in [
                            "finger",
                            "palm",
                        ]
                    )
                ):
                    print(
                        "GEOM",
                        name,
                        np.round(
                            data.geom_xpos[i],
                            6,
                        ),
                    )

        # -----------------------------------------------------
        # Phase 0: settle
        # -----------------------------------------------------

        print(
            "\n=== SETTLE ===",
            flush=True,
        )

        hold(
            initial_q,
            seconds=0.5,
            gripper_cmd=OPEN_GRIPPER if args.descent_only else 0.0,
            phase="SETTLE",
        )

        # -----------------------------------------------------
        # Open gripper gradually
        # -----------------------------------------------------

        print(
            "\n=== OPEN GRIPPER ===",
            flush=True,
        )

        for aperture in np.linspace(
            OPEN_GRIPPER if args.descent_only else 0.0,
            OPEN_GRIPPER,
            80,
        ):
            step_to_joint_target(
                initial_q,
                float(aperture),
                "OPENING",
            )

        hold(
            initial_q,
            seconds=0.2,
            gripper_cmd=OPEN_GRIPPER,
            phase="OPEN HOLD",
        )

        # -----------------------------------------------------
        # GT task state
        # -----------------------------------------------------

        cube_start = (
            current_cube()
        )

        tcp_start = (
            current_tcp()
        )

        print(
            "cube world xyz:",
            cube_start,
            flush=True,
        )

        print(
            "tcp  world xyz:",
            tcp_start,
            flush=True,
        )

        # -----------------------------------------------------
        # Physical grasp-point target positions
        # -----------------------------------------------------

        pregrasp_xyz = (
            cube_start.copy()
        )

        pregrasp_xyz[2] += (
            PREGRASP_HEIGHT
        )

        grasp_xyz = (
            cube_start.copy()
        )

        grasp_xyz[2] += (
            GRASP_Z_OFFSET
        )

        lift_xyz = (
            grasp_xyz.copy()
        )

        lift_xyz[2] += (
            LIFT_HEIGHT
        )

        print(
            "pregrasp physical grasp xyz:",
            pregrasp_xyz,
            flush=True,
        )

        print(
            "alignment physical grasp xyz:",
            grasp_xyz,
            flush=True,
        )

        print(
            "lift physical grasp xyz:",
            lift_xyz,
            flush=True,
        )

        # -----------------------------------------------------
        # Phase 1: fixed READY -> horizontal approach -> vertical pregrasp
        # -----------------------------------------------------

        grasp_rotation = TOP_DOWN_BASE_ROT.copy()
        approach_path, approach_report = plan_ready_approach(env, q_ready, cube_start)
        print("APPROACH PLAN", json.dumps(approach_report), flush=True)
        print("APPROACH WAYPOINTS", json.dumps(approach_path), flush=True)
        approach_q_path = [
            np.asarray(
                waypoint["q"],
                dtype=float,
            )
            for waypoint in approach_path
        ]

        execute_joint_path(
            approach_q_path,
            OPEN_GRIPPER,
            "APPROACH",
            steps_per_waypoint=2,
            final_tolerance=0.008,
        )

        q_pregrasp = np.asarray(approach_path[-1]["q"])

        hold(
            q_pregrasp,
            seconds=0.2,
            gripper_cmd=OPEN_GRIPPER,
            phase="PREGRASP HOLD",
        )

        pregrasp_actual = current_grasp_point()
        actual_rotation = current_tcp_rotation()
        grasp_error = float(np.linalg.norm(pregrasp_actual - pregrasp_xyz))
        rotation_error = float(Rotation.from_matrix(
            grasp_rotation @ actual_rotation.T
        ).magnitude())
        print(
            "\nPREGRASP SNAPSHOT",
            json.dumps(snapshot(env, diagnostic)),
            flush=True,
        )
        print(
            "PREGRASP GEOMETRY",
            json.dumps({
                "cube_xyz": current_cube().tolist(),
                "physical_grasp_target_xyz": pregrasp_xyz.tolist(),
                "converted_legacy_tcp_target_xyz": grasp_point_to_tcp_target(
                    pregrasp_xyz, grasp_rotation
                ).tolist(),
                "actual_legacy_tcp_xyz": current_tcp().tolist(),
                "actual_physical_grasp_xyz": pregrasp_actual.tolist(),
                "physical_grasp_error_m": grasp_error,
                "orientation_error_deg": float(np.rad2deg(rotation_error)),
                "tcp_to_grasp_world_offset": (actual_rotation @ TCP_TO_GRASP).tolist(),
                "local_positive_x_world": actual_rotation[:, 0].tolist(),
                "robot_contacts_seen": list(robot_contacts_seen.values()),
            }),
            flush=True,
        )
        if grasp_error > 0.005 or rotation_error > np.deg2rad(3.0):
            raise BringupStop("pregrasp geometry differs from target; diagnose before proceeding")

        print_gripper_debug()
        print("APPROACH EXECUTION", json.dumps({
            "max_orientation_error_deg": max_orientation_error_deg,
            "samples": len(approach_samples),
            "robot_contacts": list(robot_contacts_seen.values()),
        }), flush=True)
        if args.trajectory_log:
            args.trajectory_log.parent.mkdir(parents=True, exist_ok=True)
            args.trajectory_log.write_text(json.dumps({
                "ready": ready_report, "plan_summary": approach_report,
                "waypoints": approach_path, "samples": approach_samples,
                "max_orientation_error_deg": max_orientation_error_deg,
                "robot_contacts": list(robot_contacts_seen.values()),
            }, indent=2) + "\n")
            print("Trajectory saved:", args.trajectory_log, flush=True)

        # -----------------------------------------------------
        # Optional pre-grasp-only test
        # -----------------------------------------------------

        if args.pregrasp_only:
            print(
                "\nPREGRASP-ONLY COMPLETE.\n"
                "Inspect GUI and logs before "
                "attempting the full grasp.",
                flush=True,
            )

            hold(
                q_pregrasp,
                seconds=5.0,
                gripper_cmd=OPEN_GRIPPER,
                phase="VISUAL INSPECTION",
            )

            print(
                "PREGRASP robot contacts over entire run:",
                json.dumps(list(robot_contacts_seen.values())),
                flush=True,
            )
            return 0

        # -----------------------------------------------------
        # Phase 2: descend with fixed grasp orientation
        # -----------------------------------------------------

        approach_active = False
        print(
            "\n=== SOLVE DESCENT PATH ===",
            flush=True,
        )

        descent_path = (
            solve_cartesian_path(
                start_xyz=(
                    pregrasp_xyz
                ),
                end_xyz=(
                    grasp_xyz
                ),
                q_seed=(
                    q_pregrasp
                ),
                target_rotation=(
                    grasp_rotation
                ),
                label="DESCENT",
                num_waypoints=(
                    max(1, int(np.ceil(np.linalg.norm(grasp_xyz - pregrasp_xyz)
                                       / DESCENT_MAX_STEP_M - 1e-10)))
                ),
            )
        )

        print(
            "\n=== EXECUTE DESCENT ===",
            flush=True,
        )

        if descent_monitor is not None:
            descent_monitor.begin_descent()
            descent_monitor.plan = {
                "waypoint_count": len(descent_path),
                "cartesian_step_m": float(np.linalg.norm(grasp_xyz - pregrasp_xyz) / len(descent_path)),
                "maximum_adjacent_delta_q_rad": float(np.max(np.abs(np.diff(
                    np.vstack([q_pregrasp, *descent_path]), axis=0)))),
                "physical_grasp_target_xyz": grasp_xyz.tolist(),
                "waypoint_q": [q.tolist() for q in descent_path],
            }
            print("DESCENT PLAN", json.dumps(descent_monitor.plan), flush=True)
        execute_joint_path(
            descent_path,
            OPEN_GRIPPER,
            "DESCENT",
            tolerance=0.002,
        )

        q_grasp = (
            descent_path[-1]
        )

        hold(
            q_grasp,
            seconds=0.2,
            gripper_cmd=OPEN_GRIPPER,
            phase="GRASP ALIGNMENT",
        )

        actual_grasp_point = current_grasp_point()
        print("actual legacy TCP xyz:", current_tcp(), flush=True)

        print(
            "physical grasp target xyz:",
            grasp_xyz,
            flush=True,
        )

        print(
            "actual physical grasp xyz:",
            actual_grasp_point,
            flush=True,
        )

        print(
            "grasp position error:",
            float(
                np.linalg.norm(
                    actual_grasp_point
                    - grasp_xyz
                )
            ),
            flush=True,
        )

        if args.descent_only:
            # STOP before the closing/lifting code. Keep monitoring while holding open.
            hold(q_grasp, seconds=2.0, gripper_cmd=OPEN_GRIPPER, phase="DESCENT FINAL OPEN HOLD")
            env.sim.forward()
            final = descent_monitor.capture()
            rotation = current_tcp_rotation()
            prefix = env.robots[0].gripper["right"].naming_prefix
            positions = {}
            for key, suffix in (("left_finger_front_xyz", "finger_left_front_collision"),
                                ("right_finger_front_xyz", "finger_right_front_collision"),
                                ("palm_geom_xyz", "gripper_palm_collision")):
                positions[key] = env.sim.data.geom_xpos[env.sim.model.geom_name2id(prefix + suffix)].tolist()
            q = current_q()
            workspace = env.placement_initializer.workspace
            final.update(positions)
            final.update(
                desired_physical_grasp_xyz=grasp_xyz.tolist(),
                physical_grasp_error_m=float(np.linalg.norm(current_grasp_point() - grasp_xyz)),
                local_positive_x_world=rotation[:, 0].tolist(),
                minimum_joint_limit_margin_rad=float(np.min(np.minimum(q - workspace.lower, workspace.upper - q))),
                finger_qpos=env.sim.data.qpos[fingers].tolist(),
                **descent_monitor.summary(),
            )
            print("DESCENT FINAL", json.dumps(final), flush=True)
            (args.descent_output / "descent_final.json").write_text(json.dumps(final, indent=2) + "\n")
            if final["physical_grasp_error_m"] > .005 or final["orientation_error_deg"] > 3.0:
                raise BringupStop("final descent pose differs from target; diagnose before closing")
            print("DESCENT-ONLY COMPLETE: gripper OPEN; no closing or lifting.", flush=True)
            if args.render:
                hold(q_grasp, seconds=5.0, gripper_cmd=OPEN_GRIPPER, phase="DESCENT SCREENSHOT HOLD")
            return 0

        # -----------------------------------------------------
        # Phase 3: close gripper
        # -----------------------------------------------------

        print(
            "\n=== CLOSE GRIPPER ===",
            flush=True,
        )

        close_steps = max(
            2,
            int(round(
                DEMO_GRIPPER_CLOSE_SECONDS
                / DEMO_SERVO_DT
            )) + 1,
        )

        for close_index in range(
            close_steps
        ):
            u = (
                close_index
                / float(
                    close_steps - 1
                )
            )

            # Minimum-jerk S-curve
            # zero velocity / acceleration at both ends
            smooth = (
                10.0 * u**3
                - 15.0 * u**4
                + 6.0 * u**5
            )

            aperture = (
                OPEN_GRIPPER
                + (
                    CLOSE_GRIPPER
                    - OPEN_GRIPPER
                )
                * smooth
            )

            step_to_joint_target(
                q_grasp,
                float(aperture),
                "CLOSING",
            )

        hold(
            q_grasp,
            seconds=0.5,
            gripper_cmd=CLOSE_GRIPPER,
            phase="GRASP HOLD",
        )

        cube_after_close = (
            current_cube()
        )

        print(
            "cube after close:",
            cube_after_close,
            flush=True,
        )

        print(
            "contacts after close:",
            json.dumps(
                diagnostic.contacts()
            ),
            flush=True,
        )

        if args.close_only:
            print(
                "\nCLOSE-ONLY COMPLETE: "
                "gripper closed; lift was NOT executed.",
                flush=True,
            )

            print(
                "cube after close:",
                current_cube(),
                flush=True,
            )

            print(
                "contacts after close:",
                json.dumps(
                    diagnostic.contacts()
                ),
                flush=True,
            )

            print(
                "CLOSE-ONLY SNAPSHOT",
                json.dumps(
                    snapshot(
                        env,
                        diagnostic,
                    )
                ),
                flush=True,
            )

            if args.render:
                hold(
                    q_grasp,
                    seconds=1.0,
                    gripper_cmd=CLOSE_GRIPPER,
                    phase="CLOSE-ONLY VISUAL INSPECTION",
                )

            return 0

        # -----------------------------------------------------
        # Phase 4: vertical lift
        # -----------------------------------------------------

        actual_lift_start = current_grasp_point()

        lift_target = (
            actual_lift_start.copy()
        )

        lift_target[2] += (
            LIFT_HEIGHT
        )

        print(
            "\n=== SOLVE LIFT PATH ===",
            flush=True,
        )

        print(
            "lift start:",
            actual_lift_start,
            flush=True,
        )

        print(
            "lift target:",
            lift_target,
            flush=True,
        )

        lift_path = (
            solve_cartesian_path(
                start_xyz=(
                    actual_lift_start
                ),
                end_xyz=(
                    lift_target
                ),
                q_seed=(
                    q_grasp
                ),
                target_rotation=(
                    grasp_rotation
                ),
                label="LIFT",
                num_waypoints=(
                    LIFT_WAYPOINTS
                ),
            )
        )

        print(
            "\n=== EXECUTE LIFT ===",
            flush=True,
        )

        execute_joint_path(
            lift_path,
            CLOSE_GRIPPER,
            "LIFT",
        )

        q_lift = (
            lift_path[-1]
        )

        hold(
            q_lift,
            seconds=1.5,
            gripper_cmd=CLOSE_GRIPPER,
            phase="LIFT HOLD",
        )

        # -----------------------------------------------------
        # Phase 5: simple cube carry
        # -----------------------------------------------------

        #
        # This is intentionally NOT robosuite PickPlace.
        #
        # Goal:
        # prove the complete manipulation primitive once:
        #
        # pick -> lift -> carry -> place -> release -> retreat
        #
        # Use a nearby point on the same Lift table so that
        # RoboCasa-specific geometry is not involved.
        #

        PLACE_OFFSET_WORLD = np.array(
            [0.0, 0.10, 0.0],
            dtype=float,
        )

        MIN_PLACE_JOINT_MARGIN = 0.10

        place_cube_target = (
            cube_start.copy()
            + PLACE_OFFSET_WORLD
        )

        place_grasp_target = (
            place_cube_target.copy()
        )

        place_grasp_target[2] += (
            GRASP_Z_OFFSET
        )

        carry_target = (
            place_grasp_target.copy()
        )

        carry_target[2] += (
            LIFT_HEIGHT
        )

        carry_start = (
            current_grasp_point()
        )

        print(
            "\n=== SIMPLE CUBE PICK-AND-PLACE ===",
            flush=True,
        )

        print(
            "cube start:",
            cube_start,
            flush=True,
        )

        print(
            "desired cube place xyz:",
            place_cube_target,
            flush=True,
        )

        print(
            "carry start grasp xyz:",
            carry_start,
            flush=True,
        )

        print(
            "carry target grasp xyz:",
            carry_target,
            flush=True,
        )


        # --------------------------------------------------
        # Safety helper for planned joint paths
        # --------------------------------------------------

        workspace = (
            env.placement_initializer.workspace
        )

        def check_path_margin(
            q_path,
            label,
        ):
            min_margin = np.inf

            for i, q in enumerate(q_path):

                q = np.asarray(
                    q,
                    dtype=float,
                )

                margin = float(
                    np.min(
                        np.minimum(
                            q - workspace.lower,
                            workspace.upper - q,
                        )
                    )
                )

                min_margin = min(
                    min_margin,
                    margin,
                )

                if (
                    margin
                    < MIN_PLACE_JOINT_MARGIN
                ):
                    raise BringupStop(
                        f"{label}_{i + 1:02d}: "
                        f"joint-limit margin "
                        f"{margin:.6f} < "
                        f"{MIN_PLACE_JOINT_MARGIN:.3f} rad"
                    )

            print(
                f"{label}: minimum joint margin "
                f"{min_margin:.6f} rad",
                flush=True,
            )


        # --------------------------------------------------
        # Carry horizontally while holding cube
        # --------------------------------------------------

        carry_distance = float(
            np.linalg.norm(
                carry_target
                - carry_start
            )
        )

        carry_waypoints = max(
            10,
            int(
                np.ceil(
                    carry_distance / 0.005
                )
            ),
        )

        print(
            "\n=== SOLVE CARRY PATH ===",
            flush=True,
        )

        carry_path = (
            solve_cartesian_path(
                start_xyz=carry_start,
                end_xyz=carry_target,
                q_seed=q_lift,
                target_rotation=grasp_rotation,
                label="CARRY",
                num_waypoints=carry_waypoints,
            )
        )

        check_path_margin(
            carry_path,
            "CARRY",
        )

        print(
            "\n=== EXECUTE CARRY ===",
            flush=True,
        )

        execute_joint_path(
            carry_path,
            CLOSE_GRIPPER,
            "CARRY",
        )

        q_carry = (
            carry_path[-1]
        )

        hold(
            q_carry,
            seconds=0.5,
            gripper_cmd=CLOSE_GRIPPER,
            phase="CARRY HOLD",
        )


        # --------------------------------------------------
        # Descend vertically to placement height
        # --------------------------------------------------

        place_descent_start = (
            current_grasp_point()
        )

        descent_distance = float(
            np.linalg.norm(
                place_descent_start
                - place_grasp_target
            )
        )

        place_waypoints = max(
            10,
            int(
                np.ceil(
                    descent_distance / 0.002
                )
            ),
        )

        print(
            "\n=== SOLVE PLACE DESCENT ===",
            flush=True,
        )

        print(
            "place descent start:",
            place_descent_start,
            flush=True,
        )

        print(
            "place grasp target:",
            place_grasp_target,
            flush=True,
        )

        place_path = (
            solve_cartesian_path(
                start_xyz=place_descent_start,
                end_xyz=place_grasp_target,
                q_seed=q_carry,
                target_rotation=grasp_rotation,
                label="PLACE_DESCENT",
                num_waypoints=place_waypoints,
            )
        )

        check_path_margin(
            place_path,
            "PLACE_DESCENT",
        )

        print(
            "\n=== EXECUTE PLACE DESCENT ===",
            flush=True,
        )

        execute_joint_path(
            place_path,
            CLOSE_GRIPPER,
            "PLACE_DESCENT",
        )

        q_place = (
            place_path[-1]
        )

        hold(
            q_place,
            seconds=0.5,
            gripper_cmd=CLOSE_GRIPPER,
            phase="PLACE HOLD CLOSED",
        )


        # --------------------------------------------------
        # Smooth release
        # --------------------------------------------------

        print(
            "\n=== RELEASE CUBE ===",
            flush=True,
        )

        #
        # 150 servo updates = 2.5 s at the existing
        # 60 Hz demo servo rate.
        #
        # Same minimum-jerk shape used for the successful
        # smooth gripper close, but reversed.
        #

        RELEASE_STEPS = 150

        for i in range(
            RELEASE_STEPS + 1
        ):
            u = (
                i
                / RELEASE_STEPS
            )

            # Minimum-jerk S-curve:
            # h(0)=0, h(1)=1,
            # zero velocity / acceleration at ends.
            h = (
                10.0 * u**3
                - 15.0 * u**4
                + 6.0 * u**5
            )

            gripper_cmd = (
                CLOSE_GRIPPER
                + (
                    OPEN_GRIPPER
                    - CLOSE_GRIPPER
                )
                * h
            )

            step_to_joint_target(
                q_place,
                float(gripper_cmd),
                "RELEASING",
            )

        hold(
            q_place,
            seconds=1.0,
            gripper_cmd=OPEN_GRIPPER,
            phase="RELEASE SETTLE",
        )


        # --------------------------------------------------
        # Retreat vertically with gripper open
        # --------------------------------------------------

        retreat_start = (
            current_grasp_point()
        )

        retreat_target = (
            retreat_start.copy()
        )

        retreat_target[2] += 0.08

        retreat_waypoints = max(
            10,
            int(
                np.ceil(
                    0.08 / 0.002
                )
            ),
        )

        print(
            "\n=== SOLVE RETREAT ===",
            flush=True,
        )

        retreat_path = (
            solve_cartesian_path(
                start_xyz=retreat_start,
                end_xyz=retreat_target,
                q_seed=q_place,
                target_rotation=grasp_rotation,
                label="RETREAT",
                num_waypoints=retreat_waypoints,
            )
        )

        check_path_margin(
            retreat_path,
            "RETREAT",
        )

        print(
            "\n=== EXECUTE RETREAT ===",
            flush=True,
        )

        execute_joint_path(
            retreat_path,
            OPEN_GRIPPER,
            "RETREAT",
        )

        q_retreat = (
            retreat_path[-1]
        )

        hold(
            q_retreat,
            seconds=1.0,
            gripper_cmd=OPEN_GRIPPER,
            phase="FINAL SETTLE",
        )


        # --------------------------------------------------
        # Custom simple Pick-and-Place success
        # --------------------------------------------------

        cube_final = (
            current_cube()
        )

        place_xy_error = float(
            np.linalg.norm(
                cube_final[:2]
                - place_cube_target[:2]
            )
        )

        place_z_error = float(
            abs(
                cube_final[2]
                - cube_start[2]
            )
        )

        moved_xy = float(
            np.linalg.norm(
                cube_final[:2]
                - cube_start[:2]
            )
        )

        success = bool(
            place_xy_error < 0.03
            and place_z_error < 0.03
            and moved_xy > 0.06
        )

        print(
            "\n========================================"
        )
        print(
            "SIMPLE CUBE PICK-AND-PLACE RESULT"
        )
        print(
            "========================================"
        )

        print(
            "cube start        :",
            cube_start,
        )

        print(
            "desired place xyz :",
            place_cube_target,
        )

        print(
            "cube final        :",
            cube_final,
        )

        print(
            "moved xy m        :",
            moved_xy,
        )

        print(
            "place xy error m  :",
            place_xy_error,
        )

        print(
            "place z error m   :",
            place_z_error,
        )

        print(
            "success           :",
            success,
        )

        print(
            "final contacts    :",
            json.dumps(
                diagnostic.contacts()
            ),
        )

        if args.render:
            hold(
                q_retreat,
                seconds=5.0,
                gripper_cmd=OPEN_GRIPPER,
                phase="FINAL PICKPLACE VIEW",
            )

        return (
            0
            if success
            else 2
        )


    except (
        BringupStop,
        KeyboardInterrupt,
    ) as exc:
        if descent_monitor is not None:
            descent_monitor.failure = str(exc)
            print("DESCENT STOP HISTORY", json.dumps(descent_monitor.summary()), flush=True)
        print(
            f"\nSTOP: {exc}",
            flush=True,
        )

        try:
            if (
                env.diagnostics
                is not None
            ):
                print(
                    "STOP SNAPSHOT",
                    json.dumps(
                        snapshot(
                            env,
                            env.diagnostics,
                        )
                    ),
                    flush=True,
                )

        except Exception:
            pass

        return 1

    finally:
        if descent_monitor is not None:
            descent_monitor.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )