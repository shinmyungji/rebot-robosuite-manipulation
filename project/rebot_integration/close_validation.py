"""Instrumented gripper closing at a fixed, already validated arm target."""

import json
import os
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .descent_validation import DescentMonitor, CATEGORIES, contact_category
from .guarded_empty_env import BringupStop


class CloseMonitor(DescentMonitor):
    MAX_ONE_SIDED_SECONDS = 0.15
    MAX_SIDE_FORCE_N = 5.0
    MAX_CUBE_XY_M = 0.001
    MAX_CUBE_Z_M = 0.002
    MAX_CUBE_ROTATION_DEG = 2.0

    def __init__(self, env, diagnostic, offset, rotation, directory, arm_target, render=False):
        super().__init__(env, diagnostic, offset, rotation, directory, log_name="close_steps.jsonl")
        self.arm_target = np.asarray(arm_target).copy()
        self.command = -1.0
        self.begin_descent()  # Reset cube-motion reference to this pre-close state.
        self.started_at = float(env.sim.data.time)
        self.first = {"left": None, "right": None}
        self.first_loaded = {"left": None, "right": None}
        self.peaks = {"left": 0.0, "right": 0.0}
        self.peak_impulses = {"left": 0.0, "right": 0.0}
        self.one_sided_since = None
        self.first_bilateral = None
        self.hold_samples = 0
        self.hold_bilateral = 0
        self.hold_between = 0
        self.hold_baseline = None
        self.hold_max_motion = {"position_m": 0.0, "rotation_deg": 0.0}
        self.max_arm_error = 0.0
        self.last = None
        self.render = render
        self.snapshots = []
        self.contacts_history = {k: {} for k in CATEGORIES}
        robot = env.robots[0]
        gripper = robot.gripper["right"]
        model = env.sim.model._model
        self.fingers = [env.sim.model.get_joint_qpos_addr(n) for n in gripper.joints]
        self.front_ids = [model.geom(gripper.naming_prefix + f"finger_{side}_front_collision").id
                          for side in ("left", "right")]
        mujoco.mj_saveModel(model, str(self.directory / "close_scene.mjb"), None)
        initial = self.capture()
        self.snapshot("before_close", initial)
        (self.directory / "before_close.json").write_text(json.dumps(self.public(initial), indent=2) + "\n")

    @staticmethod
    def public(state):
        return {k: v for k, v in state.items() if not k.startswith("_")}

    def capture(self):
        state = super().capture()
        data = self.env.sim.data
        rotation = data.site_xmat[self.site].reshape(3, 3).copy()
        q = data.qpos[self.diagnostic.arm_qpos].copy()
        fingers = data.qpos[self.fingers].copy()
        front = data.geom_xpos[self.front_ids].copy()
        cube = data.body_xpos[self.env.cube_body_id].copy()
        lateral = rotation[:, 1]
        cube_between = bool(np.dot(front[0] - cube, lateral) > 0 and
                            np.dot(front[1] - cube, lateral) < 0)
        state.update(
            close_time_s=float(data.time - self.started_at), gripper_command=float(self.command),
            arm_qpos=q.tolist(), arm_target=self.arm_target.tolist(),
            max_arm_target_error_rad=float(np.max(np.abs(q - self.arm_target))),
            left_finger_qpos_m=float(fingers[0]), right_finger_qpos_m=float(fingers[1]),
            slide_travel_separation_m=float(fingers[0] - fingers[1]),
            front_geom_center_separation_m=float(np.linalg.norm(front[0] - front[1])),
            front_geom_xyz=front.tolist(), cube_between_fingers=cube_between,
            cube_xy_displacement_m=float(np.linalg.norm(state["cube_delta_xyz_m"][:2])),
            cube_z_displacement_m=float(state["cube_delta_xyz_m"][2]),
            _cube_rotation=data.body_xmat[self.env.cube_body_id].reshape(3, 3).copy(),
            _lateral=lateral, _qpos=data.qpos.copy(), _qvel=data.qvel.copy(),
        )
        return state

    def snapshot(self, name, state):
        # Save pre-integration qpos: it is the state that generated this contact solve.
        path = self.directory / (name + ".npz")
        np.savez(path, qpos=state["_qpos"], qvel=state["_qvel"], time=state["time_s"],
                 lookat=np.asarray(state["cube_xyz"]) + [0, 0, .06])
        self.snapshots.append(name)

    def begin_hold(self):
        state = self.capture()
        self.hold_baseline = (np.asarray(state["cube_xyz"]), state["_cube_rotation"].copy())

    def record(self, state, *, enforce=True, solver_completed=True):
        data, model = self.env.sim.data._data, self.env.sim.model._model
        contacts = []
        sides = {"left": [], "right": []}
        unexpected = []
        for index, c in enumerate(data.contact):
            names = [model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name]
            category = contact_category(names)
            if category is None:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            normal = np.asarray(c.frame[:3]).copy()
            entry = dict(category=category, geoms=names, position_world=c.pos.tolist(),
                         normal_geom0_to_geom1_world=normal.tolist(),
                         normal_force_N=float(force[0]), distance_m=float(c.dist),
                         penetration_m=max(0.0, -float(c.dist)))
            contacts.append(entry)
            if category in ("left_finger_cube", "right_finger_cube"):
                side = "left" if category.startswith("left") else "right"
                entry["cube_outward_normal_world"] = (normal if names[0].startswith("cube") else -normal).tolist()
                entry["lateral_position_relative_to_cube_m"] = float(np.dot(
                    np.asarray(c.pos) - state["cube_xyz"], state["_lateral"]))
                sides[side].append(entry)
            elif category != "cube_table":
                unexpected.append(entry)
        totals = {side: sum(c["normal_force_N"] for c in values) for side, values in sides.items()}
        bilateral = all(totals[side] > 1e-4 for side in sides)
        opposite = False
        if bilateral:
            centers = {side: np.mean([c["lateral_position_relative_to_cube_m"] for c in values])
                       for side, values in sides.items()}
            normals = {side: np.mean([c["cube_outward_normal_world"] for c in values], axis=0)
                       for side, values in sides.items()}
            dot = float(np.dot(normals["left"], normals["right"]) /
                        max(1e-12, np.linalg.norm(normals["left"]) * np.linalg.norm(normals["right"])))
            opposite = bool(centers["left"] > 0 and centers["right"] < 0 and dot < -0.9)
        row = dict(self.public(state), physics_step=self.steps, solver_completed=solver_completed,
                   contacts=contacts, side_normal_force_N=totals, bilateral_contact=bilateral,
                   opposite_cube_sides=opposite, actuator_ctrl=self.env.sim.data.ctrl.tolist())
        self.stream.write(json.dumps(row) + "\n")
        self.last = row
        self.steps += 1
        for c in contacts:
            history = self.contacts_history[c["category"]]
            key = tuple(c["geoms"])
            if key not in history:
                history[key] = dict(first_time_s=state["time_s"], first_close_time_s=state["close_time_s"],
                                    first_physics_step=row["physics_step"], first_contact=c,
                                    samples=0, peak_point_force_N=0.0, max_penetration_m=0.0)
            entry = history[key]
            entry["samples"] += 1
            entry["last_time_s"] = state["time_s"]
            entry["peak_point_force_N"] = max(entry["peak_point_force_N"], c["normal_force_N"])
            entry["max_penetration_m"] = max(entry["max_penetration_m"], c["penetration_m"])
        for side, values in sides.items():
            event = dict(time_s=state["time_s"], close_time_s=state["close_time_s"],
                         physics_step=row["physics_step"], contacts=values,
                         gripper_command=state["gripper_command"],
                         physical_grasp_xyz=state["physical_grasp_xyz"])
            if values and self.first[side] is None:
                self.first[side] = event
            if totals[side] > 1e-4 and self.first_loaded[side] is None:
                self.first_loaded[side] = event
            self.peaks[side] = max(self.peaks[side], totals[side])
            self.peak_impulses[side] = max(self.peak_impulses[side], totals[side] * self.env.model_timestep)
        if bilateral and self.first_bilateral is None:
            self.first_bilateral = dict(time_s=state["time_s"], close_time_s=state["close_time_s"],
                                        physics_step=row["physics_step"], opposite_cube_sides=opposite)
            self.snapshot("first_bilateral_contact", state)
            print("FIRST BILATERAL CONTACT", json.dumps(self.first_bilateral), flush=True)
        self.max_motion["xy_m"] = max(self.max_motion["xy_m"], row["cube_xy_displacement_m"])
        self.max_motion["z_m"] = max(self.max_motion["z_m"], abs(row["cube_z_displacement_m"]))
        self.max_motion["rotation_deg"] = max(self.max_motion["rotation_deg"], row["cube_rotation_change_deg"])
        self.min_clearance = min(self.min_clearance, row["gripper_table_clearance_m"])
        self.max_arm_error = max(self.max_arm_error, row["max_arm_target_error_rad"])
        if self.hold_baseline is not None:
            self.hold_samples += 1
            self.hold_bilateral += int(bilateral and opposite)
            self.hold_between += int(row["cube_between_fingers"])
            self.hold_max_motion["position_m"] = max(self.hold_max_motion["position_m"],
                float(np.linalg.norm(np.asarray(row["cube_xyz"]) - self.hold_baseline[0])))
            self.hold_max_motion["rotation_deg"] = max(self.hold_max_motion["rotation_deg"],
                float(np.rad2deg(Rotation.from_matrix(state["_cube_rotation"] @ self.hold_baseline[1].T).magnitude())))
        if not enforce:
            return
        if unexpected:
            raise BringupStop(f"unexpected closing contact: {unexpected}")
        touching = [side for side, values in sides.items() if values]
        if len(touching) == 1:
            if self.one_sided_since is None:
                self.one_sided_since = state["time_s"]
            if state["time_s"] - self.one_sided_since > self.MAX_ONE_SIDED_SECONDS:
                raise BringupStop(f"one-sided contact exceeded {self.MAX_ONE_SIDED_SECONDS} s: {touching}")
        else:
            self.one_sided_since = None
        if max(totals.values()) > self.MAX_SIDE_FORCE_N:
            raise BringupStop(f"finger force exceeded {self.MAX_SIDE_FORCE_N} N per side: {totals}")
        if row["cube_xy_displacement_m"] > self.MAX_CUBE_XY_M or abs(row["cube_z_displacement_m"]) > self.MAX_CUBE_Z_M or row["cube_rotation_change_deg"] > self.MAX_CUBE_ROTATION_DEG:
            raise BringupStop(f"cube moved during closing: XYZ={row['cube_delta_xyz_m']}, rotation={row['cube_rotation_change_deg']} deg")
        if row["max_arm_target_error_rad"] > .01:
            raise BringupStop(f"arm deviated from fixed alignment q: {row['max_arm_target_error_rad']} rad")
        if bilateral and not opposite:
            raise BringupStop("bilateral contacts are not on opposite cube sides")
        if self.hold_baseline is not None and (not bilateral or not row["cube_between_fingers"]):
            raise BringupStop("cube is not retained bilaterally during closed hold")
        if self.hold_baseline is not None and (self.hold_max_motion["position_m"] > .0002 or self.hold_max_motion["rotation_deg"] > .5):
            raise BringupStop(f"cube unstable during closed hold: {self.hold_max_motion}")

    def summary(self):
        delta = None if any(v is None for v in self.first.values()) else abs(
            self.first["left"]["time_s"] - self.first["right"]["time_s"])
        ratio = None
        if self.last and max(self.last["side_normal_force_N"].values()) > 0:
            forces = self.last["side_normal_force_N"]
            ratio = abs(forces["left"] - forces["right"]) / max(forces.values())
        return dict(physics_samples=self.steps, timestep_s=self.env.model_timestep,
                    first_contact=self.first, first_loaded_contact=self.first_loaded,
                    first_contact_time_difference_s=delta, first_bilateral=self.first_bilateral,
                    peak_side_normal_force_N=self.peaks, peak_side_normal_impulse_Ns=self.peak_impulses,
                    maximum_cube_motion=self.max_motion, maximum_arm_target_error_rad=self.max_arm_error,
                    minimum_gripper_table_clearance_m=self.min_clearance,
                    closed_hold_samples=self.hold_samples,
                    closed_hold_bilateral_samples=self.hold_bilateral,
                    closed_hold_cube_between_samples=self.hold_between,
                    closed_hold_maximum_cube_motion=self.hold_max_motion,
                    final_force_relative_asymmetry=ratio,
                    approximately_symmetric=bool(delta is not None and delta <= .15 and ratio is not None and ratio < .25),
                    contact_history={k: list(v.values()) for k, v in self.contacts_history.items()},
                    snapshots=self.snapshots, failure=self.failure)

    def finish(self):
        if not self.last or self.first_bilateral is None:
            raise BringupStop("gripper reached closed command without bilateral contact")
        if self.hold_samples * self.env.model_timestep < .5 or self.hold_bilateral != self.hold_samples:
            raise BringupStop("bilateral grasp did not persist for the closed hold")
        self.snapshot("after_closed_hold", self.capture())

    def close(self):
        self.stream.close()
        (self.directory / "contact_summary.json").write_text(json.dumps(self.summary(), indent=2) + "\n")
        if self.last:
            (self.directory / "final_measurement.json").write_text(json.dumps(self.last, indent=2) + "\n")
        # Separate GL process renders exact recorded states, without advancing physics.
        if self.render:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
            env["MUJOCO_GL"] = "glfw"
            subprocess.run([sys.executable, "-B", "-m", "rebot_integration.close_validation",
                            str(self.directory), *self.snapshots], env=env, check=True)


def render_snapshots(directory, names):
    from PIL import Image
    directory = Path(directory)
    model = mujoco.MjModel.from_binary_path(str(directory / "close_scene.mjb"))
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        for name in names:
            state = np.load(directory / (name + ".npz"))
            data.qpos[:] = state["qpos"]
            data.qvel[:] = state["qvel"]
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = state["lookat"]
            camera.distance, camera.azimuth, camera.elevation = .55, 135, -20
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render()).save(directory / (name + ".png"))
            print("Saved rendered snapshot:", directory / (name + ".png"), flush=True)


if __name__ == "__main__":
    render_snapshots(sys.argv[1], sys.argv[2:])
