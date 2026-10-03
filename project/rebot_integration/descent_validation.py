"""Per-physics-step logging for the open-gripper descent validation only."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .guarded_empty_env import BringupStop
from .lift_validation import ReBotLiftValidation


CATEGORIES = ("left_finger_cube", "right_finger_cube", "palm_cube",
              "gripper_table", "arm_table", "cube_table", "other_robot")


def contact_category(names):
    """Classify both orderings of the named contact pair."""
    cube = any(n.startswith("cube") for n in names)
    table = "table_collision" in names
    gripper = any(n.startswith("gripper") for n in names)
    arm = any(n.startswith("robot") for n in names)
    if table:
        if gripper:
            return "gripper_table"
        if arm:
            return "arm_table"
        if cube:
            return "cube_table"
    if cube and gripper:
        if any("finger_left" in n for n in names):
            return "left_finger_cube"
        if any("finger_right" in n for n in names):
            return "right_finger_cube"
        return "palm_cube"
    return "other_robot" if gripper or arm else None


class DescentMonitor:
    """Log contact solver results with the pre-integration poses they belong to.

    step1 computes geometry at t; step2 solves contacts and integrates to t+dt.
    Capture poses after step1, then read contact forces after step2. This avoids
    pairing old geometry with newly integrated qpos or altering the simulation.
    """

    def __init__(self, env, diagnostic, tcp_to_grasp, target_rotation, directory, log_name="descent_steps.jsonl"):
        self.env, self.diagnostic = env, diagnostic
        self.offset = np.asarray(tcp_to_grasp)
        self.rotation = np.asarray(target_rotation)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / log_name).open("w")
        self.phase = "RESET"
        self.baseline = None
        self.history = {key: {} for key in CATEGORIES}
        self.steps = 0
        self.max_motion = dict(xy_m=0.0, z_m=0.0, rotation_deg=0.0)
        self.min_clearance = float("inf")
        self.max_orientation = 0.0
        self.plan = {}
        self.failure = None
        self.site = env.robots[0].eef_site_id["right"]
        self.hulls = {}
        model = env.sim.model._model
        for name in env.robots[0].gripper["right"].contact_geoms:
            gid = model.geom(name).id
            mid = model.geom_dataid[gid]
            start = model.mesh_vertadr[mid]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mid]].astype(float)
            self.hulls[gid] = vertices[ConvexHull(vertices).vertices]

    def capture(self):
        data = self.env.sim.data
        rotation = data.site_xmat[self.site].reshape(3, 3)
        tcp = data.site_xpos[self.site].copy()
        cube_R = data.body_xmat[self.env.cube_body_id].reshape(3, 3).copy()
        cube = data.body_xpos[self.env.cube_body_id].copy()
        table = self.diagnostic.table
        table_R = data.geom_xmat[table].reshape(3, 3)
        half = self.env.sim.model._model.geom_size[table, 2]
        clearance = min(float(np.min(((vertices @ data.geom_xmat[gid].reshape(3, 3).T
                        + data.geom_xpos[gid] - data.geom_xpos[table]) @ table_R)[:, 2]) - half)
                        for gid, vertices in self.hulls.items())
        delta = np.zeros(3) if self.baseline is None else cube - self.baseline[0]
        cube_angle = 0.0 if self.baseline is None else float(np.rad2deg(
            Rotation.from_matrix(cube_R @ self.baseline[1].T).magnitude()))
        return {
            "time_s": float(data.time), "phase": self.phase,
            "tcp_xyz": tcp.tolist(), "physical_grasp_xyz": (tcp + rotation @ self.offset).tolist(),
            "cube_xyz": cube.tolist(), "cube_delta_xyz_m": delta.tolist(),
            "cube_rotation_change_deg": cube_angle,
            "orientation_error_deg": float(np.rad2deg(Rotation.from_matrix(
                self.rotation @ rotation.T).magnitude())),
            "gripper_table_clearance_m": clearance,
        }

    def begin_descent(self):
        data = self.env.sim.data
        self.baseline = (data.body_xpos[self.env.cube_body_id].copy(),
                         data.body_xmat[self.env.cube_body_id].reshape(3, 3).copy())

    def record(self, state, *, enforce=True, solver_completed=True):
        # Deduplicate table/robot entries while retaining all individual contact points.
        contacts = self.diagnostic.contacts()
        entries = contacts["robot_contacts"] + [c for c in contacts["table_contacts"]
                   if contact_category(c["geoms"]) == "cube_table"]
        grouped = {}
        for contact in entries:
            key = tuple(sorted(contact["geoms"]))
            category = contact_category(key)
            if category is None:
                continue
            entry = grouped.setdefault(key, dict(category=category, geoms=list(key),
                                      distance_m=contact["distance_m"], normal_force_N=0.0, points=0))
            entry["distance_m"] = min(entry["distance_m"], contact["distance_m"])
            entry["normal_force_N"] += contact["normal_force_N"]
            entry["points"] += 1
        state = dict(state, contacts=list(grouped.values()), solver_completed=solver_completed)
        # Write the offending sample before raising any guard.
        self.stream.write(json.dumps(state) + "\n")
        self.steps += 1
        self.min_clearance = min(self.min_clearance, state["gripper_table_clearance_m"])
        self.max_orientation = max(self.max_orientation, state["orientation_error_deg"])
        if self.baseline is not None:
            delta = state["cube_delta_xyz_m"]
            for key, value in (("xy_m", float(np.linalg.norm(delta[:2]))),
                               ("z_m", abs(delta[2])), ("rotation_deg", state["cube_rotation_change_deg"])):
                self.max_motion[key] = max(self.max_motion[key], value)
        for key, contact in grouped.items():
            history = self.history[contact["category"]]
            if key not in history:
                history[key] = dict(geoms=list(key), first_time_s=state["time_s"],
                    first_phase=state["phase"], first_physical_grasp_xyz=state["physical_grasp_xyz"],
                    first_cube_xyz=state["cube_xyz"], first_distance_m=contact["distance_m"],
                    first_force_N=contact["normal_force_N"], max_force_N=0.0,
                    minimum_distance_m=contact["distance_m"], samples=0)
            item = history[key]
            item["last_time_s"] = state["time_s"]
            item["samples"] += 1
            item["max_force_N"] = max(item["max_force_N"], contact["normal_force_N"])
            item["minimum_distance_m"] = min(item["minimum_distance_m"], contact["distance_m"])
        if not enforce:
            return
        bad = [c for c in grouped.values() if c["category"] in
               ("palm_cube", "gripper_table", "arm_table", "other_robot")]
        fingers = [c for c in grouped.values() if c["category"] in ("left_finger_cube", "right_finger_cube")]
        if fingers:
            forces = {k: sum(c["normal_force_N"] for c in fingers if c["category"] == k)
                      for k in ("left_finger_cube", "right_finger_cube")}
            expected_height = abs(state["physical_grasp_xyz"][2] - state["cube_xyz"][2]) < 0.04
            symmetric = set(c["category"] for c in fingers) == set(forces)
            light = max(forces.values()) <= 0.5 and abs(forces["left_finger_cube"] - forces["right_finger_cube"]) <= 0.2
            front_only = all(any("front_collision" in name for name in c["geoms"]) for c in fingers)
            if not (symmetric and light and front_only and expected_height):
                bad += fingers
        if bad:
            raise BringupStop(f"premature descent contact at physical grasp z={state['physical_grasp_xyz'][2]:.9f} m: {bad}")
        if self.baseline is not None and (self.max_motion["xy_m"] > .001 or
                self.max_motion["z_m"] > .001 or self.max_motion["rotation_deg"] > 1.0):
            raise BringupStop(f"cube moved during open descent: {self.max_motion}")
        if state["orientation_error_deg"] > 3.0:
            raise BringupStop(f"descent orientation error: {state['orientation_error_deg']} deg")
        if state["gripper_table_clearance_m"] < 0:
            raise BringupStop("gripper collision hull below tabletop")

    def summary(self):
        return dict(physics_samples=self.steps, model_timestep_s=self.env.model_timestep,
                    maximum_cube_motion=self.max_motion, minimum_gripper_table_clearance_m=self.min_clearance,
                    maximum_orientation_error_deg=self.max_orientation,
                    contact_history={k: list(v.values()) for k, v in self.history.items()}, plan=self.plan,
                    failure=self.failure)

    def close(self):
        self.stream.close()
        (self.directory / "descent_summary.json").write_text(json.dumps(self.summary(), indent=2) + "\n")


class ReBotDescentValidation(ReBotLiftValidation):
    """Add logging around each physics integration, retaining stock task/control."""

    def __init__(self, *args, **kwargs):
        self.descent_monitor = None
        self._descent_pending = None
        super().__init__(*args, **kwargs)

    def _pre_action(self, action, policy_step=False):
        if self.descent_monitor is not None:
            self._descent_pending = self.descent_monitor.capture()
        try:
            super()._pre_action(action, policy_step)
        except BringupStop:
            if self._descent_pending is not None:
                self.descent_monitor.record(self._descent_pending, enforce=False, solver_completed=False)
                self._descent_pending = None
            raise

    def _update_observables(self, force=False):
        # Stock env invokes this immediately after every step2 / step.
        if self._descent_pending is not None:
            pending, self._descent_pending = self._descent_pending, None
            self.descent_monitor.record(pending)
        super()._update_observables(force=force)
