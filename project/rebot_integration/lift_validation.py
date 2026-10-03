"""Contact diagnostics and safety hooks; all Lift task methods are inherited."""

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from robosuite.environments.manipulation.lift import Lift

from .guarded_empty_env import BringupStop


class LiftDiagnostics:
    def __init__(self, env, arm_excursion_limit=0.1):
        self.env = env
        self.arm_excursion_limit = arm_excursion_limit
        self.model = env.sim.model._model
        self.table = self.model.geom("table_collision").id
        robot = env.robots[0]
        self.arm_qpos = np.array([env.sim.model.get_joint_qpos_addr(n) for n in robot.robot_model.joints])
        self.arm_qvel = np.array([env.sim.model.get_joint_qvel_addr(n) for n in robot.robot_model.joints])
        self.finger_qvel = np.array([env.sim.model.get_joint_qvel_addr(n) for n in robot.gripper["right"].joints])
        names = robot.robot_model.contact_geoms + robot.gripper["right"].contact_geoms
        self.robot_geoms = {self.model.geom(n).id for n in names}
        prefix = robot.robot_model.naming_prefix
        self.arm_geoms = [self.model.geom(prefix + n + "_collision").id
                          for n in ["base_link"] + [f"link{i}" for i in range(1, 7)]]
        self.hulls = {}
        for gid in self.arm_geoms:
            mid = self.model.geom_dataid[gid]
            start = self.model.mesh_vertadr[mid]
            vertices = self.model.mesh_vert[start:start + self.model.mesh_vertnum[mid]].astype(float)
            # Convex-hull support vertices give the same extrema as the full mesh.
            self.hulls[gid] = vertices[ConvexHull(vertices).vertices]

    def arm_table_clearance(self):
        """Conservative hull extrema in table coordinates, including deep tunneling.

        XY bounding-box overlap can over-report near an edge; it never certifies
        a collision-free path. Values are metres above the tabletop, not forces.
        """
        data = self.env.sim.data
        table_rot = data.geom_xmat[self.table].reshape(3, 3)
        half = self.model.geom_size[self.table]
        result = {}
        for gid, vertices in self.hulls.items():
            rotation = data.geom_xmat[gid].reshape(3, 3)
            world = vertices @ rotation.T + data.geom_xpos[gid]
            local = (world - data.geom_xpos[self.table]) @ table_rot
            lo, hi = local.min(axis=0), local.max(axis=0)
            overlaps = bool(np.all(hi[:2] >= -half[:2]) and np.all(lo[:2] <= half[:2]))
            result[self.model.geom(gid).name] = {
                "lowest_above_table_m": float(lo[2] - half[2]),
                "overlaps_table_xy": overlaps,
                "below_tabletop": overlaps and bool(lo[2] < half[2] - 0.001),
            }
        return result

    def contacts(self):
        data = self.env.sim.data._data
        table_contacts, robot_contacts = [], []
        for index, contact in enumerate(data.contact):
            ids = {int(contact.geom1), int(contact.geom2)}
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, force)
            entry = {
                "geoms": [self.model.geom(int(contact.geom1)).name, self.model.geom(int(contact.geom2)).name],
                "distance_m": float(contact.dist), "normal_force_N": float(force[0]),
            }
            if self.table in ids:
                table_contacts.append(entry)
            if ids & self.robot_geoms:
                robot_contacts.append(entry)
        return {"table_contacts": table_contacts, "robot_contacts": robot_contacts}

    def check(self):
        data = self.env.sim.data
        for name in ("qpos", "qvel", "qacc", "ctrl", "actuator_force"):
            if not np.isfinite(getattr(data, name)).all():
                raise BringupStop(f"non-finite {name}")
        if np.max(np.abs(data.qvel[self.arm_qvel])) > 1.0:
            raise BringupStop("arm speed exceeded 1 rad/s")
        if np.max(np.abs(data.qvel[self.finger_qvel])) > 0.15:
            raise BringupStop("finger speed exceeded 0.15 m/s")
        initial = self.env.robots[0].robot_model.init_qpos

        if (
            self.arm_excursion_limit is not None
            and np.max(
                np.abs(
                    data.qpos[self.arm_qpos] - initial
                )
            ) > self.arm_excursion_limit
        ):
            raise BringupStop(
                f"unexpected arm excursion above "
                f"{self.arm_excursion_limit:.2f} rad"
            )
        intrusion = [name for name, values in self.arm_table_clearance().items() if values["below_tabletop"]]
        if intrusion:
            raise BringupStop(f"arm hull below tabletop: {intrusion}")
        for contact in data.contact:
            ids = {int(contact.geom1), int(contact.geom2)}
            if ids & self.robot_geoms and contact.dist < -0.001:
                raise BringupStop(f"robot contact penetration: {[self.model.geom(i).name for i in ids]}")
        if np.any(data._data.warning.number):
            raise BringupStop(f"MuJoCo warning: {data._data.warning.number}")


class ReBotLiftValidation(Lift):
    """Only add guards around control; retain stock model/reset/reward/success."""

    def __init__(self, *args, **kwargs):
        self.diagnostics = None
        self.saturated_for = np.zeros(6)
        self.last_safety_time = 0.0
        super().__init__(*args, **kwargs)

    def _pre_action(self, action, policy_step=False):
        if self.diagnostics is None:
            self.diagnostics = LiftDiagnostics(self)
        if self.sim.data.time == 0:
            self.saturated_for[:] = 0
            self.last_safety_time = 0.0
        if self.sim.data.time < self.last_safety_time:
            raise BringupStop("simulation clock reset")
        self.last_safety_time = self.sim.data.time
        if not np.isfinite(action).all():
            raise BringupStop("non-finite action")
        self.diagnostics.check()
        super()._pre_action(action, policy_step)
        torque = self.robots[0].part_controllers["right"].torques
        if not np.isfinite(torque).all():
            raise BringupStop("non-finite controller torque")
        saturated = np.abs(torque) >= 0.98 * np.array([27, 27, 27, 7, 7, 7])
        self.saturated_for = np.where(saturated, self.saturated_for + self.model_timestep, 0)
        if np.any(self.saturated_for >= 0.25):
            raise BringupStop("sustained arm torque saturation")

    def _post_action(self, action):
        self.diagnostics.check()
        return super()._post_action(action)


def snapshot(env, diagnostics):
    robot = env.robots[0]
    data, model = env.sim.data, env.sim.model
    base = model.body_name2id(robot.robot_model.root_body)
    site = robot.eef_site_id["right"]
    tcp_quat = np.empty(4)
    mujoco.mju_mat2Quat(tcp_quat, data.site_xmat[site])
    cube = data.body_xpos[env.cube_body_id]
    return {
        "base_pos": data.body_xpos[base].tolist(), "base_quat_wxyz": data.body_xquat[base].tolist(),
        "tcp_pos": data.site_xpos[site].tolist(), "tcp_quat_wxyz": tcp_quat.tolist(),
        "cube_pos": cube.tolist(), "cube_quat_wxyz": data.body_xquat[env.cube_body_id].tolist(),
        "tcp_to_cube_m": float(np.linalg.norm(data.site_xpos[site] - cube)),
        "arm_qpos": data.qpos[diagnostics.arm_qpos].tolist(),
        "arm_qvel": data.qvel[diagnostics.arm_qvel].tolist(),
        "finger_qpos": data.qpos[6:8].tolist(), "ctrl": data.ctrl.tolist(),
        "arm_table_clearance": diagnostics.arm_table_clearance(),
        **diagnostics.contacts(), "stock_lift_success": bool(env._check_success()),
    }
