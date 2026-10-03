"""Bring-up-only guards around stock robosuite controllers and physics steps."""

import numpy as np

from .empty_env import ReBotB601DMEmpty


class BringupStop(RuntimeError):
    """A guard failed; the caller must stop stepping and close the viewer."""


class GuardedReBotB601DMEmpty(ReBotB601DMEmpty):
    # Deliberately strict for the demo's < 0.01 rad motions.
    arm_speed_limit = 1.0  # rad/s
    finger_speed_limit = 0.15  # m/s
    arm_excursion_limit = 0.10  # rad from reset
    saturation_fraction = 0.98
    saturation_seconds = 0.25

    def __init__(self, **kwargs):
        self.guard_armed = False
        super().__init__(**kwargs)

    def arm_guards(self):
        robot = self.robots[0]
        self.arm_qpos = np.array([self.sim.model.get_joint_qpos_addr(n) for n in robot.robot_model.joints])
        self.arm_qvel = np.array([self.sim.model.get_joint_qvel_addr(n) for n in robot.robot_model.joints])
        self.finger_qvel = np.array([self.sim.model.get_joint_qvel_addr(n) for n in robot.gripper["right"].joints])
        self.arm_actuators = np.array([self.sim.model.actuator_name2id(n) for n in robot.robot_model.actuators])
        self.initial_arm_qpos = self.sim.data.qpos[self.arm_qpos].copy()
        self.saturated_for = np.zeros(6)
        self.saturated = np.zeros(6, dtype=bool)
        self.last_guard_time = self.sim.data.time
        self.guard_armed = True
        self.check_state()

    def reset(self):
        self.guard_armed = False
        obs = super().reset()
        self.arm_guards()
        return obs

    def torque_output(self):
        torque = self.robots[0].part_controllers["right"].torques
        return np.zeros(6) if torque is None else np.asarray(torque)

    def check_state(self):
        data = self.sim.data
        for name in ("qpos", "qvel", "qacc", "ctrl", "actuator_force"):
            if not np.isfinite(getattr(data, name)).all():
                raise BringupStop(f"non-finite {name}")
        if np.max(np.abs(data.qvel[self.arm_qvel])) > self.arm_speed_limit:
            raise BringupStop("arm speed exceeded 1 rad/s")
        if np.max(np.abs(data.qvel[self.finger_qvel])) > self.finger_speed_limit:
            raise BringupStop("finger speed exceeded 0.15 m/s")
        if np.max(np.abs(data.qpos[self.arm_qpos] - self.initial_arm_qpos)) > self.arm_excursion_limit:
            raise BringupStop("arm excursion exceeded 0.10 rad from initial pose")
        # All eight joints are scalar; tolerate only small soft-limit excursions.
        ranges = self.sim.model.jnt_range
        tolerance = np.array([0.01] * 6 + [0.002] * 2)
        if np.any(data.qpos < ranges[:, 0] - tolerance) or np.any(data.qpos > ranges[:, 1] + tolerance):
            raise BringupStop("joint-limit violation")
        if abs(float(data.qpos[6] + data.qpos[7])) > 0.002:
            raise BringupStop("finger coupling error exceeded 2 mm")
        if not np.isfinite(data.time) or data.time < self.last_guard_time:
            raise BringupStop("simulation clock reset or became non-finite")
        if np.any(data._data.warning.number):
            raise BringupStop(f"MuJoCo warning counters: {data._data.warning.number}")
        self.last_guard_time = data.time

    def check_torques(self):
        torque = self.torque_output()
        if not np.isfinite(torque).all():
            raise BringupStop("non-finite controller output torque")
        limits = self.sim.model.actuator_ctrlrange[self.arm_actuators]
        self.saturated = (torque <= self.saturation_fraction * limits[:, 0]) | (
            torque >= self.saturation_fraction * limits[:, 1]
        )
        self.saturated_for = np.where(self.saturated, self.saturated_for + self.model_timestep, 0.0)
        if np.any(self.saturated_for >= self.saturation_seconds):
            joints = (np.flatnonzero(self.saturated_for >= self.saturation_seconds) + 1).tolist()
            raise BringupStop(f"sustained torque saturation on joints {joints} (98% for 0.25 s)")

    def step(self, action):
        if not self.guard_armed:
            raise BringupStop("reset and arm the guards before stepping")
        if not np.isfinite(action).all():
            raise BringupStop("non-finite policy action")
        self.check_state()
        return super().step(action)

    def _pre_action(self, action, policy_step=False):
        # Called before EVERY 2 ms physics integration, not just every 50 ms policy step.
        self.check_state()
        super()._pre_action(action, policy_step)
        self.check_torques()
        self.check_state()

    def _post_action(self, action):
        self.check_state()
        return super()._post_action(action)
