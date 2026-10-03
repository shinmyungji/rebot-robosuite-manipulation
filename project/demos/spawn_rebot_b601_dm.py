"""Guarded, real-time B601-DM empty-scene visual bring-up; use --render."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rebot_integration  # noqa: E402,F401 -- registration
from rebot_integration.guarded_empty_env import BringupStop, GuardedReBotB601DMEmpty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    env = GuardedReBotB601DMEmpty(has_renderer=args.render)
    phase = "reset"
    try:
        env.reset()
        robot = env.robots[0]
        controller = robot.part_controllers["right"]
        initial = env.initial_arm_qpos.copy()
        delta_scale = np.asarray(controller.output_max)
        print("Arm joints:", robot.robot_model.joints, flush=True)
        print("Actuators:", robot.robot_model.actuators + robot.gripper["right"].actuators, flush=True)
        print("qpos: 6 arm rad + 2 finger m; qvel: rad/s + m/s; ctrl: 6 Nm + finger m; torque_raw: Nm", flush=True)
        print("STOP: non-finite state/torque, arm speed >1 rad/s, finger speed >0.15 m/s, "
              "excursion >0.10 rad, limits/coupling violation, MuJoCo warning/reset, "
              "or any arm at >=98% torque limit for 0.25 s. Ctrl-C/window close also stops.", flush=True)
        if args.render:
            env.viewer.update()
            handle = env.viewer.viewer
            with handle.lock():
                handle.cam.lookat[:] = [0.10, 0.0, 0.22]
                handle.cam.distance = 0.95
                handle.cam.azimuth = 135
                handle.cam.elevation = -22
            handle.sync()
        next_report = 0.0
        peak_speed = np.zeros(8)
        peak_torque = np.zeros(6)
        any_saturation = False

        def report():
            data = env.sim.data
            raw = env.torque_output()
            limits = env.sim.model.actuator_ctrlrange[env.arm_actuators]
            clipped = (raw < limits[:, 0]) | (raw > limits[:, 1])
            print(f"t={data.time:.2f} phase={phase} "
                  f"qpos={np.round(data.qpos, 6).tolist()} "
                  f"qvel={np.round(data.qvel, 6).tolist()} "
                  f"ctrl={np.round(data.ctrl, 6).tolist()} "
                  f"torque_raw={np.round(raw, 6).tolist()} "
                  f"saturated_98pct={env.saturated.tolist()} "
                  f"clipped={clipped.tolist()} "
                  f"sat_seconds={np.round(env.saturated_for, 3).tolist()} "
                  f"gripper_force_N={data.actuator_force[-1]:.4f}", flush=True)

        def step_toward(target, aperture=0.0):
            nonlocal next_report, peak_speed, peak_torque, any_saturation
            started = time.monotonic()
            if args.render and not env.viewer.viewer.is_running():
                raise BringupStop("viewer closed")
            # Convert absolute waypoints to bounded stock JOINT_POSITION deltas.
            # This holds all other joints and returns physically, without qpos writes.
            action = np.zeros(7)
            action[:6] = np.clip((target - env.sim.data.qpos[env.arm_qpos]) / delta_scale, -1, 1)
            action[-1] = aperture
            env.step(action)
            peak_speed = np.maximum(peak_speed, np.abs(env.sim.data.qvel))
            peak_torque = np.maximum(peak_torque, np.abs(env.torque_output()))
            any_saturation |= bool(env.saturated.any())
            if env.sim.data.time >= next_report:
                report()
                next_report = env.sim.data.time + 0.5
            if args.render:
                if not env.viewer.viewer.is_running():
                    raise BringupStop("viewer closed")
                time.sleep(max(0.0, env.control_timestep - (time.monotonic() - started)))

        def hold(target, seconds, aperture=0.0):
            for _ in range(round(seconds / env.control_timestep)):
                step_toward(target, aperture)

        def return_to_initial():
            nonlocal phase
            phase = "return to initial"
            settled = 0
            for _ in range(round(20 / env.control_timestep)):
                step_toward(initial)
                error = np.max(np.abs(env.sim.data.qpos[env.arm_qpos] - initial))
                speed = np.max(np.abs(env.sim.data.qvel[env.arm_qvel]))
                settled = settled + 1 if error < 0.001 and speed < 0.002 else 0
                if settled >= 10:
                    print(f"RETURN VERIFIED: max error={error:.6f} rad, max speed={speed:.6f} rad/s", flush=True)
                    return
            raise BringupStop("failed to settle at initial pose within 20 seconds")

        phase = "initial hold"
        hold(initial, 4.0)
        for joint in range(6):
            for direction in (1, -1):
                phase = f"joint{joint + 1} {'positive' if direction > 0 else 'negative'}"
                start = env.sim.data.qpos[env.arm_qpos].copy()
                target = initial.copy()
                target[joint] += direction * 0.008  # 0.46 degrees
                for blend in np.linspace(0, 1, round(3 / env.control_timestep)):
                    step_toward(start + blend * (target - start))
                hold(target, 2.0)
                displacement = env.sim.data.qpos[env.arm_qpos[joint]] - initial[joint]
                if direction * displacement < 0.0005:
                    raise BringupStop(f"joint{joint + 1} did not move in the requested direction")
                print(f"MOTION VERIFIED: joint{joint + 1}, signed displacement={displacement:.6f} rad", flush=True)
                return_to_initial()

        for label, start, stop in (("open", 0.0, -1.0), ("close", -1.0, 1.0), ("reopen", 1.0, -1.0)):
            phase = f"gripper {label}"
            for aperture in np.linspace(start, stop, round(4 / env.control_timestep)):
                step_toward(initial, aperture)
            hold(initial, 1.5, stop)
            left = env.sim.data.qpos[6]
            if (stop < 0 and left < 0.045) or (stop > 0 and left > 0.005):
                raise BringupStop(f"gripper failed to reach {label} endpoint")
            report()
        phase = "final hold"
        hold(initial, 3.0, -1.0)
        print(f"PASS: all 12 signed joint motions and returns, slow gripper open/close/reopen; "
              f"peak_policy_sample_qvel={peak_speed.tolist()}, "
              f"peak_policy_sample_torque={peak_torque.tolist()}, "
              f"any_policy_sample_saturation={any_saturation}; no guard triggered.", flush=True)
    except (BringupStop, KeyboardInterrupt) as exc:
        data = env.sim.data
        print(f"STOP IMMEDIATELY: {exc}; phase={phase}; qpos={data.qpos}; "
              f"qvel={data.qvel}; ctrl={data.ctrl}; torque_raw={env.torque_output()}; "
              f"saturated={getattr(env, 'saturated', None)}", flush=True)
        return 1
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
