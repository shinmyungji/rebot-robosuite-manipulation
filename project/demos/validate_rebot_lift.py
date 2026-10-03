"""Milestone 2: visualize stock Lift placement and contacts, without grasping."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebot_integration.guarded_empty_env import BringupStop
from rebot_integration.lift_setup import BASE_POSITION, make_rebot_lift
from rebot_integration.lift_validation import LiftDiagnostics, ReBotLiftValidation, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--show-collisions", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    env = make_rebot_lift(has_renderer=args.render, seed=args.seed, env_class=ReBotLiftValidation)
    try:
        env.reset()
        env.sim.forward()
        diagnostic = LiftDiagnostics(env)
        env.diagnostics = diagnostic
        diagnostic.check()
        print("RESET", json.dumps(snapshot(env, diagnostic)), flush=True)
        cube = env.sim.data.body_xpos[env.cube_body_id].copy()
        reach = env.placement_initializer.workspace.check(cube - BASE_POSITION)
        print(f"cube_TCP_position_reachable={reach.reachable}, error_m={reach.error_m:.8f}; "
              "position-only scratch-kinematics check, not grasp/path certification", flush=True)
        if not reach.reachable:
            raise BringupStop("cube position failed reachability validation")
        if args.render:
            env.viewer.update()
            with env.viewer.viewer.lock():
                env.viewer.viewer.cam.lookat[:] = [-0.1, 0.0, 0.95]
                env.viewer.viewer.cam.distance = 1.6
                env.viewer.viewer.cam.azimuth = 135
                env.viewer.viewer.cam.elevation = -25
                env.viewer.viewer.opt.geomgroup[0] = int(args.show_collisions)
            env.viewer.viewer.sync()

        initial = env.robots[0].robot_model.init_qpos
        phase = "initial hold"
        next_report = 0.0

        def step(target, aperture=0.0):
            nonlocal next_report
            started = time.monotonic()
            if args.render and not env.viewer.viewer.is_running():
                raise BringupStop("viewer closed")
            action = np.zeros(env.action_dim)
            action[:6] = np.clip((target - env.sim.data.qpos[diagnostic.arm_qpos]) / 0.005, -1, 1)
            action[-1] = aperture
            env.step(action)
            if env.sim.data.time >= next_report:
                print(phase, json.dumps(snapshot(env, diagnostic)), flush=True)
                next_report = env.sim.data.time + 1.0
            if args.render:
                time.sleep(max(0.0, env.control_timestep - (time.monotonic() - started)))

        def hold(target, seconds, aperture=0):
            for _ in range(round(seconds / env.control_timestep)):
                step(target, aperture)

        hold(initial, 4)
        # Only milliradian perturbations around the elevated reset pose.
        for joint in range(6):
            for direction in (1, -1):
                phase = f"joint{joint + 1} small {'positive' if direction > 0 else 'negative'}"
                target = initial.copy()
                target[joint] += direction * 0.003
                hold(target, 2)
                phase = "return toward reset"
                hold(initial, 3)
        for label, start, stop in (("open", 0, -1), ("close", -1, 1), ("reopen", 1, -1)):
            phase = "gripper " + label
            for aperture in np.linspace(start, stop, 80):
                step(initial, aperture)
            hold(initial, 1, stop)
        print("VALIDATION COMPLETE", json.dumps(snapshot(env, diagnostic)), flush=True)
        print("No grasp attempted; a false stock Lift success flag is expected.", flush=True)
        return 0
    except (BringupStop, KeyboardInterrupt) as exc:
        print(f"STOP: {exc}", flush=True)
        if env.diagnostics is not None:
            print(json.dumps(snapshot(env, env.diagnostics)), flush=True)
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
