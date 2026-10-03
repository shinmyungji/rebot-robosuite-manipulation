# Minimal reBot B601-DM integration

Import `rebot_integration` with `project/` on `sys.path` to register the model,
FixedBaseRobot wrapper mapping, and gripper. The robot uses NullMount and the DM
ready pose `[0, -0.75, -0.55, 0, 0, 0]`.

`empty_env.ReBotB601DMEmpty` combines this robot with EmptyArena and explicitly
loads `project/configs/rebot_b601_dm_joint_position.json`. The BASIC composite
uses JOINT_POSITION with fixed impedance, kp=20, critical damping ratio, gravity
compensation, and at most 0.005 rad per policy command. Arm actuator controls
are torques, not joint angle targets. No external generalized forces are applied.

The seven policy actions are joint1 through joint6, then gripper aperture.
The stateless gripper action is -1 open, +1 closed, 0 half open. Its actuator
control is a left-finger position in metres, with equality-coupled right travel.
Both reset finger positions and a zero action correspond to 0.025 m opening per
finger. The stock joint delta controller sets goals relative to measured qpos;
zero deltas do not latch a previous absolute goal.

From the workspace root, model/unit tests only:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/rebot-numba-cache python -B -m unittest discover -s project/tests -p 'test_rebot_b601_dm*.py' -v
```

The guarded GUI demo holds for four seconds, moves each joint to +0.008 and
-0.008 rad relative to its initial pose, and verifies a physical return within
0.001 rad between every motion. Commands use bounded stock controller deltas;
there are no reset/qpos writes between motions. The fingers open, close, and
reopen with four-second ramps. The camera is centered on the small robot and
simulation is paced to wall time.

From the workspace root in the `rebot-sim` environment:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/rebot-numba-cache python -B project/demos/spawn_rebot_b601_dm.py --render
```

Telemetry prints every 0.5 simulation seconds: all qpos/qvel, actuator ctrl,
raw arm controller torque before clipping, per-joint saturation/clipping flags,
saturation duration, and gripper actuator force. Arm ctrl is Nm; gripper ctrl
is metres. Closing the viewer or Ctrl-C stops the demo. It exits after the
verification sequence.

Guards check each physics substep and stop on non-finite state/control/torque,
arm speed above 1 rad/s, finger speed above 0.15 m/s, arm excursion above 0.10 rad,
joint-limit/coupling violations, a simulation clock reset or MuJoCo warning,
or torque at 98% of an actuator limit for 0.25 seconds. Failure to return to the
initial pose or reach a commanded endpoint also stops. These thresholds apply
to this small-motion bring-up, not general robot tasks.

There are no task objects, Lift/PickPlace,
or RoboCasa dependencies. Table/bin offsets and robot envelope dimensions are
provisional; only the empty scene is tested. Milestone 2 adds arm collision hulls, but finite-state checks alone still do
not establish collision safety. Asset origins
and modifications are documented in `assets/README.md`.


## Milestone 2: stock Lift validation

`lift_setup.make_rebot_lift()` instantiates stock `Lift` with the existing BASIC /
JOINT_POSITION config. The validation-only subclass adds safety hooks before
physics steps; `_load_model`, `_reset_internal`, reward, observations, and success
remain the stock Lift implementations. No OSC, autonomous grasp, demonstrations,
PickPlace, RoboCasa, or AutoFocus integration is included.

For the stock 0.8 x 0.8 m table, top z=0.8 m, NullMount places the base at
(-0.30, 0, 0.80). The source base hull extends +/-0.07 m in x and +/-0.10 m in y,
with its bottom at zero, leaving 3 cm from the left table edge. This factory is
intentionally limited to the stock table dimensions and height.

Cube CENTER ranges are world x=[-0.02, 0.04], y=[-0.03, 0.03]: 28--34 cm in
front of the base. Rotation is fixed for this validation stage. The stock cube
size and 1 cm reset drop remain unchanged. A custom placement sampler rejects
positions without a bounded numerical TCP-position witness at both reset and
settled heights (0.5 mm tolerance). The solver uses independent robot MjData,
never steps physics, and never commands the live arm. This is a position-only
reachability check, not orientation, collision-free approach, or grasp certification.

Arm collisions use seven source convex mesh hulls, with default MuJoCo adjacent
body filtering. The original gripper collision segments and their exclusion are
preserved. Diagnostics report named table / robot contact pairs and normal forces.
A separate conservative XY-overlap / lowest-hull-point check flags arm geometry
more than 1 mm below the tabletop, including deep penetration without contacts.
It can over-report at table edges; it is a stop guard, not a path planner.

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/rebot-numba-cache python -B -m unittest discover -s project/tests -p 'test_rebot*.py' -v
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/rebot-numba-cache python -B project/demos/validate_rebot_lift.py --render
```

Add `--show-collisions` to display collision hulls and `--seed 7` to select cube
sampling. The demo prints base/TCP/cube poses, distance, joint states, table and
robot contacts, and hull clearances. It holds, makes +/-0.003 rad perturbations
around reset, and ramps the gripper. It never approaches or grasps the cube.
A false success flag is expected. Unit tests verify the unchanged success rule
using synthetic cube heights: center z > 0.84 m, with no grasp requirement.

The base/link1 mechanical flange overlaps by about 3 mm. That single joined
body pair is explicitly excluded because the base is world-welded. All arm
hulls retain environment collision masks; other non-adjacent robot contacts
remain enabled.
