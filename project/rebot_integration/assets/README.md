# B601-DM asset provenance

Adapted on 2026-10-01 from local Seeed repositories:

- `third_party/reBot_Arm_Mujoco-DM`, revision
  `3f3e98d0682fcc382d5607f92890f44918d097d9`:
  `reBotArm_ros2_DM/src/rebotarm_mujoco/models/rebotarm_b601_colored.xml`.
- `third_party/reBot-DevArm`, revision
  `bafae7341755fd72586450ddbfdbbb4fc1b19310`:
  `Rebot_Arm_description/DM/meshes/` and `urdf/ReBot_Arm_DM.urdf`.

Mesh bytes are copied unchanged from the mechanical description. The bundled
`LICENSE` is the upstream CERN-OHL-W-2.0 license. No upstream file is modified.

Changes to MJCF: remove scene objects, lights, cameras and mocap; split the arm
and gripper at end_link; retain the end_link transform in the arm's right_hand
attachment; preserve link transforms and explicit inertials; inline joint and
geom defaults; mark visuals group 1 and collisions group 0; make mesh paths
relative to each XML; bake the canonical URDF colors/material assignments into
the assets instead of relying on ROS runtime recoloring.

Add six unit-gear torque motors (27/27/27/7/7/7 Nm caps), one position actuator
(100 N/m gain, 8 N cap), a negative finger coupling equality, and robosuite
reference sites and force/torque sensors. Preserve the source finger-to-finger
contact exclusion. Milestone 2 adds seven unchanged source collision meshes from the same DM
mechanical description under meshes/arm_collision: base_link and link1..link6.
Each becomes one MuJoCo convex mesh geom, group 0, contype/conaffinity 1,
with explicit mass=0 so the existing body inertials are preserved. Visual geoms
remain non-colliding. Segmented gripper collision geometry is unchanged.
These convex approximations do not certify arbitrary arm motions or grasping.

Mesh paths resolve wholly inside this assets directory. Runtime loading does
not need either source repository or ROS. Source-checking unit tests use the
workspace URDF to catch accidental changes to joint limits and axes.

The base/link1 mechanical flange overlaps by about 3 mm. That single joined
body pair is explicitly excluded because the base is world-welded. All arm
hulls retain environment collision masks; other non-adjacent robot contacts
remain enabled.
