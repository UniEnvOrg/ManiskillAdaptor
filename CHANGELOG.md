# Changelog

All notable changes to this project will be documented in this file.

## 0.0.1a4 (2026-10-17)

- Relicensed under the Apache License 2.0 (previously MIT); `NOTICE` carries the copyright and third-party attributions.
- Describe the spaces with unienv space metadata (requires `unienv>=0.0.1b16`), from what ManiSkill knows about the data rather than from key names:
  - `agent/qpos` / `agent/qvel`: `kind="proprio"` with the robot's active joint names and per-joint units from the joint types (`rad` / `m`, `rad/s` / `m/s`); controller targets (`agent/controller/...`) likewise.
  - `sensor_data/<camera>/...`: `source` = camera uid and `fps` = `control_freq`; `rgb` (and `albedo`) as `uint8` HWC RGB images, `depth` in `mm` (HWC, one channel), `segmentation`, `position` (`mm`, OpenGL camera frame), `normal`; in `obs_mode="sensor_data"` the raw shader textures (`Color` RGBA, `PositionSegmentation` / `Position` / `Segmentation`) of the `minimal`, `default` and `rt*` shader packs.
  - `sensor_param/<camera>/...` as `extrinsics` / `intrinsics`; `pointcloud/xyzw` (world frame, `w` = validity), `pointcloud/rgb`, `pointcloud/segmentation`.
  - The flat `state` vector (`obs_mode="state"`, or the `state` key) as `kind="state"` with a name per element (`agent/qpos/panda_joint1`, `extra/tcp_pose/qw`, ...).
  - `extra/*_pose` (7 values) as poses with a `quat_wxyz` rotation and `extra/*_pos` (3 values) in `m`, following ManiSkill's naming convention; other task extras are left undescribed.
  - The action from the controllers of the control mode: `control` / `mode` (`joint_position` / `joint_velocity` / `eef_pose`, `absolute` / `delta`), `frame` and `rotation` (`euler_rxyz` for `pd_ee_*pose`), names and units per element, and `normalized` / `range` (the controllers' bounds) when the controllers normalize to [-1, 1] (no `range` for `pd_ee_*pose`: ManiSkill clips their rotation by its norm and scales it by `rot_lower`, which is not a linear map onto the bounds).
- Fixed the observation space going stale: it was converted once, at construction. `observation_space` / `action_space` now follow the ManiSkill env's spaces (converted and described again when ManiSkill regenerates them, e.g. via `update_obs_space`), and `reset()` regenerates ManiSkill's observation space after a reconfiguration that changed the observations (ManiSkill itself keeps the space it built from the first observation, e.g. after changing the camera resolution).
- `reset()` accepts `options` (e.g. `{"reconfigure": True}`) and passes them on, merged with the `env_idx` of `mask`; passing `options` used to raise a `TypeError`.

## 0.0.1a3 (2026-02-10)

- Make a separate repository for maniskill adaptor.

## 0.0.1a2 (2025-11-21)

- Fix dependency

## 0.0.1a1 (2025-11-13)

- Initial release with maniskill environment adaptor