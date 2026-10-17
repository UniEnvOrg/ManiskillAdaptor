"""Space metadata for ManiSkill 3 environments.

ManiSkill builds its gymnasium spaces from sample data, so they say nothing
about what the data means. The functions here describe the converted unienv
spaces with the shared metadata vocabulary of
``unienv_interface.space.space_utils.metadata_utils``, using what ManiSkill
itself knows about the data:

* the robot's active joints (names and types) for ``agent/qpos`` / ``agent/qvel``,
* the controllers of the current control mode for the action,
* the camera sensors (uid, resolution, shader pack) for ``sensor_data``,
  ``sensor_param`` and ``pointcloud``,
* ``control_freq`` as the capture rate of the sensors,
* the standard texture formats documented in ``mani_skill.render.shaders``
  (``rgb`` uint8 HWC, ``depth`` / ``position`` int16 millimetres, ...).

Only leaves whose shape matches what ManiSkill describes are annotated; data
added or reshaped by observation wrappers is left undescribed.
"""
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv as ManiSkillBaseEnv
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.controllers.base_controller import BaseController, DictController
from mani_skill.agents.controllers.passive_controller import PassiveController
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController, PDJointPosMimicController
from mani_skill.agents.controllers.pd_joint_pos_vel import PDJointPosVelController
from mani_skill.agents.controllers.pd_joint_vel import PDJointVelController
from mani_skill.agents.controllers.pd_base_vel import PDBaseVelController, PDBaseForwardVelController
from mani_skill.agents.controllers.pd_ee_pose import PDEEPosController, PDEEPoseController
from mani_skill.sensors.camera import Camera

from unienv_interface.space import Space, BoxSpace, DictSpace
from unienv_interface.space.space_utils.metadata_utils import (
    metadata, image_metadata, depth_metadata, proprio_metadata, action_metadata,
)

__all__ = [
    "maniskill_observation_metadata",
    "maniskill_action_metadata",
]

SpacePath = Tuple[str, ...]

# ArticulationJoint.type values (sapien.physx) -> units of the joint position / velocity
_JOINT_POSITION_UNITS = {"revolute": "rad", "revolute_unwrapped": "rad", "prismatic": "m"}
_JOINT_VELOCITY_UNITS = {"revolute": "rad/s", "revolute_unwrapped": "rad/s", "prismatic": "m/s"}

# Pose.raw_pose: position followed by a (w, x, y, z) quaternion
_RAW_POSE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
_RAW_POSE_UNITS = ["m", "m", "m", "normalized", "normalized", "normalized", "normalized"]
_XYZ = ["x", "y", "z"]

# ===================================================================== helpers

def _shape(space: Space) -> Optional[Tuple[int, ...]]:
    return tuple(space.shape) if isinstance(space, BoxSpace) else None

def _is_float(space: Space) -> bool:
    return isinstance(space, BoxSpace) and space.backend.dtype_is_real_floating(space.dtype)

def _is_int(space: Space, dtype: Optional[torch.dtype] = None) -> bool:
    """An integer array space (of ``dtype``, if given)."""
    if not isinstance(space, BoxSpace):
        return False
    if dtype is not None:
        return space.dtype == dtype
    return space.backend.dtype_is_real_integer(space.dtype)

def _all_units(units: Sequence[Optional[str]]) -> Optional[List[str]]:
    """``units`` as a list if every entry is known, else ``None``."""
    return list(units) if all(unit is not None for unit in units) else None

def _joint_units(joints, table: Mapping[str, str]) -> List[Optional[str]]:
    return [table.get(joint.type[0]) for joint in joints]

def _compact(value: float) -> float:
    """A float32 bound as the shortest float that round-trips it (``0.1`` instead of ``0.10000000149``)."""
    return float(f"{float(value):.7g}")

def _single_agent(env: ManiSkillBaseEnv) -> Optional[BaseAgent]:
    agent = getattr(env, "agent", None)
    if agent is None or isinstance(agent, MultiAgent):
        return None
    return agent

def _camera_sensors(env: ManiSkillBaseEnv) -> Dict[str, Camera]:
    return {
        uid: sensor for uid, sensor in getattr(env, "_sensors", {}).items()
        if isinstance(sensor, Camera)
    }

def _is_image_of(space: Space, camera: Camera, channels: Optional[int] = None) -> bool:
    """``space`` is a batch of (H, W, C) images of ``camera``'s resolution."""
    shape = _shape(space)
    if shape is None or len(shape) < 3:
        return False
    height, width = shape[-3], shape[-2]
    return (
        height == camera.config.height and width == camera.config.width
        and (channels is None or shape[-1] == channels)
    )

# =============================================================== observations

def _agent_leaf_metadata(agent: BaseAgent, path: SpacePath, last_dim: Optional[int]) -> Optional[Dict[str, Any]]:
    """Metadata of a leaf of ``agent/`` (``path`` is relative to ``agent``)."""
    joints = agent.robot.get_active_joints()
    names = [joint.name for joint in joints]
    if path == ("qpos",) and last_dim == len(joints):
        return proprio_metadata(units=_all_units(_joint_units(joints, _JOINT_POSITION_UNITS)), names=names)
    if path == ("qvel",) and last_dim == len(joints):
        return proprio_metadata(units=_all_units(_joint_units(joints, _JOINT_VELOCITY_UNITS)), names=names)
    if len(path) >= 2 and path[0] == "controller":
        # BaseController.get_state(): {"target_qpos" | "target_pose": ...}, keyed by
        # the sub-controller uid for combined controllers
        controller = agent.controller
        if isinstance(controller, DictController) and len(path) == 3:
            controller = controller.controllers.get(path[1])
        elif len(path) != 2:
            controller = None
        if controller is None:
            return None
        if path[-1] == "target_qpos" and last_dim == len(controller.joints):
            return proprio_metadata(
                units=_all_units(_joint_units(controller.joints, _JOINT_POSITION_UNITS)),
                names=[joint.name for joint in controller.joints],
                label="controller target joint positions",
            )
        if path[-1] == "target_pose" and last_dim == 7 and isinstance(controller, PDEEPosController):
            # the target end-effector pose relative to the controller's root link
            return metadata(
                "pose", units=_RAW_POSE_UNITS, names=_RAW_POSE_NAMES, frame="base",
                rotation="quat_wxyz", label="controller target end-effector pose",
            )
    return None

def _extra_leaf_metadata(path: SpacePath, last_dim: Optional[int], is_float: bool) -> Optional[Dict[str, Any]]:
    """Metadata of a leaf of ``extra/`` (task specific data).

    ManiSkill tasks follow a naming convention for their extra observations:
    ``<name>_pose`` is a ``Pose.raw_pose`` (position + wxyz quaternion, 7 values)
    and ``<name>_pos`` a position (3 values), both in metres. Only leaves that
    follow the convention *and* have the matching size are described.
    """
    if not is_float or not path:
        return None
    key = path[-1]
    if key.endswith("_pose") and last_dim == 7:
        return metadata("pose", units=_RAW_POSE_UNITS, names=_RAW_POSE_NAMES, rotation="quat_wxyz")
    if key.endswith("_pos") and last_dim == 3:
        return metadata(None, units="m", names=_XYZ)
    return None

def _state_dict_leaf_metadata(env: ManiSkillBaseEnv, path: SpacePath, last_dim: Optional[int], is_float: bool) -> Optional[Dict[str, Any]]:
    """Metadata of a leaf of the state dict (``agent/...`` or ``extra/...``)."""
    if len(path) >= 2 and path[0] == "agent":
        agent = _single_agent(env)
        return None if agent is None else _agent_leaf_metadata(agent, path[1:], last_dim)
    if len(path) >= 2 and path[0] == "extra":
        return _extra_leaf_metadata(path[1:], last_dim, is_float)
    return None

def _flat_state_metadata(env: ManiSkillBaseEnv, size: int) -> Dict[str, Any]:
    """Metadata of the flattened state vector (``obs_mode="state"`` / the ``state`` key).

    The vector is ``common.flatten_state_dict(dict(agent=..., extra=...))``;
    the names (and units, if all are known) of its elements follow that order.
    """
    try:
        state_dict = env._get_obs_state_dict(env.get_info())
    except Exception:
        return metadata("state")

    names : List[str] = []
    units : List[Optional[str]] = []

    def visit(value: Any, path: SpacePath) -> None:
        if isinstance(value, dict):
            for key, sub_value in value.items():
                visit(sub_value, path + (key,))
            return
        if not isinstance(value, torch.Tensor):
            names.append("/".join(path))
            units.append(None)
            return
        count = 1 if value.ndim == 1 else int(value.shape[-1])
        leaf = _state_dict_leaf_metadata(env, path, count, torch.is_floating_point(value)) or {}
        leaf_names = leaf.get("names")
        leaf_units = leaf.get("units")
        if count == 1:
            names.append("/".join(path))
        elif leaf_names is not None and len(leaf_names) == count:
            names.extend("/".join(path + (name,)) for name in leaf_names)
        else:
            names.extend("/".join(path + (str(i),)) for i in range(count))
        if isinstance(leaf_units, str):
            units.extend([leaf_units] * count)
        elif isinstance(leaf_units, list) and len(leaf_units) == count:
            units.extend(leaf_units)
        else:
            units.extend([None] * count)

    visit(state_dict, ())
    if len(names) != size:
        return metadata("state")
    return metadata("state", names=names, units=_all_units(units))

def _sensor_texture_metadata(
    camera_uid: str,
    camera: Camera,
    texture: str,
    space: Space,
    fps: float,
    raw: bool,
) -> Optional[Dict[str, Any]]:
    """Metadata of ``sensor_data/<camera_uid>/<texture>``.

    With ``raw=False`` the textures are the standard image modalities of
    ``mani_skill.render.shaders.ShaderConfig`` (``rgb`` uint8, ``depth`` /
    ``segmentation`` int (H, W, 1), ``position`` int16 millimetres in the
    OpenGL camera frame, ...). With ``raw=True`` (``obs_mode="sensor_data"``)
    they are the untransformed shader textures, named after the shader pack's
    textures.
    """
    common = dict(source=camera_uid, fps=fps)
    if not raw:
        if texture == "rgb" and _is_image_of(space, camera, 3) and _is_int(space, torch.uint8):
            return image_metadata(**common)
        if texture == "albedo" and _is_image_of(space, camera, 3) and _is_int(space, torch.uint8):
            return image_metadata(label="albedo", **common)
        if texture == "depth" and _is_image_of(space, camera, 1):
            # distance along the camera's viewing axis, int16 millimetres
            return depth_metadata(units="mm" if _is_int(space) else None, layout="HWC", **common)
        if texture == "segmentation" and _is_image_of(space, camera, 1):
            return metadata("segmentation", layout="HWC", label="actor / link id", **common)
        if texture == "position" and _is_image_of(space, camera, 3):
            return metadata(
                "pointcloud", layout="HWC", units="mm" if _is_int(space) else None, names=_XYZ,
                frame="camera_gl", **common,
            )
        if texture == "normal" and _is_image_of(space, camera, 3):
            return metadata("normal", layout="HWC", names=_XYZ, **common)
        return None

    shader_pack = camera.config.shader_config.shader_pack
    if shader_pack == "minimal":
        if texture == "Color" and _is_image_of(space, camera, 4) and _is_int(space, torch.uint8):
            return image_metadata(format="rgba", **common)
        if texture == "PositionSegmentation" and _is_image_of(space, camera, 4) and _is_int(space):
            return metadata(
                "pointcloud", layout="HWC", names=["x", "y", "z", "segmentation"], frame="camera_gl",
                label="position (mm) + actor / link id", **common,
            )
    elif shader_pack == "default" or shader_pack.startswith("rt"):
        if texture == "Color" and _is_image_of(space, camera, 4) and _is_float(space):
            return image_metadata(format="rgba", normalized=True, **common)
        if texture == "Albedo" and _is_image_of(space, camera, 4) and _is_float(space):
            return image_metadata(format="rgba", normalized=True, label="albedo", **common)
        if texture == "Position" and _is_image_of(space, camera, 4) and _is_float(space):
            return metadata(
                "pointcloud", layout="HWC", names=["x", "y", "z", "w"], frame="camera_gl",
                label="position (m) + z-buffer depth", **common,
            )
        if texture == "Segmentation" and _is_image_of(space, camera, 4) and _is_int(space):
            # channel 0: visual shape / mesh id, 1: actor / link id, 2: parallel scene id, 3: unused
            return metadata(
                "segmentation", layout="HWC", names=["mesh_id", "actor_id", "scene_id", "unused"], **common,
            )
    return None

def _sensor_param_metadata(camera_uid: str, key: str, space: Space) -> Optional[Dict[str, Any]]:
    """Metadata of ``sensor_param/<camera_uid>/<key>`` (``Camera.get_params()``)."""
    shape = _shape(space)
    if shape is None or len(shape) < 2:
        return None
    if key == "extrinsic_cv" and shape[-2:] == (3, 4):
        return metadata("extrinsics", source=camera_uid, frame="camera_cv", label="world to camera (OpenCV) transform")
    if key == "cam2world_gl" and shape[-2:] == (4, 4):
        return metadata("extrinsics", source=camera_uid, frame="camera_gl", label="camera (OpenGL) to world transform")
    if key == "intrinsic_cv" and shape[-2:] == (3, 3):
        return metadata("intrinsics", source=camera_uid, frame="camera_cv")
    return None

def _pointcloud_metadata(key: str, space: Space, cameras: Dict[str, Camera], fps: float) -> Optional[Dict[str, Any]]:
    """Metadata of ``pointcloud/<key>`` (``sensor_data_to_pointcloud``: the points of all cameras, in the world frame)."""
    shape = _shape(space)
    if shape is None or len(shape) < 2:
        return None
    source = next(iter(cameras)) if len(cameras) == 1 else None
    if key == "xyzw" and shape[-1] == 4 and _is_float(space):
        # homogeneous world coordinates: w is 1 for valid points and 0 for the background
        return metadata(
            "pointcloud", names=["x", "y", "z", "w"], frame="world", source=source, fps=fps,
            label="xyz (m) + validity w",
        )
    if key == "rgb" and shape[-1] == 3 and _is_int(space, torch.uint8):
        return metadata("pointcloud", format="rgb", source=source, fps=fps, label="point colors")
    if key == "segmentation" and shape[-1] == 1:
        return metadata("segmentation", source=source, fps=fps, label="actor / link id")
    return None

def maniskill_observation_metadata(env: ManiSkillBaseEnv, observation_space: Space) -> Dict[SpacePath, Dict[str, Any]]:
    """Metadata for the leaves of ``observation_space``, the converted observation space of ``env``.

    Returns a mapping ``{path: metadata}`` for :func:`annotate`. ``env`` is the
    unwrapped ManiSkill environment.
    """
    out : Dict[SpacePath, Dict[str, Any]] = {}
    obs_mode = env.obs_mode
    fps = env.control_freq

    if not isinstance(observation_space, DictSpace):
        shape = _shape(observation_space)
        if obs_mode == "state" and shape is not None and len(shape) >= 1:
            out[()] = _flat_state_metadata(env, shape[-1])
        return out

    def leaves(space: Space, path: SpacePath):
        if isinstance(space, DictSpace):
            for key, sub_space in space.spaces.items():
                yield from leaves(sub_space, path + (key,))
        else:
            yield path, space

    cameras = _camera_sensors(env)
    raw_textures = obs_mode == "sensor_data"
    for path, space in leaves(observation_space, ()):
        shape = _shape(space)
        if shape is None:
            continue
        last_dim = shape[-1] if len(shape) >= 2 else None  # leading axis is the batch axis
        md = None
        if path[0] in ("agent", "extra"):
            md = _state_dict_leaf_metadata(env, path, last_dim, _is_float(space))
        elif path == ("state",) and last_dim is not None:
            md = _flat_state_metadata(env, last_dim)
        elif path[0] == "sensor_data" and len(path) == 3 and path[1] in cameras:
            md = _sensor_texture_metadata(path[1], cameras[path[1]], path[2], space, fps, raw_textures)
        elif path[0] == "sensor_param" and len(path) == 3 and path[1] in cameras:
            md = _sensor_param_metadata(path[1], path[2], space)
        elif path[0] == "pointcloud" and len(path) == 2:
            md = _pointcloud_metadata(path[1], space, cameras, fps)
        if md:
            out[path] = md
    return out

# ===================================================================== actions

def _controller_segment(uid: str, controller: BaseController) -> Dict[str, Any]:
    """Description of one controller's part of the action vector."""
    size = int(controller.single_action_space.shape[0]) if controller.single_action_space.shape else 0
    joints = list(controller.joints)
    seg : Dict[str, Any] = dict(size=size, names=None, units=None, control=None, mode=None, frame=None, rotation=None)
    use_delta = getattr(controller.config, "use_delta", None)
    mode = None if use_delta is None else ("delta" if use_delta else "absolute")

    if isinstance(controller, PassiveController) or size == 0:
        seg["size"] = 0
    elif isinstance(controller, PDEEPoseController):
        frame = controller.config.frame
        seg.update(
            control="eef_pose", mode=mode,
            names=_XYZ + ["rx", "ry", "rz"], units=["m", "m", "m", "rad", "rad", "rad"],
            # euler_angles_to_matrix(rot, "XYZ") composes R = Rx @ Ry @ Rz: intrinsic x-y-z
            rotation="euler_rxyz",
            frame={
                "root_translation:root_aligned_body_rotation": "base",
                "body_translation:body_aligned_body_rotation": "link_local",
            }.get(frame),
        )
    elif isinstance(controller, PDEEPosController):
        seg.update(
            control="eef_pose", mode=mode, names=list(_XYZ), units=["m", "m", "m"],
            frame={"root_translation": "base", "body_translation": "link_local"}.get(controller.config.frame),
        )
    elif isinstance(controller, PDJointPosVelController):
        seg.update(
            mode=mode,
            names=[joint.name for joint in joints] + [f"{joint.name}_vel" for joint in joints],
            units=_joint_units(joints, _JOINT_POSITION_UNITS) + _joint_units(joints, _JOINT_VELOCITY_UNITS),
        )
    elif isinstance(controller, PDJointPosController):
        if isinstance(controller, PDJointPosMimicController):
            joints = [joints[int(i)] for i in controller.control_joint_indices.tolist()]
        seg.update(
            control="gripper" if "gripper" in uid else "joint_position", mode=mode,
            names=[joint.name for joint in joints], units=_joint_units(joints, _JOINT_POSITION_UNITS),
        )
    elif isinstance(controller, PDBaseForwardVelController):
        seg.update(names=["forward", "yaw"], units=["m/s", "rad/s"])
    elif isinstance(controller, PDBaseVelController):
        seg.update(names=[joint.name for joint in joints], units=_joint_units(joints, _JOINT_VELOCITY_UNITS))
    elif isinstance(controller, PDJointVelController):
        seg.update(
            control="joint_velocity", mode="absolute",
            names=[joint.name for joint in joints], units=_joint_units(joints, _JOINT_VELOCITY_UNITS),
        )
    elif len(joints) == size:
        seg.update(names=[joint.name for joint in joints])

    for key in ("names", "units"):
        if seg[key] is not None and len(seg[key]) != seg["size"]:
            seg[key] = None

    seg["normalized"] = bool(getattr(controller, "_normalize_action", False))
    original = getattr(controller, "_original_single_action_space", None)
    # gym_utils.clip_and_scale_action maps [-1, 1] linearly onto [low, high], except for the
    # rotation of PDEEPoseController, which is clipped by its norm and scaled by rot_lower
    linear = not isinstance(controller, PDEEPoseController)
    if seg["normalized"] and linear and original is not None and original.shape == (seg["size"],):
        seg["low"] = [_compact(value) for value in np.asarray(original.low).reshape(-1)]
        seg["high"] = [_compact(value) for value in np.asarray(original.high).reshape(-1)]
    return seg

def maniskill_action_metadata(env: ManiSkillBaseEnv, action_space: Space) -> Dict[str, Any]:
    """Metadata for ``action_space``, the converted (batched) action space of ``env``.

    The action is the concatenation of the actions of the controllers of the
    current control mode (``CombinedController.action_mapping``). ``names``
    and ``units`` describe every element; ``control`` / ``mode`` / ``frame`` /
    ``rotation`` are those of the main controller (the first one that is not
    the gripper). ``normalized`` is given when every controller normalizes its
    action to [-1, 1], and ``range`` (the controllers' unnormalized bounds)
    when every one of them maps [-1, 1] linearly onto those bounds (not the
    ``pd_ee_*pose`` controllers: their rotation is clipped by its norm and
    scaled by ``rot_lower``).
    """
    agent = _single_agent(env)
    shape = _shape(action_space)
    if agent is None or shape is None or len(shape) < 1:
        return {}
    controller = agent.controller
    if isinstance(controller, DictController):
        items = list(controller.controllers.items())
    else:
        items = [("", controller)]
    segments = [seg for seg in (_controller_segment(uid, sub) for uid, sub in items) if seg["size"] > 0]
    if not segments or sum(seg["size"] for seg in segments) != shape[-1]:
        return metadata("action")

    main = next((seg for seg in segments if seg["control"] != "gripper"), segments[0])
    names = None if any(seg["names"] is None for seg in segments) else [n for seg in segments for n in seg["names"]]
    units = None if any(seg["units"] is None for seg in segments) else _all_units([u for seg in segments for u in seg["units"]])
    normalized = all(seg["normalized"] for seg in segments)
    value_range = None
    if normalized and all("low" in seg for seg in segments):
        low = [value for seg in segments for value in seg["low"]]
        high = [value for seg in segments for value in seg["high"]]
        if all(np.isfinite(low)) and all(np.isfinite(high)):
            value_range = [low, high]
    return action_metadata(
        control=main["control"], mode=main["mode"], units=units,
        normalized=True if normalized else None, range=value_range, names=names,
        frame=main["frame"], rotation=main["rotation"],
    )
