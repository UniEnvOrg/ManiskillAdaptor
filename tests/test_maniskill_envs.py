import pytest
import typing
from unienv_maniskill import FromManiSkillEnv
from unienv_interface.space.space_utils.batch_utils import *
from unienv_interface.space.space_utils.metadata_utils import validate_metadata

import os.path
import gymnasium as gym
import numpy as np
import torch
import copy

# Env Imports
import mani_skill.envs.tasks.tabletop.pick_cube
import mani_skill.envs.tasks.control.cartpole

TEST_ENV_NAMES = [
    "PickCube-v1",
    "MS-CartpoleBalance-v1"
]
TEST_EPISODES = 2
TEST_MAX_STEPS = 100
TEST_GPU_SIM = True if torch.cuda.is_available() else False

# Units of agent/qpos: the Panda arm's revolute joints and prismatic fingers,
# the cart-pole's prismatic slider and revolute hinge
EXPECTED_QPOS_UNITS = {
    "PickCube-v1": ["rad"] * 7 + ["m"] * 2,
    "MS-CartpoleBalance-v1": ["m", "rad"],
}
# Units of the action of the default control mode (pd_joint_delta_pos): the
# Panda arm joints and the finger (mimic) gripper, the cart-pole slider
EXPECTED_ACTION_UNITS = {
    "PickCube-v1": ["rad"] * 7 + ["m"],
    "MS-CartpoleBalance-v1": ["m"],
}

def leaf(space, path : str):
    for key in path.split("/"):
        space = space[key]
    return space

def assert_metadata(space, **expected):
    for key, value in expected.items():
        assert key in space.metadata, f"{key!r} missing from {dict(space.metadata)}"
        actual = space.metadata[key]
        if isinstance(value, list):
            actual = list(actual)
        assert actual == value, f"{key!r}: {actual!r} != {value!r}"

def check_space_metadata(env : FromManiSkillEnv, env_name : str, obs_mode : str):
    """The spaces are described with the shared vocabulary, and key leaves carry the expected metadata."""
    validate_metadata(env.observation_space)
    validate_metadata(env.action_space)
    fps = env.env.unwrapped.control_freq

    assert_metadata(
        env.action_space, kind="action", control="joint_position", mode="delta",
        normalized=True, units=EXPECTED_ACTION_UNITS[env_name],
    )
    assert len(env.action_space.metadata["names"]) == env.action_space.shape[-1]
    assert len(env.action_space.metadata["range"][0]) == env.action_space.shape[-1]

    obs_space = env.observation_space
    if obs_mode == "state":
        assert_metadata(obs_space, kind="state")
        assert len(obs_space.metadata["names"]) == obs_space.shape[-1]
        return
    assert_metadata(leaf(obs_space, "agent/qpos"), kind="proprio", units=EXPECTED_QPOS_UNITS[env_name])
    assert_metadata(leaf(obs_space, "agent/qvel"), kind="proprio")
    if env_name == "PickCube-v1":
        assert_metadata(leaf(obs_space, "extra/tcp_pose"), kind="pose", rotation="quat_wxyz")
    if obs_mode == "state_dict":
        return

    textures = set(obs_mode.split("+"))
    if obs_mode == "rgbd":
        textures = {"rgb", "depth"}
    camera = "sensor_data/base_camera"
    assert_metadata(leaf(obs_space, "sensor_param/base_camera/intrinsic_cv"), kind="intrinsics", source="base_camera")
    assert_metadata(leaf(obs_space, "sensor_param/base_camera/extrinsic_cv"), kind="extrinsics", frame="camera_cv")
    if obs_mode == "sensor_data":
        # raw textures of the default ("minimal") shader pack
        assert_metadata(leaf(obs_space, f"{camera}/Color"), kind="image", format="rgba", layout="HWC", source="base_camera", fps=fps)
        assert_metadata(leaf(obs_space, f"{camera}/PositionSegmentation"), kind="pointcloud", frame="camera_gl")
    elif obs_mode == "pointcloud":
        assert_metadata(leaf(obs_space, "pointcloud/xyzw"), kind="pointcloud", frame="world", names=["x", "y", "z", "w"])
        assert_metadata(leaf(obs_space, "pointcloud/rgb"), format="rgb")
        assert_metadata(leaf(obs_space, "pointcloud/segmentation"), kind="segmentation")
    else:
        if "rgb" in textures:
            assert_metadata(leaf(obs_space, f"{camera}/rgb"), kind="image", format="rgb", layout="HWC", source="base_camera", fps=fps)
        if "depth" in textures:
            assert_metadata(leaf(obs_space, f"{camera}/depth"), kind="depth", units="mm", layout="HWC", source="base_camera", fps=fps)
        if "segmentation" in textures:
            assert_metadata(leaf(obs_space, f"{camera}/segmentation"), kind="segmentation", layout="HWC", source="base_camera")

@pytest.mark.parametrize("env_name", TEST_ENV_NAMES)
@pytest.mark.parametrize("obs_mode", ["state", "state_dict", "sensor_data", "rgb", "depth", "segmentation", "rgbd", "rgb+depth", "rgb+depth+segmentation", "rgb+segmentation", "depth+segmentation", "pointcloud"])
@pytest.mark.parametrize("render_mode", ["rgb_array"])
@pytest.mark.parametrize(
    ("sim_backend", "num_envs"), [
        ("cpu", 1)
    ] if not TEST_GPU_SIM else [
        ("gpu", 10),
        ("auto", 5)
    ]
)
def test_maniskill_env(
    env_name : str,
    obs_mode : str,
    render_mode : str,
    num_envs : int,
    sim_backend : str,
    render_backend : str = "gpu" if TEST_GPU_SIM else "cpu",
    max_steps : int = TEST_MAX_STEPS,
    episodes : int = TEST_EPISODES,
):
    env = gym.make(
        env_name, 
        obs_mode=obs_mode,
        render_mode=render_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
        num_envs=num_envs,
        max_episode_steps=max_steps
    )
    env = FromManiSkillEnv(env)
    check_space_metadata(env, env_name, obs_mode)
    done = None
    
    for _ in range(episodes):
        if done is None:
            _, obs, info = env.reset()
        else:
            _, part_obs, info = env.reset(mask=done)
            obs = env.update_observation_post_reset(
                obs, part_obs, done
            )
        
        done = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

        assert obs in env.observation_space
        for _ in range(max_steps):
            action = env.sample_action()
            assert action in env.action_space
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs in env.observation_space
            done = torch.logical_or(terminated, truncated)
            if torch.any(done):
                break

@pytest.mark.parametrize(
    ("control_mode", "expected"), [
        ("pd_joint_pos", dict(control="joint_position", mode="absolute")),
        ("pd_joint_vel", dict(control="joint_velocity", normalized=True)),
        ("pd_ee_delta_pos", dict(control="eef_pose", mode="delta", frame="base", normalized=True, names=["x", "y", "z", "panda_finger_joint1"])),
        # euler_angles_to_matrix(rot, "XYZ"): intrinsic x-y-z Euler angles
        ("pd_ee_delta_pose", dict(control="eef_pose", mode="delta", frame="base", rotation="euler_rxyz", units=["m"] * 3 + ["rad"] * 3 + ["m"])),
    ]
)
def test_maniskill_action_metadata(control_mode : str, expected : typing.Dict[str, typing.Any]):
    env = FromManiSkillEnv(gym.make(
        "PickCube-v1",
        obs_mode="state_dict",
        control_mode=control_mode,
        sim_backend="gpu" if TEST_GPU_SIM else "cpu",
        num_envs=2 if TEST_GPU_SIM else 1,
    ))
    try:
        validate_metadata(env.action_space)
        assert_metadata(env.action_space, kind="action", **expected)
        assert len(env.action_space.metadata["names"]) == env.action_space.shape[-1]
        if control_mode == "pd_ee_delta_pose":
            # the rotation is not mapped linearly onto the controller bounds
            assert "range" not in env.action_space.metadata
    finally:
        env.env.close()


def test_maniskill_observation_space_follows_reconfigure():
    """The observation space is converted (and described) again when ManiSkill's observations change."""
    num_envs = 2 if TEST_GPU_SIM else 1
    env = FromManiSkillEnv(gym.make(
        "PickCube-v1",
        obs_mode="rgb+depth",
        sim_backend="gpu" if TEST_GPU_SIM else "cpu",
        render_backend="gpu" if TEST_GPU_SIM else "cpu",
        num_envs=num_envs,
    ))
    try:
        camera = "sensor_data/base_camera"
        assert leaf(env.observation_space, f"{camera}/rgb").shape == (num_envs, 128, 128, 3)

        # Change the camera resolution; it takes effect when the scene is reconfigured
        env.env.unwrapped._custom_sensor_configs = dict(width=64, height=48)
        _, obs, info = env.reset(options=dict(reconfigure=True))
        assert info["reconfigure"]
        assert obs["sensor_data"]["base_camera"]["rgb"].shape == (num_envs, 48, 64, 3)
        assert leaf(env.observation_space, f"{camera}/rgb").shape == (num_envs, 48, 64, 3)
        assert leaf(env.observation_space, f"{camera}/depth").shape == (num_envs, 48, 64, 1)
        assert obs in env.observation_space
        # ... and described again
        validate_metadata(env.observation_space)
        assert_metadata(leaf(env.observation_space, f"{camera}/rgb"), kind="image", format="rgb", source="base_camera")
        assert_metadata(leaf(env.observation_space, f"{camera}/depth"), kind="depth", units="mm")
        assert_metadata(leaf(env.observation_space, "agent/qpos"), kind="proprio", units=EXPECTED_QPOS_UNITS["PickCube-v1"])

        obs, _, _, _, _ = env.step(env.sample_action())
        assert obs in env.observation_space

        # A partial reset keeps working with the new space
        mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        mask[0] = True
        _, part_obs, _ = env.reset(mask=mask)
        obs = env.update_observation_post_reset(obs, part_obs, mask)
        assert obs in env.observation_space

        # ManiSkill observation wrappers regenerate the space with `update_obs_space`
        space_before = env.observation_space
        base_env = env.env.unwrapped
        base_env.update_obs_space(base_env._init_raw_obs)
        assert env.observation_space is not space_before
        assert env.observation_space == space_before
        assert_metadata(leaf(env.observation_space, f"{camera}/rgb"), kind="image", source="base_camera")
    finally:
        env.env.close()
