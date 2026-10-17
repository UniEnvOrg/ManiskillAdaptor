from typing import Any, Optional, Tuple, Dict, Union, SupportsFloat, Sequence
from mani_skill.envs.sapien_env import BaseEnv as ManiSkillBaseEnv
from mani_skill.utils import common as maniskill_common
import gymnasium as gym
import torch
import numpy as np

from unienv_interface.env_base.env import Env
from unienv_interface.space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
from unienv_interface.space.space_utils import batch_utils as space_batch_utils, gym_utils as space_gym_utils
from unienv_interface.space.space_utils.metadata_utils import annotate

from .maniskill_metadata import maniskill_observation_metadata, maniskill_action_metadata

MANISKILL_ENV_ARRAYTYPE = Union[PyTorchArrayType, NumpyArrayType]
MANISKILL_ENV_DEVICETYPE = Union[PyTorchDeviceType, NumpyDeviceType]
MANISKILL_ENV_DTYPET = Union[PyTorchDtypeType, NumpyDtypeType]
MANISKILL_ENV_RNGTYPE = Union[PyTorchRNGType, NumpyRNGType]

def convert_maniskill_dict_to_backend(
    dict: Dict[str, MANISKILL_ENV_ARRAYTYPE],
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    device: Optional[BDeviceType] = None,
):
    return {
        key: convert_maniskill_array_to_backend(value, backend, device)
        for key, value in dict.items()
    }

def convert_maniskill_array_to_backend(
    array: MANISKILL_ENV_ARRAYTYPE,
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    device: Optional[BDeviceType] = None,
):
    if isinstance(array, torch.Tensor):
        source_backend = PyTorchComputeBackend
    elif isinstance(array, np.ndarray):
        source_backend = NumpyComputeBackend
    else:
        return array
    
    if backend is source_backend:
        if device is not None:
            return backend.to_device(array, device)
        else:
            return array
    else:
        return backend.from_other_backend(
            array,
            source_backend
        )

def convert_maniskill_to_backend(
    data: Any,
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    device: Optional[BDeviceType] = None,
):
    if isinstance(data, dict):
        return convert_maniskill_dict_to_backend(data, backend, device)
    else:
        return convert_maniskill_array_to_backend(data, backend, device)

def _data_matches_gym_space(data: Any, gym_space: gym.Space) -> bool:
    """Whether the (batched) ManiSkill data has the structure, shapes and dtypes of ``gym_space``."""
    if isinstance(gym_space, gym.spaces.Dict):
        return (
            isinstance(data, dict)
            and data.keys() == gym_space.spaces.keys()
            and all(_data_matches_gym_space(data[key], sub_space) for key, sub_space in gym_space.spaces.items())
        )
    if isinstance(gym_space, gym.spaces.Box):
        if isinstance(data, torch.Tensor):
            dtype = torch.empty(0, dtype=data.dtype).numpy().dtype
        elif isinstance(data, np.ndarray):
            dtype = data.dtype
        else:
            return False
        return tuple(data.shape) == tuple(gym_space.shape) and dtype == gym_space.dtype
    return True

class FromManiSkillEnv(
    Env[
        PyTorchComputeBackend.ARRAY_TYPE,
        None,
        Any,
        Any,
        PyTorchComputeBackend.ARRAY_TYPE,
        PyTorchComputeBackend.DEVICE_TYPE,
        PyTorchComputeBackend.DTYPE_TYPE,
        PyTorchComputeBackend.RNG_TYPE
    ]
):
    def __init__(
        self,
        env: ManiSkillBaseEnv,
    ) -> None:
        self.env = env
        self.backend = PyTorchComputeBackend
        self.device = env.get_wrapper_attr("device")
        self.batch_size = env.get_wrapper_attr("num_envs")

        # The spaces are converted from (and kept in sync with) the gymnasium spaces of `env`
        self._gym_action_space = None
        self._gym_observation_space = None
        self._refresh_spaces()
        self.context_space = None
        self.rng = torch.Generator(device=self.device)

    # =========== Spaces ==========
    @property
    def action_space(self) -> Space:
        self._refresh_spaces()
        return self._action_space

    @action_space.setter
    def action_space(self, space: Space) -> None:
        # Kept until the action space of the ManiSkill env changes
        self._action_space = space

    @property
    def observation_space(self) -> Space:
        self._refresh_spaces()
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: Space) -> None:
        # Kept until the observation space of the ManiSkill env changes
        self._observation_space = space

    def _refresh_spaces(self) -> None:
        """Convert (and describe) the gymnasium spaces of the ManiSkill env again if they changed.

        ManiSkill replaces its observation space when it is regenerated
        (``BaseEnv.update_obs_space``, used by its observation wrappers and by
        :meth:`reset` after a reconfiguration).
        """
        gym_action_space = self.env.action_space
        if gym_action_space is not self._gym_action_space:
            action_space = space_gym_utils.from_gym_space(
                gym_action_space,
                self.backend,
                device=self.device
            )
            if self.env.get_wrapper_attr("num_envs") <= 1:
                # Weirdly Maniskill doesn't batch the action space when num_envs is 1 but will batch the observation space
                action_space = space_batch_utils.batch_space(
                    action_space,
                    1
                )
            self._action_space = annotate(action_space, {
                "": maniskill_action_metadata(self.env.unwrapped, action_space)
            })
            self._gym_action_space = gym_action_space

        gym_observation_space = self.env.observation_space
        if gym_observation_space is not self._gym_observation_space:
            observation_space = space_gym_utils.from_gym_space(
                gym_observation_space,
                self.backend,
                device=self.device
            )
            self._observation_space = annotate(
                observation_space,
                maniskill_observation_metadata(self.env.unwrapped, observation_space)
            )
            self._gym_observation_space = gym_observation_space

    def _update_maniskill_observation_space(self, obs: Any) -> None:
        """Regenerate ManiSkill's observation space if ``obs`` (a full reset after a reconfiguration) no longer fits it.

        ManiSkill builds its observation space once, from the first
        observation, and keeps it when a reconfiguration (e.g. with changed
        sensor configs) changes the observations. It is regenerated here the way
        ManiSkill's own observation wrappers do. A wrapper that defines its own
        observation space is left alone.
        """
        base_env = self.env.unwrapped
        if self.env.observation_space is not base_env.observation_space:
            return
        if _data_matches_gym_space(obs, base_env.observation_space):
            return
        base_env.update_obs_space(maniskill_common.to_cpu_tensor(obs))

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.env.metadata

    def get_render_camera_params(
        self
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        ret = {}
        for name, camera in getattr(self.env, "_human_render_cameras", {}).items():
            ret[name] = camera.get_params()
        return ret

    @property
    def render_mode(self) -> Optional[str]:
        return self.env.render_mode
    
    @property
    def render_fps(self) -> Optional[int]:
        return self.env.get_wrapper_attr("control_freq")

    def reset(
        self,
        *args,
        mask: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[
        None,
        Any,
        Dict[str, Any]
    ]:
        """Reset the ManiSkill env; ``options`` are passed on (e.g. ``{"reconfigure": True}``)."""
        options = dict(options) if options else {}
        if mask is not None:
            options["env_idx"] = torch.nonzero(mask).flatten()
        obs, info = self.env.reset(
            *args,
            seed=seed,
            options=options or None,
            **kwargs
        )
        if isinstance(info, dict) and bool(info.get("reconfigure", False)):
            self._update_maniskill_observation_space(obs)
        self._refresh_spaces()
        # We don't convert the obs using from_gym_data here 
        # because the array may not be numpy array
        obs = convert_maniskill_to_backend(
            obs,
            self.backend,
            self.device
        )
        if mask is not None:
            obs = space_batch_utils.get_at(
                self.observation_space,
                obs,
                mask
            )

        return None, obs, info

    def step(
        self,
        action: Any
    ) -> Tuple[
        Any,
        SupportsFloat,
        bool,
        bool,
        Dict[str, Any]
    ]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        c_obs = convert_maniskill_to_backend(
            obs,
            self.backend,
            self.device
        )
        c_rew = convert_maniskill_array_to_backend(
            rew,
            self.backend,
            self.device
        )
        c_terminated = convert_maniskill_array_to_backend(
            terminated,
            self.backend,
            self.device
        )
        c_truncated = convert_maniskill_array_to_backend(
            truncated,
            self.backend,
            self.device
        )
        c_info = convert_maniskill_dict_to_backend(
            info,
            self.backend,
            self.device
        )
        return c_obs, c_rew, c_terminated, c_truncated, c_info

    def render(
        self
    ) -> Optional[
        MANISKILL_ENV_ARRAYTYPE
    ]:
        render_ret = self.env.render()
        if render_ret is None:
            return None
        else:
            dat = convert_maniskill_to_backend(
                render_ret,
                self.backend,
                self.device
            )
            if PyTorchComputeBackend.is_backendarray(dat) and dat.shape[0] == 1:
                return dat.squeeze(0)
            else:
                return None
        
    # =========== Wrapper methods ==========
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or hasattr(self.env, name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        elif hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError(f"Attribute {name} not found in the environment.")
        
    def set_wrapper_attr(self, name: str, value: Any):
        if hasattr(self, name):
            setattr(self, name, value)
        elif hasattr(self.env, name):
            setattr(self.env, name, value)
        else:
            raise AttributeError(f"Attribute {name} not found in the environment.")