# Maniskill3 Environment Adaptor

This repository contains the implementation of the adaptor that connects the Maniskill3 environment to the UniEnv interface.

## Installation

You can install the adaptor from PyPI:

```bash
pip install unienv_maniskill
```

## Usage

```python
from unienv_maniskill import ManiskillEnv
import mani_skill.envs.tasks.control.cartpole
import gymnasium as gym

# Create the underlying Maniskill3 environment
maniskill_env = gym.make("CartPole-v0")
# Create the ManiskillEnv wrapper
env = ManiskillEnv(maniskill_env)

# Now you can use the env as a standard UniEnv environment
ctx, obs, info = env.reset()
while True:
    action = env.sample_action()
    ctx, obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```