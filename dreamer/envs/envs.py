import gym
import dmc2gym

from dreamer.envs.wrappers import *

import sys
import os

dir = os.path.dirname(__file__)
sys.path.append(dir)
from walk_in_the_park.env_utils import make_mujoco_env


def make_dmc_env(
    domain_name,
    task_name,
    seed,
    visualize_reward,
    from_pixels,
    height,
    width,
    frame_skip,
    pixel_norm=True,
):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=visualize_reward,
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
    )
    if pixel_norm:
        env = PixelNormalization(env)
    return env


def make_atari_env(task_name, skip_frame, width, height, seed, pixel_norm=True):
    env = gym.make(task_name)
    env = gym.wrappers.ResizeObservation(env, (height, width))
    env = ChannelFirstEnv(env)
    env = SkipFrame(env, skip_frame)
    if pixel_norm:
        env = PixelNormalization(env)
    env.seed(seed)
    return env


def make_a1_env(task_name, seed):
    assert task_name == "run"
    env_name = "A1Run-v0"
    control_frequency = 50
    randomize_ground = True
    env = make_mujoco_env(
        env_name,
        control_frequency=control_frequency,
        action_filter_high_cut=None,
        action_history=1,
        randomize_ground=randomize_ground
    )
    # state_dim = env.observation_space.shape[0]
    # env = SkipFrame(env, skip_frame)
    env.seed(seed)
    return env


def get_env_infos(env):
    obs_shape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete_action_bool = True
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discrete_action_bool = False
        action_size = env.action_space.shape[0]
    else:
        raise Exception
    return obs_shape, discrete_action_bool, action_size
