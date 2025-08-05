import time
import os
import gym
import numpy as np
import random

from collections import deque
from torch.utils.tensorboard import SummaryWriter

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

import torch

from networks import CriticNetwork, ActorNetwork
from buffer import ReplyBuffer
from td3_torch_ap import Agent  # your modified Agent class with noise annealing

# Set MuJoCo rendering backend
os.environ["MUJOCO_GL"] = "egl"

# ✅ Random Seed Setup
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 42
# Change for multiple seeds
set_seed(SEED)

# Create Robosuite environment
def make_env(env_name="Door"):
    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )
    return GymWrapper(env)

if __name__ == '__main__':
    env_name = "Door"  # or "Door"
    log_dir = f"logs/{env_name}_TD350_TQC_seed{SEED}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(env_name)

    actor_lr = 1e-3
    critic_lr = 1e-3
    batch_size = 128
    layer1 = 256
    layer2 = 128

    n_episodes = 500  # Used for noise annealing and training length
    max_timesteps = 300

    agent = Agent(
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1,
        layer2_size=layer2,
        batch_size=batch_size,
        total_episodes= n_episodes,  # used for noise annealing
    )

    writer = SummaryWriter(log_dir)
    best_score = -np.inf
    score_history = deque(maxlen=100)

    print(f"\n--- Starting training on {env_name} with seed {SEED} ---\n")

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        score = 0
        t = 0

        while not done and t < max_timesteps:
            action = agent.choose_action(obs, episode=ep)  # ✅ Pass current episode
            next_obs, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            agent.learn()

            obs = next_obs
            score += reward
            t += 1

        score_history.append(score)
        avg_score = np.mean(score_history)

        writer.add_scalar("Score", score, global_step=ep)
        writer.add_scalar("Avg100_Score", avg_score, global_step=ep)

        print(f"Episode {ep:03d} | Score: {score:.2f} | Avg100: {avg_score:.2f}")

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            print(f"✔ Best avg score {best_score:.2f} at episode {ep}, models saved.")

    print(f"\n--- Training finished on {env_name} with seed {SEED} ---")
    writer.close()