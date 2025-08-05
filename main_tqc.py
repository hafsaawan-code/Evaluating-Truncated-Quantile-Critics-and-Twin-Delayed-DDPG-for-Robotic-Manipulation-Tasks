import time
import os
import gym
import numpy as np

print("Importing TensorBoard...")
from torch.utils.tensorboard import SummaryWriter
print("Importing robosuite...")
import robosuite as suite

print("Importing wrappers and custom modules...")
from robosuite.wrappers.gym_wrapper import GymWrapper
print("About to import CriticNetwork, ActorNetwork from networks.py")
from networks import CriticNetwork, ActorNetwork, QuantileCriticNetwork
print("Imported CriticNetwork, ActorNetwork.")

print("About to import ReplyBuffer from buffer.py")
from buffer import ReplyBuffer
print("Imported ReplyBuffer.")

print("About to import Agent from td3_torch.py")
from td3_torch import Agent
print("Imported Agent.")


print("Setting MUJOCO_GL to egl...")
os.environ["MUJOCO_GL"] = "egl"

if __name__ == '__main__':
    print("Starting main...")

    if not os.path.exists("tmp1/td_main3"):
        os.makedirs("tmp1/td3_main3")
        print("Created directory tmp1/td32")
    else:
        print("Directory tmp1/td32 already exists")

    env_name = "Door"
    print(f"Environment name: {env_name}")

    print("Creating robosuite environment...")
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
    print("robosuite environment created.")

    print("Wrapping environment with GymWrapper...")
    env = GymWrapper(env)
    print("Environment wrapped.")

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    print("Initializing agent...")
    agent = Agent(
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size
    )
    print("Agent initialized.")

    print("Setting up TensorBoard writer...")
    writer = SummaryWriter('logs/Door_main2')
    n_games = 100000
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate = {actor_learning_rate}  critic_learning_rate = {critic_learning_rate}  layer1_size = {layer1_size} layer2_size = {layer2_size}"

    print("Loading agent models (if any)...")
    agent.load_models()
    print("Models loaded.")

    print("Starting training loop...")
    for i in range(n_games):
        print(f"Resetting environment for episode {i}...")
        observation = env.reset()
        done = False
        score = 0

        step_count = 0
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation
            step_count += 1
            if step_count % 50 == 0:
                print(f"Episode {i}, Step {step_count}: Current score {score}")

        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if i % 10 == 0:
            print(f"Saving models at episode {i}...")
            agent.save_models()

        print(f"Episode: {i} finished with Score: {score}")

    print("Training complete.")