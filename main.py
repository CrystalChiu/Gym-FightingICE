import gym
import sys
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sys.path.append('gym-fightingice')
from gym_fightingice.envs.Machete import Machete

# Custom wrapper to ensure that the observations are always flattened and scaled
class CustomFightingICEEnv(gym.Env):
    def __init__(self):
        super(CustomFightingICEEnv, self).__init__()
        logger.debug("Initializing environment...")
        self.env = gym.make("FightingiceDataFrameskip-v0", java_env_path="", port=4242)

        # set action and obs space
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print("Observation space:", self.env.observation_space)
        logger.debug(f"Action space: {self.action_space}, Observation space: {self.observation_space}")

    def reset(self):
        # logger.debug("Resetting environment...")
        obs = self.env.reset()

        # observation returned as a list containing numpy array --> must extract
        obs = np.array(obs[0]) if isinstance(obs, tuple) or isinstance(obs, list) else np.array(obs)

        # scale image data observation
        obs = obs / 255.0

        # flatten observation
        obs = np.reshape(obs, -1)

        return obs

    def step(self, action):
        # action is numpy.int64, must convert to python int
        action = int(action)

        obs, reward, done, info = self.env.step(action)

        # make sure observation is 1d array
        obs = np.reshape(obs, -1)
        obs = obs / 255.0 # scale observation

        return obs, reward, done, info


    def render(self):
        logger.debug("Rendering environment...")
        self.env.render()

# custom callback to log rewards
class LogRewardsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LogRewardsCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards']
        self.rewards.append(reward)
        return True

    def get_rewards(self):
        return self.rewards

def main():
    # need custom env due to incompatibility in data returned
    logger.debug("Creating custom environment...")
    env = DummyVecEnv([lambda: CustomFightingICEEnv()])

    # reset environment
    logger.debug("Resetting environment...")
    obs = env.reset()
    print(f"Observation shape: {obs.shape}, type: {type(obs)}")

    # init PPO model
    logger.debug("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_fightingice_tensorboard/"
    )

    logger.debug("Starting training...")
    reward_callback = LogRewardsCallback(verbose=1)

    #NOTE: will adjust timestamps higher later
    model.learn(total_timesteps=100, callback=reward_callback)
    logger.debug("Training complete.")

    model.save("ppo_fightingice")
    logger.debug("Model saved as 'ppo_fightingice'.")

    rewards = reward_callback.get_rewards()

    # plot reward graph
    plt.plot(rewards)
    plt.title("Reward over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.show()

    # test model now
    logger.debug("Testing trained model...")
    obs = env.reset()
    for _ in range(300):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if done:
            logger.debug("Episode done, resetting environment...")
            obs = env.reset()
    logger.debug("Testing complete.")

if __name__ == "__main__":
    main()
