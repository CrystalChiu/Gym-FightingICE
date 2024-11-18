import gym
import sys

sys.path.append('gym-fightingice')
from gym_fightingice.envs.Machete import Machete

def main():
    # we will need to define and train player 1

    # set opponent
    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="", port=4242)
    env.reset(p2=Machete)
    
    for i in range(300):
        env.step(0)


if __name__ == "__main__":
    main()
