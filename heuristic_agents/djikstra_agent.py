import itertools
import sys

from src.Environment.Environment import *

from src.Environment.Actions import *
from src.Environment.Vizualization import *
from src.Environment.WallFollowing import *
import random
import yaml


def read_integer():
    while True:
        try:
            num = int(input("Enter an integer: "))
            if num == 8:
                return Actions.NORTH
            elif num == 2:
                return Actions.SOUTH
            elif num == 6:
                return Actions.EAST
            elif num == 4:
                return Actions.WEST
            else:
                return Actions.NORTH
            return num
        except ValueError:
            print("Invalid input. Please enter an integer.")


with open('../configs/training_obstacles.yaml', 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file

env = Environment(EnvironmentParams(conf['env4']))
Viz = Vizualization()
sys.setrecursionlimit(2000)
while True:
    observation, _ = env.reset()
    env.render()
    for t in itertools.count():
        observation_, reward, done, truncated, info = env.step(env.get_heuristic_action())
        #print(reward)
        # print(env.state.remaining)
        env.render()
        # time.sleep(5)

        observation = observation_
        if done or truncated:
            break
