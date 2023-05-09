import itertools

from src.Environment.Environment import *

from src.Environment.Actions import *
from src.Environment.Vizualization import *
from src.Environment.WallFollowing import *
import random
import yaml

with open('../configs/training_obstacles.yaml', 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file

env = Environment(EnvironmentParams(conf['env1']))
Viz = Vizualization()

while True:
    observation, _ = env.reset()
    env.render()
    for t in itertools.count():
        observation_, reward, done, truncated, info = env.step(env.get_heuristic_action())
        # print(action)
        # print(env.state.remaining)
        env.render()

        observation = observation_
        if done or truncated:
            break

