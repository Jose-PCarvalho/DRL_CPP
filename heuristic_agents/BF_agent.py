import itertools

from src.Environment.Environment import *
from src.Environment.Vizualization import *
from src.Environment.BackAndForth import *
import random
import yaml
import time
with open('../configs/training_test.yaml', 'rb') as f:
    conf = yaml.safe_load(f.read())    # load the config file

env = Environment(EnvironmentParams(conf['env1']))
Viz = Vizualization()
agent = BackForth()
while True:
    observation, _ = env.reset()
    #env.render()
    Viz.render(env.state.local_map, env.state.position)
    agent.init(observation[0][-1],env.state.params.real_size)
    for t in itertools.count():
        action = agent.select_action(observation[0][-1])
        observation_, reward, done, truncated, info = env.step(action.value)
        #print(action)
        # print(env.state.remaining)
        #env.render()
        Viz.render(env.state.local_map,env.state.position)
        observation = observation_
        if done or truncated:
            break
