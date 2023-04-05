import itertools
from src.Environment.Environment import *
from src.Environment.Actions import *
from src.Environment.Vizualization import *
from src.Rainbow.DQNAgent import *
import random

num_frames = 200000
memory_size = 10000
batch_size = 64
target_update = 200

# environment

env = Environment(EnvironmentParams())

agent = DQNAgent([4, 9, 9], 4, memory_size, batch_size, target_update)
agent.load_models()
env = Environment(EnvironmentParams())
Viz = Vizualization()
while True:
    observation, _ = env.reset()
    for t in itertools.count():
        action = Actions(agent.select_action(observation))
        observation, reward, done, truncated, info = env.step(action)
        print("começa")
        print(observation)
        print("acaba")
        # print(env.state.remaining)
        Viz.render_center(observation)
        if done or truncated:
            break
