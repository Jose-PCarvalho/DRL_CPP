import itertools

from src.Environment.Environment import *
from src.Environment.Grid import *
import numpy as np
from PIL import Image
from src.Environment.State import *
from src.Environment.Actions import *
from src.Environment.Vizualization import *
from src.Rainbow.DQNAgent import *
import random

num_frames = 1000000
memory_size = 50000
batch_size = 128
target_update = 200

# environment

env = Environment(EnvironmentParams())

agent = DQNAgent([1,9, 9], 4, memory_size, batch_size, target_update)

n_steps = 0
scores, eps_history, steps_array = [], [], []
n_episodes = 400
best_score = -np.inf
i = 0
Viz = Vizualization()
while i < num_frames:
    done = False
    observation, info = env.reset()
    score = 0
    j = 0
    while True:
        i += 1
        j += 1
        action = agent.select_action(observation[np.newaxis,:])
        observation_, reward, done, truncated, info = env.step(Actions(action))
        # if action==0:
        #     print(action)
        agent.store_experience(observation, action, reward, observation_, done or truncated)
        agent.learn()
        agent.update_beta(i, num_frames)
        observation = observation_
        n_steps += 1
        if done or truncated:
            print(i)
            scores.append(env.rewards.get_cumulative_reward())
            print(env.rewards.get_cumulative_reward())
            if env.rewards.get_cumulative_reward() > best_score:
                print(env.rewards.get_cumulative_reward(), best_score)
                agent.save_models()
                best_score = env.rewards.get_cumulative_reward()
            score = 0
            break
    steps_array.append(n_steps)
    avg_score = np.mean(scores[-n_episodes:])

print(scores)
