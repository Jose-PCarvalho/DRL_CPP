import itertools

from src.Environment.Environment import *
from src.Environment.Grid import *
import numpy as np
from PIL import Image
from src.Environment.State import *
from src.Environment.Actions import *
from src.Environment.Vizualization import *
import random
env = Environment(EnvironmentParams())
Viz = Vizualization()
while True:
    env.reset()
    for t in itertools.count():
        observation_, reward, done, truncated, info = env.step(random.choice(list(Actions)))
        print(reward)
        #print(env.state.remaining)
        Viz.render_center(observation_)
        if done or truncated:
            break
