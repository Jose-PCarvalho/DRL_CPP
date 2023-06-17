import time

from src.Environment.Reward import *
from src.Environment.Actions import *
from src.Environment.Grid import *
from src.Environment.State import *
from src.Environment.Actions import *
from src.Environment.Vizualization import *
import tqdm
import copy

class EnvironmentParams:
    def __init__(self, args):
        self.state_params = StateParams(args)
        self.reward_params = RewardParams(self.state_params.size)


class Environment:
    def __init__(self, params: EnvironmentParams):
        self.rewards = GridRewards(params.reward_params)
        self.state = State(params.state_params)
        self.episode_count = 0
        self.viz = Vizualization()
        self.params = params.state_params
        self.stall_counter = 0
        self.remaining = 0  ## Only for stalling purposes, the actual variable is in the state.
        self.interesting_states = []
        self.was_partial = False
        self.heuristic_position = None
        self.position_locked = False

    def reset(self, training=True):

        if self.state.truncated and training:
            self.state.partial_reset()
            self.was_partial = True
        else:
            len_states = len(self.interesting_states)
            if not self.was_partial and len_states > 0 and training:
                self.interesting_states.pop(-1)
                len_states -= 1
            self.was_partial = False
            if len_states > 1 and np.random.random() < 0.5 and training:
                self.state = self.interesting_states.pop(0)
            else:
                self.state.init_episode()
                if training:
                    self.interesting_states.append(copy.deepcopy(self.state))

        self.rewards.reset(self.state)
        self.remaining = self.state.remaining
        self.heuristic_position = None
        self.position_locked = False
        return self.get_observation(), self.get_info()

    def step(self, action):
        events = self.state.move_agent(Actions(action))
        reward = self.rewards.compute_reward(events, self.state)
        return self.get_observation(), reward, self.state.terminated, self.state.truncated, self.get_info()

    def action_space(self):
        return len(Actions)

    def render(self):
        self.viz.render_center(self.get_observation()[0][-1, :, :, :])
        # self.viz.render_center(self.state.local_map.center_map(self.stateposition.get_position()).transpose(2, 0, 1))

    def get_observation(self):
        oob = np.array(self.state.out_of_bounds)
        oob = oob[:, :, 0:2]
        return (np.array(self.state.state_array), self.state.t_to_go, np.array(self.state.last_action),oob)

    def get_info(self):
        if self.remaining == self.state.remaining:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        self.remaining = self.state.remaining
        if self.stall_counter > 15:
            return True
        else:
            self.position_locked = False
            return False

    def get_heuristic_action(self):
        if self.position_locked == False:
            positions, indices = self.state.local_map.path_min_manhattan(self.state.position.get_position())
            for i in indices:
                path = self.state.local_map.dijkstra_search(self.state.position.get_position(),
                                                            (positions[i][0], positions[i][1]))
                if len(path) != 0:
                    self.position_locked = True
                    self.heuristic_position = positions[i]
                    break
        else:
            path = self.state.local_map.dijkstra_search(self.state.position.get_position(),
                                                        (self.heuristic_position[0], self.heuristic_position[1]))
        next = path[0]
        if np.array_equal(next, self.heuristic_position):
            self.position_locked = False
            self.heuristic_position = None
        diff = np.array(next) - np.array(self.state.position.get_position())
        diff = (diff[0], diff[1])
        if diff == (1, 0):
            return Actions.SOUTH
        elif diff == (-1, 0):
            return Actions.NORTH
        if diff == (0, 1):
            return Actions.EAST
        if diff == (0, -1):
            return Actions.WEST
