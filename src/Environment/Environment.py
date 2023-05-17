from src.Environment.Reward import *
from src.Environment.Actions import *
from src.Environment.Grid import *
from src.Environment.State import *
from src.Environment.Actions import *
from src.Environment.Vizualization import *
import tqdm


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

    def reset(self):
        if self.state.truncated:
            self.state.partial_reset()
        else:
            self.state.init_episode()
        self.rewards.reset(self.state)
        self.remaining=self.state.remaining
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
        return (np.array(self.state.state_array), self.state.t_to_go)

    def get_info(self):
        if self.remaining == self.state.remaining:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        self.remaining = self.state.remaining
        if self.stall_counter > 15:
            return True
        else:
            return False


    def get_heuristic_action(self):
        closest = self.state.local_map.min_manhattan_distance(self.state.position.get_position())[1]
        path = self.state.local_map.dijkstra_search(self.state.position.get_position(), (closest[0], closest[1]))
        if len(path) == 0:
            print('wtf')
        next = path[0]
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
