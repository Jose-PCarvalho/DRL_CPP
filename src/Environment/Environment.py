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

    def reset(self):
        self.state.init_episode()
        self.rewards.reset(self.state.params.real_size)
        return self.get_observation(), self.get_info()

    def step(self, action):
        events = self.state.move_agent(Actions(action))
        reward = self.rewards.compute_reward(events, self.state.position.get_position())
        return self.get_observation(), reward, self.state.terminated, self.state.truncated, self.get_info()

    def action_space(self):
        return len(Actions)

    def render(self):
        self.viz.render_center(self.get_observation()[0][-1, :, :, :])
        # self.viz.render_center(self.state.local_map.center_map(self.stateposition.get_position()).transpose(2, 0, 1))

    def get_observation(self):
        return (np.array(self.state.state_array), self.state.t_to_go)

    def get_info(self):
        return {}
