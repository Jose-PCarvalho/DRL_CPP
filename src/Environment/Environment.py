from src.Environment.Reward import *
from src.Environment.Actions import *
from src.Environment.Grid import *
from src.Environment.State import *
import tqdm


class EnvironmentParams:
    def __init__(self):
        self.state_params = StateParams()
        self.reward_params = RewardParams(1/(self.state_params.size**2))


class Environment:
    def __init__(self, params: EnvironmentParams):
        self.rewards = GridRewards(params.reward_params)
        self.state = State(params.state_params)
        self.episode_count = 0

    def reset(self):
        self.state.init_episode()
        self.rewards.reset(1 / self.state.params.size)
        return self.state.get_observation(), self.state.get_info()

    def step(self, action):
        events = self.state.move_agent(action)
        reward = self.rewards.compute_reward(events)
        return self.state.get_observation(), reward, self.state.terminated, self.state.truncated, self.state.get_info()

    def run(self):
        # self.fill_replay_memory()

        print('Running ')  # , self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.episode_count < self.trainer.params.num_steps:
            bar.update(self.episode_count - last_step)
            last_step = self.episode_count
            self.train_episode()

            if self.episode_count % self.trainer.params.eval_period == 0:
                self.test_episode()

            # self.stats.save_if_best()

        # self.stats.training_ended()

    def train_episode(self):
        pass

    def test_episode(self):
        pass
