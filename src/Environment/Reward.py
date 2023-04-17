from src.Environment.Actions import Events


class RewardParams:
    def __init__(self, scaling, max_size=25):
        self.blocked_reward = -0.5
        self.repeated_field_reward = -0.5
        self.new_tile_reward = 1.0
        self.map_complete = 10 #max_size ** 2 - scaling ** 2
        self.timeout = -10#scaling ** 2
        self.close_to_wall_reward = 1.0
        self.repeated_action_reward = 1.0
        self.finished_row_col = 1.0


class GridRewards:
    def __init__(self, scaling):
        self.params = scaling
        self.cumulative_reward: float = 0.0
        self.overlap = 0
        self.steps = 0

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def get_overlap(self):
        return self.overlap / self.steps

    def reset(self, scaling):
        self.cumulative_reward = 0
        self.overlap = 0
        self.steps = 0
        self.params.scaling_factor = 1  # scaling ** 2

    def compute_reward(self, events):
        r = 0
        self.steps += 1
        if Events.NEW in events:
            r += self.params.new_tile_reward
            # if Events.REPEATED in events:
            #      r += self.params.repeated_action_reward * self.params.scaling_factor

            # if action != 4:
            #     if self.agent.scanner[-1][action] <= 0 and self.agent.scanner[-2][action] > 0:
            #         r += 1 / (self.width * self.height)
            #         # print("finished row/col")
            # if adjacent_to_wall(self.agent.scanner[-1]):
            #     r += 1 / (self.width * self.height)
            #     # print("WALL")

        else:
            r += self.params.repeated_field_reward
            self.overlap += 1
        if Events.BLOCKED in events:
            r += self.params.blocked_reward
        if Events.FINISHED in events:
            r += self.params.map_complete
        if Events.TIMEOUT in events:
            r += self.params.timeout

        self.cumulative_reward += r

        return r
