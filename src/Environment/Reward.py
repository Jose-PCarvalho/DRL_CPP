from src.Environment.Actions import Events


class RewardParams:
    def __init__(self, scaling):
        self.blocked_reward = -1.0/10
        self.repeated_field_reward = -1.0/10
        self.new_tile_reward = 2.0
        self.close_to_wall_reward = 1.0
        self.repeated_action_reward = 1.0
        self.finished_row_col = 1.0
        self.scaling_factor = scaling


class GridRewards:
    def __init__(self, scaling):
        self.params = RewardParams(scaling)
        self.cumulative_reward: float = 0.0

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def reset(self, scaling):
        self.cumulative_reward = 0
        self.params.scaling_factor = scaling

    def compute_reward(self, events):
        r = 0
        if Events.NEW in events:
            r += self.params.new_tile_reward * self.params.scaling_factor
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
            r += self.params.repeated_field_reward * self.params.scaling_factor
            if Events.BLOCKED in events:
                r += self.params.blocked_reward * self.params.scaling_factor

        self.cumulative_reward += r

        return r
