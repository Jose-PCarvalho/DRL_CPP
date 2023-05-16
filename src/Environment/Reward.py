from src.Environment.Actions import Events
from src.Environment.State import State


class RewardParams:
    def __init__(self, scaling, max_size=25):
        self.blocked_reward = -1
        self.repeated_field_reward = -0.5
        self.new_tile_reward = 0
        self.map_complete = 0  # max_size ** 2 - scaling ** 2
        self.timeout = 0  # scaling ** 2
        self.close_to_wall_reward = 1.0
        self.repeated_action_reward = 1.0
        self.finished_row_col = 1.0
        self.repeating_two_moves = -1.0


class GridRewards:
    def __init__(self, scaling):
        self.last_remaining_potential = None
        self.last_closest = None
        self.params = scaling
        self.cumulative_reward: float = 0.0
        self.overlap = 0
        self.steps = 0
        self.total_steps = None
        self.remaining = None
        self.closest = None

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def get_overlap(self):
        return self.overlap / self.steps  # self.overlap / (self.steps - self.overlap + 1)

    def reset(self, state: State):
        self.cumulative_reward = 0
        self.overlap = 0
        self.steps = 0
        self.params.scaling_factor = 1  # scaling ** 2
        self.total_steps = state.remaining
        self.remaining = state.remaining
        self.last_remaining_potential = -self.remaining  # / self.total_steps
        closest_cell = state.local_map.min_manhattan_distance(state.position.get_position())[1]
        self.closest = -len(
            state.local_map.dijkstra_search(state.position.get_position(), (closest_cell[0], closest_cell[1])))

    def compute_reward(self, events, state: State):
        r = 0
        self.steps += 1
        self.remaining = state.remaining
        new_remaining_potential = - self.remaining  # / self.total_steps
        new_closest_cell = state.local_map.min_manhattan_distance(state.position.get_position())[1]
        new_closest = -len(state.local_map.dijkstra_search(state.position.get_position(), (new_closest_cell[0], new_closest_cell[1])))

        if Events.NEW in events:
            r += self.params.new_tile_reward
        else:
            # r += self.params.repeated_field_reward
            r += 0.25 * (new_closest - self.closest)
            self.overlap += 1
        if Events.BLOCKED in events:
            r += self.params.blocked_reward
        if Events.FINISHED in events:
            r += self.params.map_complete
        if Events.TIMEOUT in events:
            r += self.params.timeout
        r += self.params.repeated_field_reward
        r += (new_remaining_potential - self.last_remaining_potential) * 1.5
        self.last_remaining_potential = new_remaining_potential
        self.cumulative_reward += r
        self.closest = new_closest
        return r
