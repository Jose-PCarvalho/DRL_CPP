from src.Environment.Actions import Actions, Events
import random
import numpy as np

from src.Environment.Grid import GridMap


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_position(self):
        return self.x, self.y


class StateParams:
    def __init__(self):
        self.max_size = 10
        self.min_size = 5
        self.obstacles = True
        self.number_of_obstacles_random = False
        self.max_number_obstacles = 5
        self.obstacles_number_non_random = 1
        self.starting_position_random = True
        self.starting_position = (0, 0)
        self.random_size = False
        self.size = 5


class State:
    def __init__(self, Params):
        self.local_map: GridMap = None
        self.global_map = None
        self.position = None
        self.remaining = None
        self.last_action = None
        self.timesteps = None
        self.t_to_go = None
        self.optimal_steps = None
        self.terminated = False
        self.truncated = False
        self.params = StateParams()
        self.state_array = []

    def move_agent(self, action: Actions):
        events = []
        old_x = self.position.x
        old_y = self.position.y

        self.position.x += -1 if action == Actions.NORTH else 1 if action == Actions.SOUTH else 0
        # Change the column: 1=right (+1), 3=left (-1)
        self.position.y += 1 if action == Actions.EAST else -1 if action == Actions.WEST else 0
        blocked = False

        if self.position.x < 0:
            self.position.x = 0
            blocked = True
        elif self.position.x >= self.local_map.height:
            self.position.x = self.local_map.height - 1
            blocked = True
        if self.position.y < 0:
            self.position.y = 0
            blocked = True
        elif self.position.y >= self.local_map.width:
            self.position.y = self.local_map.width - 1
            blocked = True
        elif (self.position.x, self.position.y) in self.local_map.obstacle_list:
            self.position.x = old_x
            self.position.y = old_y
            blocked = True
        if blocked:
            events.append(Events.BLOCKED)
            self.position.x = old_x
            self.position.y = old_y

        if self.position.get_position() not in self.local_map.visited_list and not blocked:
            self.local_map.visit_tile((self.position.x, self.position.y))
            self.remaining -= 1
            events.append(Events.NEW)
            if self.remaining <= 0:
                self.terminated = True
                events.append(Events.FINISHED)

        if action == self.last_action:
            events.append(Events.REPEATED)

        self.last_action = action
        self.local_map.laser_scanner(self.position.get_position(), self.global_map)
        self.state_array.pop(0)
        self.state_array.append(self.local_map.center_map(self.position.get_position()))
        self.timesteps += 1
        self.t_to_go -= 1
        if self.t_to_go <= 0:
            self.truncated = True
        return events

    def init_episode(self):
        if self.params.random_size:
            width = random.randint(4, 10)
            height = width
        else:
            width = self.params.size
            height = width

        if self.params.starting_position_random:
            self.position = Position(random.randint(0, height - 1), random.randint(0, width - 1))
        else:
            self.position = Position(self.params.starting_position[0], self.params.starting_position[1])
        mapa = np.zeros((height, width), dtype=int)
        mapa[self.position.x, self.position.y] = 1

        obstacles = 0
        obstacle_number = 0
        if self.params.obstacles:
            if self.params.number_of_obstacles_random:
                obstacle_number = random.randint(0, self.params.max_number_obstacles)
            else:
                obstacle_number = self.params.obstacles_number_non_random
            while obstacles != obstacle_number:
                coord = (random.randint(0, height - 1), random.randint(0, width - 1))
                if coord != self.position.get_position():
                    mapa[coord[0], coord[1]] = -1
                    obstacles += 1
        self.global_map = GridMap(mapa)
        self.local_map = GridMap(start=self.position.get_position())
        self.remaining = height * width - 1 - obstacle_number
        self.optimal_steps = self.remaining
        self.timesteps = 0
        self.t_to_go = self.params.size ** 2 * 20
        self.terminated = False
        self.truncated = False
        s = self.local_map.center_map(self.position.get_position())
        self.state_array = [s, s, s, s]


