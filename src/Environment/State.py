import time

from numpy import ceil

from src.Environment.Actions import Actions, Events
import random
import numpy as np
import yaml
from src.Environment.Grid import GridMap


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_position(self):
        return self.x, self.y


class StateParams:
    def __init__(self, args):
        self.size = args['size']
        self.min_size = args['min_size']
        self.random_size = args['random_size']
        self.obstacles_random = args['obstacles_random']
        self.number_obstacles = args['number_obstacles']
        self.starting_position_random = args['starting_position_random']
        self.starting_position = args['starting_position']
        self.starting_position_corner = args['starting_position_corner']
        self.real_size = None
        self.sensor_range = args['sensor_range']
        self.sensor = args['sensor']
        self.random_coverage = args['random_coverage']
        if args['map_config'] != 'empty':
            with open(args['map_config'], 'r') as file:
                yaml_data = yaml.safe_load(file)
            self.map_data = np.array(yaml_data['map'])
        else:
            self.map_data = None


class State:
    def __init__(self, Params):
        self.local_map: GridMap = None
        self.global_map = None
        self.position = None
        self.remaining = None
        self.last_action = 0
        self.timesteps = None
        self.t_to_go = None
        self.optimal_steps = None
        self.terminated = False
        self.truncated = False
        self.params = Params
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

        if action == self.last_action:
            events.append(Events.REPEATED)

        self.last_action = action
        if self.params.sensor == "laser":
            self.local_map.laser_scanner(self.position.get_position(), self.global_map, self.params.sensor_range)
        elif self.params.sensor == "camera":
            self.local_map.camera(self.position.get_position(), self.global_map, self.params.sensor_range)

        self.state_array.pop(0)
        self.state_array.append(self.local_map.center_map(self.position.get_position()))
        self.timesteps += 1
        self.t_to_go -= 1
        if self.t_to_go <= 0:
            self.truncated = True
            events.append(Events.TIMEOUT)

        if self.remaining < 1:
            self.terminated = True
            events.append(Events.FINISHED)
        return events

    def init_episode(self):
        if self.params.random_size:
            width = random.randint(self.params.min_size, self.params.size)
            height = width
            self.params.real_size = width
        else:
            width = self.params.size
            height = width
            self.params.real_size = self.params.size

        if self.params.starting_position_random:
            self.position = Position(random.randint(0, height - 1), random.randint(0, width - 1))
        elif self.params.starting_position_corner:
            corners = [(0, 0), (self.params.real_size - 1, 0), (self.params.real_size - 1, self.params.real_size - 1),
                       (0, self.params.real_size - 1)]
            pos = random.choice(corners)
            self.position = Position(pos[0], pos[1])
        else:
            self.position = Position(self.params.starting_position[0], self.params.starting_position[1])

        if self.params.map_data is not None:
            mapa = self.params.map_data
        else:
            mapa = np.zeros((height, width), dtype=int)
            if self.params.number_obstacles > 0:
                if self.params.obstacles_random:
                    obstacle_number = random.randint(0, self.params.number_obstacles)
                else:
                    obstacle_number = self.params.number_obstacles
                obstacles = 0
                while obstacles != obstacle_number:
                    coord = (random.randint(0, height - 1), random.randint(0, width - 1))
                    if coord != self.position.get_position():
                        mapa[coord[0], coord[1]] = -1
                        obstacles += 1

        self.global_map = GridMap(mapa)
        if self.params.map_data is not None or self.params.number_obstacles > 0:
            self.global_map.fix_map(self.position.get_position())
        if self.params.sensor == "full information":
            self.local_map = self.global_map
        else:
            self.local_map = GridMap(start=self.position.get_position())

        self.local_map.visit_tile(self.position.get_position())
        if self.params.random_coverage and np.random.random() < 0.5:
            for i in range(0, random.randint(0, ceil(self.params.real_size ** 2 / 1.5))):
                tile = (random.randint(0, self.params.real_size-1), random.randint(0, self.params.real_size-1))
                if tile not in self.local_map.visited_list and tile in set(self.global_map.getTiles()).difference(self.global_map.obstacle_list):
                    self.local_map.visit_tile(tile)

        if self.params.sensor == "laser":
            self.local_map.laser_scanner(self.position.get_position(), self.global_map, self.params.sensor_range)
        elif self.params.sensor == "camera":
            self.local_map.camera(self.position.get_position(), self.global_map, self.params.sensor_range)

        self.remaining = len(set(self.global_map.getTiles()).difference(self.global_map.obstacle_list)) - len(self.local_map.visited_list)
        if self.remaining < 1:
            self.init_episode()
        self.optimal_steps = self.remaining
        self.timesteps = 0
        self.t_to_go = self.params.size ** 2 * 4
        self.terminated = False
        self.truncated = False
        s = self.local_map.center_map(self.position.get_position())
        self.state_array = [s]
        self.last_action = Actions.NORTH

    def partial_reset(self):
        self.t_to_go = self.params.size ** 2 * 2
        self.terminated = False
        self.truncated = False
        self.timesteps = 0
        s = self.local_map.center_map(self.position.get_position())
        self.state_array = [s]
        if self.remaining < 1:
            self.init_episode()
