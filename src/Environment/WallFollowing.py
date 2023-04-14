from src.Environment.Actions import *
import numpy as np


class WallFollower:
    def __init__(self):

        self.clockwise = None
        self.lastAction = None
        self.size = None
        self.n_actions = None
        self.found_wall = None
        self.found_second_wall = False
        self.counter = 0
        self.second_wall = None
        self.second_action = None

    def select_action(self, obs):

        center = obs.shape[1] // 2
        north = (center - 1, center)
        south = (center + 1, center)
        west = (center, center - 1)
        east = (center, center + 1)

        if not self.found_second_wall:
            if not self.found_wall:
                if obs[2][north] == 1:
                    self.found_wall = True
                    self.second_action = Actions.SOUTH
                    self.lastAction = Actions.WEST
                    self.second_wall = west
            if self.found_wall:
                if obs[2][self.second_wall] == 1:
                    self.found_second_wall = True
                    self.lastAction = self.second_action

            if self.found_second_wall:
                self.n_actions = self.size
        else:
            self.n_actions -= 1
            if self.n_actions == 0 or (self.n_actions == 1 and self.counter == 3):
                self.counter += 1
                if self.counter == 4 and self.n_actions == 1:
                    self.size -= 2
                    self.counter = 0
                self.n_actions += self.size
                if self.lastAction == Actions.NORTH:
                    self.lastAction = Actions.WEST
                elif self.lastAction == Actions.WEST:
                    self.lastAction = Actions.SOUTH
                elif self.lastAction == Actions.SOUTH:
                    self.lastAction = Actions.EAST
                elif self.lastAction == Actions.EAST:
                    self.lastAction = Actions.NORTH
        print(self.lastAction, self.n_actions, self.size, self.counter, self.found_wall, self.found_second_wall)
        return self.lastAction

    def init(self, obs, size):
        self.n_actions = size
        self.size = size - 1
        center = obs.shape[1] // 2
        north = (center - 1, center)
        south = (center + 1, center)
        west = (center, center - 1)
        east = (center, center + 1)
        self.found_wall = False
        self.found_second_wall = False
        self.counter = 0
        self.second_wall = False
        self.second_action = False

        if obs[2][north] == 1:
            self.found_wall = True
            if obs[2][east] == obs[2][west] == 0:
                self.lastAction = Actions.WEST
                self.second_wall = west
                self.second_action = Actions.SOUTH
            elif obs[2][east] == 1:
                self.lastAction = Actions.WEST
                self.found_second_wall = True
            elif obs[2][west] == 1:
                self.lastAction = Actions.SOUTH
                self.found_second_wall = True

        elif obs[2][south] == 1:
            self.found_wall = True
            if obs[2][east] == obs[2][west] == 0:
                self.lastAction = Actions.EAST
                self.second_wall = east
                self.second_action = Actions.WEST

            elif obs[2][east] == 1:
                self.lastAction = Actions.NORTH
                self.found_second_wall = True
            elif obs[2][west] == 1:
                self.lastAction = Actions.EAST
                self.found_second_wall = True

        elif obs[2][west]:
            self.lastAction = Actions.SOUTH
            self.found_wall = True
            self.second_wall = south
            self.second_action = Actions.EAST
        elif obs[2][east]:
            self.lastAction = Actions.NORTH
            self.found_wall = True
            self.second_wall = north
            self.second_action = Actions.WEST

        else:
            self.lastAction = Actions.NORTH
            self.found_wall = False
            self.found_second_wall = False
