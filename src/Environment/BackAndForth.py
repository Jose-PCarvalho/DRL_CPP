from src.Environment.Actions import *
import random


class BackForth:
    def __init__(self):
        self.action_seq = []
        self.action_idx = 0
        self.center = None
        self.north = None
        self.south = None
        self.west = None
        self.east = None
        self.wall_list = []
        self.dirs = None
        self.counter = 0

    def select_action(self, obs):

        if len(self.wall_list) < 2:
            for d in self.dirs:
                if obs[2][d] == 1:
                    if d not in self.wall_list:
                        self.wall_list.append(d)
                        self.assign_actions()

        if len(self.wall_list) == 2:
            a = self.action_seq[self.counter]
            if self.counter == 1 or self.counter == 3:
                self.counter += 1
            elif (a == Actions.NORTH and obs[2][self.north]) or (a == Actions.SOUTH and obs[2][self.south]) or \
                    (a == Actions.EAST and obs[2][self.east]) or (a == Actions.WEST and obs[2][self.west]):
                self.counter += 1
        if self.counter > 3:
            self.counter = 0
        action = self.action_seq[self.counter]
        print(action, self.counter)
        return action

    def init(self, obs):
        self.center = obs.shape[1] // 2
        self.north = (self.center - 1, self.center)
        self.south = (self.center + 1, self.center)
        self.west = (self.center, self.center - 1)
        self.east = (self.center, self.center + 1)
        self.dirs = [self.north, self.south, self.west, self.east]
        self.counter = 0
        self.wall_list = []

        if obs[2][self.north] == 1:
            self.wall_list.append(self.north)
        if obs[2][self.south] == 1:
            self.wall_list.append(self.south)
        if obs[2][self.west]:
            self.wall_list.append(self.west)
        if obs[2][self.east]:
            self.wall_list.append(self.east)

        self.assign_actions()

    def assign_actions(self):
        if len(self.wall_list) == 2:
            if self.north in self.wall_list and self.east in self.wall_list:
                self.action_seq = random.choice([[Actions.SOUTH, Actions.WEST, Actions.NORTH, Actions.WEST],
                                                 [Actions.WEST, Actions.SOUTH, Actions.EAST, Actions.SOUTH]])

            elif self.north in self.wall_list and self.west in self.wall_list:
                l1 = [Actions.SOUTH, Actions.EAST, Actions.NORTH, Actions.EAST]
                l2 = [Actions.EAST, Actions.SOUTH, Actions.WEST, Actions.SOUTH]
                self.action_seq = random.choice([l1, l2])

            elif self.south in self.wall_list and self.east in self.wall_list:
                self.action_seq = random.choice([[Actions.NORTH, Actions.WEST, Actions.SOUTH, Actions.WEST],
                                                 [Actions.WEST, Actions.NORTH, Actions.EAST, Actions.NORTH]])

            elif self.south in self.wall_list and self.west in self.wall_list:
                self.action_seq = random.choice([[Actions.NORTH, Actions.EAST, Actions.SOUTH, Actions.EAST],
                                                 [Actions.EAST, Actions.NORTH, Actions.WEST, Actions.NORTH]])

        elif len(self.wall_list) == 1:
            if self.north in self.wall_list:
                self.action_seq = [random.choice([Actions.WEST, Actions.EAST])]

            elif self.west in self.wall_list:
                self.action_seq = [random.choice([Actions.SOUTH, Actions.NORTH])]

            elif self.east in self.wall_list:
                self.action_seq = [random.choice([Actions.NORTH, Actions.SOUTH])]

            elif self.south in self.wall_list:
                self.action_seq = [random.choice([Actions.WEST, Actions.EAST])]

        else:
            self.action_seq = [random.choice(list(Actions))]
