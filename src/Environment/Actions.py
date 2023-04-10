from enum import Enum


class Actions(Enum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3


class Events(Enum):
    BLOCKED = 0
    NEW = 1
    REPEATED = 2
    FINISHED = 3

