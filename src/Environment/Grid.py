import numpy as np
from math import ceil


class GridMap:
    def __init__(self, a=None, start=None):

        if start is not None:
            self.map = {start: []}
            self.visited_list = [start]
        else:
            self.map = {}
            self.visited_list = []
        self.obstacle_list = []

        if a is not None:
            self.height = a.shape[0]
            self.width = a.shape[1]
            # self.map_array = np.zeros((self.height, self.width), dtype=int)
            for i in range(self.width):
                for j in range(self.height):
                    self.new_tile((i, j), a[i, j] == -1)
            for t in self.getTiles():
                if not self.map[t] and t not in self.obstacle_list:
                    self.obstacle_list.append(t)

        else:
            self.height = max(start[0] + 1, start[1] + 1, 2)
            self.width = self.height
        self.map_array = self.graph_to_array()

    @staticmethod
    def adjacentTiles(tile):

        return [(tile[0] + 1, tile[1]), (tile[0] - 1, tile[1]), (tile[0], tile[1] + 1), (tile[0], tile[1] - 1)]

    def getTiles(self):
        return list(self.map.keys())

    def new_tile(self, tile, obstacle=False):
        if tile not in self.getTiles():
            if tile[0] >= self.height:
                self.height = tile[0] + 1
                self.width = self.height


            elif tile[1] >= self.width:
                self.width = tile[1] + 1
                self.height = self.width

            self.map[tile] = []
            if not obstacle:
                adjacent = self.adjacentTiles(tile)
                for adj in adjacent:
                    if adj in set(self.getTiles()).difference(set(self.obstacle_list)):
                        self.map[tile].append(adj)
                        self.map[adj].append(tile)
            else:
                self.obstacle_list.append(tile)
            self.map_array = self.graph_to_array()

    def print_graph(self):
        tiles = self.getTiles()
        for t in tiles:
            print(t, ": ", self.map[t])
        print("end\n")

    def visit_tile(self, tile):
        if tile not in self.visited_list:
            self.visited_list.append(tile)
            self.map_array[tile[0], tile[1], :] = [255, 0, 0]

    def graph_to_array(self):
        a = np.zeros((self.height, self.width, 3),
                     dtype=np.uint8)  # 0 -> visited , #1 ->non-visited, #2 -> obstacles 3->not-seen
        for i in range(self.height):
            for j in range(self.width):
                tile = (i, j)
                if tile in self.visited_list:
                    a[i, j, 0] = 255
                elif tile in (self.getTiles()) and tile not in (set(self.visited_list).union(set(self.obstacle_list))):
                    a[i, j, :] = [255, 255, 255]
                elif tile in self.obstacle_list:
                    pass
                else:
                    a[i, j, 2] = 255
        return a

    def graph_to_RGB_array(self):
        rg = np.zeros((self.height, self.width, 2), dtype=bool)  # default True
        b = np.ones((self.height, self.width), dtype=bool)
        rgb = np.dstack((rg, b))

        for tile in self.getTiles():

            if tile in self.visited_list:  # RED
                rgb[tile[0], tile[1], 0] = True
                rgb[tile[0], tile[1], 2] = False

            elif tile in self.obstacle_list:  # black
                rgb[tile[0], tile[1], 0] = False
                rgb[tile[0], tile[1], 1] = False
                rgb[tile[0], tile[1], 2] = False

        rgb = (rgb.astype(np.uint8) * 255).astype(np.uint8)
        return rgb

    def laser_scanner(self, tile, full_map):
        r = 4
        tiles = {"up": [],
                 "down": [],
                 "right": [],
                 "left": [],
                 "up-right": [],
                 "up-left": [],
                 "down-right": [],
                 "down-left": []
                 }
        for i in range(1, r):
            tiles["up"].append((tile[0] - i, tile[1]))
            tiles["down"].append((tile[0] + i, tile[1]))
            tiles["right"].append((tile[0], tile[1] + i))
            tiles["left"].append((tile[0], tile[1] - i))
        for i in range(1, ceil(r / np.sqrt(r) + 1)):
            tiles["up-right"].append((tile[0] - i, tile[1] + i))
            tiles["up-left"].append((tile[0] - i, tile[1] - i))
            tiles["down-right"].append((tile[0] + i, tile[1] + i))
            tiles["down-left"].append((tile[0] + i, tile[1] - i))
        directions = tiles.keys()
        to_remove = []
        obstacles = []
        full_map_tiles = full_map.getTiles()
        local_map_tiles = self.getTiles()
        for dir in directions:
            if tiles[dir][0] not in full_map_tiles or tiles[dir][0] in full_map.obstacle_list:
                obstacles.append(True)
            else:
                obstacles.append(False)

        for dir in directions:
            to_remove.clear()
            remove_further = False
            for t in tiles[dir]:
                if t in full_map.getTiles() and t not in local_map_tiles:
                    if t in full_map.obstacle_list:
                        self.new_tile(t, obstacle=True)
                    else:
                        self.new_tile(t)
                elif t not in full_map_tiles:
                    to_remove.append(t)

                if t in (self.visited_list + self.obstacle_list) or (remove_further and t in full_map_tiles):
                    to_remove.append(t)
                    remove_further = True
            for rem in to_remove:
                if rem in tiles[dir]:
                    tiles[dir].remove(rem)
        ranges = []
        for dir in directions:
            ranges.append(len(tiles[dir]))

        for n in range(len(ranges)):
            if ranges[n] == 0 and obstacles[n] == True:
                ranges[n] = -1

        return ranges

    def center_map(self, position):
        new_size = 9
        # calculate the center index of the new array
        center_index = new_size // 2

        # create a new array of zeros with the desired size
        new_arr = np.zeros((new_size, new_size, 3), dtype=np.uint8)

        # calculate the indices of the original array that should be copied to the new array
        start_i = center_index - position[0]
        end_i = start_i + self.map_array.shape[0]
        start_j = center_index - position[1]
        end_j = start_j + self.map_array.shape[1]

        # copy the original array to the center of the new array
        new_arr[start_i:end_i, start_j:end_j, :] = self.map_array

        #gray_scale = np.dot(new_arr, [0.299, 0.587, 0.114])

        return new_arr