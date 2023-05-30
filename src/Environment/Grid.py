import numpy as np
from math import ceil
import heapq


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return not self.elements

    def put(self, item, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


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
            self.map_array[:, tile[0], tile[1]] = [1, 0, 0]

    def graph_to_array(self):
        a = np.zeros((3, self.height, self.width),
                     dtype=np.uint8)  # 0 -> visited, #2 -> obstacles 3->not-seen
        for i in range(self.height):
            for j in range(self.width):
                tile = (i, j)
                if tile in self.visited_list:
                    a[:, i, j] = [1, 0, 0]
                elif tile in self.obstacle_list:
                    a[:, i, j] = [0, 1, 0]
                elif tile in (self.getTiles()) and tile not in (set(self.visited_list).union(set(self.obstacle_list))):
                    a[:, i, j] = [0, 0, 1]
        return a

    def laser_scanner(self, tile, full_map, r):
        tiles = {"up": [],
                 "down": [],
                 "right": [],
                 "left": [],
                 "up-right": [],
                 "up-left": [],
                 "down-right": [],
                 "down-left": []
                 }
        for i in range(1, r + 1):
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
        full_map_tiles = full_map.getTiles()
        local_map_tiles = self.getTiles()
        for dir in directions:
            for t in tiles[dir]:
                obst = t in full_map.obstacle_list
                if t in full_map_tiles and t not in local_map_tiles:
                    self.new_tile(t, obst)
                if obst:
                    break

    def camera(self, tile, full_map, r):
        tiles = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                tiles.append((tile[0] + i, tile[1] + j))

        full_map_tiles = full_map.getTiles()
        local_map_tiles = self.getTiles()
        for t in tiles:
            if t in full_map_tiles and t not in local_map_tiles and all(cord >= 0 for cord in t):
                self.new_tile(t, obstacle=t in full_map.obstacle_list)

    def center_map(self, position):
        new_size = 37
        # calculate the center index of the new array
        center_index = new_size // 2

        # create a new array of zeros with the desired size
        new_arr = np.zeros((3, new_size, new_size), dtype=np.uint8)
        new_arr[1, :, :] = 1
        # calculate the indices of the original array that should be copied to the new array
        start_i = center_index - position[0]
        end_i = start_i + self.map_array.shape[1]
        start_j = center_index - position[1]
        end_j = start_j + self.map_array.shape[2]

        # copy the original array to the center of the new array
        new_arr[:, start_i:end_i, start_j:end_j] = self.map_array
        return new_arr

    def fix_map(self, start):
        visited = self.dfs(start)
        non_obs = set(self.getTiles()).difference(self.obstacle_list)
        should_be_obs = non_obs.difference(visited)
        for t in should_be_obs:
            for neighbor in self.map[t]:
                self.map[neighbor].remove(t)
            self.map[t] = []
            self.obstacle_list.append(t)
        self.map_array = self.graph_to_array()

    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        non_obs = set(self.getTiles()).difference(self.obstacle_list)
        l_non_obs = len(non_obs)
        if len(visited) == l_non_obs:
            return visited

        for neighbor in self.map[start]:
            if neighbor not in visited:
                self.dfs(neighbor, visited)

        return visited

    def min_manhattan_distance(self, agent_pos):
        not_seen_list = set(self.getTiles()).difference(set(self.visited_list).union(set(self.obstacle_list)))
        if len(not_seen_list) == 0:
            return 0 , [0,0]
        positions = np.array(list(not_seen_list))
        distances = positions - agent_pos
        distances = np.linalg.norm(distances, ord=1, axis=1)
        dist = min(distances) - 1
        dist_idx = np.argmin(distances)
        closest = positions[dist_idx]
        return dist, closest

    def path_min_manhattan(self,agent_pos):
        not_seen_list = set(self.getTiles()).difference(set(self.visited_list).union(set(self.obstacle_list)))
        if len(not_seen_list) == 0:
            return 0, [0, 0]
        positions = np.array(list(not_seen_list))
        distances = positions - agent_pos
        distances = np.linalg.norm(distances, ord=1, axis=1)
        indices = sorted(range(len(distances)), key=lambda x: distances[x])
        return positions, indices

    def dijkstra_search(self, start, finish):
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not frontier.empty():
            current = frontier.get()

            if current == finish:
                break

            for next in self.map[current]:
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost
                    frontier.put(next, priority)
                    came_from[next] = current

        return self.reconstruct_path(came_from, start, finish)

    @staticmethod
    def reconstruct_path(came_from, start, finish):
        current = finish
        path = []
        if finish not in came_from.keys():  # no path was found
            return []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()  # optional
        return path
