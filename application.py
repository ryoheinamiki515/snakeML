from snakeEnv import SnakeEnv
from snake import Snake
import numpy as np
from PIL import Image
import cv2
import argparse
import copy

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
directions = ["left", "right", "up", "down"]

class App:
    def __init__(self, size, fps, method):
        self.size = size
        self.fps = fps
        self.method = method
        self.snake_env = SnakeEnv(self.size, Snake(self.size))

    def initialize(self):
        self.snake_env.create_food()
        self.snake_env.snake.create_start()
        self.draw()

    def draw(self):
        display = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        display[self.snake_env.food_loc[1], self.snake_env.food_loc[0]] = (150, 100, 50)
        for i, coord in enumerate(self.snake_env.snake.body):
            display[coord[1], coord[0]] = (50, 100, 255-2*i)
        img = Image.fromarray(display, 'RGB')
        img = img.resize((600, 600))
        cv2.imshow("image", np.array(img))
        cv2.waitKey(round(1000/self.fps))

    def run(self):
        while True:
            prev = self.snake_env.snake.prev_dir
            if self.method == "astar":
                next_pos = astar(self.snake_env.snake.head, self.snake_env.food_loc, self.snake_env.snake.body, self.size)
            diff = np.subtract(next_pos, self.snake_env.snake.head)
            if diff[0] != 0:
                if diff[0] > 0:
                    curr = RIGHT
                else:
                    curr = LEFT
            elif diff[1] != 0:
                if diff[1] > 0:
                    curr = UP
                else:
                    curr = DOWN

            if curr == RIGHT and prev == LEFT:
                curr = prev
            elif curr == LEFT and prev == RIGHT:
                curr = prev
            elif curr == UP and prev == DOWN:
                curr = prev
            elif curr == DOWN and prev == UP:
                curr = prev

            self.snake_env.snake.prev_dir = curr
            self.process_move(curr)
            self.draw()

    def process_move(self, move):
        self.snake_env.snake.make_move(move)
        if not self.snake_env.is_valid(self.snake_env.snake.head):
            exit()
        self.snake_env.check_food()


def astar(start, goal, walls, size):
    open_set = set()
    open_set.add(start)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan(start, goal)}
    while open_set:
        current = min(open_set, key=f_score.get)
        if current == goal:
            return reconstruct_path(came_from, current)[1]
        open_set.remove(current)
        for neighbor in get_neighbors(current, walls, size):
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + manhattan(neighbor, goal)
                if neighbor not in open_set:
                    open_set.add(neighbor)
    return get_neighbors(start, walls, size)[0]


def get_neighbors(current, walls, size):
    candidates = [(current[0] - 1, current[1]), (current[0] + 1, current[1]),
                  (current[0], current[1] + 1), (current[0], current[1] - 1)]
    valid = []
    for x in candidates:
        if is_valid(x, walls, size):
            valid.append(x)
    return valid


def is_valid(curr, walls, size):
    if curr in walls:
        return False
    x, y = curr
    if x < 0 or x >= size or y < 0 or y >= size:
        return False
    return True


def reconstruct_path(cameFrom, current):
    total_path = [current]
    while current in cameFrom.keys():
        current = cameFrom[current]
        total_path.insert(0, current)
    return total_path


def manhattan(node, goal):
    return abs(node[0]-goal[0])+abs(node[1]-goal[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python Snake')

    parser.add_argument('--method', dest="search", type=str, default="astar",
                        choices=["bfs", "astar_corner", "astar", "extra", "astar_multi"],
                        help='search method - default bfs')
    parser.add_argument('--size', dest="scale", type=int, default=20,
                        help='scale - default: 20')
    parser.add_argument('--fps', dest="fps", type=int, default=30,
                        help='fps for the display - default 30')

    args = parser.parse_args()
    application = App(args.scale, args.fps, args.search)
    application.initialize()
    application.run()
