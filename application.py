from snakeEnv import SnakeEnv
from snake import Snake
import numpy as np
from PIL import Image
import cv2
import argparse

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


class App:
    def __init__(self, size, fps):
        self.size = size
        self.fps = fps
        self.snake_env = SnakeEnv(self.size, Snake(self.size))

    def initialize(self):
        self.snake_env.create_food()
        self.snake_env.snake.create_start()
        self.draw()

    def draw(self):
        display = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        display[self.snake_env.food_loc[1], self.snake_env.food_loc[0]] = (150, 100, 50)
        for coord in self.snake_env.snake.body:
            display[coord[1], coord[0]] = (50, 100, 175)
        img = Image.fromarray(display, 'RGB')
        img = img.resize((600, 600))
        cv2.imshow("image", np.array(img))
        cv2.waitKey(round(1000/self.fps))

    def run(self):
        while True:
            prev = self.snake_env.snake.prev_dir
            curr = np.random.randint(0, 4)

            if curr == RIGHT and prev == LEFT:
                curr = prev
            elif curr == LEFT and prev == RIGHT:
                curr = prev
            elif curr == UP and prev == DOWN:
                curr = prev
            elif curr == DOWN and prev == UP:
                curr = prev

            self.snake_env.snake.prev_dir = curr
            self.snake_env.snake.make_move(curr)
            if not self.snake_env.is_valid():
                break
            self.snake_env.check_food()
            self.snake_env.snake.update()
            self.draw()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python Snake')

    parser.add_argument('--method', dest="search", type=str, default="bfs",
                        choices=["bfs", "astar_corner", "astar", "extra", "astar_multi"],
                        help='search method - default bfs')
    parser.add_argument('--size', dest="scale", type=int, default=20,
                        help='scale - default: 20')
    parser.add_argument('--fps', dest="fps", type=int, default=30,
                        help='fps for the display - default 30')

    args = parser.parse_args()
    application = App(args.scale, args.fps)
    application.initialize()
    application.run()
