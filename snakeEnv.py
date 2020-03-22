import numpy as np
import copy


class SnakeEnv:
    def __init__(self, size, snake):
        self.food_loc = ()
        self.size = size
        self.snake = snake

    def create_food(self):
        candidate = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        while candidate in self.snake.body or candidate == self.food_loc:
            candidate = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        self.food_loc = candidate

    def is_valid(self, head):
        if self.snake.body.count(head) != 1:
            return False
        x, y = head
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return True

    def check_food(self):
        if self.food_loc == self.snake.head:
            self.snake.score += 1
            self.snake.update()
            self.create_food()
        else:
            self.snake.update()
