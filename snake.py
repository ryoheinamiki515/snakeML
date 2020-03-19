import numpy as np

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

class Snake:
    def __init__(self, size):
        self.size = size
        self.body = []
        self.prev_dir = -1
        self.head = ()
        self.score = 1

    def create_start(self):
        random_loc = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        self.body.append(random_loc)
        self.head = random_loc

    def make_move(self, move):
        if move == LEFT:
            self.head = (self.head[0]-1, self.head[1])
        elif move == RIGHT:
            self.head = (self.head[0]+1, self.head[1])
        elif move == UP:
            self.head = (self.head[0], self.head[1]+1)
        elif move == DOWN:
            self.head = (self.head[0], self.head[1]-1)
        self.body.append(self.head)

    def update(self):
        self.body = self.body[-self.score:]
