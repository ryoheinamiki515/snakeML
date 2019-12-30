import cv2
import numpy as np
from PIL import Image
import curses
import time


count = 0
SHOW = 10000
score_dict = {}
while True:
    SIZE = 50
    body = [[np.random.randint(0, SIZE), np.random.randint(0, SIZE)]]
    food = [np.random.randint(0, SIZE), np.random.randint(0, SIZE)]
    food_count = 1
    direction = np.random.randint(0, 4)
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
    env[food[1]][food[0]] = (150, 100, 50)
    env[body[0][1]][body[0][0]] = (50, 100, 175)
    end_game = False
    prev_dir = -1
    while True:
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[food[1]][food[0]] = (150, 100, 50)
        for coord in body:
            env[coord[1]][coord[0]] = (50, 100, 175)
        if direction == 0:
            #right
            potential = [body[-1][0]+1, body[-1][1]]
            if potential in body:
                end_game = True
            if potential[0] > SIZE-1:
                end_game = True
            body.append(potential)
        elif direction == 1:
            # left
            potential = [body[-1][0] - 1, body[-1][1]]
            if potential in body:
                end_game = True
            if potential[0] < 0:
                end_game = True
            body.append(potential)
        elif direction == 2:
            # up
            potential = [body[-1][0], body[-1][1]-1]
            if potential in body:
                end_game = True
            if potential[1] < 0:
                end_game = True
            body.append(potential)
        else:
            # down
            potential = [body[-1][0], body[-1][1] + 1]
            if potential in body:
                end_game = True
            if potential[1] > SIZE-1:
                end_game = True
            body.append(potential)

        if body[-1] == food:
            temp = [np.random.randint(0, SIZE), np.random.randint(0, SIZE)]
            while temp in body:
                temp = [np.random.randint(0, SIZE), np.random.randint(0, SIZE)]
            food = temp
            food_count += 1
        body = body[-food_count:]
        if count % SHOW == 0:
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            cv2.waitKey(50)
        if end_game:
            break
        prev_dir = direction
        direction = np.random.randint(0, 4)
        if prev_dir == 0 and direction == 1:
            direction = 0
        elif prev_dir == 1 and direction == 0:
            direction = 1
        elif prev_dir == 2 and direction == 3:
            direction = 2
        elif prev_dir == 3 and direction == 2:
            direction = 3
    score_dict[food_count] = score_dict.get(food_count, 0) + 1
    if count % SHOW == 0:
        print(score_dict)
    count += 1



