import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import os
import time
from tqdm import tqdm


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True

if not os.path.isdir('models'):
    os.makedirs('models')

ep_rewards = [-200]

# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=(SIZE, SIZE, 3)))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(4, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

SIZE = 10
agent = DQNAgent()
SHOW = False

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    body = [[np.random.randint(0, SIZE), np.random.randint(0, SIZE)]]
    food = [np.random.randint(0, SIZE), np.random.randint(0, SIZE)]
    food_count = 1
    direction = np.random.randint(0, 4)
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    env[food[1]][food[0]] = (150, 100, 50)
    env[body[0][1]][body[0][0]] = (50, 100, 175)
    done = False
    prev_dir = -1
    current_state = env

    # Reset flag and start iterating until episode ends
    while not done:
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            direction = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            direction = np.random.randint(0, 4)

        if direction == 0:
            #right
            potential = [body[-1][0]+1, body[-1][1]]
            if potential in body:
                done = True
            if potential[0] > SIZE-1:
                done = True
            body.append(potential)
        elif direction == 1:
            # left
            potential = [body[-1][0] - 1, body[-1][1]]
            if potential in body:
                done = True
            if potential[0] < 0:
                done = True
            body.append(potential)
        elif direction == 2:
            # up
            potential = [body[-1][0], body[-1][1]-1]
            if potential in body:
                done = True
            if potential[1] < 0:
                done = True
            body.append(potential)
        else:
            # down
            potential = [body[-1][0], body[-1][1] + 1]
            if potential in body:
                done = True
            if potential[1] > SIZE-1:
                done = True
            body.append(potential)

        if body[-1] == food:
            temp = [np.random.randint(0, SIZE), np.random.randint(0, SIZE)]
            while temp in body:
                temp = [np.random.randint(0, SIZE), np.random.randint(0, SIZE)]
            food = temp
            food_count += 1
            reward = 25
        elif done:
            reward = -300
        else:
            reward = -1
        body = body[-food_count:]
        if not done:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food[1]][food[0]] = (150, 100, 50)
            for coord in body:
                env[coord[1]][coord[0]] = (50, 100, 175)

        new_state = env

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward
        if SHOW and not episode % 200:
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            cv2.waitKey(50)
        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, direction, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)



"""
count = 1
SHOW = 500
SIZE = 50
score_dict = {}
while True:
    body = [[np.random.randint(0, SIZE), np.random.randint(0, SIZE)]]
    food = [np.random.randint(0, SIZE), np.random.randint(0, SIZE)]
    food_count = 1
    direction = np.random.randint(0, 4)
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    env[food[1]][food[0]] = (150, 100, 50)
    env[body[0][1]][body[0][0]] = (50, 100, 175)
    end_game = False
    prev_dir = -1
    while True:
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        env[food[1]][food[0]] = (150, 100, 50)
        for coord in body:
            env[coord[1]][coord[0]] = (50, 100, 175)
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
    score_dict[food_count] = score_dict.get(food_count, 0) + 1
    print(score_dict)
    count += 1
"""

