import tensorflow as tf
import numpy as np

class Agent:

    possible_actions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    def __init__(self, game_width):
        self.game_width = game_width

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=(5, 5), strides=(1, 1),
                padding="same",activation="elu",
                input_shape=(self.game_width, self.game_width, 1)
            ),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=(3, 3), strides=(1, 1),
                padding="same", activation="elu"
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(4)
        ])

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.max_memory_length = 100000

    def predict(self, game_frame):
        img = np.array(game_frame)[np.newaxis, :]
        return self.model(img)

    def predict_action(self, game_frame):
        pred = self.predict(game_frame)
        pred = np.argmax(pred[0])
        return self.possible_actions[pred], pred

    def add_observation(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        if len(self.states) > self.max_memory_length: self.states.pop(0)
        if len(self.actions) > self.max_memory_length: self.actions.pop(0)
        if len(self.rewards) > self.max_memory_length: self.rewards.pop(0)
        if len(self.next_states) > self.max_memory_length: self.next_states.pop(0)
        if len(self.dones) > self.max_memory_length: self.dones.pop(0)