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

        self.max_memory_length = 1000000
        self.batch_size = 50
        self.gamma = 0.90

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        self.loss_fn = tf.keras.losses.mean_squared_error

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
        self.dones.append(int(done))

        if len(self.states) > self.max_memory_length: self.states.pop(0)
        if len(self.actions) > self.max_memory_length: self.actions.pop(0)
        if len(self.rewards) > self.max_memory_length: self.rewards.pop(0)
        if len(self.next_states) > self.max_memory_length: self.next_states.pop(0)
        if len(self.dones) > self.max_memory_length: self.dones.pop(0)

    def get_random_batch(self, batch_size=20):
        index = np.random.randint(low=0, high=len(self.states), size=batch_size)
        return np.array(self.states)[index], np.array(self.actions)[index], np.array(self.rewards)[index], np.array(self.next_states)[index], np.array(self.dones)[index]

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)

    def save(self):
        self.model.save("model.h5")

    def train(self):
        states, actions, rewards, next_states, dones = self.get_random_batch()
        max_next_next_states = self.gamma * np.max(self.model.predict(next_states), axis=1)
        mask = tf.one_hot(actions, depth=len(self.possible_actions))
        #target_q_values = rewards * (1 - dones) + max_next_next_states
        target_q_values = rewards + max_next_next_states
        target_q_values = target_q_values.reshape(-1, 1)
        with tf.GradientTape() as tape:
            predictions = self.model(states) * mask
            predictions = tf.reduce_sum(predictions, axis = 1, keepdims=True)
            loss = self.loss_fn(target_q_values, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables), verbose=0)