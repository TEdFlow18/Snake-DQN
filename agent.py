import tensorflow as tf
import numpy as np

class Agent:

    actions = [
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

    def predict(self, game_frame):
        img = np.array(game_frame)[np.newaxis, :]
        print("debug : img.shape :", img.shape)
        return self.model(img)

    def predict_action(self, game_frame):
        pred = self.predict(game_frame)
        return self.actions[np.argmax(pred[0])]