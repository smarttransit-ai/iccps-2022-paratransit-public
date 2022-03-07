import os
import tensorflow as tf
from tensorflow.keras import layers
from constants import Constants


class VFA:
    def __init__(self, base_path):
        self._counter = 0
        self._constants = Constants()
        self._number_of_vehicles = self._constants.NUM_VEHICLES

        self._save_weight_directory = os.path.join("./", base_path, "save_model")
        self._load_weight_directory = os.path.join("./", "rnslab2", "seed_" + str(self._constants.LOAD_SEED_VALUE), "save_model")
        self._create_dir()

        self.vf = self.model()
        self._target_vf = self.model()
        self._target_vf.set_weights(self.vf.get_weights())
        self._optimizer = tf.keras.optimizers.Adam(self._constants.LEARNING_RATE)

        if self._constants.LOAD_MODEL:
            self.load_weight(version=self._constants.LOAD_MODEL_VERSION, episode_num=self._constants.LOAD_MODEL_AT)

    def _create_dir(self):
        try:
            os.makedirs(self._save_weight_directory)
        except OSError as error:
            print(error)

        try:
            os.makedirs(self._load_weight_directory)
        except OSError as error:
            print(error)

    def save_weight(self, version, episode_num):
        self.vf.save_weights(f"{self._save_weight_directory}/vfa_{version}_{episode_num}.h5")
        self._target_vf.save_weights(f"{self._save_weight_directory}/target_vfa_{version}_{episode_num}.h5")

    def load_weight(self, version, episode_num):
        self.vf.load_weights(f"{self._load_weight_directory}/vfa_{version}_{episode_num}.h5")
        self._target_vf.load_weights(f"{self._load_weight_directory}/target_vfa_{version}_{episode_num}.h5")
        print("weights are loaded successfully from:", self._load_weight_directory)

    def _get_tf_state(self, state):
        tf_time = tf.expand_dims(tf.convert_to_tensor(state["time"]), 0)
        tf_cost_remain = tf.expand_dims(tf.convert_to_tensor(state["cost_remain"]), 0)
        tf_cost_penalty = tf.expand_dims(tf.convert_to_tensor(state["cost_penalty"]), 0)
        return [tf_time, tf_cost_remain, tf_cost_penalty]

    def get_predict_value(self, vehicle_state):
        tf_vehicle_state = self._get_tf_state(vehicle_state)
        value = self.vf(tf_vehicle_state)
        return self._constants.USE_NEGATIVE_REWARD * value[0][0]           # Convert from negative to positive

    def train(self, pre_state_batch, post_state_batch, reward_batch):
        with tf.GradientTape() as tape:
            target_value = self._target_vf(post_state_batch)
            y = reward_batch + self._constants.GAMMA * target_value
            value = self.vf(pre_state_batch)
            loss = tf.math.reduce_mean(tf.math.square(y - value))
        grad = tape.gradient(loss, self.vf.trainable_variables)
        self._optimizer.apply_gradients(zip(grad, self.vf.trainable_variables))

        # reset \hat(v) = v for every C steps
        if self._counter % self._constants.C == 0:
            self._target_vf.set_weights(self.vf.get_weights())

        return loss

    def model(self):
        st_time_step = layers.Input(shape=(1,))
        st_cost_remain = layers.Input(shape=(self._number_of_vehicles,))
        st_cost_penalty = layers.Input(shape=(self._number_of_vehicles,))

        state = layers.Concatenate() ([st_time_step, st_cost_remain, st_cost_penalty])
        hidden = layers.Dense(64, activation="relu") (state)
        reward = layers.Dense(1, activation="linear") (hidden)

        model = tf.keras.Model([st_time_step, st_cost_remain, st_cost_penalty], reward)
        return model

