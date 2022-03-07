import os
import copy
import numpy as np
import tensorflow as tf
from constants import Constants


class ReplayBuffer:
    def __init__(self, base_path, buffer_capacity=50000, batch_size=32):
        self._counter = 0
        self._constants = Constants()

        self._number_of_vehicle = self._constants.NUM_VEHICLES
        self._capacity = buffer_capacity
        self._batch_size = batch_size

        self._replay_buffer_dir = os.path.join("./", base_path, "replay_buffer")
        self._create_dir()

        if not self._constants.LOAD_REPLAY_BUFFER:
            self._initialize_buffer()
        else:
            self._load_buffer(self._constants.LOAD_REPLAY_BUFFER_VERSION)

    def _create_dir(self):
        try:
            os.makedirs(self._replay_buffer_dir)
        except OSError as error:
            print(error)

        try:
            os.makedirs(self._replay_buffer_dir)
        except OSError as error:
            print(error)

    def _initialize_buffer(self):
        self.pre_st_time = np.zeros((self._capacity, 1))
        self.pre_st_cost_remain = np.zeros((self._capacity, self._number_of_vehicle))
        self.pre_st_cost_penalty = np.zeros((self._capacity, self._number_of_vehicle))

        self.next_st_time = np.zeros((self._capacity, 1))
        self.next_st_cost_remain = np.zeros((self._capacity, self._number_of_vehicle))
        self.next_st_cost_penalty = np.zeros((self._capacity, self._number_of_vehicle))

        self.rewards = np.zeros((self._capacity, 1))

        self.np_counter = np.zeros((1))

    def save_buffer(self, version):
        np.save(f'{self._replay_buffer_dir}/pre_st_time_v{version}.npy', self.pre_st_time)
        np.save(f'{self._replay_buffer_dir}/pre_st_cost_remain_v{version}.npy', self.pre_st_cost_remain)
        np.save(f'{self._replay_buffer_dir}/pre_st_cost_penalty_v{version}.npy', self.pre_st_cost_penalty)

        np.save(f'{self._replay_buffer_dir}/next_st_time_v{version}.npy', self.next_st_time)
        np.save(f'{self._replay_buffer_dir}/next_st_cost_remain_v{version}.npy', self.next_st_cost_remain)
        np.save(f'{self._replay_buffer_dir}/next_st_cost_penalty_v{version}.npy', self.next_st_cost_penalty)

        np.save(f'{self._replay_buffer_dir}/rewards_v{version}.npy', self.rewards)

        self.np_counter[0] = self._counter
        np.save(f'{self._replay_buffer_dir}/counter_v{version}.npy', self.np_counter)

    def _load_buffer(self, version):
        self.pre_st_time = np.load(f'{self._replay_buffer_dir}/pre_st_time_v{version}.npy')
        self.pre_st_cost_remain = np.load(f'{self._replay_buffer_dir}/pre_st_cost_remain_v{version}.npy')
        self.pre_st_cost_penalty = np.load(f'{self._replay_buffer_dir}/pre_st_cost_penalty_v{version}.npy')

        self.next_st_time = np.load(f'{self._replay_buffer_dir}/next_st_time_v{version}.npy')
        self.next_st_cost_remain = np.load(f'{self._replay_buffer_dir}/next_st_cost_remain_v{version}.npy')
        self.next_st_cost_penalty = np.load(f'{self._replay_buffer_dir}/next_st_cost_penalty_v{version}.npy')

        self.rewards = np.load(f'{self._replay_buffer_dir}/rewards_v{version}.npy')

        self.np_counter = np.load(f'{self._replay_buffer_dir}/counter_v{version}.npy')
        self._counter = int(self.np_counter[0])
        print("Replay buffer loaded successfully from:", self._replay_buffer_dir)
        print("Counter set at: ", self._counter)

    def get_num_records(self):
        record_size = min(self._capacity, self._counter)
        return record_size

    def add_record(self, record):
        index = self._counter % self._capacity

        self.pre_st_time[index] = copy.deepcopy(record[0]["time"])
        self.pre_st_cost_remain[index] = copy.deepcopy(record[0]["cost_remain"])
        self.pre_st_cost_penalty[index] = copy.deepcopy(record[0]["cost_penalty"])

        self.next_st_time[index] = copy.deepcopy(record[1]["time"])
        self.next_st_cost_remain[index] = copy.deepcopy(record[1]["cost_remain"])
        self.next_st_cost_penalty[index] = copy.deepcopy(record[1]["cost_penalty"])

        self.rewards[index] = copy.deepcopy(record[2])

        self._counter = self._counter + 1

    def get_batch(self):
        record_size = self.get_num_records()
        batch_indices = np.random.choice(record_size, self._batch_size)

        pre_st_tf_time = tf.convert_to_tensor(self.pre_st_time[batch_indices])
        pre_st_tf_remain = tf.convert_to_tensor(self.pre_st_cost_remain[batch_indices])
        pre_st_tf_penalty = tf.convert_to_tensor(self.pre_st_cost_penalty[batch_indices])

        next_st_tf_time = tf.convert_to_tensor(self.next_st_time[batch_indices])
        next_st_tf_remain = tf.convert_to_tensor(self.next_st_cost_remain[batch_indices])
        next_st_tf_penalty = tf.convert_to_tensor(self.next_st_cost_penalty[batch_indices])

        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)

        pre_st_batch = [pre_st_tf_time, pre_st_tf_remain, pre_st_tf_penalty]
        next_st_batch = [next_st_tf_time, next_st_tf_remain, next_st_tf_penalty]

        return pre_st_batch, next_st_batch, reward_batch

