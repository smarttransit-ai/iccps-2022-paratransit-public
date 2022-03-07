from enum import Enum


class NodeType(Enum):
    PICK_UP = 1
    DROP_OFF = 2


class Constants:
    def __init__(self):
        self.NUM_VEHICLES = 5
        self.K_MAX = 500                   # Simulated annealing number of steps

        self.TOTAL_EPISODES = 100000000
        self.GAMMA = 0.99                 # discount factor
        self.LEARNING_RATE = 0.01
        self.C = 1                        # number of steps to update target value function
        self.EPSILON = 0.1                # select a random decision

        self.DEPOT_LOCATION = 3606
        self.EARLY_PICKUP_LIMIT = 15 * 60           # pickup_time - time_window
        self.REDUCE_LATE_DROPOFF_LIMIT = 10 * 60
        self.LATE_DROPOFF_LIMIT = 15 * 60 - self.REDUCE_LATE_DROPOFF_LIMIT      # dropoff_time + time_window
        self.REQUEST_ARRIVAL_TIME = 60 * 60         # 60 minutes before the schedule time
        self.MAX_NUM_PASSENGER = 8

        self.MAX_NUM_REQUESTS = 400
        self.USE_NEGATIVE_REWARD = 1
        self.NUM_SECONDS_IN_DAY = 86400.0
        self.SCALE_EXTRA_TIME_PENALTY_BY = 1        # set 3 during testing

#------------------------------------------------------------------------------------------------#
        # Testing or training
        self.TEST_NETWORK = False         # change for testing
        self.TEST_AT_DURING_TRAINING = 20
        self.TEST_FOR = 3
        self.SAVE_VERSION = 1.0
        self.LOAD_SEED_VALUE = 111         # change for testing

        # save and load Model
        self.SAVE_MODEL = True
        self.SAVE_MODEL_AT = self.TEST_AT_DURING_TRAINING
        self.SAVE_MODEL_VERSION = self.SAVE_VERSION
        self.LOAD_MODEL = False            # change for testing
        self.LOAD_MODEL_AT = 960          # change for testing;
        self.LOAD_MODEL_VERSION = 1.0     # change for testing

        # save and load Replay Buffer
        self.SAVE_REPLAY_BUFFER = True
        self.SAVE_REPLAY_BUFFER_AT = self.TEST_AT_DURING_TRAINING
        self.SAVE_REPLAY_BUFFER_VERSION = self.SAVE_VERSION
        self.LOAD_REPLAY_BUFFER = False
        self.LOAD_REPLAY_BUFFER_VERSION = 1.0