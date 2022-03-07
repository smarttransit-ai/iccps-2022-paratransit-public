import copy
import random
import pandas as pd
from constants import Constants


class Requests:
    def __init__(self, distance_time_table):
        self._distance_time_table = distance_time_table
        self._constants = Constants()
        self._load_whole_dataset()

    def get_requests(self, chain_id):
        processed_requests = []
        all_requests = self._get_dataset_mike(chain_id)

        for index, row in all_requests.iterrows():
            request = {
                "booking_id": int(row["chain_order"]),
                "num_of_passengers": 1,

                "pickup_location": int(row["pickup_node_id"]),
                "pickup_schedule_time": int(row["pickup_time_since_midnight"]),

                "dropoff_locaton": int(row["dropoff_node_id"]),
                "dropoff_schedule_time": int(row["dropoff_time_since_midnight"]),

                "request_arrival_time": int(row["pickup_time_since_midnight"]) - self._constants.REQUEST_ARRIVAL_TIME,
                "run": random.randint(0, self._constants.NUM_VEHICLES - 1)
            }
            processed_requests.append(copy.deepcopy(request))
        return processed_requests

    def _get_dataset_mike(self, chain_id):
        return self._df[self._df['chain_id'] == chain_id]

    def _load_whole_dataset(self):
        if self._constants.TEST_NETWORK == True:
            # self._df = pd.read_csv("./dataset/MIKE/CARTA/processed/validation_chains.csv")
            self._df = pd.read_csv("./dataset/MIKE/CARTA/processed/test_chains.csv")
        else:
            self._df = pd.read_csv("./dataset/MIKE/CARTA/processed/train_chains.csv")

