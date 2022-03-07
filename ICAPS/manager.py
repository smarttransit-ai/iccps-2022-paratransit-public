import os
import copy
import math
import sys
import random
import numpy as np
from datetime import datetime
from constants import Constants
from requests import Requests
from vehicle import Vehicle
from node import Node
from cost import Cost
from tensor_board import Tensorboard, LogWriter
from approximated_value_function import VFA
from replay_bufferr import ReplayBuffer


class Manager:
    def __init__(self, base_path):
        self._constants = Constants()
        self._node = Node()
        self._distance_time_table = self._get_travel_time_matrix()

        self._requests = Requests(self._distance_time_table)
        self._cost = Cost(self._distance_time_table)

        self._tensorboard = Tensorboard(base_path)
        self._log_writer = LogWriter(base_path)

        self._vfa = VFA(base_path)
        self._replay_buffer = ReplayBuffer(base_path)

        self._vehicles = [None] * self._constants.NUM_VEHICLES
        for vehicle_id in range(self._constants.NUM_VEHICLES):
            self._vehicles[vehicle_id] = Vehicle(vehicle_id, self._distance_time_table, self._cost, self._vfa,
                                                 self._replay_buffer, self._log_writer)

    def manage(self):
        test_network = self._constants.TEST_NETWORK
        if test_network:
            self._constants.TOTAL_EPISODES = 15

        total_service_count_filtered_naive = 0
        total_service_count_filtered = 0

        for episode in range(self._constants.TOTAL_EPISODES):
            if episode % self._constants.TEST_AT_DURING_TRAINING == 0 and episode != 0 and self._constants.TEST_NETWORK == False:
                test_network = True
            if episode % self._constants.TEST_AT_DURING_TRAINING == self._constants.TEST_FOR and self._constants.TEST_NETWORK == False:
                test_network = False

            if self._constants.TEST_NETWORK == False:
                chain_id = random.randint(0, 99)    # for training
            else:
                chain_id = episode                    # for testing (0 - 14)

            start_time = datetime.now()
            orders = self._get_online_orders(chain_id)
            for vehicle_id in range(self._constants.NUM_VEHICLES):
                self._vehicles[vehicle_id].reset_vehicle(episode)
            vehicle_states = self._get_vehicle_states()

            vfa_loss = -1
            num_requests = len(orders)
            for order in orders:
                previous_state = copy.deepcopy(vehicle_states)

                current_time = order[0]["request_arrival_time"]
                vehicle_states["time"][0] = current_time / self._constants.NUM_SECONDS_IN_DAY

                # print("time:", current_time, "state:", vehicle_states)

                for vehicle_id, vehicle in enumerate(self._vehicles):
                    self._vehicles[vehicle_id].update_vehicle_location(current_time)

                min_reward = math.inf
                min_reward_vehicle_id = -1
                min_results = (-1, -1, -1, -1)

                for vehicle_id, vehicle in enumerate(self._vehicles):
                    vehicle_states1 = copy.deepcopy(vehicle_states)
                    results = self._vehicles[vehicle_id].add_new_order(order, test_network, vehicle_states1)
                    # print("id:", vehicle_id, ", reward:", results[1])

                    if results[1] != -1 and min_reward > results[1]:
                        min_reward = results[1]
                        min_reward_vehicle_id = vehicle_id
                        min_results = copy.deepcopy(results)

                if min_reward_vehicle_id != -1:
                    self._vehicles[min_reward_vehicle_id].update_route(min_results[0])

                    vehicle_states["cost_remain"][min_reward_vehicle_id] = min_results[2]
                    vehicle_states["cost_penalty"][min_reward_vehicle_id] = min_results[3]
                    self._replay_buffer.add_record([previous_state, vehicle_states, self._constants.USE_NEGATIVE_REWARD * min_reward])    # save as negative
                else:
                    print("could not assign order:", order[0]["booking_id"])

                if test_network == False:
                    vfa_loss = self.train_network()
                    self._tensorboard.vfa_loss(vfa_loss)

            # save model weights
            if (episode % self._constants.SAVE_MODEL_AT == 0) and self._constants.SAVE_MODEL:
                self._vfa.save_weight(version=self._constants.SAVE_MODEL_VERSION, episode_num=episode)

            if (episode % self._constants.SAVE_REPLAY_BUFFER_AT == 0) and self._constants.SAVE_REPLAY_BUFFER:
                self._replay_buffer.save_buffer(self._constants.SAVE_REPLAY_BUFFER_VERSION)

            computation_time = (datetime.now() - start_time).total_seconds()

            if self._constants.TEST_NETWORK:
                self._save_routes(chain_id)

            total_valid_service_count_filtered_naive = 0
            total_valid_service_count_filtered = 0
            for vehicle_id in range(self._constants.NUM_VEHICLES):
                total_valid_service_count_filtered_naive += self._vehicles[vehicle_id].get_valid_service_count()
                total_valid_service_count_filtered += self._vehicles[vehicle_id].get_valid_service_count_filtered()

            total_cost = self._get_total_cost()
            self._tensorboard.total_cost(total_cost)
            self._log_writer.add_reward_info(episode, chain_id, num_requests, total_cost, vfa_loss, computation_time,
                                             total_valid_service_count_filtered_naive, total_valid_service_count_filtered)

            service_rate_filtered_naive = round(total_valid_service_count_filtered_naive/num_requests, 2)
            service_rate_filtered = round(total_valid_service_count_filtered/num_requests, 2)

            total_service_count_filtered_naive += service_rate_filtered_naive
            total_service_count_filtered += service_rate_filtered

            print("Episode:", episode, "test_network:", test_network, ", date:", chain_id, ", num_requests:", num_requests,
                  ", total_cost:", total_cost, ", vfa_loss:", vfa_loss, ", total_time:", computation_time,
                  " seconds, time_per_req:", round(computation_time/num_requests, 3),
                  "seconds, total_valid_service_count_filtered_naive:", total_valid_service_count_filtered_naive,
                  ", service_rate_filtered_naive:", service_rate_filtered_naive, ", total_valid_service_count_filtered:",
                  total_valid_service_count_filtered, ", service_rate_filtered:", service_rate_filtered)

        print("Average_service_rate_filtered_naive:", total_service_count_filtered_naive/15, ", average_service_rate_filtered:",
              total_service_count_filtered/15)


    def train_network(self):
        pre_st_batch, next_st_batch, reward_batch = self._replay_buffer.get_batch()
        loss = self._vfa.train(pre_st_batch, next_st_batch, reward_batch)
        return loss.numpy()

    def _get_online_orders(self, chain_id):
        requests = self._requests.get_requests(chain_id)

        all_orders = []
        for req in requests:
            pickup_node, dropoff_node = self._node.get_nodes(req)
            all_orders.append((pickup_node, dropoff_node))

        return all_orders

    def _get_vehicle_states(self):
        vehicle_states = {
            "time": np.zeros(1, float),
            "cost_remain": np.zeros(self._constants.NUM_VEHICLES, float),
            "cost_penalty": np.zeros(self._constants.NUM_VEHICLES, float),
        }

        for i, vehicle in enumerate(self._vehicles):
            vehicle_states["cost_remain"][i] = 0
            vehicle_states["cost_penalty"][i] = 0

        return vehicle_states

    def _get_travel_time_matrix(self):
        filepath = os.path.join(os.getcwd(), 'dataset', 'travel_time_matrix', 'travel_time_matrix.csv')
        with open(filepath, 'rb') as fd:
            travel_time_matrix = np.loadtxt(fd, delimiter=",", dtype=float)
            travel_time_matrix = np.rint(travel_time_matrix)  # round to integer
            travel_time_matrix = np.where(travel_time_matrix < 0, 86400, travel_time_matrix)

        # print("size travel_time_matrix:", round(travel_time_matrix.nbytes/1024/1024, 2), "MB")
        return travel_time_matrix

    def _save_routes(self, chain_id):
        for i in range(self._constants.NUM_VEHICLES):
            if self._vehicles[i] is not None:
                self._vehicles[i].save_routes(chain_id)

    def _print_order(self, orders):
        for i, order in enumerate(orders):
            print(i, ".", order)

    def _get_total_cost(self):
        total_cost = 0
        for id, vehicle in enumerate(self._vehicles):
            if vehicle is not None:
                cost, *__ = vehicle.get_cost()
                # print("vehicle:", id, ", cost:", vehicle.get_cost())
                total_cost += cost
        return total_cost
