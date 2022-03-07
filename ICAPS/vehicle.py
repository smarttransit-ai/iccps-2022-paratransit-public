import math
import random
import numpy as np
import copy
import tensorflow as tf
from constants import Constants, NodeType


class Vehicle:
    def __init__(self, vehicle_id, distance_time_table, cost, vfa, replay_buffer, log_writer):
        self._vehicle_id = vehicle_id
        self._route = []
        self._vehicle_location = -1
        self._num_passenger = 0
        self._previous_route_cost = 0

        self._cost = cost
        self._distance_time_table = distance_time_table
        self._constants = Constants()

        self._vfa = vfa
        self._replay_buffer = replay_buffer
        self._log_writer = log_writer

        self._sa_scale = 750
        self._max = -math.inf
        self._min = math.inf

    def reset_vehicle(self, episode):
        self._route = []
        self._vehicle_location = -1
        self._num_passenger = 0
        self._previous_route_cost = 0
        self._episode = episode
        self._total_rejection_count = 0

    def update_vehicle_location(self, req_arrival_time):
        for i in range(self._vehicle_location + 1, len(self._route)):
            if self._route[i]["assign_time"] < req_arrival_time:
                self._vehicle_location = i

                if self._route[i]["node_type"] == NodeType.PICK_UP:
                    self._num_passenger += 1
                else:
                    self._num_passenger -= 1
            else:
                break

        if self._vehicle_location + 1 < len(self._route):
            if self._route[self._vehicle_location]["assign_time"] < req_arrival_time:
                self._vehicle_location += 1

                if self._route[self._vehicle_location]["node_type"] == NodeType.PICK_UP:
                    self._num_passenger += 1
                else:
                    self._num_passenger -= 1

        # if len(self._route):
        #     print("vehicle_id:", self._vehicle_id, " route_len:", len(self._route), " vehicle_location:", self._vehicle_location,
        #         "vehi_cur_node_time:", self._route[self._vehicle_location]["assign_time"], "req_arrival_time:", req_arrival_time,
        #           "num_passengers:", self._num_passenger)

    def add_new_order(self, order, test_network, vehicle_states):
        request = copy.deepcopy(order)
        request[0]["vehicle_id"] = self._vehicle_id
        request[1]["vehicle_id"] = self._vehicle_id
        route = copy.deepcopy(self._route)

        if (random.uniform(0, 1) < self._constants.EPSILON) and test_network is False:        # explore network
            reward, cost_remain, cost_penalty = self._randomly_insert_new_order(route, request)
            return (route, reward, cost_remain, cost_penalty)
        else:
            if self._insert_new_order(route, request):      # next state pre-decision
                min_route, reward, cost_remain, cost_penalty = self._apply_SA(route, vehicle_states)  # pre-decision to post-decision
                return (min_route, reward, cost_remain, cost_penalty)
            else:
                return ([], -1, -1, -1)

    def update_route(self, min_reward_route):
        self._route = copy.deepcopy(min_reward_route)

    def _randomly_insert_new_order(self, route, request):
        # lower_bound = self._get_lower_bound_index(order)
        if len(route) == 0:
            route.append(request[0])
            route.append(request[1])
        else:
            pickup_index = random.randint(self._vehicle_location + 1, len(route))
            route[pickup_index:pickup_index] = [request[0]]

            dropoff_index = random.randint(pickup_index + 1, len(route))
            route[dropoff_index:dropoff_index] = [request[1]]

            if self._number_of_passenger_constraints(route) == False:
                route.pop(dropoff_index)
                route.pop(pickup_index)

                # insert at the end if random location does not work
                route.append(request[0])
                route.append(request[1])
                if self._number_of_passenger_constraints(route) == False:
                    route.pop()
                    route.pop()
                    return -1, -1, -1

        self._cost.update_run_time(route)
        pre_cost = self._previous_route_cost
        self._previous_route_cost, *__ = self._cost.calculate_cost(route)
        reward = self._previous_route_cost - pre_cost

        __, cost_remain, cost_penalty = self._cost.calculate_cost(route[self._vehicle_location+1:])
        return reward, cost_remain, cost_penalty

    def _insert_new_order(self, route, request):
        pickup_index = self._get_index_with_closer_assign_time(route, request[0], self._vehicle_location+1)
        route[pickup_index:pickup_index] = [request[0]]

        dropoff_index = self._get_index_with_closer_assign_time(route, request[1], pickup_index+1)
        route[dropoff_index:dropoff_index] = [request[1]]

        if self._number_of_passenger_constraints(route) == False:
            route.pop(dropoff_index)
            route.pop(pickup_index)

            # append at the end
            route.append(request[0])
            route.append(request[1])
            if self._number_of_passenger_constraints(route) == False:
                route.pop()
                route.pop()
                return False

        self._cost.update_run_time(route)
        # self.print_route()
        return True

    def _get_index_with_closer_assign_time(self, route, order, from_index):
        index = len(route)
        min_time_diff = math.inf
        for i in range(from_index, len(route)):
            from_node = route[i]
            travel_time = self._distance_time_table[from_node["location"], order["location"]]
            reach_time = from_node["assign_time"] + travel_time

            time_diff = abs(reach_time - order["schedule_time"])
            if min_time_diff > time_diff:
                index = i + 1
                min_time_diff = time_diff
        return index

    def _apply_SA(self, route, vehicle_states):
        if self._vehicle_location + 2 >= len(route):      # only one node remaining at max
            return self._get_reward_for_not_having_enough_remaining_route(route, vehicle_states)

        # if self._vehicle_location + 7 >= len(route):      # greedy is faster than SA when rem_route_len is less than 6
        #     return self._get_reward_using_greedy(route, vehicle_states)

        # update_count = 0
        # reject_count = 0
        # constraint_failed_count = 0
        # before_route_cost , *__ = self._cost.calculate_cost(route)
        #
        # diff = 100
        # min_obj = math.inf
        # min_objs = {i: math.inf for i in range(int(self._constants.k_max / 100))}

        e_reward = self._get_objective_reward(route, vehicle_states)
        min_reward = e_reward
        min_route = copy.deepcopy(route)

        for k in range(self._constants.K_MAX):
            T = (1 - k / self._constants.K_MAX) * self._sa_scale

            flag, pos1, pos2 = self._do_mutation(route)
            if flag == False:
                # constraint_failed_count += 1
                continue

            e_ref_reward = self._get_objective_reward(route, vehicle_states)
            if self._P(e_reward, e_ref_reward, T) >= random.uniform(0, 1):
                e_reward = e_ref_reward

                if e_ref_reward < min_reward:
                    min_reward = e_ref_reward
                    min_route = copy.deepcopy(route)

                # update_count += 1
                # if min_obj > e_reward:
                #     min_obj = e_reward
                #     min_objs[int(k / diff)] = min_obj
            else:
                self._revert_in_infeasibility(route, pos1, pos2)
                # reject_count += 1
                # self._total_rejection_count += 1

        # after_route_cost , *__ = self._cost.calculate_cost(route)
        # total_improvement = round(before_route_cost - after_route_cost, 2)
        # percent_improvement = round(total_improvement / before_route_cost, 2)
        #
        # self._log_writer.add_sa_info(self._episode, self._vehicle_id, round(before_route_cost), round(after_route_cost),
        #                              total_improvement, percent_improvement, update_count, min_obj, min_objs[0],
        #                              min_objs[1], min_objs[2], min_objs[3], min_objs[4], reject_count,
        #                              self._total_rejection_count, constraint_failed_count, len(route), len(route) - self._vehicle_location - 1)

        reward, cost_remain, cost_penalty = self._get_final_update(min_route)
        return min_route, reward, cost_remain, cost_penalty

    def _get_final_update(self, route):
        current_state_cost, *__ = self._cost.calculate_cost(route)
        reward = current_state_cost - self._previous_route_cost

        self._previous_route_cost = current_state_cost

        __, cost_remain, cost_penalty = self._cost.calculate_cost(route[self._vehicle_location + 1:])
        return reward, cost_remain, cost_penalty

    def _P(self, e, e_ref, T):
        # dif = -(e_ref - e)
        # if self._min > dif:
        #     self._min = dif
        # if self._max < dif:
        #     self._max = dif
        # print("dif:", dif, ", max:", self._max, ", min:", self._min)

        return 1 if e_ref < e else math.exp(-(e_ref-e)/T)

    # @profile
    def _get_objective_reward(self, route, vehicle_states):
        current_route_cost, *__ = self._cost.calculate_cost(route)
        reward = current_route_cost - self._previous_route_cost           # reward based on full route

        __, cost_remain, cost_penalty = self._cost.calculate_cost(route[self._vehicle_location + 1:])
        vehicle_states["cost_remain"][self._vehicle_id] = cost_remain
        vehicle_states["cost_penalty"][self._vehicle_id] = cost_penalty

        future_value = self._vfa.get_predict_value(vehicle_states)        # vehicle state based on remaining route
        total_reward = reward + self._constants.GAMMA * future_value

        return total_reward.numpy()

    def _revert_in_infeasibility(self, route, pos1, pos2):
        route[pos2], route[pos1] = route[pos1], route[pos2]
        self._cost.update_run_time(route)

    def _do_mutation(self, route):
        pos1 = 0
        pos2 = 0
        while pos1 == pos2:
            pos1 = random.randint(self._vehicle_location + 1, len(route) - 1)
            pos2 = random.randint(self._vehicle_location + 1, len(route) - 1)

        route[pos1], route[pos2] = route[pos2], route[pos1]
        if self._pickup_dropoff_contraints(route) == False or self._number_of_passenger_constraints(route) == False:
            self._revert_in_infeasibility(route, pos1, pos2)
            return False, pos1, pos2

        self._cost.update_run_time(route)
        return True, pos1, pos2

    def _pickup_dropoff_contraints(self, route):
        flags = np.zeros(self._constants.MAX_NUM_REQUESTS)

        for node in route:
            node_id = node["booking_id"]

            if flags[node_id] == 0:
                if node["node_type"] == NodeType.DROP_OFF:
                    # print("failed in pickup-dropoff constraints")
                    return False
                flags[node_id] = node["booking_id"]

        return True

    def _number_of_passenger_constraints(self, route):
        num_passenger = self._num_passenger
        for i in range(self._vehicle_location + 1, len(route)):
            if route[i]["node_type"] == NodeType.PICK_UP:
                num_passenger += 1
            else:
                num_passenger -= 1

            if num_passenger > self._constants.MAX_NUM_PASSENGER:
                # print("failed in number of passenger constraints")
                return False

        return True

    def _get_reward_for_not_having_enough_remaining_route(self, route, vehicle_states):
        cost, *__ = self._cost.calculate_cost(route)
        reward = cost - self._previous_route_cost
        self._previous_route_cost = cost

        __, cost_remain, cost_penalty = self._cost.calculate_cost(route[self._vehicle_location + 1:])
        return reward, cost_remain, cost_penalty

    def _get_reward_using_greedy(self, route, vehicle_states):
        route_len = len(route)
        min_reward = self._get_objective_reward(route, vehicle_states)

        for pos1 in range(self._vehicle_location + 1, route_len - 1):
            for pos2 in range(pos1 + 1, route_len):
                route[pos1], route[pos2] = route[pos2], route[pos1]

                if self._pickup_dropoff_contraints(route) == False or self._number_of_passenger_constraints(route) == False:
                    self._revert_in_infeasibility(route, pos1, pos2)
                    continue

                self._cost.update_run_time(route)
                reward = self._get_objective_reward(route, vehicle_states)

                if reward<min_reward:
                    min_reward = reward
                else:
                    self._revert_in_infeasibility(route, pos1, pos2)

        reward, cost_remain, cost_penalty = self._get_final_update(route)
        return reward, cost_remain, cost_penalty

    def get_cost(self):
        return self._cost.calculate_cost(self._route)

    def get_valid_service_count(self):
        valid_nodes_count = 0
        valid_requests_count = 0
        arr = np.zeros(self._constants.MAX_NUM_REQUESTS)

        for index, node in enumerate(self._route):
            if node["exceed_time"] <= self._constants.REDUCE_LATE_DROPOFF_LIMIT:  # node["wait_time"] == 0
                valid_nodes_count += 1

                node_id = node["booking_id"]
                arr[node_id] += 1
                if arr[node_id] == 2:
                    valid_requests_count += 1

        # print("vehicle_id:", self._vehicle_id, ", total_valid_requests:", valid_requests_count, ", valid_nodes:", valid_nodes_count)
        return valid_requests_count

    def get_valid_service_count_filtered(self):
        while True:
            valid_nodes_count = 0
            valid_requests_count = 0
            arr = np.zeros(self._constants.MAX_NUM_REQUESTS)

            for index, node in enumerate(self._route):
                if node["exceed_time"] <= self._constants.REDUCE_LATE_DROPOFF_LIMIT:       # node["wait_time"] == 0
                    valid_nodes_count += 1

                    node_id = node["booking_id"]
                    arr[node_id] += 1
                    if arr[node_id] == 2:
                        valid_requests_count += 1
                else:
                    # print("remove index:", index, ", node_type:", node["node_type"])
                    booking_id = node["booking_id"]
                    if node["node_type"] == NodeType.PICK_UP:
                        for i in range(index+1, len(self._route)):
                            if self._route[i]["booking_id"] == booking_id:
                                self._route.pop(i)
                                break
                        self._route.pop(index)
                    elif node["node_type"] == NodeType.DROP_OFF:
                        self._route.pop(index)
                        for i in range(len(self._route)):
                            if self._route[i]["booking_id"] == booking_id:
                                self._route.pop(i)
                                break

                    self._cost.update_run_time(self._route)
                    break

                if len(self._route) == index + 1:
                    # print("vehicle_id:", self._vehicle_id, ", total_valid_requests:", valid_requests_count, ", valid_nodes:", valid_nodes_count)
                    return valid_requests_count

    def save_routes(self, chain_id):
        # print("vehicle:", self._vehicle_id)
        for i, node in enumerate(self._route):
            # print(i, ": ", node)

            if i==0 and self._vehicle_id == 0:
                self._log_writer.add_route_plan(chain_id, node, True)
            else:
                self._log_writer.add_route_plan(chain_id, node, False)
