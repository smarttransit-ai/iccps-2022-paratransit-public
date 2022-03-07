from constants import Constants, NodeType


class Cost:
    def __init__(self, distance_time_table):
        self._distance_time_table = distance_time_table
        self._constants = Constants()

    def update_node_time(self, from_node, to_node):
        travel_time = self._distance_time_table[from_node["location"], to_node["location"]]
        reach_time = from_node["assign_time"] + travel_time

        if reach_time <= to_node["early_pickup_window"]:
            to_node["assign_time"] = to_node["early_pickup_window"]
            to_node["wait_time"] = to_node["early_pickup_window"] - reach_time
            to_node["exceed_time"] = 0
        elif reach_time >= to_node["late_dropoff_window"]:
            to_node["assign_time"] = reach_time
            to_node["wait_time"] = 0
            to_node["exceed_time"] = reach_time - to_node["late_dropoff_window"]
        else:
            to_node["assign_time"] = reach_time
            to_node["wait_time"] = 0
            to_node["exceed_time"] = 0

    def update_run_time(self, vehicle):
        if vehicle is None:
            return

        for j, to_node in enumerate(vehicle):
            if j == 0:
                to_node["assign_time"] = to_node["schedule_time"]
                to_node["wait_time"] = 0
                to_node["exceed_time"] = 0
            else:
                self.update_node_time(from_node, to_node)
            from_node = to_node

    def calculate_cost(self, vehicle):
        if vehicle == None or len(vehicle) == 0:
            return 0, 0, 0

        depot_to_first_node = self._distance_time_table[self._constants.DEPOT_LOCATION, vehicle[0]["location"]]
        num_nodes = len(vehicle)
        last_node_to_depot = self._distance_time_table[vehicle[num_nodes - 1]["location"], self._constants.DEPOT_LOCATION]
        total_travel_time = vehicle[num_nodes - 1]["assign_time"] - vehicle[0]["assign_time"] + \
                            depot_to_first_node + last_node_to_depot

        extra_time = 0            # extra time is the penalty for the time window violations
        for node in vehicle:
            extra_time += node["exceed_time"]
        extra_time = extra_time * self._constants.SCALE_EXTRA_TIME_PENALTY_BY

        total_penalty = total_travel_time + extra_time
        remaining_travel_time = total_travel_time - depot_to_first_node  # considering route from the vehicle location; based on caller function

        remaining_travel_time /= self._constants.NUM_SECONDS_IN_DAY        # preprocessing state value
        extra_time /= self._constants.NUM_SECONDS_IN_DAY                   # preprocessing state value

        return total_penalty, remaining_travel_time, extra_time