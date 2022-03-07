from constants import Constants, NodeType


class Node:
    def __init__(self):
        self._constants = Constants()

    def get_nodes(self, request):
        pickup_node = {
            "node_type": NodeType.PICK_UP,
            "booking_id": request["booking_id"],
            "location": request["pickup_location"],
            "schedule_time": request["pickup_schedule_time"],
            "request_arrival_time": request["request_arrival_time"],
            "early_pickup_window": request["pickup_schedule_time"] - self._constants.EARLY_PICKUP_LIMIT,
            "late_dropoff_window": request["dropoff_schedule_time"] + self._constants.LATE_DROPOFF_LIMIT,
            "vehicle_id": int(request["run"]),
            "num_passengers": 1,

            "assign_time": -1,
            "wait_time": 0,
            "exceed_time": 0,
        }

        dropoff_node = {
            "node_type": NodeType.DROP_OFF,
            "booking_id": request["booking_id"],
            "location": request["dropoff_locaton"],
            "schedule_time": request["dropoff_schedule_time"],
            "request_arrival_time": request["request_arrival_time"],
            "early_pickup_window": request["pickup_schedule_time"] - self._constants.EARLY_PICKUP_LIMIT,
            "late_dropoff_window": request["dropoff_schedule_time"] + self._constants.LATE_DROPOFF_LIMIT,
            "vehicle_id": int(request["run"]),
            "num_passengers": 1,

            "assign_time": -1,
            "wait_time": 0,
            "exceed_time": 0,
        }

        return pickup_node, dropoff_node
