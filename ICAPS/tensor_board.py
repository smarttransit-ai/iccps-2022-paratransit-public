import os
import csv
import datetime
import tensorflow as tf
from constants import Constants


class Tensorboard:                 # $ tensorboard --logdir ./logs  # $ python /Users/smsalahuddinkadir/.local/lib/python3.8/site-packages/tensorboard/main.py --logdir logs
    def __init__(self, base_path):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self._counter_loss = 0
        log_dir = os.path.join("./", 'logs', base_path, current_time, "loss")
        self._fw_loss = tf.summary.create_file_writer(log_dir)

        self._counter_cost = 0
        log_dir = os.path.join("./", 'logs', base_path, current_time, "total_cost")
        self._fw_total_cost = tf.summary.create_file_writer(log_dir)

    def vfa_loss(self, loss):
        with self._fw_loss.as_default():
            tf.summary.scalar('vfa_loss', loss, step=self._counter_loss)
        self._counter_loss += 1

    def total_cost(self, cost):
        with self._fw_total_cost.as_default():
            tf.summary.scalar('total_cost', cost, step=self._counter_cost)
        self._counter_cost += 1


class LogWriter:
    def __init__(self, base_path):
        self._constants = Constants()
        self._model_version = self._constants.SAVE_VERSION
        self._dir_name = os.path.join("./", base_path, "test_result")

        self._reward_list = os.path.join(self._dir_name, "reward_list")
        self._sa_info = os.path.join(self._dir_name, "sa_info")
        self._route_plan = os.path.join(self._dir_name, "route_plan")

        self._create_dir()
        self._initialize()

    def _create_dir(self):
        try:
            os.makedirs(self._dir_name)
        except OSError as error:
            print(error)

        try:
            os.makedirs(self._sa_info)
        except OSError as error:
            print(error)

        try:
            os.makedirs(self._route_plan)
        except OSError as error:
            print(error)


    def _initialize(self):
        with open(f'{self._reward_list}_v-{self._model_version}.csv', 'w') as fd_reward_info:
            writer = csv.writer(fd_reward_info)
            writer.writerow(["model_version", "episode", "chain_id", "num_requests", "total_cost", "load_loss",
                             "total_time", "time_per_request", "total_valid_service_count_filtered_naive",
                             "valid_service_ratio_filtered_naive", "total_valid_service_count_filtered",
                             "valid_service_ratio_filtered"])

        with open(f'{self._sa_info}_v-{self._model_version}.csv', 'w') as fd_sa_info:
            writer = csv.writer(fd_sa_info)
            writer.writerow(["episode", "vehicle_id", "before_route_cost", "after_route_cost", "total_improvement",
                             "percent_improvement", "update_count", "min_obj", "100", "200", "300", "400", "500",
                             "reject_count", "total_rejection_count", "constraints_failed_count", "route_len", "rem_route_len"])

    def add_reward_info(self, episode, chain_id, num_req, total_cost, load_loss, computation_time,
                        total_valid_service_count_filtered_naive, total_valid_service_count_filtered):
        with open(f'{self._reward_list}_v-{self._model_version}.csv', 'a') as fd_reward_info:
            writer = csv.writer(fd_reward_info)
            writer.writerow([str(self._model_version), str(episode), str(chain_id), str(num_req), str(total_cost),
                             str(load_loss), str(computation_time), str(round(computation_time/num_req, 2)),
                             str(total_valid_service_count_filtered_naive), str(round(total_valid_service_count_filtered_naive/num_req, 2)),
                             str(total_valid_service_count_filtered), str(round(total_valid_service_count_filtered/num_req, 2))])

    def add_sa_info(self, episode, vehicle_id, before_route_cost, after_route_cost, total_improvement, percent_improvement,
                    update_count, min_obj, mo_100, mo_200, mo_300, mo_400, mo_500,
                    reject_count, total_rejection_count, constraint_failed_count, route_len, rem_route_len):
        with open(f'{self._sa_info}_v-{self._model_version}.csv', 'a') as fd_sa_info:
            writer = csv.writer(fd_sa_info)
            writer.writerow([str(episode), str(vehicle_id), str(before_route_cost), str(after_route_cost),
                             str(total_improvement), str(percent_improvement), str(update_count), str(min_obj),
                             str(mo_100), str(mo_200), str(mo_300), str(mo_400), str(mo_500), str(reject_count),
                             str(total_rejection_count), str(constraint_failed_count), str(route_len), str(rem_route_len)])

    def initialize_route_plan(self, chain_id):
        with open(f'{self._route_plan}_chain_id-{chain_id}.csv', 'w') as fd_route_plan:
            writer = csv.writer(fd_route_plan)
            writer.writerow(['chain_id', 'booking_id', 'node_type', 'vehicle_id', 'location', 'schedule_time',
                             'request_arrival_time', 'early_pickup_window', 'late_dropoff_window', 'num_passengers',
                             'assign_time', 'wait_time', 'exceed_time'])


    def add_route_plan(self, chain_id, node, initialize_flag):
        if initialize_flag:
            self.initialize_route_plan(chain_id)

        with open(f'{self._route_plan}_chain_id-{chain_id}.csv', 'a') as fd_route_plan:
            writer = csv.writer(fd_route_plan)
            writer.writerow([str(chain_id), node["booking_id"], node["node_type"], node["vehicle_id"],
                             node["location"], node["schedule_time"],
                             node["request_arrival_time"], node["early_pickup_window"],
                             node["late_dropoff_window"], node["num_passengers"],
                             node["assign_time"], node["wait_time"], node["exceed_time"]])