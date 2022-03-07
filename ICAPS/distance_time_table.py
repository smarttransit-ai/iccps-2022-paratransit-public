import pandas as pd
import numpy as np


class DistanceTimeTable:
    def __init__(self, num_input_files = 32):
        self._num_input_files = num_input_files

        self._find_unique_locations()
        self._create_hash_table_for_locations()
        self._create_travel_time_matrix()

        self._SECONDS_IN_MINUTES = 60

    def _find_unique_locations(self):
        self._unique_locations = set()

        for j in range(self._num_input_files):
            file_path = './dataset/CARTA/travel_times/otp/travel_times_' + str(j) + '.pickle'
            object = pd.read_pickle(file_path)
            print(file_path)

            for obs in object:
                self._unique_locations.add(str(obs[0]))
                self._unique_locations.add(str(obs[1]))

    def _create_hash_table_for_locations(self):
        self._num_unique_locs = len(self._unique_locations)
        self._hash_table = [""] * self._num_unique_locs
        # print("number of unique locations: ", self._num_unique_locs)

        for loc in self._unique_locations:
            val = hash(loc) % self._num_unique_locs
            if self._hash_table[val] == "":
                self._hash_table[val] = loc
            else:
                while self._hash_table[val] != "":
                    val += 1
                    val = val % self._num_unique_locs
                self._hash_table[val] = loc

    def _get_hash_index(self, location):
        if (location in self._unique_locations) == False:
            # print("does not exist:", location)
            return  -1

        val = hash(location) % self._num_unique_locs
        while self._hash_table[val] != location:
            val += 1
            val = val % self._num_unique_locs

        return val

    def _create_travel_time_matrix(self):
        self._travel_time_matrix = np.full((self._num_unique_locs, self._num_unique_locs), -1, dtype=float)

        for j in range(self._num_input_files):
            file_path = './dataset/CARTA/travel_times/otp/travel_times_' + str(j) + '.pickle'
            object = pd.read_pickle(file_path)
            print(file_path)

            for obs in object:
                row = self._get_hash_index(str(obs[0]))
                col = self._get_hash_index(str(obs[1]))
                self._travel_time_matrix[row][col] = int(object[obs] * 60)
                # print("row:", row, ", col:", col, ", val:", object[obs])

    def get_travel_time(self, loc1, loc2):
        row = self._get_hash_index(str(loc1))
        col = self._get_hash_index(str(loc2))
        return self._travel_time_matrix[row][col]      # -1 if unreachable path


if __name__ == "__main__":
    total_input_files = 32
    travel_time = DistanceTimeTable(total_input_files)

    row = str((35.065586, -85.309052))
    col = str((35.089539, -85.287305))
    print(travel_time.get_travel_time(row, col))


