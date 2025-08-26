
import numpy as np

class Meter(object):

    def __init__(self):
        self.measure_dicts = {}

    def add_attributes(self, new_attribute):
        self.measure_dicts[new_attribute] = []

    def clear_data(self):
        for key, item in self.measure_dicts.items():
            self.measure_dicts[key] = []

    def add_data(self, attribute, data):

        assert attribute in self.measure_dicts
        self.measure_dicts[attribute].append(data)

    def return_avg_dict(self):

        return_dict = {}
        for key, item in self.measure_dicts.items():
            return_dict[key] = f'{np.mean(self.measure_dicts[key]):.9f}'

        return return_dict