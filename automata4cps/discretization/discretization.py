"""
    The module provides base classes to represent the dynamics of cyber-physical systems (CPS): CPS and
    CPSComponent.

    Author: Nemanja Hranisavljevic, hranisan@hsu-hh.de
"""
import numpy as np
import pandas as pd


class TimeSeriesDiscretizer:
    """
        Abstract class that encapsulates methods used for the discretization of time series.
    """
    def train(self, data, *args):
        pass

    def discretize(self, data):
        pass


class EqualWidthDiscretizer (TimeSeriesDiscretizer):
    """
    A class that implements equal-width interval (EWI) discretization method. It calculates range of every variable
    (column) of the input data into a predefined number of equal-width intervals.
    """
    def __init__(self):
        self.intervals = None

    def train(self, data, number_of_intervals=10):
        """
        Estimate model parameters, thresholds that divide each variable into equal-width intervals.
        :param data: Data to calculate model parameters from.
        :param number_of_intervals: Number of equal-width intervals per variable.
        :return:
        """
        data = pd.DataFrame(data)
        min_d = data.min(axis=0)
        max_d = data.max(axis=0)

    def discretize(self, data):
        """
        Discretize data into equal width intervals.
        :param data: Data to discretize.
        :return: Discretized data.
        """
        data = np.asarray(data)
        return data


class EqualFrequencyDiscretizer (TimeSeriesDiscretizer):
    def __init__(self, intervals=None):
        self.intervals = intervals

    def train(self, data, number_of_intervals=10):
        data = np.asarray(data)
        min_d = data.min(axis=0)
        max_d = data.max(axis=0)

    def discretize(self, data):
        data = np.asarray(data)
        return data


class ThresholdDiscretizer (TimeSeriesDiscretizer):
    def discretize(self, states):
        pass


if __name__ == "__main__":
    from automata4cps import examples
    from automata4cps.discretization import catvae

    discrete_data, time_col, discrete_cols = examples.conveyor_system_sfowl(variable_type="discrete")
    cont_data, _, cont_cols = examples.conveyor_system_sfowl(variable_type="continuous")
    catvae.train_cat_vae(cont_data, )
