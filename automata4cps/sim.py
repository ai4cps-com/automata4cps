"""
    Simulation.

    Author:
    - Nemanja Hranisavljevic, hranisan@hsu-hh.de, nemanja@ai4cps.com
"""

from collections import OrderedDict
import copy


class Simulator():
    def simulate(self, finish_time):
        pass

    def reinitialize(self, time, state):
        pass

    def stop_condition(self, t):
        """
        Simulation stop condition which can be overridden by the subclasses.
        """
        return False

    # def simulate_alternatives(self, number_of_sim=100, finish_time=100):
        # res = []
        # state = self.state
        # for i in range(number_of_sim):
        #     self.reinitialize(0, copy.deepcopy(state))
        #     res.append(self.simulate(finish_time=finish_time))
        # self.state = copy.deepcopy(state)
        # return res