"""
    The module provides base classes to represent the dynamics of cyber-physical systems (CPS): CPS and
    CPSComponent.

    Author: Nemanja Hranisavljevic, hranisan@hsu-hh.de
"""

from collections import OrderedDict


class DecisionMaker:
    def __init__(self, model, shared_alternatives_set=False):
        self.model = model
        self.shared_alternatives_set = shared_alternatives_set
        if shared_alternatives_set:
            self._alternatives_set = OrderedDict()
        else:
            self._alternatives_set = None

    def optimize(self, initial_state, initial_time=0, number_of_runs=100):
        for i in range(number_of_runs):
            self.model.reinitialize(initial_time, initial_state)

            discrete_state_data, discrete_output_data, cont_state_data, cont_output_data, finish_time = (
                self.model.simulate(finish_time=100, verbose=True))

    def get_decision_making_state(self):
        pass

    def calculate_decision(self, state):
        pass

    def make_random_decision(self):
        pass


