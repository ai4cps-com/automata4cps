"""
    The module provides base classes to represent the dynamics of cyber-physical systems (CPS): CPS and
    CPSComponent.

    Author: Nemanja Hranisavljevic, hranisan@hsu-hh.de
"""

from collections import OrderedDict


class DecisionMaker:
    def __init__(self, shared_alternatives_set):
        self.shared_alternatives_set = shared_alternatives_set
        if shared_alternatives_set:
            self._alternatives_set = OrderedDict()
        else:
            self._alternatives_set = None

    def get_decision_making_state(self):
        pass

    def calculate_decision(self, state):
        pass

    def make_random_decision(self):
        pass


