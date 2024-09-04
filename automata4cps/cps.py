"""
    Author: Nemanja Hranisavljevic, hranisan@hsu-hh.de
            Tom Westermann, tom.westermann@hsu-hh.de
"""

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
import warnings
from scipy.integrate import solve_ivp
import simpy
from mlflow.pyfunc import PythonModel


class CPS:
    """
    CPS class represents a cyber-physical system. It is used to model the hierarchy in the system as each CPS can
    contain a mixture of other CPS and CPS components. The leaves of the hierarchy are only CPS component objects
    which define their dynamics and communication with other components.

    Attributes
    ----------
    id : str
        Identifier of the object
    com : OrderedDict
        A collection of components
    parent_system : CPS
        Parent CPS that this (sub) system belongs to
    """
    def __init__(self, id, components):
        """
        Parameters
        ----------
        id : str
            ID of the system
        components : iterable
            Child components of the system
        """
        self.id = id
        self.com = OrderedDict()
        self.parent_system = None
        self.decision_logic = None
        for s in components:
            s.parent_system = self
            self.com[s.id] = s

    def __getitem__(self, key):
        return self.com[key]

    def __setitem__(self, key, value):
        self.com[key] = value

    @property
    def _q(self):
        return {k: x.x for k, x in self.com.items()}

    @property
    def state(self):
        return self._q

    @property
    def overall_system(self):
        s = self
        while s.parent_system:
            s = s.parent_system
        return s

    def stop_condition(self, t):
        """
        Simulation stop condition which can be overridden by the subclasses.
        """
        return False

    def get_choice_state(self, comp):
        return self.state

    def get_components(self, exclude=None):
        """Returns a list of all direct child components except those with ids possibly provided in exclude.

        Parameters
        ----------
        exclude : iterable
            A list of component ids to exclude in the returned list (default is None).
        """
        if exclude is not None and type(exclude) is str:
            exclude = [exclude]
        return list(c for c in self.com.values() if exclude is None or c.id not in exclude)

    def set_child_component(self, id, com):
        """Set component with the id.

        Parameters
        ----------
        id : str
            ID of the component to add.
        com : (CPS, CPSComponent)
            Component of subsystem to add.
        """
        self.com[id] = com
        com.parent_system = self

    def get_all_components(self, exclude=None):
        """Returns a list of all child components (not only direct) except those with ids possibly provided in exclude.

        Parameters
        ----------
        exclude : iterable
            A list of component ids to exclude in the returned list.
        """
        comp = []
        for c in self.com.values():
            if issubclass(type(c), CPS):
                comp += c.get_all_components(exclude=exclude)
            elif exclude is None or c.id not in exclude:
                comp.append(c)
        return comp

    def get_component(self, name):
        for k, c in self.com.items():
            if type(c) is CPS:
                cc = c.get_component(name)
                if cc is not None:
                    return cc
            else:
                if k == name:
                    return c
        return None

    def get_component_by_full_id(self, full_id):
        ids = full_id.split('.')
        c = self
        for ii in ids:
            c = c[ii]
        return c

    def get_execution_data(self, flat=False):
        exe_data = OrderedDict()
        if flat:
            for c in self.get_all_components():
                exe_data[c.full_id] = c.get_execution_data()
        else:
            for k, c in self.com.items():
                exe_data[c.id] = c.get_execution_data()
        return exe_data


class CPSComponent(PythonModel):
    """
    General hybrid system class based on scipy and simpy.
    Based on
    """

    def __init__(self, id, initial_q=(), initial_xt: list = (), initial_xk: list = (), dt=-1., p=None,
                 blocked_states=None, cont_state_names=None, discr_state_names=None, discr_output_names=None,
                 cont_output_names=None, use_observed_timings=False):
        self.parent_system = None
        self.decision_logic = None
        self.id = id
        self._q = initial_q
        self._xt = initial_xt
        self._xk = initial_xk
        self._timing = None
        self._timing_event = None
        self._y = None
        self.dt = dt
        self._e = None
        self._u = None
        self._d = ()
        if p is None:
            p = {}
        self._p = p
        self._past_t = []
        self._past_p = []
        self._pending_events = []
        self._block_event = None
        self.choices_set = OrderedDict()
        self.use_observed_timings = use_observed_timings
        self.observed_event_data = None
        self._discrete_state_data = []
        self._discrete_output_data = []
        self._continuous_state_data = []
        self._continuous_output_data = []
        self.cont_state_names = cont_state_names
        self.cont_output_names = cont_output_names
        self.discrete_state_names = discr_state_names
        self.discrete_output_names = discr_output_names
        self._blocked_states = blocked_states

    @property
    def full_id(self):
        full_id = self.id
        s = self.parent_system
        while s.parent_system:
            full_id = s.id + "." + full_id
            s = s.parent_system
        return full_id

    @property
    def overall_system(self):
        s = self.parent_system
        while s.parent_system:
            s = s.parent_system
        return s

    def __str__(self):
        return '{}: {}'.format(self.id, self._q)

    def predict(self, context, model_input):
        pass

    def get_sim_state(self):
        return self._q, self._e, self._p, self._y, self._block_event

    def set_sim_state(self, q, e, p, y, block_event):
        self._q = q
        self._e = e
        self._p = p
        self._y = y
        self._block_event = block_event

    def finish_condition(self):
        pass

    def start(self, t, q=None):
        self._p = self.context(self._q, None, None, 0)
        self._q = self.UNKNOWN_STATE if q is None else q
        self._u = self.input(self._q, 0)
        self._past_t = [t]
        self._past_p = [self._p]
        self.output(self._q, self._e, 0, self._xt, self._xk, self._u, self._p)
        self._continuous_state_data = []
        self._continuous_output_data = []
        self._discrete_state_data = []
        self._discrete_state_data.append(dict(Timestamp=t, State=self._q, Event=self._e, **self._p))
        self._discrete_output_data = []

    def get_execution_data(self):
        data = pd.DataFrame(self._discrete_state_data) #, columns=['Timestamp', 'Finish', 'State', 'Event'] + list(self._p.keys()))
        data['Finish'] = data['Timestamp'].shift(-1)
        data['Duration'] = pd.to_timedelta(data['Finish'] - data['Timestamp']).dt.total_seconds()
        return data

    def discrete_event_dynamics(self, previous_q, e, xt, xk, p) -> tuple:
        """
        The function updates system state given the event e.
        :param event:
        :return:
        """
        pass

    def react(self, q, e, clock, xt, xk, u, p):
        """
        You can set self._d and self.x, but you must not
        """
        pass

    def update_input(self, u): # Set i if you want from outside
        self._u = u

    def output(self, q, e, clock, xt, xk, u, p) -> tuple:
        pass

    def input(self, q, clock) -> tuple:
        pass

    def context(self, q, past_t, past_p, t) -> tuple:
        return dict()

    def time_continuous_dynamics(self, t, xt, xk, q, u, p, y):
        pass

    def time_discrete_dynamics(self, t, xt, xk, q, u, p, y):
        pass

    def on_entry(self, q, context):
        pass

    def __step_continuous(self, clock):
        """
    Simulates one time step of continuous behavior from t to t+dt. Underlying function is solve_ivp with method is 'RK23'.
        :param x0: Initial state vector.
        :param t: Time at start of the step simulation.
        :param u: Arguments passed.....
        :return: Time t+dt, value of state at t+dt
        """

        s = solve_ivp(self.time_continuous_dynamics, t_span=(clock, clock + self.dt), y0=self._xt, method='RK23',
                      args=(self._xk, self._q, self._u, self._p, self._y))
        return s.y[:, -1]

    def invariants(self, q, clock, xc, xd, y):
        pass

    def timed_transition(self, q, xc, xd, y, use_observed_timings=False):
        return None, None

    def wait(self, q):
        pass

    def apply_event(self, e):
        if not self._block_event:
            raise Exception()
        if self._block_event.triggered:
            pass
            # if self._e != e:
            #     raise Exception('Should not happen')
        else:
            self._e = e
            self._block_event.succeed()

    def read_event(self, t, e, clear_p=False, keep_p=None, **kwargs):
        if keep_p:
            for k in keep_p:
                kwargs[k] = self._p[k]
        if clear_p or keep_p:
            self._p = kwargs
        else:
            self._p.update(kwargs)
        self._p.pop('Error Message', None)
        self._t = t
        self._e = e

        new_q, self._xt, self._xk = self.discrete_event_dynamics(self._q, self._e, self._xt, self._xk, self._p)

        if new_q == self.UNKNOWN_STATE:
            text1 = '{} at {}:'.format(self.id, t)
            text2 = '{}->{}->{}'.format(self._q, self._e, new_q)
            self._p['Error Message'] = text2
            # warnings.warn(text1 + text2)
        else:
            dl = self.decision_logic
            # d = self.check_decisions(self.x, self.overall_system.state)
            if dl and not self.overall_system.is_unknown(self):
                try:
                    state_key = json.dumps(self.overall_system.get_choice_state(self), sort_keys=True)
                except:
                    print('Problem getting choice state')
                old_e = self.choices_set.get(state_key, None)
                if old_e is not None:
                    if self._e not in old_e:
                        # warnings.warn(
                        #     '{}: Different decision for same state {}: {} vs {}'.format(t, state_key, old_e, self._e))
                        self.choices_set[state_key][self._e] = [str(t)]
                    else:
                        self.choices_set[state_key][self._e].append(str(t))
                else:
                    self.choices_set[state_key] = {self._e: [str(t)]}
                # if self._e not in (x[0] for x in d):
                #     warnings.warn('{}: Not allowed decision in state {}: {}'.format(t, state_key, self._e))

        self._q = new_q
        self._past_t.append(t)
        self._past_p.append(self._p)
        if len(self._discrete_state_data):
            self._discrete_state_data[-1]["Finish"] = t
        self._discrete_state_data.append(dict(Timestamp=t, Event=self._e, State=self._q, **self._p))
        self._discrete_output_data.append([t, *self._d, self._e])

    def finish(self, t, **kwargs):
        if self._p is None:
            self._p = kwargs
        else:
            self._p.update(kwargs)
        self._t = t
        self._e = None
        self._q = None
        self._past_p.append(self._p)
        self._discrete_state_data[-1]['Finish'] = t


def data_list_to_dataframe(element, data, signal_names, prefix=None, last_var=None):
    data = pd.DataFrame(data)
    if data.shape[1] == 0:
        return data
    if signal_names is None:
        signal_names = [prefix + str(i + 1) for i in range(len(data.columns) - 1)]
        if last_var is not None and len(signal_names) > 0:
            signal_names[-1] = last_var
    else:
        if last_var is not None:
            signal_names += [last_var + str(i + 1) for i in range(len(data.columns) - len(signal_names) - 1)]
    signal_names.insert(0, 'Time')

    if len(data.columns) != len(signal_names):
        raise Exception('Wrong number of column names: length of {} != {}'.format(signal_names, len(data.columns)))

    data.columns = signal_names
    data.set_index(data.columns[0], inplace=True)
    if element is not None:
        data.columns = element + '.' + data.columns
    return data


def group_components(comp, *states):
    if type(comp) is str:
        return
    res = ([k for k in comp if comp[k] == state]
           for state in states)
    if len(states) == 1:
        res = next(res)
        return res
    else:
        return tuple(res)


def flatten_dict_data(stateflow, reduce_keys_if_possible=True):
    d = pd.json_normalize(stateflow).to_dict('records')[0]
    if reduce_keys_if_possible:
        for k in list(d.keys()):
            k_new = k.split('.')[-1]
            d[k_new] = d.pop(k)
    return d



if __name__ == '__main__':
    print('__main__ sim')
