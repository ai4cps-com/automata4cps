"""
    The module provides base classes to represent the dynamics of cyber-physical systems (CPS):
    - CPS, and,
    - CPSComponent.

    Author: Nemanja Hranisavljevic, hranisan@hsu-hh.de
"""

import json
from traceback import print_exc
import numpy as np
from collections import OrderedDict
import pandas as pd
import warnings
from scipy.integrate import solve_ivp
import simpy
from mlflow.pyfunc import PythonModel
from automata4cps import tools, sim


class CPS:
    """
    CPS class represents a cyber-physical system. It is used to model the hierarchy in the system as each CPS can
    contain a mixture of other CPS and CPS components. The leaves of the hierarchy are only CPS component objects
    which define their dynamics and communication with other components.

    Attributes:
        id (str): Identifier of the object.
        com (OrderedDict): Property. A collection of components.
        parent_system (CPS): Parent CPS that this (sub)system belongs to.
    """

    def __init__(self, sys_id, components):
        """
        Initializes CPS with the given attributes.

        Args:
            sys_id (str): ID of the system.
            components (iterable): Child components of the system.
        """
        super().__init__()
        self._env = simpy.Environment()  # It is a simpy environment used internally for simulation
        self.id = sys_id
        self._parent_system = None  # It is always referenced two-way from children to parent and vice versa
        self._com = OrderedDict()  # "Private" in order to avoid setting value without reference in other way
        for s in components:
            s._parent_system = self
            self._com[s.id] = s

    @property
    def parent_system(self):
        """
        Gets the parent system by accessing _parent_system private attribute.

        Returns:
            (CPS): The parent system.
        """
        return self._parent_system

    @parent_system.setter
    def parent_system(self, parent_system):
        """
        Sets the _parent_system of this component and adds the child of the parent system.

        Args:
            parent_system (CPS): The new parent system.

        Raises:
            AttributeError: If we cannot set child of the parent system or if the component with the same id is already
            a child of the parent system.
        """
        if id in parent_system._com:
            raise AttributeError("Component with that id is already a child of the parent system.")
        if self._parent_system is not None and self.id in self._parent_system:
            self._parent_system.pop(self.id)
        try:
            self._parent_system = parent_system
            parent_system._com[id] = self
        except AttributeError as ex:
            raise ex

    def __getitem__(self, key):
        """
        Gets the component with the given key in the collection.

        Args:
            key (string): The key where the value will be stored. Must be a hashable type (e.g., int, str).
        Return:
            (CPS, CPSComponent): Returned component or subsystem.

        """
        return self._com[key]

    def __setitem__(self, key, value):
        self._com[key] = value
        self._parent_system[value] = self

    @property
    def state(self):
        return {k: x.state for k, x in self._com.items()}

    @state.setter
    def state(self, value): # If some component is not specified in value dict it will stay the same
        for k, v in value.items():
            self[k].state = v

    @property
    def overall_system(self):
        s = self
        while s.parent_system:
            s = s.parent_system
        return s

    def get_components(self, exclude=None):
        """Returns a list of all direct child components except those with ids possibly provided in exclude.

        Parameters
        ----------
        exclude : iterable
            A list of component ids to exclude in the returned list (default is None).
        """
        if exclude is not None and type(exclude) is str:
            exclude = [exclude]
        return list(c for c in self._com.values() if exclude is None or c.id not in exclude)

    def set_child_component(self, id, com):
        """Set component with the id.

        Parameters
        ----------
        id : str
            ID of the component to add.
        com : (CPS, CPSComponent)
            Component of subsystem to add.
        """
        self._com[id] = com
        com.parent_system = self

    def get_all_components(self, exclude=None):
        """Returns a list of all child components (not only direct) except those with ids possibly provided in exclude.

        Parameters
        ----------
        exclude : iterable
            A list of component ids to exclude in the returned list.
        """
        comp = []
        for c in self._com.values():
            if issubclass(type(c), CPS):
                comp += c.get_all_components(exclude=exclude)
            elif exclude is None or c.id not in exclude:
                comp.append(c)
        return comp

    def get_component(self, name):
        for k, c in self._com.items():
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

    def reinitialize(self, t=0, state=None):
        """
        The function re-initializes the CPS components of this CPS with the given state values.
        :param t: Current time to set to the components.
        :param state: State of the CPS (it's components) given as a dictionary of dictionaries ... of tuples (according
        to the CPS hierarchy). The tuplus are created as values concatenated values of discrete-event state variables,
        time-continuous state variables and time-discrete state variables.

        :return:
        """
        if state is None:
            state = {}
        for k, v in self._com.items():
            v.reinitialize(t, state.get(k, None))

    def simulate(self, finish_time, verbose=False, save_choices=False):
        """Simulates behaviour of the system until the finish_time is reached.

        Parameters
        ----------
        finish_time : Time when simulation finishes.
        verbose : Should the execution log be printed in detail (default is False).
        save_choices : Should the choices for each component be saved to json files after
        the simulation is finished (default is False).
        """

        env = simpy.Environment()
        finish_time = float(finish_time)
        print('Simulation started: ')

        for s in self.get_all_components():
            if not s._q:
                s._q = np.random.choice(list(s.q0.keys()))
            env.process(s.simulation_process_simpy(env, finish_time, verbose))

        env.run(until=finish_time)

        stateflow_data = {}
        discr_output_data = {}
        cont_state_data = {}
        cont_output_data = {}

        decisions = {}
        for s in self.get_all_components():
            if save_choices and len(s.choices_set) > 0:
                out_file = open(s.id + str(env.now) + ".json", "w")
                json.dump(s.choices_set, out_file, indent=6)
            try:
                s.finish(env.now)
                # if s.parent_system.shared_alternatives_set:
                #     if s.parent_system.id not in decisions:
                #         decisions[s.parent_system.id] = s.choices_set
                # else:
                #     decisions[s.id] = s.choices_set
                stateflow_data[s.id] = pd.DataFrame(s._discrete_state_data)
                discr_output_data[s.id] = tools.data_list_to_dataframe(None, s._discrete_output_data,
                                                                       s.discrete_output_names,
                                                                       'd', 'e')
                cont_state_data[s.id] = tools.data_list_to_dataframe(s.id, s._continuous_state_data, s.cont_state_names,
                                                                     'x')
                cont_output_data[s.id] = tools.data_list_to_dataframe(s.id, s._continuous_output_data,
                                                                      s.cont_output_names, 'y')
            except Exception as ex:
                print_exc()
                warnings.warn('Simulation failed.')
        print('Simulation finished.')
        return stateflow_data, discr_output_data, cont_state_data, cont_output_data, env.now, decisions

<<<<<<< Updated upstream
class CPSComponent(PythonModel):
=======

class CPSComponent(PythonModel, sim.Simulator):
>>>>>>> Stashed changes
    """
    General hybrid system class based on scipy and simpy.
    """

    def __init__(self, id, t=0, initial_q=(), initial_xt: list = (), initial_xk: list = (), dt=-1., p=None,
                 cont_state_names=None, discr_state_names=None, discr_output_names=None,
                 cont_output_names=None, unknown_state="Unknown"):
        self.parent_system = None
        self.decision_logic = None
        self.id = id
        self.dt = dt

        # State variables
        self._t = t
        self._q = initial_q
        self._xt = initial_xt
        self._xk = initial_xk

        # Other variables
        self._y = None
        self._u = None
        self._d = ()
        self._p = {} if p is None else p

        # simpy events
        self._pending_events = []
        self._block_event = None

        # List to append the past values
        self._past_p = []
        self._past_t = []
        self._discrete_state_data = []
        self._discrete_output_data = []
        self._continuous_state_data = []
        self._continuous_output_data = []

        # Variable names
        self.cont_state_names = cont_state_names
        self.cont_output_names = cont_output_names
        self.discrete_state_names = discr_state_names
        self.discrete_output_names = discr_output_names

        self.UNKNOWN_STATE = unknown_state

        super(sim.Simulator).__init__()

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

    @property
    def state(self):
        """
        System state is a vector of all state variables.
        :return:
        """
        return self._q, self._xt, self._xk

    @state.setter
    def state(self, value):
        self._q = value[0]
        self._xt = value[1]
        self._xk = value[2]

    def is_decision(self, state, overall_state):
        return False

    def __str__(self):
        return '{}: {}'.format(self.id, self._q)

    def simulation_process_simpy(self, env, max_time, verbose=False):
        """
            The function simulates single concurrent thread of one component (CPSComponent).
            Parameters
            ----------
            env : simpy.Environment
                It is simpy environment used to synchronize multiple parallel threads during simulation.
            max_time : float
                It is the maximum time that the simulation will perform.
            verbose : bool
                Should the function print execution logs (default is False).
        """

        self._env = env
        self._u = self.input(self._q, 0)
        if self._p is None:
            self._p = self.context(self._q, None, None, 0)
        self._past_t = [0]
        self._past_p = [self._p]
        self._y = ()

        self._continuous_state_data = [[env.now, *self._xt, *self._xk]]
        self._continuous_output_data = [[env.now, *self._y]]
        self._discrete_state_data = []
        self._discrete_state_data.append(dict(Timestamp=env.now, State=self._q, Finish=None, **self._p))
        self._discrete_output_data = []

        while True:
            # Stop the simulation if the max time is reached
            if env.now > max_time:
                if verbose:
                    print(f'Stop because of maximum time {max_time} is reached.')
                break

            # Stop if the stopping condition of the overall system is met
            elif self.overall_system.stop_condition(env.now):
                if verbose:
                    print('Stop because stop condition of the overall system is met.')
                break

            # DETERMINE THE NEXT EVENT
            # 1. CHECK IF TIMED EVENT
            try:
                event_delay, new_state_value = self.timed_event(t, self.state)
            except Exception as ex:
                print(f"{self.id}: Exception during self.timed_transition in state {self._q}")
                raise ex

            try:
                if len(self._xt)>0: # there is time-continuous state variable
                    self.__step_continuous(t, *self.state)



            if event_delay is not None:
                yield env.timeout(event_delay)
            self.state = new_state_value
            continue

            # 2. IF NOT TIMED
            try:


            # 2. IF NO TIMED EVENT -> BLOCK EVENT
            if time is None:
                if verbose:
                    print('{}: Waiting in: {}'.format(self.id, self._q))
                if self._block_event is None:
                    self._block_event = env.event()

                    yield self._block_event
                else:
                    raise Exception("{}: State: {}. Exception during block event.".format(self.id, self._q))
                self._block_event = None
         # The one who unblocks must set the ._e

            # EXECUTE THE NEXT EVENT
            if verbose:
                print('--------------------')
            old_q = self._q
            try:
                self._q, self._xt, self._xk = \
                    self.discrete_event_dynamics(self._q, self._xt, self._xk, self._p)

                self.on_entry(self._q, self._p)
                # self._d, self._y = self.o(self.x, 0, self._xt, self._xk, self._u, self._p)
            except Exception as ex:
                print(f"{self.id}: Exception during discrete event dynamics '{old_q}'->")
                raise ex

            # self._p = self.context(self._past_t, self._past_p, env.now)
            self._past_t.append(env.now)
            self._past_p.append(self._p)
            self._discrete_state_data[-1]['Finish'] = env.now
            self._discrete_state_data.append(
                dict(Timestamp=env.now, State=self._q, Finish=None, **self._p))
            self._discrete_output_data.append([env.now, *self._d])

            if verbose:
                print('{}: Time: {} Event: {} State: {}'.format(self.id, env.now, self._q))
            continue

            if self._xt is not None:
                self._continuous_state_data.append([env.now, *self._xt, *self._xk])
            if self._y is not None:
                self._continuous_output_data.append([env.now, *self._y])
            if self._xt is not None:
                self._xt = self.__step_continuous(env.now - clock_start)
            self._d, self._y = self.output(self._q, 0, self._xt, self._xk, self._u, self._p)

        print('--------------------')
        print('{}: Simulation finished'.format(self.id))

    def predict(self, context, model_input):
        pass

    def get_sim_state(self):
        return self._q, self._p, self._y, self._block_event

    def set_sim_state(self, q, e, p, y, block_event):
        self._q = q
        self._p = p
        self._y = y
        self._block_event = block_event

    def finish_condition(self):
        pass

    def reinitialize(self, t, state=None):
        self._p = self.context(self._q, None, None, 0)
        if state is not None:
            self.state = state

        self._u = self.input(self._q, 0)
        self._past_t = [t]
        self._past_p = [self._p]
        # self.output(self._q, 0, self._xt, self._xk, self._u, self._p)
        self._continuous_state_data = []
        self._continuous_output_data = []
        self._discrete_state_data = []
        self._discrete_state_data.append(dict(Timestamp=t, State=self._q, **self._p))
        self._discrete_output_data = []

    def get_execution_data(self):
<<<<<<< Updated upstream
        data = pd.DataFrame(self._discrete_state_data) #, columns=['Timestamp', 'Finish', 'State', 'Event'] + list(self._p.keys()))
=======
        data = pd.DataFrame(self._discrete_state_data,
                            columns=['Timestamp', 'Finish', 'State', 'Event'] + list(self._p.keys()))
>>>>>>> Stashed changes
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

    def update_input(self, u):  # Set i if you want from outside
        self._u = u

    def output(self, q, e, clock, xt, xk, u, p) -> tuple:
        return (), ()

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

    def timed_event(self, t, q, xc, xd):
        """
        Calculates if and when the next time event will happen and the new state values.
        :param t: Current time.
        :param q: Discrete-event part of the state.
        :param xc: Time-continuous part of the state.
        :param xd: Time-discrete part of the state.
        :return: Time delay of the timed event, new state value.
        """
        return None, None

    def wait(self, q):
        pass

    def apply_sim_event(self, e):
        """
        The event e is applied in this component's simpy execution, this means that the process must wait for an event.
        :param e: Event to apply.
        :return:
        """
        if not self._block_event:
            # self._block_event = self._env.now
            self._block_event = self._env.now
        # if not self._block_event:
        #     raise Exception("Block event is not set, but event is applied.")
        elif self._block_event.triggered:
            # pass
            # if self._e != e:
            raise Exception('Should not happen')
        else:
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

        new_q, self._xt, self._xk = self.discrete_event_dynamics(self._q, self._xt, self._xk, self._p)

        if new_q == self.UNKNOWN_STATE:
            text1 = '{} at {}:'.format(self.id, t)
            text2 = '{}->{}'.format(self._q, new_q)
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

    def get_alternatives(self, state, system_state):
        return None

    def get_decision_state(self, state, overall_state):
        return state
