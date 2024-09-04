"""
    The module provides a class Automaton which inherits CPSComponent and implements the dynamics of different kinds of
    automata.

    Authors:
    - Nemanja Hranisavljevic, hranisan@hsu-hh.de, nemanja@ai4cps.com
    - Tom Westermann, tom.westermann@hsu-hh.de, tom@ai4cps.com
"""

import numpy as np
from collections import OrderedDict
import pandas as pd
from scipy.integrate import solve_ivp
import networkx as nx
import warnings
from automata4cps.cps import CPSComponent


class Automaton (CPSComponent):
    """
    Automaton class is the main class for modeling various kinds of hybrid systems.
    """

    def __init__(self, states: list = None, events: list = None, transitions: list = None,
                 unknown_state: str = 'raise', id="", initial_q=(), super_states=(), decision_states=(), **kwargs):
        """
        Class initialization from lists of elements.
        :param states: Discrete states / modes of continuous behavior.
        :param events: The events that trigger state transitions.
        :param transitions: The transition information. If a collection of dicts then dict should contain "source",
        "dest" and "event". The other attributes will be added as data of that transition. Alternatively, a collection
        of tuples can be used of the form (source, event, dest, *).
        :param unknown_state: The name of unknown states during "play in", if "raise", an exception will be raised.
        """
        self._G = nx.MultiDiGraph()
        if initial_q:
            initial_q = [initial_q]
        self.q0 = OrderedDict.fromkeys(initial_q)
        self.Sigma = set()
        self.previous_node_positions = None
        self.UNKNOWN_STATE = unknown_state
        if super_states is not None:
            if type(super_states) is str:
                self.__super_states = [super_states]
            else:
                self.__super_states = list(super_states)

            # self._G.add_nodes_from(self.__super_states)

        if decision_states is not None:
            if type(decision_states) is str:
                self.decision_states = [decision_states]
            else:
                self.decision_states = list(decision_states)

        if states is not None:
            self._G.add_nodes_from(states)

        if events is not None:
            self.Sigma = set(events)

        if transitions is not None:
            for tr in transitions:
                if type(tr) is dict:
                    self._G.add_edge(tr.pop('source'), tr.pop('dest'), event=tr.pop('event'), **tr)
                else:
                    self._G.add_edge(tr[0], tr[2], event=tr[1])

        CPSComponent.__init__(self, id, **kwargs)

    @property
    def discrete_states(self):
        return self._G.nodes

    @property
    def transitions(self):
        return self._G.edges

    @property
    def discrete_state(self):
        return self._q

    @property
    def state(self):
        """
        Automata discrete state is uni-variate.
        :return:
        """
        return self._q[0], self._xt, self._xk

    @state.setter
    def state(self, state):
        if type(state) is not tuple:
            new_states = (state, (), ())
        elif len(state) < 3:
            new_states = [(), (), ()]
            for i, v in enumerate(state):
                new_states[i] = v
        else:
            new_states = state

        if new_states[0] not in self.discrete_states and new_states[0] not in self.__super_states:
            raise ValueError(f'State "{new_states[0]}" is not a valid state. Valid states are: {self.discrete_states}')

        self._q = (new_states[0],)
        self._xt = new_states[1]
        self._xk = new_states[2]

    @property
    def num_modes(self):
        """
        Returns the number of modes in the automaton.
        :return: number of states.
        """
        return self._G.number_of_nodes()

    @property
    def num_transitions(self):
        """
        Returns the number of transitions in the automaton.
        :return: number of transitions.
        """
        return self._G.number_of_edges()

    def remove_transition(self, source, dest):
        """
        Remove the transition(s) from source to dest.
        :param source:
        :param dest:
        :return:
        """
        self._G.remove_edge(source, dest)

    def is_decision(self, state, overall_state):
        return state in self.decision_states

    def get_alternatives(self, state, system_state):
        if self.is_decision(state, system_state):
            trans = [(x[3]['event'], dict()) for x in self.out_transitions(state)]
            return [(None, None)] + trans
        else:
            return None

    def remove_rare_transitions(self, min_p=0, min_num=0, keep_from_initial=False, keep_states=False, keep=None):
        self.learn_transition_probabilities()

        for source, dest, event, data in self.get_transitions():
            if keep_from_initial and source in self.q0:
                continue
            if (len(data['timing']) <= min_num or data['probability'] < min_p) and \
                    ((keep is None) or (keep is not None and source not in keep and dest not in keep)):
                self.remove_transition(source, dest)

        if not keep_states:
            for s in list(self.discrete_states):
                # if s in self.q0:
                #     continue
                if len(self.in_transitions(s)) == 0 and len(self.out_transitions(s)) == 0:
                    self.remove_state(s)

            # if self.DummyInitial:
            #     for s in list(self.InitialState):
            #         if len(self.out_transitions(s)) == 0:
            #             self.States.pop(s)
            #             self.InitialState.pop(s)

        # recalculate probabilities
        if min_p:
            self.learn_transition_probabilities()
        print('Remove rare transitions')

    def learn_transition_probabilities(self):
        for s in self.discrete_states:
            total_num = sum([len(data['timing']) for s, d, e, data in self.out_transitions(s)])
            for s, d, e, data in self.out_transitions(s):
                data['probability'] = len(data['timing']) / total_num

    def state_is_deterministic(self, q):
        events = set()
        for tr in self.out_transitions(q):
            if tr[2] in events:
                return False
            else:
                events.add(tr[2])
        return True

    def update_timing_boundaries(self, source, destination, event, newTiming):
        edge_data = self._G.get_edge_data(source, destination, event)
        try:
            if newTiming < edge_data['minTiming']:
                edge_data['minTiming'] = newTiming
            elif newTiming > edge_data['maxTiming']:
                edge_data['maxTiming'] = newTiming
        except KeyError:
            edge_data['minTiming'] = newTiming
            edge_data['maxTiming'] = newTiming

    def is_deterministic(self):
        for q in self.discrete_states:
            if not self.state_is_deterministic(q):
                print('State', q, 'not deterministic:')
                for tr in sorted(self.out_transitions(q), key=lambda x: x[2]):
                    print(tr[0], '->', tr[2], '->', tr[1])
                return False
        return True

    def add_single_transition(self, s, d, e, timing=None):
        edge_data = self._G.get_edge_data(s, d, e)
        if edge_data is None:
            if timing is None:
                self._G.add_edge(s, d, key=e)
            else:
                try:
                    timing = list(timing)
                    self._G.add_edge(s, d, key=e, event=e, timing=timing)
                except:
                    self._G.add_edge(s, d, key=e, event=e, timing=[timing])
        elif timing is not None:
            try:
                timing = list(timing)
                edge_data['timing'] += timing
            except:
                edge_data['timing'].append(timing)

    def add_state_data(self, s: str, d: object):
        """
    Add state data to a state s the automaton.
        :param s: state
        :param d: data to be added to s
        :return:
        """
        self.Q[s] = d

    def add_state(self, new_state, **kwargs):
        """
    Add state to the automaton.
        :param new_state: State to be added.
        """
        self._G.add_node(new_state, **kwargs)

    def add_states_from(self, new_state, **kwargs):
        """
    Add multiple states to the automaton.
        :param new_state: States to be added.
        """
        self._G.add_nodes_from(new_state, **kwargs)

    def add_event(self, new_event):
        """
    Add multiple events to the automaton.
        :param new_event: Events to be added.
        """
        self.Sigma.add(new_event)

    def add_transitions_from(self, list_of_tuples, **other):
        """
    Add multiple transition.
        :param list_of_tuples: List of transitions in the form (source_state, destination_state, event, ...<unused>...).
        """
        self._G.add_edges_from(list_of_tuples, **other)

    def add_transition(self, s, d, e, **other):
        """
    Add multiple transition.
        :param list_of_tuples: List of transitions in the form (source_state, destination_state, event, ...<unused>...).
        """
        self._G.add_edge(s, d, e, **other)

    def add_initial_state(self, states):
        """
    Add initial state(s) of the automaton.
        :param states: States to add.
        """
        if type(states) is str:
            states = (states,)
        for s in states:
            if s is not None and s not in self.q0:
                self.q0[s] = None
            self._G.add_node(s)

    def is_transition(self, s, d, e):
        """
    Check if a transition (s,d,e) exists in the automaton.
        :param s: Source.
        :param d: Destination.
        :param e: Event.
        :return:
        """
        transitions = [trans for trans in self.T.values() if trans['source'] == s and
                       trans['destination'] == d and trans['event'] == e]

        is_transition = len(transitions) != 0
        return is_transition

    def num_occur(self, q, e):
        tr = self.get_transition(q, e=e)
        if tr[-1]:
            return len(tr[-1]['timing'])
        else:
            return ""

    def num_timings(self):
        return sum(len(tr[-1]['timing']) for tr in self.get_transitions())

    def get_num_in(self, q):
        """
        Returns the number of in transitions of state q in the automaton.
        :return: number of transitions.
        """
        if self._G.has_node(q):
            return self._G.in_degree(q)
        else:
            raise Exception(f'State {q} not in the automaton.')

    def get_num_out(self, q):
        """
        Returns the number of out transitions of state q in the automaton.
        :return: number of transitions.
        """
        if self._G.has_node(q):
            return self._G.out_degree(q)
        else:
            raise Exception(f'State {q} not in the automaton.')

    def is_state(self, q):
        return self._G.has_node(q)

    def remove_state(self, s):
        self._G.remove_node(s)
        if s in self.q0:
            self.q0.pop(s)

    def in_transitions(self, s):
        """
    Get all incoming transitions of state s.
        :param s:
        :return:
        """
        return self._G.in_edges(s, data=True, keys=True)

    def out_transitions(self, s, event=None):
        """
    Get all outgoing transitions of state s.
        :param event:
        :param s:
        :return:
        """
        if event is None:
            return self._G.out_edges(s, data=True, keys=True)
        else:
            return (e for e in self._G.out_edges(s, data=True, keys=True) if e[2] == event)

    def discrete_event_dynamics(self, q, e, xt, xk, p) -> tuple:
        # self.discrete_event_dynamics(previous_q, e, xt, xk, p)
        new_q = self.UNKNOWN_STATE
        possible_destinations = set(d for s, d, ev in self._G.out_edges(q, data='event') if ev == e)
        if len(possible_destinations) == 1:
            new_q = possible_destinations.pop()
        else:
            dests = set(d for s, d, ev in self._G.edges(data='event') if ev == e)
            if len(dests) == 1:
                new_q = dests.pop()
        return new_q, None, None

    def timed_event(self, q, xc, xd, y, use_observed_timings=False):
        possible_destinations = list(ev for s, d, ev in self._G.out_edges(q, data=True) if s == q)
        if possible_destinations:
            if len(possible_destinations) == 1:
                dest = possible_destinations[0]
                if 'time' in dest:
                    return dest['time'], dest['event']
            else:
                choice = np.random.choice(possible_destinations, p=[p['prob'] if 'prob' in p else 1/len(possible_destinations) for p in possible_destinations])
                if 'time' in choice:
                    return choice['time'], choice['event']
        return None, None

    def get_transition(self, s, d=None, e=None, if_more_than_one='raise'):
        """
    Get all transitions with source state s, destination state __d. In case when e is provided, the returned list
    contains transitions where event is e.
        :param if_more_than_one:
        :param s: Source state.
        :param d: Destination state.
        :param e: Event.
        :return:
        """
        transitions = self._G.out_edges(s, keys=True, data=True)
        if e is None and d is not None:
            transitions = [trans for trans in transitions if trans[1] == d]
        elif d is None and e is not None:
            transitions = [trans for trans in transitions if trans[2] == e]
        else:
            transitions = [trans for trans in transitions if trans[1] == d and trans[2] == e]

        if len(transitions) > 1:
            if if_more_than_one == 'raise':
                raise Exception('There are multiple transitions which satisfy the condition.')
            else:
                return transitions
        elif len(transitions) == 0:
            return None
        else:
            return transitions[0]

    def rename_events(self, prefix="e_"):
        """
    Rename events to become e_0, e_1... The old id is stored in the field 'old_symbol' of the state data.
        """
        i = 0
        new_events_dict = OrderedDict()
        for k, v in self.Sigma.items():
            new_key = f'{prefix}{i}'
            new_value = v
            if new_value is None:
                new_value = {}
            new_value['old_symbol'] = k
            i += 1
            new_events_dict[new_key] = new_value
            for t in self.T.values():
                if t['event'] == k:
                    t['event'] = new_key
        self.Sigma = new_events_dict

    def step(self, x0, t, u):
        """
    Simulates one time step of continuous behavior from t to t+dt. Underlying function is solve_ivp with method is 'RK23'.
        :param x0: Initial state vector.
        :param t: Time at start of the step simulation.
        :param u: Arguments passed.....
        :return: Time t+dt, value of state at t+dt
        """
        s = solve_ivp(self.flow, t_span=(t, t + self.dt), y0=x0, method='RK23', args=u)
        x = s.y[:, -1]
        t = s.t[-1]
        return t, x

    def sim_continuous(self, q, x, y):
        """
    Simulates continuous behavior.
        :param q:
        :param x:
        :param y:
        :return:
        """
        state_data = self.discrete_states[q]
        time = []
        output = []
        state = []

        while True:
            y = self.y(t, x, u)
            u = self.u(t, x, y)
            z = self.z(t, x, y)
            p = self.p(t, x, y)

            time.append(t)
            state.append(x)
            output.append(y)

            if self.inv(t, q, x, y, z, p):
                break

            x, t = self.step(x, t, u)
        return pd.DataFrame.from_records(state), pd.DataFrame(time), pd.DataFrame.from_records(output)

    def __str__(self):
        """
    String representation of the automaton.
        :return:
        """
        return f"""Automaton:
    Number of states: {self.num_modes}
    Number of transitions: {self.num_transitions}"""

    def flow(self, q, p, x, u):
        """
    Flow equation gives derivative of the continuous variables.
        :param q: Current discrete state of the model.
        :param p: Stochastic parameters generated on entry to state current_q.
        :param x: Current continuous state of the model.
        :param u: Calculated internal i signals.
        :return: Derivative of the continuous state x.
        """
        pass

    def inv(self, t, q, x, y, z, p):
        """
    Invariants.
        :param t:
        :param q:
        :param x:
        :param y:
        :param z:
        :param p:
        """
        pass

    def get_transitions(self):
        return list(self._G.edges(data=True, keys=True))

    def print_state(self, v):
        """Prints outgoing transitions of a state v.

        Args:
            v (state): 

        Returns:
            String: Description of the outgoing transitions of the state.
        """
        s = f'<b>{str(v)}</b>'
        for tr in self.out_transitions(v):
            s += f"<br>{tr[2]} -> {tr[1]} [{self.num_occur(tr[0], tr[2])}]"
        return s

    def simulate(self, finish_time=100, current_q=None):
        """
        Simulates behaviour of the system.
        :param finish_time: Time when simulation finishes.
        :return: generated data.
        """

        if current_q is None:
            if len(self.q0) == 0:
                current_q = np.random.choice(list(self.discrete_states.keys()), 1)[-1]
                warnings.warn(
                    'Initial state not defined, sampling initial state uniformly from the set of all states.')
            else:
                current_q = np.random.choice(list(self.q0.keys()), 1)[-1]

        state_sequence = []
        data_state = []
        t = 0
        last_x = None
        last_q = None
        last_output = None
        clock = 0
        states = []
        data = []
        current_e = None

        while True:
            cont_state, cont_time, cont_output = self.sim_continuous(q=current_q, x=last_x, y=last_output, last_t=last_t)
            last_x = dict(cont_state.iloc[-1])
            last_output = dict(cont_output.iloc[-1])
            cont_time = cont_time + t - cont_time.iloc[0, 0]
            clock = cont_time.iloc[-1, 0] - cont_time.iloc[0, 0]

            current_state = current_q

            observed_current_state = current_state
            state_sequence.append(
                pd.DataFrame(np.full((cont_time.size - 1, 3), (current_state, observed_current_state, current_e)),
                             index=cont_time.iloc[0:-1, 0],
                             columns=['State', 'Observed State', 'Event']))

            data_state.append(cont_output.iloc[:-1].set_index(cont_time.iloc[0:-1, 0]))

            tr = self.out_transitions(current_state)

            if len(tr) == 0:
                break
            elif len(tr) != 1:
                tr = np.random.choice(tr, 1)[-1]
                warnings.warn('Multiple transitions can occur.')
            else:
                tr = tr[0]

            last_q = current_q
            current_e = tr.event
            self.apply_sim_event(current_e)
            if cont_time.size == 0:
                break
            t += clock

            if t >= finish_time:
                break

            states.append(pd.concat(state_sequence, axis=0))
            data.append(pd.concat(data_state, axis=0))
        return states, data

    def predict_state(self, data_collection, time_col_name, discr_col_names):
        for data in data_collection:
            data["StateEstimate"] = None
            data["Event"] = None

            prev_discr_state = None
            prev_time = None
            for row in data[discr_col_names].itertuples(index=True):
                time = data[time_col_name].iloc[row[0]]
                discr_state = row[1:]
                if prev_discr_state is not None and prev_discr_state != discr_state:
                    event = np.asarray(discr_state) - np.asarray(prev_discr_state)
                    event = ' '.join(str(x) for x in event)
                    data.loc[row[0], "Event"] = event

                data.loc[row[0], "StateEstimate"] = signal_vector_to_state(discr_state)
                prev_discr_state = discr_state
                prev_time = time
        return data_collection