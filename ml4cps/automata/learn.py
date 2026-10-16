"""
    The module provides learning algorithms for creation of different kinds of automata.

    Authors:
    - Nemanja Hranisavljevic, hranisan@hsu-hh.de, nemanja@ai4cps.com
    - Tom Westermann, tom.westermann@hsu-hh.de, tom@ai4cps.com
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from ml4cps import vis, tools
from ml4cps.automata.base import Automaton
from collections import OrderedDict
import pprint

def FnShiftAndDiff(xout, udout, norm_coeff, num_var, num_ud, max_deriv, Ts):
    """
    Normalizes and processes state and input data by generating shifted versions and computing derivatives.
    This function performs the following operations:
    1. Normalizes the state output `xout` using the provided normalization coefficients.
    2. Generates shifted duplicates of `xout` up to the specified maximum derivative order.
    3. Computes numerical derivatives of `xout` up to `max_deriv` order and appends them to the state data.
    4. Strips the initial entries from `xout` and the shifted data to account for the derivative computation.
    5. Normalizes the input data `udout` (if present) and strips initial entries to match the processed state data.
    Args:
        xout (np.ndarray): State output data of shape (n_samples, num_var).
        udout (np.ndarray): Input data of shape (n_samples, num_ud).
        norm_coeff (np.ndarray): Normalization coefficients of shape (num_var + num_ud, 1).
        num_var (int): Number of state variables.
        num_ud (int): Number of input variables.
        max_deriv (int): Maximum order of derivatives to compute.
        Ts (float): Sampling time interval.
    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
            - xout (np.ndarray): Processed and augmented state data with derivatives, shape (n_samples - max_deriv, ...).
            - udout (np.ndarray): Normalized input data, shape (n_samples - max_deriv, num_ud).
            - xout_shifts (pd.DataFrame): Shifted duplicates of the normalized state data, shape (n_samples - max_deriv, ...).
    Notes:
        - The function assumes that `xout` and `udout` are NumPy arrays and that `xout` has at least `max_deriv` rows.
        - The normalization coefficients should be provided for both state and input variables.
        - The function uses zero-padding for shifted and derivative computations.
    """

    # Normalize xout
    xout = xout / norm_coeff

    xout_shifts = pd.DataFrame(np.tile(xout, (1, max_deriv + 1)))

    # Calculate shifted duplicates
    for shift in range(max_deriv + 1):
        # Fill with zeros for the first 'shift' rows, then shift the xout
        xout_shifts[:, shift * num_var:(shift + 1) * num_var] = np.vstack(
            (np.zeros((shift, num_var)), xout[:-shift if shift != 0 else None, :num_var])
        )

    # Calculate derivatives up to the max_deriv order
    for deriv in range(1, max_deriv + 1):
        for curr_var in range(num_var):
            pos_last_deriv = (deriv - 1) * num_var + curr_var
            # Compute the derivative and append it to xout
            derivative = np.vstack((np.zeros((deriv, 1)), np.diff(xout[deriv - 1:, pos_last_deriv], axis=0) / Ts))
            xout = np.hstack((xout, derivative))

    # Strip entries from the front of xout and xout_shifts to match the size after derivation
    xout = xout[max_deriv:]
    xout_shifts = xout_shifts[max_deriv:]

    # Normalize udout using normalization factors (derivatives not needed)
    if num_ud != 0:
        for j in range(num_ud):
            udout[:, j] = udout[:, j] / norm_coeff[num_var + j, 0]
        # Strip entries from the front of udout to match xout
        udout = udout[max_deriv:]

    return xout, udout, xout_shifts


# # Example usage
# xout = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]])
# udout = np.array([[0.5], [0.75], [1.0], [1.25]])
# norm_coeff = np.array([[2.0], [3.0], [1.0]])  # Normalization factors for both xout and udout
#
# num_var = 2  # Number of output variables
# num_ud = 1  # Number of input variables
# max_deriv = 2  # Maximum derivative order
# Ts = 0.1  # Sampling time
#
# xout, udout, xout_shifts = FnShiftAndDiff(xout, udout, norm_coeff, num_var, num_ud, max_deriv, Ts)
#
# print("xout:")
# print(xout)
# print("udout:")
# print(udout)
# print("xout_shifts:")
# print(xout_shifts)


def simple_learn_from_event_logs(data, initial=True, count_repetition=True, verbose=False):
    """
    Simple algorithm to learn a timed automaton from event log data.
    This function constructs a timed automaton by iterating over sequences of timestamped events.
    Each event sequence is treated as a trace, and transitions are created between states based on event occurrences and their timing.
    States are determined by the emitted events, optionally including repetition counts.
    The automaton can be initialized with an explicit initial state, and transitions can account for repeated events.
    Args:
        data (list or pandas.Series): A list of event sequences, where each sequence is a pandas Series with timestamps as indices and event labels as values.
        initial (bool, optional): If True, adds an explicit 'initial' state to the automaton. Defaults to True.
        count_repetition (bool, optional): If True, distinguishes states and transitions by counting consecutive repetitions of the same event. Defaults to True.
        verbose (bool, optional): If True, prints detailed information about the learning process. Defaults to False.
    Returns:
        Automaton: The learned timed automaton object.
    Notes:
        - Each sequence in `data` should be a pandas Series indexed by timestamps.
        - If a sequence contains fewer than two events, it is skipped.
        - The function assumes the existence of an `Automaton` class with `add_initial_state` and `add_single_transition` methods.

    """

    # Here the state is determined by the events it emits, but only the first event is taken as transition
    if type(data) is not list:
        data = [data]

    a = Automaton(id='Simple')
    sequence = 0
    if verbose:
        print('***Timed automaton learning from event logs***')

    for d in data:
        sequence += 1
        print('Sequence #{}'.format(sequence))
        if len(d) < 2:
            print('Skipping because num events: 0')
            continue
        print('Duration: {}'.format(d.index[-1] - d.index[0]))

        event_rpt = 0
        state_event = ''

        old_event_rpt = 0
        old_state_event = ''

        t_old = d.index[0]
        if initial:
            a.add_initial_state('initial')

        if type(d) is pd.DataFrame:
            d = tools.create_events_from_concurent_logs(d)

        for t, event in d:
            if state_event == event:
                event_rpt += 1
            else:
                state_event = event
                event_rpt = 0

            delta_t = t - t_old
            if old_state_event == '':
                source = 'initial'
            else:
                if count_repetition and old_event_rpt:
                    source = f'{old_state_event}#{old_event_rpt}'
                else:
                    source = old_state_event

            if count_repetition and event_rpt:
                dest = f'{state_event}#{event_rpt}'
            else:
                dest = state_event

            if source != 'initial' or initial:
                a.add_single_transition(source, dest, event, delta_t)
            t_old = t
            old_state_event = state_event
            old_event_rpt = event_rpt
            if verbose:
                print(source, dest, event, delta_t)
    return a


def simple_learn_from_signal_vectors(data, config=None, drop_no_changes=True, verbose=False):
    """
    Learns a timed automaton from a list of signal vector dataframes.
    This function processes sequences of signal vectors (as pandas DataFrames), detects changes in the specified
    signal columns, and constructs a timed automaton by adding transitions for each detected event.
    Args:
        data (list of pandas.DataFrame): 
            List of DataFrames, each representing a sequence of signal vectors. 
            The first column is assumed to be the time column, and the remaining columns are signal values.
        sig_names (list of str): 
            List of column names in the DataFrame that correspond to the signals to be considered for state transitions.
        drop_no_changes (bool, optional): 
            If True, rows where no signal changes occur are dropped before processing. Default is False.
        verbose (bool, optional): 
            If True, prints detailed information about the learning process. Default is False.
    Returns:
        Automaton: 
            An Automaton object constructed from the observed transitions in the input data.
    Notes:
        - Each transition in the automaton corresponds to a change in the signal vector, with the event label
          representing the difference between consecutive signal vectors and the transition time as the time delta.
        - The function assumes that the Automaton class and its add_single_transition method are defined elsewhere.
    """

    a = Automaton()
    sequence = 0
    if verbose:
        print('***Timed automaton learning from variable changes***')

    dummy_initial = 'initial'
    a.add_initial_state(dummy_initial)

    num_sequences = len(data)

    def _select_sequence_config(config_value, sequence_index):
        if config_value is None or np.isscalar(config_value):
            return config_value
        if isinstance(config_value, (list, tuple)) and len(config_value) == num_sequences:
            return config_value[sequence_index]
        if isinstance(config_value, np.ndarray) and config_value.ndim >= 1 and config_value.shape[0] == num_sequences:
            return config_value[sequence_index]
        return config_value

    def _subset_sequence_config(sequence_config, full_index, kept_index, keep_mask=None):
        if sequence_config is None or np.isscalar(sequence_config):
            return sequence_config

        try:
            config_len = len(sequence_config)
        except TypeError:
            return sequence_config

        # Primary path: config is per-time-step and aligned by row position.
        # This is robust even when index labels are duplicated.
        if keep_mask is not None and config_len == len(full_index):
            positions = np.flatnonzero(np.asarray(keep_mask))
            if hasattr(sequence_config, "iloc"):
                return sequence_config.iloc[positions]
            if isinstance(sequence_config, np.ndarray):
                return sequence_config[positions]
            return [sequence_config[i] for i in positions]

        # Fallback path: label-based alignment.
        if hasattr(sequence_config, "reindex"):
            try:
                return sequence_config.reindex(kept_index)
            except ValueError:
                try:
                    return sequence_config.loc[kept_index]
                except Exception:
                    return sequence_config
        return sequence_config

    for sequence_index, d in enumerate(data):
        full_index = d.index
        sequence_config = _select_sequence_config(config, sequence_index)
        sig_names = d.columns
        if drop_no_changes:
            keep_mask = (d[sig_names] != d[sig_names].shift(1)).any(axis=1)
            d = d.loc[keep_mask].copy(deep=True)
            sequence_config = _subset_sequence_config(sequence_config, full_index, d.index, keep_mask=keep_mask)
        sequence += 1
        if verbose:
            print('Sequence #{}'.format(sequence))
        if len(d) < 2:
            if verbose:
                print('Skipping because num events: 0')
            continue
        if verbose:
            print('Duration: {}'.format(d.index[-1] - d.index[0]))

        previous_state = d[sig_names].iloc[:-1]
        dest_state = d[sig_names].iloc[1:]
        mask = ~d[sig_names].isin([0, 1])
        has_invalid = mask.any().any()
        if has_invalid:
            event = d[sig_names].apply(lambda x: ' '.join(map(str, x)).replace(".0", ""), 1).iloc[1:]
        else:
            event = d[sig_names].diff().apply(lambda x: ' '.join(map(str, x)).replace(".0", ""), 1).iloc[1:]
        deltat = d.index.to_series().diff().iloc[1:]

        if sequence_config is None:
            transition_config = [None] * len(previous_state)
        elif np.isscalar(sequence_config):
            transition_config = [sequence_config] * len(previous_state)
        else:
            if hasattr(sequence_config, "iloc"):
                transition_config = sequence_config.iloc[1:]
            else:
                transition_config = sequence_config[1:]

        obs_ind = 0
        for source, dest, ev, dt, cfg in zip(
            previous_state.itertuples(index=False, name=None),
            dest_state.itertuples(index=False, name=None),
            event,
            deltat,
            transition_config,
        ):
            obs_ind += 1
            source = pprint.pformat(source, compact=True, width=10000).replace(".0", "")
            dest = pprint.pformat(dest, compact=True, width=10000).replace(".0", "")

            if cfg is not None:
                a.set_configuration(cfg)
            if obs_ind == 1:
                a.add_single_transition(dummy_initial, source, "Start", 0)

            a.add_single_transition(source, dest, ev, dt)

            if obs_ind == len(previous_state):
                a.add_final_state(dest)
    return a


def simple_learn_from_signal_updates(data, sig_names, initial=True, verbose=False):
    """
    Learns a timed automaton from sequences of signal updates.
    This function processes a list of dataframes, each representing a sequence of signal updates over time.
    For each sequence, it constructs states based on the values of the specified signals and adds transitions
    to an Automaton object whenever a signal value changes.
    Args:
        data (list of pandas.DataFrame): List of dataframes, each containing time-stamped signal updates.
            The first column is assumed to be the time column, and subsequent columns correspond to signal names.
        sig_names (list of str): List of signal names to track and use for state construction.
        initial (bool, optional): If True, adds an initial state to the automaton for each sequence. Defaults to True.
        verbose (bool, optional): If True, prints detailed information about the learning process. Defaults to False.
    Returns:
        Automaton: The learned automaton with states and transitions based on the observed signal updates.
    Notes:
        - Each state is represented as a dictionary mapping signal names to their current values.
        - Transitions are added only when all signal values are set (i.e., not None).
        - The event label for each transition is formatted as '<signal_name><-<value>'.
        - The transition is annotated with the time difference (delta_t) between consecutive events.
    """

    a = Automaton()
    sequence = 0
    if verbose:
        print('***Timed automaton learning from variable changes***')

    for d in data:
        time_col = d.columns[0]
        sequence += 1
        print('Sequence #{}'.format(sequence))
        if len(d) < 2:
            print('Skipping because num events: 0')
            continue
        print('Duration: {}'.format(d[time_col].iloc[-1] - d[time_col].iloc[0]))

        t_old = d[time_col].iloc[0]
        if initial:
            a.add_initial_state('initial')

        state = dict.fromkeys(sig_names)
        for t, signal, value in d.itertuples(index=False, name=None):
            event = f'{signal}<-{value}'
            all_values_are_set = all(value is not None for value in state.values())

            delta_t = t - t_old
            t_old = t
            source = pprint.pformat(state)
            state[signal] = value
            dest = pprint.pformat(state)

            if all_values_are_set:
                a.add_single_transition(source, dest, event, delta_t)
    return a


def build_pta(data, event_col='event', boundaries=1):
    """
    Builds a Prefix Tree Acceptor (PTA) from a collection of event sequences.
    This function constructs a PTA by iterating through each sequence of events in the provided data. 
    It adds states and transitions to the automaton based on the observed event sequences, 
    and sets the depth, in-degree, and out-degree of the states. The PTA is useful for learning 
    automata from positive examples.
    Args:
        data (iterable): An iterable of event sequences. Each sequence can be a pandas DataFrame, 
            pandas Series, or string. If a DataFrame, it should contain at least a time column and 
            an event column.
        event_col (str, optional): The name of the column containing event labels in the input 
            DataFrame or Series. Defaults to 'event'.
        boundaries (int or dict, optional): Not currently used in the function, but intended for 
            handling event boundaries or timing constraints. Defaults to 1.
    Returns:
        Automaton: The constructed Prefix Tree Acceptor representing the input event sequences.
    Notes:
        - The function expects the presence of an `Automaton` class with methods for adding states, 
          transitions, and final states.
        - If a sequence is a string, it is converted to a pandas Series of characters.
        - Timing information (dt) is calculated as the difference between consecutive time steps.
        - The function skips empty sequences.
    """

    pta = Automaton()
    pta.add_initial_state('q0')
    for seq in data:
        if len(seq) == 0:
            continue
        if not isinstance(seq, pd.DataFrame):
            if isinstance(seq, str):
                seq = pd.Series(list(seq))
            if isinstance(seq, pd.Series):
                seq.name = event_col
            seq = pd.DataFrame(seq).reset_index(drop=False)
        # old_t = seq[seq.columns[0]].iloc[0] # Instead I will assume index is time
        old_t = seq.index[0]
        curr_stat = "q0"
        # time_col = seq.columns[0]
        seq = seq[[event_col]] #.iloc[1:]
        for t, event in seq.itertuples(index=True, name=None):
            if pd.isna(event):
                continue # Maybe inform about this
            dt = t - old_t
            # if event in boundaries and curr_stat != "q0":
            #     sub_event = 1 + next(ii for ii, tt in enumerate(boundaries[event]) if dt >= tt)
            #     event = event + "'" * sub_event
            dest = pta.get_transition(curr_stat, e=event)
            if dest is None:
                dest = f"q{pta.num_modes}"
            else:
                dest = dest[1]
            pta.add_single_transition(curr_stat, dest, event, timing=dt)
            curr_stat = dest
            old_t = t
        pta.add_final_state(curr_stat)
    return pta


def FnDetectChangePoints(xout, udout, xout_shifts):
    """
    Detects change points in output and input variables, filters them, and constructs a trace structure.
    This function analyzes the provided output (`xout`) and input (`udout`) data to detect change points
    using the `findChangePoints` function. It processes both output and input variables, aggregates and filters
    the detected change points, and returns a structured trace dictionary containing the processed data and
    change point information.
    Args:
        xout (np.ndarray): Output data array of shape (n_samples, n_outputs).
        udout (np.ndarray): Input data array of shape (n_samples, n_inputs).
        xout_shifts (np.ndarray): Shifted output data array, used for further processing.
    Returns:
        dict: A dictionary with the following keys:
            - 'x': Filtered output data array.
            - 'xs': Filtered shifted output data array.
            - 'chpoints': Array of global change points detected across all variables.
            - 'chpoints_per_var': List of arrays, each containing change points for a specific variable.
            - 'ud': Filtered input data array.
            - 'labels_num': Empty list (reserved for numeric labels).
            - 'labels_trace': Empty list (reserved for trace labels).
    Notes:
        - Relies on global variables: `num_var`, `num_ud`, `max_deriv`, and `chp_depths`.
        - Uses external functions: `findChangePoints` and `filterChangePoints`.
        - The function is intended for use in time-series or sequential data analysis where detecting
          significant changes in variable values is required.
    """

    global num_var, num_ud, max_deriv, chp_depths

    # Initialize global variables
    chpoints = []  # Global changepoints
    chp_depths = np.zeros(max_deriv + 1)  # Purely for debugging
    chp_var = [None] * (num_var + num_ud)  # Local changepoints (per variable)

    # Detect change points for output variables
    for i in range(num_var):
        new_chp = findChangePoints(xout[:, i::num_var], 0, 1, xout.shape[0], max_deriv)
        chpoints = np.union1d(chpoints, new_chp)
        chp_var[i] = np.sort(new_chp)

    # Detect change points for input variables
    for i in range(num_ud):
        new_chp = findChangePoints(udout[:, i], 0, 1, udout.shape[0], 0)
        chpoints = np.union1d(chpoints, new_chp)
        chp_var[num_var + i] = new_chp

    # Filter changepoints
    xout, udout, xout_shifts, chpoints, chp_var = filterChangePoints(xout, udout, xout_shifts, chpoints, chp_var)

    # Create the trace structure
    trace = {
        'x': xout,
        'xs': xout_shifts,
        'chpoints': chpoints,
        'chpoints_per_var': chp_var,
        'ud': udout,
        'labels_num': [],
        'labels_trace': []
    }
    return trace


def computeDistance(der):
    """
    Computes a distance metric over a sliding window for a given derivative array.
    For each position in the input array `der`, the function calculates the sum of absolute differences
    between two windows of size `windowSize` before and after the current position, after normalizing
    each window by subtracting its first element. The result is an array of distances.
    Parameters:
        der (np.ndarray): The input array (e.g., derivative values) over which to compute the distance.
    Returns:
        np.ndarray: An array containing the computed distances for each valid position.
    """

    global windowSize
    dist = np.zeros(windowSize)
    for i in range(windowSize, len(der) - windowSize):
        before = der[(i - windowSize):i]
        after = der[(i + 1):(i + windowSize + 1)]
        dist_new = np.sum(np.abs((before - before[0]) - (after - after[0])))
        dist = np.append(dist, dist_new)
    return dist


def findChangePoints(xout, depth, starting, ending, max_depth):
    """
    Recursively detects change points in a multi-dimensional signal using a hierarchical approach.
    This function analyzes a segment of the input array `xout` at a given `depth` (dimension),
    computes a distance metric to identify potential change points (peaks), and then recursively
    searches for further change points in subsegments at deeper levels. The recursion stops when
    the maximum depth is reached or the segment is too small.
    Parameters
    ----------
    xout : np.ndarray
        The input array containing the signal or features to analyze. Expected shape is (n_samples, n_features).
    depth : int
        The current depth (dimension) being analyzed.
    starting : int
        The starting index of the segment to analyze.
    ending : int
        The ending index (exclusive) of the segment to analyze.
    max_depth : int
        The maximum depth (dimension) to analyze.
    Returns
    -------
    np.ndarray
        An array of indices representing detected change points within the specified segment.
    Notes
    -----
    - Uses global variables `windowSize` and `chp_depths` for windowing and tracking change points per depth.
    - Utilizes `computeDistance`, `find_peaks`, and `filterindx` helper functions.
    - At the top level (depth == 0), prepends and appends boundary indices to the result.
    """

    global windowSize, chp_depths

    locs = []
    if depth > max_depth or ending - starting - 1 < 2 * windowSize:
        return locs

    der = xout[starting:ending, depth]
    dist = computeDistance(der)

    # Find peaks in distance to detect change points
    _, locsDist = find_peaks(dist, height=5)
    locsHere = np.sort(locsDist + starting - 1)
    locsHere = filterindx(locsHere, 1.5 * windowSize)
    chp_depths[depth] += len(locsHere)
    locs.extend(locsHere)

    locsHere = np.concatenate([[starting - windowSize // 2], locsHere, [ending + windowSize // 2]])
    for i in range(len(locsHere) - 1):
        newStart = int(locsHere[i] + windowSize / 2)
        newEnd = int(locsHere[i + 1] - windowSize / 2)
        locsNew = findChangePoints(xout, depth + 1, newStart, newEnd, max_depth)
        locs.extend(locsNew)

    if depth == 0:
        locs = np.concatenate([[1], locs, [len(der)]])

    return np.array(locs)


def filterChangePoints(xout, udout, xout_shifts, chpoints, chp_var):
    """
    Filters and synchronizes detected changepoints in time series data across multiple variables.
    This function processes global and local changepoint indices to ensure consistency and remove redundant or closely spaced changepoints. It updates the provided data arrays and changepoint lists accordingly.
    Args:
        xout (np.ndarray): Output variable time series data (samples x variables).
        udout (np.ndarray): Input variable time series data (samples x variables), or an empty array if not used.
        xout_shifts (np.ndarray): Shifted output variable data (samples x variables).
        chpoints (list or np.ndarray): List of global changepoint indices.
        chp_var (list of np.ndarray): List containing arrays of changepoint indices for each variable.
    Returns:
        tuple:
            - xout (np.ndarray): Filtered output variable data.
            - udout (np.ndarray): Filtered input variable data.
            - xout_shifts (np.ndarray): Filtered shifted output variable data.
            - chpoints (np.ndarray): Filtered and synchronized global changepoint indices.
            - chp_var (list of np.ndarray): Updated list of changepoint indices for each variable.
    Notes:
        - Uses global variables: windowSize, num_var, num_ud.
        - Assumes that changepoints are sorted and that there are at least two changepoints.
        - The function modifies chp_var in place.
    """

    global windowSize, num_var, num_ud

    # Filter global changepoints detected on multiple output variables
    chpoints = filterindx(chpoints, windowSize)

    if chpoints[-1] - chpoints[-2] < 2 * windowSize:
        xout = xout[:chpoints[-2], :]
        xout_shifts = xout_shifts[:chpoints[-2], :]
        if num_ud != 0:
            udout = udout[:chpoints[-2], :]
        chpoints = chpoints[:-1]
        for i in range(num_var + num_ud):
            current_chps = np.array(chp_var[i])
            current_chps = current_chps[:-1]
            current_chps = np.append(current_chps, chpoints[-1])
            chp_var[i] = current_chps

    # Ensure consistency between global and local changepoint sets
    for i in range(num_var + num_ud):
        current_chps = np.array(chp_var[i])
        for j in range(len(current_chps)):
            idx = np.argmin(np.abs(chpoints - current_chps[j]))
            current_chps[j] = chpoints[idx]
        if current_chps[-2] == current_chps[-1]:
            current_chps = current_chps[:-1]
        chp_var[i] = current_chps

    return xout, udout, xout_shifts, chpoints, chp_var


def filterindx(indx, windw):
    """
    Filters out indices from the input array that are within a specified window of each other.
    Given a sorted array of indices, this function removes any index that is within `windw` distance
    from its predecessor, keeping only the first occurrence in each window.
    Parameters:
        indx (array-like): A sorted array or list of integer indices.
        windw (int): The minimum allowed distance between consecutive indices.
    Returns:
        numpy.ndarray: The filtered array of indices, where no two indices are within `windw` of each other.
    Example:
        >>> filterindx(np.array([1, 2, 3, 10, 12]), 2)
        array([ 1, 10, 12])
    """
    n = 0
    while n < len(indx) - 1:
        id1 = indx[n]
        while n + 1 < len(indx) and indx[n + 1] - id1 <= windw:
            indx = np.delete(indx, n + 1)
        n += 1
    return indx


# Global variables (must be initialized elsewhere in your code)
num_var = None
num_ud = None
useTime = None


def FnTraceToTrainingData(trace, num_var, num_ud, useTime):
    """
    Converts a trace dictionary into training data suitable for machine learning models.
    Parameters:
        trace (dict): A dictionary containing the following keys:
            - 'x' (np.ndarray): Array of system variables over time (shape: [timesteps, num_var]).
            - 'ud' (np.ndarray): Array of user-defined variables over time (shape: [timesteps, num_ud]).
            - 'labels_trace' (np.ndarray): Array of state labels for each segment.
            - 'chpoints' (list or np.ndarray): Indices indicating change points (state switches) in the trace.
        num_var (int): Number of system variables to include in the feature vector.
        num_ud (int): Number of user-defined variables to include in the feature vector.
        useTime (bool): If True, includes the time since the last state switch as a feature.
    Returns:
        X (np.ndarray): Feature matrix where each row corresponds to a time step and contains:
            - Current state label
            - System variables
            - User-defined variables (if num_ud > 0)
            - Time since last state switch (if useTime is True)
        Y (np.ndarray): Array of next state labels (class labels) for each feature vector in X.
        states (np.ndarray): Array of state labels for each time step in the trace.
    Notes:
        - The function skips the last time step in the trace for feature construction, as it cannot form a (X, Y) pair.
        - The function assumes that the trace data is properly aligned and that 'chpoints' and 'labels_trace' are consistent.
    """


    # Preallocate arrays for samples
    states = np.zeros((trace['x'].shape[0] - 1, 1))
    values = np.zeros((trace['x'].shape[0] - 1, num_var + num_ud))
    timeSwitch = np.zeros((trace['x'].shape[0] - 1, 1))

    # Variables to keep track of states
    lastswitch = 1
    indxStates = 0

    # Loop through trace data to generate feature vectors and class labels
    for indx in range(trace['x'].shape[0] - 1):
        # Update index associated with system mode switch
        if indxStates + 1 < len(trace['chpoints']) and indx >= trace['chpoints'][indxStates + 1]:
            indxStates += 1
            lastswitch = indx

        # Save current states and values for the feature vector
        states[indx] = trace['labels_trace'][indxStates]
        values[indx, :num_var] = trace['x'][indx, :num_var]
        if num_ud != 0:
            values[indx, num_var:num_var + num_ud] = trace['ud'][indx, :num_ud]

        timeSwitch[indx] = indx - lastswitch

    # Create matrices containing feature vectors and corresponding class labels
    points = np.arange(states.shape[0] - 1)
    if useTime:
        X = np.hstack([states[points], values[points, :], timeSwitch[points]])
    else:
        X = np.hstack([states[points], values[points, :]])

    Y = states[points + 1]

    return X, Y, states


""" RPNI algorithm"""


def rpni(positive_samples, negative_samples):
    """
    Implements the RPNI (Regular Positive and Negative Inference) algorithm for learning a DFA from positive and negative samples.
    Args:
        positive_samples (Iterable[str]): A collection of strings that should be accepted by the learned DFA.
        negative_samples (Iterable[str]): A collection of strings that should be rejected by the learned DFA.
    Returns:
        DFA: A deterministic finite automaton (DFA) that accepts all positive samples and rejects all negative samples.
    Notes:
        - The function first constructs a Prefix Tree Acceptor (PTA) from the positive samples.
        - It then attempts to merge states in the DFA, ensuring that no negative sample is accepted after each merge.
        - The merging process is guided by the constraint that all negative samples must be rejected.
    """


    # Process positive samples
    dfa = build_pta(positive_samples)

    # Process negative samples
    # Function to check if the DFA rejects all negative samples
    def rejects_negative_samples(mdl):
        for sample in negative_samples:
            if mdl.accepts(sample):
                return False
        return True

    # Attempt to merge states while preserving rejection of negative samples
    states = list(dfa.discrete_states)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            state1 = states[i]
            state2 = states[j]
            if dfa.is_state(state2):
                dfa.try_merge_states(state1, state2, rejects_negative_samples)
                print('Compare')

    return dfa



if __name__ == '__main__':
    # # Example usage
    # positive_samples = ["a", "ab", "abc", "aa"]
    # negative_samples = ["b", "ba", "ac", "abab"]
    #
    # dfa = rpni(positive_samples, negative_samples)
    # vis.plot_cps_component(dfa, output='dash', min_zoom=3, max_zoom=3)
    # # Test the DFA
    # print(dfa.accepts("a"))    # True
    # print(dfa.accepts("ab"))   # True
    # print(dfa.accepts("abc"))  # True
    # print(dfa.accepts("b"))    # False
    # print(dfa.accepts("ba"))   # False
    # print(dfa.accepts("ac"))   # False
    # print(dfa.accepts("abab"))  # False

    # Example usage
    positive_samples = ["a", "ab", "abc"]
    negative_samples = ["aa", "ac", "aba", "abb"]

    dfa = rpni(positive_samples, negative_samples)
    vis.plot_cps_component(dfa, output='dash', min_zoom=3, max_zoom=3)
    # Test the DFA
    print(dfa.accepts("a"))  # True
    print(dfa.accepts("ab"))  # True
    print(dfa.accepts("abc"))  # True
    print(dfa.accepts("b"))  # False
    print(dfa.accepts("ba"))  # False
    print(dfa.accepts("ac"))  # False
    print(dfa.accepts("abab"))  # False





if __name__ == "__main__":
    from ml4cps.examples import examples
    import tools

    discrete_data, time_col, discrete_cols = examples.conveyor_system_sfowl("discrete")
    data, _, _, cont_vars = examples.conveyor_system_sfowl("all")

    discrete_data_changes = tools.remove_timestamps_without_change(discrete_data, sig_names=discrete_cols)
    discrete_data_events = tools.create_events_from_signal_vectors(discrete_data_changes, sig_names=discrete_cols)
    discrete_data_events = tools.split_data_on_signal_value(discrete_data_events, "O_w_BRU_Axis_Ctrl", 3)


    ######################## Test simple learn from signals vectors  ###################################################
    ta = simple_learn_from_signal_vectors(discrete_data_events, sig_names=discrete_cols)

    data = ta.predict_state(data, time_col_name="timestamp", discr_col_names=discrete_cols)
    exit()

    state_sequences = tools.group_data_on_discrete_state(data, state_column="StateEstimate", reset_time=True,
                                                         time_col="timestamp")
    dd = list(state_sequences.values())[4]
    tools.plot_timeseries(dd, timestamp="timestamp", iterate_colors=False).show()
    exit()

    print("Number of sequences: ", len(discrete_data_events))
    discrete_data_events[0]

    ta = build_pta(discrete_data_events)
    print(ta)
    ta.plot_cps().show("browser")
    ta.plot_cps().show()


    # ta = simple_learn_from_signal_vectors(discrete_data, sig_names=discrete_cols)
    # ta.view_plotly(show_num_occur=True)

    ################### Test PTA #######################################################################################
    print("Number of sequences: ", len(discrete_data_events))
    discrete_data_events[0]

    ta = build_pta(discrete_data_events)
    print(ta)
    ta.plot_cps().show("browser")
    ta.plot_cps().show()

    # ta = simple_learn_from_signal_vectors(discrete_data, sig_names=discrete_cols)
    # ta.view_plotly(show_num_occur=True)


    exit()

    print("Test build_pta")

    test_data1 = [[[1, 0, 0, 0, 1.3, 9.6, 14.5],
                   [2, 0, 0, 0, 1.5, 9.5, 14.4],
                   [3, 0, 0, 1, 1.8, 9.3, 14.1],
                   [4, 0, 0, 1, 2.1, 8.9, 13.6],
                   [5, 0, 0, 1, 2.2, 8.5, 13.3],
                   [6, 0, 1, 1, 2.3, 8.4, 13.2],
                   [7, 0, 1, 1, 2.4, 8.2, 13.1],
                   [8, 0, 1, 1, 2.6, 5.1, 12.9],
                   [9, 0, 0, 1, 2.9, 7.9, 12.7],
                   [10, 0, 0, 1, 3.1, 7.8, 12.6]],
                  [[1, 0, 0, 0, 1.6, 9.9, 14.9],
                   [5, 1, 0, 0, 1.3, 9.2, 14.1],
                   [6, 1, 0, 0, 1.9, 9.6, 14.7],
                   [7, 0, 1, 1, 2.5, 8.7, 13.3],
                   [8, 0, 1, 1, 2.6, 8.2, 13.5],
                   [64, 0, 0, 1, 2.7, 8.6, 13.6],
                   [88, 0, 0, 1, 2.9, 8.1, 13.7],
                   [90, 1, 0, 1, 2.6, 5.4, 12.5],
                   [140, 1, 0, 1, 2.7, 7.2, 12.6],
                   [167, 1, 1, 1, 3.7, 7.2, 12.1]],
                  [[1, 0, 0, 0, 1.3, 9.6, 14.5],
                   [2, 0, 0, 0, 1.5, 9.5, 14.4],
                   [4, 0, 0, 1, 1.8, 9.3, 14.1],
                   [6, 0, 0, 1, 2.1, 8.9, 13.6],
                   [8, 0, 0, 1, 2.2, 8.5, 13.3],
                   [11, 0, 1, 1, 2.3, 8.4, 13.2],
                   [13, 0, 1, 1, 2.4, 8.2, 13.1],
                   [14, 0, 1, 1, 2.6, 5.1, 12.9],
                   [15, 0, 0, 1, 2.9, 7.9, 12.7],
                   [17, 0, 0, 1, 3.1, 7.8, 12.6]]]

    test_data1 = [pd.DataFrame(d) for d in test_data1]
    test_data1 = tools.remove_timestamps_without_change(test_data1, sig_names=[1, 2, 3])
    test_data1 = tools.create_events_from_signal_vectors(test_data1, sig_names=[1, 2, 3])

    # test_data1 = createEventsfromDataFrame(test_data1)
    pta = build_pta(test_data1)
    pta.plot_cps().show()
