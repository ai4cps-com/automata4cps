from automata import Automaton
from collections import OrderedDict
import pandas as pd
import pprint
import tools


def simple_learn_from_event_logs(data, initial=True, verbose=False):
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
        print('Duration: {}'.format(d[-1][0] - d[0][0]))

        event_rpt = 0
        state_event = ''

        old_event_rpt = 0
        old_state_event = ''

        t_old = d[0][0]
        if initial:
            a.add_initial_state('initial')
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
                source = '{}#{}'.format(old_state_event, old_event_rpt) if old_event_rpt else old_state_event

            dest = '{}#{}'.format(state_event, event_rpt) if event_rpt else state_event

            if source != 'initial' or initial:
                a.add_single_transition(source, dest, event, delta_t)
            t_old = t
            old_state_event = state_event
            old_event_rpt = event_rpt
            if verbose:
                print(source, dest, event, delta_t)
    return a


def simple_learn_from_signal_vectors(data, sig_names, verbose=False):
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


        previous_state = d[sig_names].iloc[:-1]
        dest_state = d[sig_names].iloc[1:]
        event = d[sig_names].diff().apply(lambda x: ' '.join(x.astype(str)).replace(".0", ""), 1).iloc[1:]
        deltat = d[time_col].diff().iloc[1:]


        for source, dest, ev, dt in zip(previous_state.itertuples(index=False, name=None),
                                        dest_state.itertuples(index=False, name=None), event, deltat):
            source = pprint.pformat(source, compact=True)
            dest = pprint.pformat(dest, compact=True)
            a.add_single_transition(source, dest, ev, dt)
    return a


def simple_learn_from_signal_updates(data, sig_names, initial=True, verbose=False):
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


def build_pta(data, boundaries=1):
    """
    In the function build_pta the prefix tree acceptor is created by going through each sequence of events in the
    learning examples. Also, the depth, in- and out-degree of the states are set.
    """
    pta = Automaton()
    pta.add_initial_state('q0')
    for seq in data:
        if len(seq) == 0:
            continue
        old_t = seq[seq.columns[0]].iloc[0]
        curr_stat = "q0"
        time_col = seq.columns[0]
        seq = seq[[time_col, "event"]].iloc[1:]
        for t, event in seq.itertuples(index=False, name=None):
            dt = t - old_t
            # if event in boundaries and curr_stat != "q0":
            #     sub_event = 1 + next(ii for ii, tt in enumerate(boundaries[event]) if dt >= tt)
            #     event = event + "'" * sub_event
            dest = pta.get_transition(curr_stat, e=event)
            if dest is None:
                dest = f"q{pta.num_states}"
            else:
                dest = dest[1]
            pta.add_single_transition(curr_stat, dest, event, timing=dt)
            curr_stat = dest
            old_t = t
    return pta


def extend_states(alphabet, bandwidth, max_density_at_split, verbose=False):
    """
    The function extend_states takes the current alphabet, which was created in the beginning and extends it via taking
    the minima of the pdfs and splitting the events, if there are at least three minima.
    """
    extended_alphabet = OrderedDict()
    boundaries = {}
    figs = []
    for symbol in alphabet:
        kde, kde_t = getKernelDensityEstimation(values=alphabet[symbol],
                                                x=np.linspace(min(alphabet[symbol]), max(alphabet[symbol]), 100),
                                                bandwidth=bandwidth)
        if verbose:
            print(f'Max density at split: {max_density_at_split}')
            print(f'Log: {np.log(max_density_at_split)}')
        minima = getExtremePoints(kde, max_density_at_split=max_density_at_split)
        if len(minima) == 0 or minima[0] != 0:
            minima.insert(0, 0)
        if len(minima) == 0 or minima[-1] != len(kde_t) - 1:
            minima.append(len(kde) - 1)
        minima_t = kde_t[minima]

        num_minima = len(minima)
        if num_minima <= 2:
            extended_alphabet[symbol] = alphabet[symbol]
        else:
            boundaries[symbol] = list(minima_t)
            event_times = np.asarray(alphabet[symbol])
            print(f'Split event {symbol} into {num_minima - 1} modes.')
            if verbose:

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=kde_t, y=kde, mode='lines', name='log density'))
                fig.add_trace(go.Scatter(x=event_times, y=np.ones_like(event_times) * kde.mean(), mode='markers',
                                        name='observed', marker_symbol='line-ns', marker_line_width=0.5,
                                        marker_line_color="midnightblue"))
                fig.update_layout(
                    title=f'{symbol}: Bandwidth:{bandwidth} Max split density: {np.log(max_density_at_split)}')
                fig.update_xaxes(title='t')
                fig.update_yaxes(title='kde')
                for minimum in minima_t:
                    fig.add_vline(x=minimum, line_width=2, line_dash="inadash", line_color="green")
                fig.add_hline(y=np.log(max_density_at_split), line_width=2, line_dash="inadash", line_color="red")
                figs.append(fig)

            for i in range(len(minima) - 1):
                new_times = event_times[(minima_t[i] <= event_times) & (event_times < minima_t[i + 1])]
                extended_alphabet[symbol + "'" * (i + 1)] = new_times
    # if verbose:
        # fn = 'state_split_report.html'
        # if os.path.exists(fn):
        #     os.remove(fn)
        # with open(fn, 'a') as f:
    return extended_alphabet, boundaries, figs


if __name__ == "__main__":
    import examples
    import tools

    data = examples.high_rack_storage_system_sfowl()

    discrete_data_changes = tools.remove_timestamps_without_change(discrete_data, sig_names=discrete_cols)
    discrete_data_events = tools.create_events_from_signal_vectors(discrete_data_changes, sig_names=discrete_cols)
    discrete_data_events = tools.split_data_on_signal_value(discrete_data_events, "O_w_BRU_Axis_Ctrl", 3)


    ######################## Test simple learn from signals vectors  ###################################################

    ta = simple_learn_from_signal_vectors(discrete_data_events, sig_names=discrete_cols)
    data = ta.predict_state(data, time_col_name="timestamp", discr_col_names=discrete_cols)

    state_sequences = tools.group_data_on_discrete_state(data, state_column="StateEstimate", reset_time=True, time_col="timestamp")
    dd = list(state_sequences.values())[4]
    tools.plot_data(dd, timestamp="timestamp", iterate_colors=False).show()
    exit()

    print("Number of sequences: ", len(discrete_data_events))
    discrete_data_events[0]

    ta = build_pta(discrete_data_events)
    print(ta)
    ta.view_plotly().show("browser")
    ta.view_plotly().show()


    # ta = simple_learn_from_signal_vectors(discrete_data, sig_names=discrete_cols)
    # ta.view_plotly(show_num_occur=True)

    ################### Test PTA #######################################################################################
    print("Number of sequences: ", len(discrete_data_events))
    discrete_data_events[0]

    ta = build_pta(discrete_data_events)
    print(ta)
    ta.view_plotly().show("browser")
    ta.view_plotly().show()

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
    pta.view_plotly().show()