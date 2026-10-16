import pandas as pd
import pytest

pytest.importorskip("simpy")
pytest.importorskip("mlflow")

from ml4cps.automata import learn as automata_learn


def test_simple_learn_from_signal_vectors_learns_start_and_state_transitions():
    sequence = pd.DataFrame(
        {"a": [0, 1, 1], "b": [0, 0, 1]},
        index=[0.0, 1.0, 2.0],
    )

    automaton = automata_learn.simple_learn_from_signal_vectors([sequence])

    assert list(automaton.q0.keys()) == ["initial"]
    assert list(automaton.final_q.keys()) == ["(1, 1)"]

    start = automaton.get_transition("initial", d="(0, 0)", e="Start")
    assert start is not None
    assert start[-1]["timing"] == [0]

    first = automaton.get_transition("(0, 0)", d="(1, 0)", e="1 0")
    assert first is not None
    assert first[-1]["timing"] == [1.0]

    second = automaton.get_transition("(1, 0)", d="(1, 1)", e="0 1")
    assert second is not None
    assert second[-1]["timing"] == [1.0]


def test_simple_learn_from_signal_vectors_uses_configuration_per_transition():
    sequence = pd.DataFrame(
        {"x": [0, 0, 1, 0]},
        index=[0, 1, 2, 3],
    )
    config = pd.Series(["C0", "C1", "C2", "C3"], index=sequence.index)

    automaton = automata_learn.simple_learn_from_signal_vectors(
        [sequence],
        config=config,
        drop_no_changes=True,
    )

    start = automaton.get_transition("initial", d="(0,)", e="Start")
    assert start is not None
    assert start[-1]["timing"] == {"C2": [0]}

    rise = automaton.get_transition("(0,)", d="(1,)", e="1")
    assert rise is not None
    assert rise[-1]["timing"] == {"C2": [2.0]}

    fall = automaton.get_transition("(1,)", d="(0,)", e="-1")
    assert fall is not None
    assert fall[-1]["timing"] == {"C3": [1.0]}


def test_simple_learn_from_signal_vectors_uses_dataframe_time_index_after_dropping_no_changes():
    sequence = pd.DataFrame(
        {"h1": [0, 0, 1, 1]},
        index=[0.5, 1.0, 1.5, 2.0],
    )

    automaton = automata_learn.simple_learn_from_signal_vectors(
        [sequence],
        drop_no_changes=True,
    )

    start = automaton.get_transition("initial", d="(0,)", e="Start")
    assert start is not None
    assert start[-1]["timing"] == [0]

    rise = automaton.get_transition("(0,)", d="(1,)", e="1")
    assert rise is not None
    assert rise[-1]["timing"] == [1.0]


def test_simple_learn_from_signal_vectors_accepts_per_timestep_config_per_sequence():
    seq1 = pd.DataFrame(
        {"x": [0, 0, 1, 1]},
        index=[0.0, 1.0, 2.0, 3.0],
    )
    seq2 = pd.DataFrame(
        {"x": [1, 1, 0]},
        index=[0.0, 0.5, 1.0],
    )

    # Per-time-step configuration, provided per sequence with different lengths.
    # This used to fail due np.asarray(...) on ragged sequence config.
    config = [
        pd.Series(["A0", "A1", "A2", "A3"], index=seq1.index),
        ["B0", "Bmid", "B1"],
    ]

    automaton = automata_learn.simple_learn_from_signal_vectors(
        [seq1, seq2],
        config=config,
        drop_no_changes=True,
    )

    tr_up = automaton.get_transition("(0,)", d="(1,)", e="1")
    assert tr_up is not None
    assert tr_up[-1]["timing"] == {"A2": [2.0]}

    tr_down = automaton.get_transition("(1,)", d="(0,)", e="-1")
    assert tr_down is not None
    assert tr_down[-1]["timing"] == {"B1": [1.0]}


def test_simple_learn_from_signal_vectors_handles_duplicate_time_index_in_config_series():
    sequence = pd.DataFrame(
        {"x": [0, 0, 1]},
        index=[0.0, 0.0, 1.0],
    )
    config = pd.Series(["A", "B", "C"], index=sequence.index)

    automaton = automata_learn.simple_learn_from_signal_vectors(
        [sequence],
        config=[config],
        drop_no_changes=True,
    )

    tr_up = automaton.get_transition("(0,)", d="(1,)", e="1")
    assert tr_up is not None
    assert tr_up[-1]["timing"] == {"C": [1.0]}
