import copy

import pytest

pytest.importorskip("simpy")
pytest.importorskip("mlflow")

from ml4cps.automata import Automaton


def test_out_transitions_respect_configuration_and_guard():
    automaton = Automaton(
        configuration="A",
        states=[
            {"name": "s0", "enabled": {"A": 1, "B": 1}},
            {"name": "s1", "enabled": {"A": 1, "B": 0}},
            {"name": "s2", "enabled": {"A": 0, "B": 1}},
            {"name": "s3", "enabled": {"A": 1, "B": 0}},
        ],
        transitions=[
            {"source": "s0", "destination": "s1", "event": "go", "enabled": {"A": 1, "B": 0}},
            {"source": "s0", "destination": "s2", "event": "go", "enabled": {"A": 0, "B": 1}},
            {
                "source": "s0",
                "destination": "s3",
                "event": "go",
                "guard": lambda cfg, *_: cfg == "A",
            },
        ],
    )

    out_a = automaton.out_transitions("s0", event="go")
    assert {tr[1] for tr in out_a} == {"s1", "s3"}

    automaton.set_configuration("B")
    out_b = automaton.out_transitions("s0", event="go")
    assert {tr[1] for tr in out_b} == {"s2"}


def test_get_transitions_active_only_filters_by_configuration():
    automaton = Automaton(
        configuration="A",
        states=[
            {"name": "s0", "enabled": {"A": 1, "B": 1}},
            {"name": "s1", "enabled": {"A": 1, "B": 1}},
            {"name": "s2", "enabled": {"A": 1, "B": 1}},
        ],
        transitions=[
            {"source": "s0", "destination": "s1", "event": "e1", "enabled": {"A": 1, "B": 0}},
            {"source": "s0", "destination": "s2", "event": "e1", "enabled": {"A": 0, "B": 1}},
            {"source": "s1", "destination": "s2", "event": "e2", "enabled": {"A": 1, "B": 1}},
        ],
    )

    assert len(automaton.get_transitions(active_only=False)) == 3
    assert len(automaton.get_transitions(active_only=True)) == 2

    automaton.set_configuration("B")
    active = automaton.get_transitions(active_only=True)
    assert len(active) == 2
    assert {(s, d, data["event"]) for s, d, _, data in active} == {("s0", "s2", "e1"), ("s1", "s2", "e2")}


def test_sigma_contains_only_active_events():
    automaton = Automaton(
        configuration="A",
        states=[
            {"name": "s0", "enabled": {"A": 1, "B": 1}},
            {"name": "s1", "enabled": {"A": 1, "B": 1}},
            {"name": "s2", "enabled": {"A": 1, "B": 1}},
        ],
        transitions=[
            {"source": "s0", "destination": "s1", "event": "go", "enabled": {"A": 1, "B": 0}},
            {"source": "s0", "destination": "s2", "event": "stop", "enabled": {"A": 0, "B": 1}},
        ],
    )

    assert set(automaton.Sigma) == {"go"}
    automaton.set_configuration("B")
    assert set(automaton.Sigma) == {"stop"}


def test_state_setter_rejects_disallowed_state():
    automaton = Automaton(
        configuration="A",
        states=[{"name": "sA", "enabled": {"A": 1}}, {"name": "sB", "enabled": {"B": 1}}],
    )

    automaton.state = ("sA", (), ())
    with pytest.raises(ValueError, match='not allowed for configuration "A"'):
        automaton.state = ("sB", (), ())


def test_discrete_event_dynamics_uses_only_allowed_transitions():
    automaton = Automaton(
        configuration="A",
        unknown_state="UNKNOWN",
        states=[
            {"name": "s0", "enabled": {"A": 1, "B": 1}},
            {"name": "s1", "enabled": {"A": 1, "B": 1}},
        ],
        transitions=[
            {"source": "s0", "destination": "s1", "event": "go", "enabled": {"A": 1, "B": 0}},
        ],
    )

    new_q, _, _ = automaton.discrete_event_dynamics(("s0",), None, None, {"event": "go"})
    assert new_q == ("s1",)

    automaton.set_configuration("B")
    new_q, _, _ = automaton.discrete_event_dynamics(("s0",), None, None, {"event": "go"})
    assert new_q == ("UNKNOWN",)


def test_sample_initial_prefers_active_initial_states():
    automaton = Automaton(
        configuration="A",
        states=[{"name": "sA", "enabled": {"A": 1}}, {"name": "sB", "enabled": {"B": 1}}],
        initial_q=["sA", "sB"],
    )

    samples = {automaton.sample_initial() for _ in range(10)}
    assert samples == {"sA"}


def test_constructor_does_not_mutate_transition_input_and_accepts_dest_alias():
    transitions = [{"source": "s0", "dest": "s1", "event": "go", "enabled": {"A": 1}}]
    original = copy.deepcopy(transitions)

    automaton = Automaton(
        configuration="A",
        states=["s0", "s1"],
        transitions=transitions,
    )

    assert transitions == original
    out = automaton.out_transitions("s0", event="go")
    assert len(out) == 1
    assert out[0][1] == "s1"


def test_add_states_from_uses_current_configuration_by_default():
    automaton = Automaton(states=["root"])
    automaton.set_configuration("CFG_A")
    automaton.add_states_from(["s1", "s2"])

    assert automaton._G.nodes["s1"]["enabled"] == {"CFG_A": 1}
    assert automaton._G.nodes["s2"]["enabled"] == {"CFG_A": 1}


def test_add_transition_uses_current_configuration_by_default():
    automaton = Automaton(
        states=[
            {"name": "s0", "enabled": {"CFG_A": 1, "CFG_B": 1}},
            {"name": "s1", "enabled": {"CFG_A": 1, "CFG_B": 1}},
        ]
    )

    automaton.set_configuration("CFG_A")
    automaton.add_transition("s0", "s1", "go")
    assert {tr[1] for tr in automaton.out_transitions("s0", event="go")} == {"s1"}

    automaton.set_configuration("CFG_B")
    assert automaton.out_transitions("s0", event="go") == []


def test_discrete_event_dynamics_uses_frequency_dict():
    automaton = Automaton(
        configuration="A",
        states=[
            {"name": "s0", "enabled": {"A": 1, "B": 1}},
            {"name": "s1", "enabled": {"A": 1, "B": 1}},
            {"name": "s2", "enabled": {"A": 1, "B": 1}},
        ],
        transitions=[
            {"source": "s0", "destination": "s1", "event": "go", "enabled": {"A": 1, "B": 1}, "frequency": {"A": 1.0, "B": 0.0}},
            {"source": "s0", "destination": "s2", "event": "go", "enabled": {"A": 1, "B": 1}, "frequency": {"A": 0.0, "B": 1.0}},
        ],
    )

    new_q_a, _, _ = automaton.discrete_event_dynamics(("s0",), None, None, {"event": "go"})
    assert new_q_a == ("s1",)

    automaton.set_configuration("B")
    new_q_b, _, _ = automaton.discrete_event_dynamics(("s0",), None, None, {"event": "go"})
    assert new_q_b == ("s2",)


def test_timed_event_uses_dynamic_min_max_interval():
    automaton = Automaton(
        configuration="A",
        states=[{"name": "s0", "enabled": {"A": 1}}, {"name": "s1", "enabled": {"A": 1}}],
        transitions=[
            {
                "source": "s0",
                "destination": "s1",
                "event": "go",
                "min_interval": {"A": 0.25},
                "max_interval": {"A": 0.25},
            }
        ],
    )

    dt, dest = automaton.timed_event("s0", None, None)
    assert dt == pytest.approx(0.25)
    assert dest == "s1"


def test_add_single_transition_uses_timing_dict_for_active_configuration():
    automaton = Automaton(configuration="A", states=["s0", "s1"])
    automaton.add_single_transition("s0", "s1", "go", timing=0.2)

    edge_data = automaton._G.get_edge_data("s0", "s1", "go")
    assert isinstance(edge_data["timing"], dict)
    assert edge_data["timing"]["A"] == [0.2]

    automaton.add_single_transition("s0", "s1", "go", timing=[0.4, 0.6])
    assert edge_data["timing"]["A"] == [0.2, 0.4, 0.6]


def test_add_single_transition_adds_timing_per_configuration_bucket():
    automaton = Automaton(configuration="A", states=["s0", "s1"])
    automaton.add_single_transition("s0", "s1", "go", timing=1.0)

    automaton.set_configuration("B")
    automaton.add_single_transition("s0", "s1", "go", timing=[2.0, 3.0])

    edge_data = automaton._G.get_edge_data("s0", "s1", "go")
    assert edge_data["timing"]["A"] == [1.0]
    assert edge_data["timing"]["B"] == [2.0, 3.0]
