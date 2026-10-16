"""
Manual visual test runner for vis.plot_cps_component.

This script opens multiple Dash browser tabs/windows, each showing a different
automaton visualization. Some automata use multiple configurations to validate
config-dependent edge expansion, colors, and labels.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ml4cps.automata import Automaton
    from ml4cps import vis
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency while loading ml4cps. "
        "Install required packages (notably simpy/mlflow) before running this visual test.\n"
        f"Original error: {exc}"
    ) from exc


def build_simple_automaton() -> Automaton:
    return Automaton(
        id="simple",
        initial_q="s0",
        final_q=["s2"],
        states=["s0", "s1", "s2"],
        transitions=[
            {"source": "s0", "destination": "s1", "event": "a", "timing": [0.2, 0.3, 0.35]},
            {"source": "s1", "destination": "s2", "event": "b", "timing": [0.5, 0.9]},
            {"source": "s2", "destination": "s0", "event": "c", "frequency": 4, "interval": [1.0, 1.4]},
        ],
    )


def build_multi_config_automaton() -> Automaton:
    return Automaton(
        id="multi_config",
        initial_q="q0",
        final_q=["q3"],
        configuration=None,
        states=[
            {"name": "q0", "enabled": {"normal": 1, "maintenance": 1, "degraded": 1}},
            {"name": "q1", "enabled": {"normal": 1, "maintenance": 0, "degraded": 1}},
            {"name": "q2", "enabled": {"normal": 0, "maintenance": 1, "degraded": 0}},
            {"name": "q3", "enabled": {"normal": 1, "maintenance": 1, "degraded": 1}},
        ],
        transitions=[
            {
                "source": "q0",
                "destination": "q1",
                "event": "boot",
                "enabled": {"normal": 1, "maintenance": 0, "degraded": 1},
                "frequency": {"normal": 12, "degraded": 4},
                "min_interval": {"normal": 0.2, "degraded": 0.8},
                "max_interval": {"normal": 0.7, "degraded": 1.4},
            },
            {
                "source": "q0",
                "destination": "q2",
                "event": "boot",
                "enabled": {"normal": 0, "maintenance": 1, "degraded": 0},
                "frequency": {"maintenance": 9},
                "interval": {"maintenance": [1.5, 2.5]},
            },
            {
                "source": "q1",
                "destination": "q3",
                "event": "finish",
                "enabled": {"normal": 1, "maintenance": 0, "degraded": 1},
                "frequency": {"normal": 8, "degraded": 2},
                "time": {"normal": 0.25, "degraded": 1.1},
            },
            {
                "source": "q2",
                "destination": "q3",
                "event": "finish_maint",
                "enabled": {"normal": 0, "maintenance": 1, "degraded": 0},
                "frequency": {"maintenance": 6},
                "min_interval": {"maintenance": 0.6},
                "max_interval": {"maintenance": 1.3},
            },
            {
                "source": "q3",
                "destination": "q0",
                "event": "reset",
                "enabled": {"normal": 1, "maintenance": 1, "degraded": 1},
                "frequency": {"normal": 7, "maintenance": 3, "degraded": 2},
                "interval": {"normal": [0.4, 0.7], "maintenance": [1.0, 1.8], "degraded": [1.6, 2.5]},
            },
        ],
    )


def build_callable_multi_config_automaton() -> Automaton:
    return Automaton(
        id="callable_multi_config",
        initial_q="idle",
        final_q=["idle"],
        configuration=None,
        states=[
            {"name": "idle", "enabled": {"fast": 1, "safe": 1}},
            {"name": "run", "enabled": {"fast": 1, "safe": 1}},
            {"name": "stop", "enabled": {"fast": 1, "safe": 1}},
        ],
        transitions=[
            {
                "source": "idle",
                "destination": "run",
                "event": "start",
                "enabled": lambda cfg, *_: cfg in {"fast", "safe"},
                "frequency": lambda cfg, *_: 14 if cfg == "fast" else 5,
                "time": lambda cfg, *_: 0.2 if cfg == "fast" else 0.9,
            },
            {
                "source": "run",
                "destination": "stop",
                "event": "halt",
                "enabled": {"fast": 1, "safe": 1},
                "frequency": {"fast": 10, "safe": 4},
                "min_interval": {"fast": 0.1, "safe": 0.6},
                "max_interval": {"fast": 0.3, "safe": 1.2},
            },
            {
                "source": "stop",
                "destination": "idle",
                "event": "reset",
                "enabled": {"fast": 1, "safe": 1},
                "frequency": {"fast": 9, "safe": 3},
                "interval": {"fast": [0.15, 0.4], "safe": [0.8, 1.5]},
            },
        ],
    )


def launch_visualizations(base_port: int = 8060, wait_between: float = 1.0) -> None:
    automaton_all_configs = build_multi_config_automaton()
    automaton_single_config = build_multi_config_automaton()
    automaton_single_config.set_configuration("normal")

    cases = [
        ("Simple (No Config Split)", build_simple_automaton()),
        ("All Configurations (Expanded Edges)", automaton_all_configs),
        ("Single Configuration = normal", automaton_single_config),
        ("Callable Config Fields (Expanded)", build_callable_multi_config_automaton()),
    ]

    for idx, (name, automaton) in enumerate(cases):
        port = base_port + idx
        print(f"[visual-test] opening '{name}' on http://127.0.0.1:{port}/")
        vis.plot_cps_component(
            automaton,
            output="dash",
            dash_port=port,
            node_labels=True,
            show_transition_freq=True,
            show_transition_timing=True,
            show_transition_data=["enabled", "frequency", "interval", "min_interval", "max_interval", "time"],
            transition_data_keys=True,
            split_edges_diff_event=True,
            freq_as_edge_thickness=True,
            title_text=f"plot_cps_component: {name}",
            max_zoom=2,
            min_zoom=0.4,
        )
        time.sleep(wait_between)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual visual tests for vis.plot_cps_component.")
    parser.add_argument("--base-port", type=int, default=8060, help="First Dash port to use.")
    parser.add_argument("--wait-between", type=float, default=1.0, help="Delay between opening apps.")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=120.0,
        help="How long to keep the process alive so Dash threads stay active.",
    )
    args = parser.parse_args()

    launch_visualizations(base_port=args.base_port, wait_between=args.wait_between)
    print(f"[visual-test] keeping apps alive for {args.hold_seconds} seconds")
    time.sleep(max(0.0, args.hold_seconds))


if __name__ == "__main__":
    main()
