"""Configuration alignment and frequency/color plotting regression checks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ml4cps import vis
from ml4cps.automata import learn


def load_script():
    path = Path('D:/Repos/kiss_bitbucket/reconfig/scripts/train_debta_models.py')
    spec = importlib.util.spec_from_file_location('train_debta_config_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigurationPlotTests(unittest.TestCase):
    def test_configuration_alignment_uses_window_end_indices(self):
        config = torch.tensor([[0., 1.], [0., 1.], [1., 0.], [1., 0.], [0., 1.], [1., 1.]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.pt'
            torch.save(config, path)
            labels = load_script().load_configuration_labels(path, 6, 5, np.array([2, 3, 4]))
        self.assertEqual(labels.tolist(), ['Config[1,0]', 'Config[1,0]', 'Config[0,1]'])
        self.assertEqual(labels.index.tolist(), [2, 3, 4])

    def test_misaligned_configuration_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.pt'
            torch.save(torch.zeros(3, 2), path)
            with self.assertRaisesRegex(ValueError, 'same number of rows'):
                load_script().load_configuration_labels(path, 5, 4, np.arange(4))

    def test_configuration_edges_have_distinct_colors_and_frequency_widths(self):
        states = pd.DataFrame({'h1': [0, 1, 0, 1, 0, 1, 0, 1, 0]})
        labels = pd.Series(['Config[0]'] * 6 + ['Config[1]'] * 3)
        automaton = learn.simple_learn_from_signal_vectors([states], config=[labels])
        automaton.set_configuration(None)
        elements = vis.plot_cps_component(
            automaton, output='elements', show_transition_freq=True,
            freq_as_edge_thickness=True,
        )
        edges = [edge['data'] for edge in elements['edges'] if edge['data'].get('freq', 0) > 0]
        colors = {}
        for edge in edges:
            colors.setdefault(edge['config'], set()).add(edge['line_color'])
        self.assertEqual(set(colors), {'Config[0]', 'Config[1]'})
        self.assertTrue(all(len(values) == 1 for values in colors.values()))
        self.assertNotEqual(colors['Config[0]'], colors['Config[1]'])
        smallest = min(edges, key=lambda edge: edge['freq'])
        largest = max(edges, key=lambda edge: edge['freq'])
        self.assertLess(smallest['freq'], largest['freq'])
        self.assertLess(smallest['thickness'], largest['thickness'])

    def test_palette_does_not_repeat_after_ten_configurations(self):
        states = pd.DataFrame({'h1': [0, 1] * 30})
        labels = pd.Series([f'Config[{i % 28}]' for i in range(len(states))])
        automaton = learn.simple_learn_from_signal_vectors([states], config=[labels])
        automaton.configurations = tuple(sorted(labels.unique()))
        automaton.set_configuration(None)
        elements = vis.plot_cps_component(automaton, output='elements', freq_as_edge_thickness=True)
        edges = [edge['data'] for edge in elements['edges'] if edge['data'].get('freq', 0) > 0]
        color_map = {edge['config']: edge['line_color'] for edge in edges}
        self.assertEqual(len(color_map), 28)
        self.assertEqual(len(set(color_map.values())), 28)
        # Removing a configuration's edges keeps the colors of the others.
        for source, target, key, data in list(automaton.get_transitions()):
            timing = data.get('timing', {})
            if isinstance(timing, dict):
                timing.pop('Config[0]', None)
        elements = vis.plot_cps_component(automaton, output='elements', freq_as_edge_thickness=True)
        for edge in elements['edges']:
            if edge['data'].get('freq', 0) > 0:
                self.assertEqual(edge['data']['line_color'], color_map[edge['data']['config']])


if __name__ == '__main__':
    unittest.main()
