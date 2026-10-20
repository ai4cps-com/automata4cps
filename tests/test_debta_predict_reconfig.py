"""Regression checks for configuration recommendations from latent transitions."""

import unittest
from unittest.mock import patch

import pandas as pd

from ml4cps.debta import DEBTA


class PredictReconfigTests(unittest.TestCase):
    def setUp(self):
        self.model = DEBTA(num_y=2, num_h=2, num_sigm_layers=1)
        self.model._G.add_edge('a', 'b', timing={
            'Config[0,0]': [1] * 20, 'Config[0,1]': [1] * 3, 'Config[1,0]': [1] * 2,
            None: [1] * 30, 'Config[2,2]': [],
        })
        self.model._G.add_edge('a', 'c', timing={'Config[1,0]': [1] * 3})
        self.model._G.add_edge('b', 'c', timing={'Config[1,1]': [1] * 50})
        self.model._G.add_edge('a', 'd', timing=[1, 2])

    def test_rank_aggregate_exclude_and_use_final_mode(self):
        self.model.set_configuration('Config[0,0]')
        with patch.object(self.model, 'predict', return_value=pd.Series(['b', 'a'])):
            self.assertEqual(self.model.predict_reconfig([[0., 0.]], 'Config[0,0]'),
                             [(1, 0), (0, 1)])
            self.assertEqual(self.model.predict_reconfig([[0., 0.]], 'Config[0,0]', n=1), [(1, 0)])
            self.assertEqual(self.model.predict_reconfig([[0., 0.]], 'Config[0,0]', n=10),
                             [(1, 0), (0, 1)])
        self.assertEqual(self.model.configuration, 'Config[0,0]')

    def test_unknown_empty_and_unlabelled_modes(self):
        for modes in ([], ['unknown'], ['c']):
            with self.subTest(modes=modes), patch.object(self.model, 'predict', return_value=modes):
                self.assertEqual(self.model.predict_reconfig([], 'Config[0,0]', n=2), [])

    def test_sensor_input_with_real_encoder(self):
        inputs = [[0., 0.], [1., 1.]]
        state = self.model.predict(None, inputs).iloc[-1]
        self.model._G.add_edge(state, 'destination', timing={'Config[1,2]': [1., 2.]})
        self.assertEqual(self.model.predict_reconfig(inputs, 'Config[0,0]', n=2),
                         [(1, 2)])

    def test_zero_and_invalid_limits(self):
        with patch.object(self.model, 'predict') as predict:
            self.assertEqual(self.model.predict_reconfig([], 'Config[0,0]', n=0), [])
            predict.assert_not_called()
        for n in (-1, 1.5, True, '2'):
            with self.subTest(n=n), self.assertRaises(ValueError):
                self.model.predict_reconfig([], 'Config[0,0]', n=n)

    def test_allowed_values_filter_and_tuple_current_configuration(self):
        with patch.object(self.model, 'predict', return_value=['a']):
            self.assertEqual(self.model.predict_reconfig([], (0, 0), n=3,
                                                        allowed_values=[(0,), (0, 1)]),
                             [(0, 1)])
            self.assertEqual(self.model.predict_reconfig([], (0, 0), n=3,
                                                        allowed_values=[(), (0, 1)]), [])
            for allowed in ([(0,)], [[0], [1]], ((0,), (1,))):
                with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                    self.model.predict_reconfig([], (0, 0), allowed_values=allowed)

    def test_legacy_and_tuple_labels_merge(self):
        self.model._G.clear()
        self.model._G.add_edge('a', 'b', timing={'Config2': [1], (2,): [2], (3,): [3]})
        with patch.object(self.model, 'predict', return_value=['a']):
            self.assertEqual(self.model.predict_reconfig([], 'Config1', n=5), [(2,), (3,)])


if __name__ == '__main__':
    unittest.main()
