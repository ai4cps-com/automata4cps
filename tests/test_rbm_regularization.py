"""Checks for sparsity and within-window temporal persistence during RBM pretraining."""

from contextlib import redirect_stdout
import io
import math
import unittest
import warnings
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from plotly import graph_objects as go

from ml4cps.debta import base
from ml4cps.debta.rbms.bb import BinaryBinaryRBM
from ml4cps.debta.rbms.training import (train_rbm, _plot_hidden_unit_diagnostics,
                                       _save_learning_log, _loss_gradient_norms)


class RegularizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.rbm = BinaryBinaryRBM(2, 2)
        with torch.no_grad():
            self.rbm.weights.copy_(torch.eye(2))
            self.rbm.hidden_bias.zero_()

    def test_sparsity_uses_per_unit_mean(self):
        x = torch.tensor([[math.log(1 / 3), math.log(3.)]]).repeat(4, 1)
        penalty = self.rbm.regularization_loss(x, sparsity_target=.25, sparsity_weight=2.)
        target = torch.tensor(.25)
        mean = torch.tensor([.25, .75])
        expected = (target * (target / mean).log() +
                    (1 - target) * ((1 - target) / (1 - mean)).log()).mean() * 2
        torch.testing.assert_close(penalty, expected)
        penalty.backward()
        self.assertGreater(self.rbm.hidden_bias.grad[1].item(), 0)

    def test_sparsity_has_corrective_gradients_for_saturated_units(self):
        with torch.no_grad():
            self.rbm.weights.zero_()
            self.rbm.hidden_bias.copy_(torch.tensor([-1000., 1000.]))
        penalty = self.rbm.regularization_loss(torch.zeros(8, 2), sparsity_weight=1.)
        self.assertTrue(torch.isfinite(penalty))
        penalty.backward()
        # Gradient descent must raise the always-off bias and lower the always-on bias.
        torch.testing.assert_close(self.rbm.hidden_bias.grad, torch.tensor([-.125, .375]))

    def test_log_space_sparsity_matches_probability_formula_and_gradcheck(self):
        self.rbm.double()
        x = (torch.randn(4, 2, dtype=torch.float64) * 3).requires_grad_()
        actual = self.rbm.regularization_loss(x, sparsity_target=.25, sparsity_weight=1.)
        mean = self.rbm.v2h(x).mean(0)
        expected = (.25 * (.25 / mean).log() + .75 * (.75 / (1 - mean)).log()).mean()
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(
            lambda values: self.rbm.regularization_loss(values, sparsity_weight=1.), (x,),
        ))

    def test_gradient_diagnostics_preserve_training_gradients_and_do_not_resample(self):
        x = torch.rand(3, 4, 2)
        parts = self.rbm.compute_loss(x, sparsity_weight=.2,
                                      temporal_persistence_weight=.3, return_components=True)
        parameters = list(self.rbm.parameters())
        expected = torch.autograd.grad(parts['Loss'], parameters, retain_graph=True)
        sparse_bias = torch.autograd.grad(parts['WeightedSparsityKL'], self.rbm.hidden_bias,
                                          retain_graph=True)[0].norm().item()
        with patch.object(self.rbm, 'gibbs_chain', wraps=self.rbm.gibbs_chain) as chain:
            norms = _loss_gradient_norms(self.rbm, parts, weight_decay=.01)
        chain.assert_not_called()
        self.assertTrue(all(parameter.grad is None for parameter in parameters))
        self.assertAlmostEqual(norms['WeightedSparsityKLHiddenBiasGradNorm'], sparse_bias, places=7)
        expected_decay = .01 * torch.cat([p.detach().flatten() for p in parameters]).norm().item()
        self.assertAlmostEqual(norms['WeightDecayGradNorm'], expected_decay, places=7)
        self.assertTrue(all(math.isfinite(value) for value in norms.values()))
        parts['Loss'].backward()
        for parameter, gradient in zip(parameters, expected):
            torch.testing.assert_close(parameter.grad, gradient)

    def test_temporal_persistence_uses_all_adjacent_steps_inside_windows(self):
        x = torch.tensor([[[.2, -.3], [.4, .8], [-.4, .5]],
                          [[-1., 2.], [.3, -.7], [1., -2.]]], requires_grad=True)
        penalty = self.rbm.regularization_loss(x, temporal_persistence_weight=.7)
        logits = self.rbm.v2h(x.reshape(-1, 2), pre_sigmoid=True).reshape(2, 3, 2)
        expected = F.binary_cross_entropy_with_logits(
            logits[:, 1:, :], logits[:, :-1, :].sigmoid().detach(),
        ) * .7
        torch.testing.assert_close(penalty, expected)
        penalty.backward()
        torch.testing.assert_close(x.grad[:, 0], torch.zeros(2, 2))
        self.assertGreater(x.grad[:, 1:].abs().sum().item(), 0)

    def test_zero_weights_preserve_cd(self):
        x = torch.rand(5, 2)
        torch.manual_seed(10)
        expected = self.rbm.cd_loss(x, num_gibbs=3)
        torch.manual_seed(10)
        actual = self.rbm.compute_loss(x, num_gibbs=3)
        torch.testing.assert_close(actual, expected)

    def test_penalties_are_added_to_cd(self):
        x = torch.rand(2, 3, 2)
        options = dict(sparsity_target=.25, sparsity_weight=.2, temporal_persistence_weight=.3)
        torch.manual_seed(10)
        expected = self.rbm.cd_loss(x.reshape(-1, 2)) + self.rbm.regularization_loss(x, **options)
        torch.manual_seed(10)
        actual = self.rbm.compute_loss(x, **options)
        torch.testing.assert_close(actual, expected)

    def test_extreme_logits_and_one_step_windows(self):
        x = torch.tensor([[[100., -100.], [-100., 100.]]])
        penalty = self.rbm.regularization_loss(x, sparsity_weight=.1, temporal_persistence_weight=.2)
        self.assertTrue(torch.isfinite(penalty))
        penalty.backward()
        self.assertTrue(torch.isfinite(self.rbm.weights.grad).all())
        for inputs in (x[:, :1], x[:, 0]):
            self.assertEqual(self.rbm.regularization_loss(inputs, temporal_persistence_weight=.2).item(), 0.)

    def test_temporal_persistence_never_compares_different_windows(self):
        x = torch.tensor([[[.1, .2], [.3, .4]], [[-.8, .9], [-.7, .8]]])
        batched = self.rbm.regularization_loss(x, temporal_persistence_weight=.3)
        separately = sum(self.rbm.regularization_loss(window[None], temporal_persistence_weight=.3)
                         for window in x) / len(x)
        torch.testing.assert_close(batched, separately)
        torch.testing.assert_close(batched, self.rbm.regularization_loss(x.flip(0), temporal_persistence_weight=.3))

    def test_window_dataset_retains_all_windows_without_crossing_sequences(self):
        first, second = torch.arange(10).reshape(5, 2).float(), torch.arange(8).reshape(4, 2).float() + 100
        dataset = base.WindowedSequenceDataset([first, second], window_size=3)
        self.assertEqual(len(dataset), 5)
        self.assertEqual([len(x) for x in dataset.windowed_sequences()], [3, 2])
        self.assertEqual([len(x) for x in dataset.windowed_time_indices()], [3, 2])
        torch.testing.assert_close(dataset[2], first[-3:])
        torch.testing.assert_close(dataset[3], second[:3])
        torch.testing.assert_close(dataset[4], second[-3:])

    def test_cd_uses_every_window_step(self):
        x = torch.rand(3, 4, 2)
        torch.manual_seed(10)
        expected = self.rbm.cd_loss(x.reshape(-1, 2), num_gibbs=3)
        torch.manual_seed(10)
        actual = self.rbm.compute_loss(x, num_gibbs=3)
        torch.testing.assert_close(actual, expected)

    def test_window_persistence_warnings_once_at_training_start(self):
        cases = [
            (torch.rand(5, 2), .3, "has no effect with window_size=1"),
            (torch.rand(5, 1, 2), .3, "has no effect with window_size=1"),
            (torch.rand(5, 3, 2), 0., "temporal regularization is disabled"),
            (torch.rand(5, 3, 2), .3, None),
            (torch.rand(5, 2), 0., None),
        ]
        for inputs, weight, message in cases:
            with self.subTest(shape=inputs.shape, temporal_persistence_weight=weight), \
                 patch.object(go.Figure, 'show'), redirect_stdout(io.StringIO()), \
                 warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', UserWarning)
                train_rbm(self.rbm, inputs, max_epoch=2, batch_size=2,
                          temporal_persistence_weight=weight, verbose=False)
                relevant = [w for w in caught if 'temporal_persistence_weight' in str(w.message)]
                self.assertEqual(len(relevant), 1 if message else 0)
                if message:
                    self.assertIn(message, str(relevant[0].message))
                    self.assertEqual(relevant[0].category, UserWarning)

    def test_direct_window_dataset_training_and_metrics(self):
        dataset = base.WindowedSequenceDataset([torch.rand(5, 2)], window_size=3)
        with patch.object(go.Figure, 'show'), redirect_stdout(io.StringIO()):
            train_rbm(self.rbm, dataset, dataset, max_epoch=1, batch_size=2,
                      temporal_persistence_weight=.3, verbose=False)
        expected = self.rbm.reconstruction_loss(dataset.sequences.reshape(-1, 2)).item()
        self.assertAlmostEqual(self.rbm.learning_curve[-1]['MSE'], expected, places=6)
        self.assertAlmostEqual(self.rbm.valid_curve[-1]['MSE'], expected, places=6)
        _, histogram = _plot_hidden_unit_diagnostics(self.rbm, DataLoader(dataset, batch_size=2))
        self.assertTrue(all(sum(trace.y) == 9 for trace in histogram.data))

    def test_loss_components_match_total_without_resampling(self):
        x = torch.rand(3, 4, 2)
        options = dict(sparsity_target=.25, sparsity_weight=.2,
                       temporal_persistence_weight=.3)
        torch.manual_seed(10)
        expected = self.rbm.compute_loss(x, **options)
        torch.manual_seed(10)
        with patch.object(self.rbm, 'gibbs_chain', wraps=self.rbm.gibbs_chain) as chain:
            parts = self.rbm.compute_loss(x, return_components=True, **options)
        self.assertEqual(chain.call_count, 1)
        torch.testing.assert_close(parts['Loss'], expected)
        torch.testing.assert_close(parts['WeightedSparsityKL'], .2 * parts['SparsityKL'])
        torch.testing.assert_close(parts['WeightedTemporalBCE'], .3 * parts['TemporalBCE'])
        torch.testing.assert_close(parts['Loss'], parts['CD'] + parts['WeightedSparsityKL']
                                   + parts['WeightedTemporalBCE'])
        parts = self.rbm.compute_loss(x, return_components=True)
        self.assertGreater(parts['SparsityKL'].item(), 0.)
        self.assertGreater(parts['TemporalBCE'].item(), 0.)
        self.assertEqual(parts['WeightedSparsityKL'].item(), 0.)
        self.assertEqual(parts['WeightedTemporalBCE'].item(), 0.)
        torch.testing.assert_close(parts['Loss'], parts['CD'])

    def test_learning_log_is_saved_and_console_output_is_preserved(self):
        self.rbm.name = 'layer 1'

        @_save_learning_log
        def learning(rbm, log_mlflow=False):
            print('Epoch 1/1: CD=0.2, SparsityKL=0.1, TemporalBCE=0.3')
            warnings.warn('example learning warning', UserWarning)
            return 42

        console, errors = io.StringIO(), io.StringIO()
        from contextlib import redirect_stderr
        with patch('ml4cps.debta.rbms.training.mlflow.log_text') as log_text, \
             redirect_stdout(console), redirect_stderr(errors), warnings.catch_warnings():
            warnings.simplefilter('always')
            self.assertEqual(learning(self.rbm, True), 42)
        text, artifact = log_text.call_args.args
        self.assertEqual(artifact, 'logs/layer_1_learning.log')
        self.assertIn(console.getvalue(), text)
        self.assertIn('example learning warning', errors.getvalue())
        self.assertIn('example learning warning', text)
        with patch('ml4cps.debta.rbms.training.mlflow.log_text') as log_text, \
             redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            learning(self.rbm, log_mlflow=False)
        log_text.assert_not_called()

    def test_partial_learning_log_is_saved_on_failure(self):
        @_save_learning_log
        def learning(rbm, log_mlflow=False):
            print('Epoch started')
            raise RuntimeError('example training failure')

        with patch('ml4cps.debta.rbms.training.mlflow.log_text') as log_text, \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'example training failure'):
            learning(self.rbm, log_mlflow=True)
        text = log_text.call_args.args[0]
        self.assertIn('Epoch started', text)
        self.assertIn('Traceback', text)
        self.assertIn('example training failure', text)

    def test_epoch_sparsity_is_global_and_loss_means_weight_partial_batches(self):
        inputs = torch.cat((torch.zeros(4, 3, 2), torch.full((1, 3, 2), 10.)))
        observed = []
        original = self.rbm.compute_loss

        def capture(x, **kwargs):
            parts = original(x, **kwargs)
            observed.append((len(x), {key: value.detach().item() for key, value in parts.items()}))
            return parts

        with patch.object(self.rbm, 'compute_loss', side_effect=capture), \
             patch.object(go.Figure, 'show'), redirect_stdout(io.StringIO()), \
             patch('ml4cps.debta.rbms.training.mlflow.log_param'), \
             patch('ml4cps.debta.rbms.training.mlflow.log_figure'), \
             patch('ml4cps.debta.rbms.training.mlflow.log_text'), \
             patch('ml4cps.debta.rbms.training.mlflow.log_metrics') as log_metrics:
            train_rbm(self.rbm, inputs, inputs, max_epoch=1, batch_size=2, shuffle=False,
                      lr=0., sparsity_weight=.2, temporal_persistence_weight=.3,
                      log_mlflow=True, verbose=False)
        progress = self.rbm.learning_curve[-1]
        expected = self.rbm.v2h(inputs.reshape(-1, 2)).mean().item()
        self.assertAlmostEqual(progress['Sparsity'], expected, places=6)
        self.assertGreater(abs(progress['Sparsity'] - self.rbm.v2h(inputs[:2].reshape(-1, 2)).mean().item()), .05)
        self.assertEqual([count for count, _ in observed], [2, 2, 1])
        for key in observed[0][1]:
            weighted = sum(count * parts[key] for count, parts in observed) / len(inputs)
            self.assertAlmostEqual(progress['UpdateMean' + key], weighted, places=5)
        train_calls = [call for call in log_metrics.call_args_list
                       if '/train/Sparsity' in call.args[0]]
        self.assertEqual(len(train_calls), 1)
        logged = train_calls[0].args[0]
        self.assertIn('/train/UpdateMeanCD', logged)
        self.assertIn('/train/Sparsity', logged)
        self.assertTrue(all(value is not None for value in logged.values()))
        self.assertEqual(train_calls[0].kwargs['step'], 1)
        self.assertIn('/train/hidden/h1/mean', logged)
        self.assertIn('/train/hidden/h1/high_fraction', logged)
        stats = self.rbm.hidden_activation_curve[-1]
        torch.testing.assert_close(torch.tensor(stats['mean']),
                                   self.rbm.v2h(inputs.reshape(-1, 2)).mean(0))
        torch.testing.assert_close(torch.tensor(stats['high_fraction']), torch.tensor([.2, .2]))
        torch.testing.assert_close(torch.tensor(stats['low_fraction']), torch.zeros(2))
        self.assertEqual(len(self.rbm.gradient_curve), 3)
        self.assertTrue(any('/gradients/CDHiddenBiasGradNorm' in call.args[0]
                            for call in log_metrics.call_args_list))

    def test_pretraining_forwards_penalties_to_every_layer(self):
        model = base.DebtaEbm(num_y=2, num_h=2, num_sigm_layers=2,
                              window_size=3, sparsity_target=.25, sparsity_weight=.2, temporal_persistence_weight=.3)
        dataset = base.WindowedSequenceDataset([torch.rand(3, 2), torch.rand(4, 2)], window_size=3)
        recorded = []
        original = base.train_rbm

        def capture(rbm, train, valid, **kwargs):
            recorded.append(kwargs)
            with patch.object(rbm, 'compute_loss', wraps=rbm.compute_loss) as compute:
                result = original(rbm, train, valid, **kwargs)
            self.assertTrue(all(c.args[0].ndim == 3 and c.args[0].shape[1] == 3
                                for c in compute.call_args_list))
            self.assertTrue(all('previous_visible' not in c.kwargs
                                for c in compute.call_args_list))
            return result

        with patch.object(base, 'train_rbm', side_effect=capture), \
             patch.object(go.Figure, 'show'), redirect_stdout(io.StringIO()):
            model.pretrain_layers(dataset, dataset, max_epoch=1, batch_size=3, verbose=False)
        self.assertEqual(len(recorded), 2)
        for options in recorded:
            self.assertEqual(options['sparsity_target'], .25)
            self.assertEqual(options['sparsity_weight'], .2)
            self.assertEqual(options['temporal_persistence_weight'], .3)
            self.assertNotIn('sequence_starts', options)


if __name__ == '__main__':
    unittest.main()
