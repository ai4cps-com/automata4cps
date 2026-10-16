"""Small execution checks for RBM sampling, CD gradients and the training script."""

from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from plotly import graph_objects as go

from ml4cps.debta import base
from ml4cps.debta.rbms.gb import GaussianBinaryRBM
from ml4cps.debta.rbms.bb import BinaryBinaryRBM


class ExecutionAuditTests(unittest.TestCase):
    def test_cd_gradient_and_gibbs_steps(self):
        torch.manual_seed(123)
        for cls in (GaussianBinaryRBM, BinaryBinaryRBM):
            rbm = cls(3, 2)
            if isinstance(rbm, GaussianBinaryRBM):
                rbm.log_sigma_v.copy_(torch.tensor([[.3, .5, .8]]).log())
            x = torch.rand(7, 3)
            for use_mean in (False, True):
                with patch.object(rbm, 'sample_h', wraps=rbm.sample_h) as sample_h, \
                     patch.object(rbm, 'sample_v', wraps=rbm.sample_v) as sample_v, \
                     patch.object(rbm, 'gibbs_chain', wraps=rbm.gibbs_chain) as chain:
                    loss = rbm.compute_loss(x, num_gibbs=10,
                                           use_probability_last_v_update=use_mean)
                    self.assertEqual(sample_h.call_count, 10)
                    self.assertEqual(sample_v.call_count, 9 if use_mean else 10)
                    self.assertFalse(chain.call_args.kwargs['v_init'].requires_grad)
                # Fix the negative samples to check the CD surrogate analytically.
                negative = torch.rand_like(x)
                with patch.object(rbm, 'gibbs_chain', return_value=negative):
                    loss = rbm.cd_loss(x)
                rbm.zero_grad()
                loss.backward()
                scale = rbm.sigma_v() if isinstance(rbm, GaussianBinaryRBM) else 1.
                positive_h = rbm.v2h(x).detach()
                negative_h = rbm.v2h(negative).detach()
                expected_w = ((negative / scale).T @ negative_h -
                              (x / scale).T @ positive_h) / len(x)
                torch.testing.assert_close(rbm.weights.grad, expected_w)
                torch.testing.assert_close(rbm.hidden_bias.grad,
                                           (negative_h - positive_h).mean(0))
                if isinstance(rbm, GaussianBinaryRBM):
                    expected_b = ((negative - x) / scale.square()).mean(0)
                else:
                    expected_b = (negative - x).mean(0)
                torch.testing.assert_close(rbm.visible_bias.grad, expected_b)

    def test_sampling_moments(self):
        torch.manual_seed(123)
        rbm = GaussianBinaryRBM(2, 2, log_sigma_v=torch.tensor([[.3, .8]]).log())
        mean = torch.tensor([[.2, -.4]]).expand(100000, -1)
        samples = rbm.sample_v(mean)
        torch.testing.assert_close(samples.mean(0), mean[0], atol=.01, rtol=0)
        torch.testing.assert_close(samples.std(0), rbm.sigma_v()[0], atol=.01, rtol=0)
        hidden = rbm.sample_h(torch.tensor([[.2, .8]]).expand(100000, -1))
        torch.testing.assert_close(hidden.mean(0), torch.tensor([.2, .8]), atol=.01, rtol=0)

    def test_script_both_presets(self):
        script_path = Path('D:/Repos/kiss_bitbucket/reconfig/scripts/train_debta_models.py')
        spec = importlib.util.spec_from_file_location('audit_train_debta', script_path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        original_train = base.train_rbm
        for preset in ('shallow', 'deep'):
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                data = torch.randn(97, 16) * 2
                data_path = Path(directory) / 'data.pt'
                torch.save(data, data_path)
                config = torch.stack((torch.arange(len(data)) // 13 % 2,
                                      torch.arange(len(data)) // 23 % 2), dim=1).float()
                torch.save(config, Path(directory) / 'config.pt')
                settings = script.PRESETS[preset].copy()
                settings.update(max_epoch=1, batch_size=16)
                stack.enter_context(patch.dict(script.PRESETS, {preset: settings}))
                stack.enter_context(patch('torch.cuda.is_available', return_value=False))
                for name in ('set_tracking_uri', 'start_run', 'active_run', 'log_params',
                             'log_param', 'log_metric', 'log_metrics', 'log_figure'):
                    stack.enter_context(patch.object(script.mlflow, name))
                learning_logs = stack.enter_context(patch.object(script.mlflow, 'log_text'))
                log_model = stack.enter_context(patch.object(script.mlflow.pyfunc, 'log_model'))
                show = stack.enter_context(patch.object(go.Figure, 'show'))
                automaton_plot = stack.enter_context(patch.object(script.vis, 'plot_cps_component'))
                stack.enter_context(patch.object(base.vis, 'plot_cps_plotly', return_value=go.Figure()))
                recorded = []

                def checked_train(rbm, train, valid, **kwargs):
                    x = train.squeeze(1)
                    recorded.append((rbm, x.clone(), valid.clone(), kwargs.copy()))
                    with patch.object(rbm, 'compute_loss', wraps=rbm.compute_loss) as compute:
                        result = original_train(rbm, train, valid, **kwargs)
                    seen = torch.cat([call.args[0] for call in compute.call_args_list])
                    self.assertEqual(len(seen), len(x))
                    torch.testing.assert_close(seen.reshape_as(x).sort(dim=0).values, x.sort(dim=0).values)
                    self.assertTrue(all(not call.kwargs['use_probability_last_v_update']
                                        for call in compute.call_args_list))
                    return result

                stack.enter_context(patch.object(base, 'train_rbm', side_effect=checked_train))
                with redirect_stdout(io.StringIO()):
                    script.main(['--data', str(data_path), '--preset', preset,
                                 '--tracking-uri', 'file:///' + directory.replace('\\', '/')])
                self.assertEqual(len(recorded), settings['num_hidden_layers'])
                self.assertEqual(learning_logs.call_count, settings['num_hidden_layers'])
                for layer, call in enumerate(learning_logs.call_args_list, start=1):
                    text, artifact = call.args
                    self.assertEqual(artifact, f'logs/layer_{layer}_learning.log')
                    for label in ('UpdateMeanCD', 'SparsityKL', 'TemporalBCE',
                                  'Hidden biases', 'Hidden activation means',
                                  'Hidden fraction <= 0.01', 'Hidden fraction >= 0.99',
                                  'CDHiddenBiasGradNorm', 'Valid:', 'Training finished'):
                        self.assertIn(label, text)
                split = int(len(data) * settings['train_fraction'])
                window = settings.get('window_size', 1)
                expected_windows = torch.stack([data[i:i + window] for i in range(split - window + 1)])
                torch.testing.assert_close(recorded[0][1], expected_windows.squeeze(1))
                for rbm, x, valid, kwargs in recorded:
                    self.assertEqual(kwargs['num_gibbs'], settings.get('num_gibbs', 1))
                    self.assertEqual(kwargs['weight_decay'], settings.get('weight_decay', 0.))
                    self.assertEqual(kwargs['lr'], settings['lr'])
                    self.assertTrue(kwargs['log_mlflow'])
                    self.assertEqual(kwargs['loss_name'], 'cd')
                    expected = rbm.reconstruction_loss(x.reshape(-1, x.shape[-1])).item()
                    self.assertAlmostEqual(rbm.learning_curve[-1]['MSE'], expected, places=6)
                    self.assertTrue(math.isfinite(rbm.valid_curve[-1]['MSE']))
                    print(f'{preset} {rbm.name}: rows={len(x)}, '
                          f'batches={math.ceil(len(x)/16)}, gibbs={kwargs["num_gibbs"]}, '
                          f'weight_decay={kwargs["weight_decay"]}, MSE={expected:.6f}')
                model = log_model.call_args.kwargs['python_model']
                self.assertEqual(model.ebm.num_h(), settings['num_h'])
                self.assertEqual(automaton_plot.call_count, 2)
                for call in automaton_plot.call_args_list:
                    self.assertTrue(call.kwargs['freq_as_edge_thickness'])
                    self.assertTrue(call.kwargs['show_transition_freq'])
                print(f'{preset}: script completed; browser shows={show.call_count}; '
                      f'automaton states after pruning={model.num_modes}')


if __name__ == '__main__':
    unittest.main()
