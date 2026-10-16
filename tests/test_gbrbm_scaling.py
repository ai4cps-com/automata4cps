"""Regression checks for the shared Gaussian RBM / DEBTA parameterization."""

import itertools
import unittest

import torch

from ml4cps.debta.base import DebtaEbm
from ml4cps.debta.rbms.gb import GaussianBinaryRBM


class GaussianScalingTests(unittest.TestCase):
    def test_transfer_preserves_encoder_decoder_and_free_energy(self):
        for sigma in (torch.tensor([[0.5, 0.5]]), torch.tensor([[0.3, 0.8]])):
            with self.subTest(sigma=sigma):
                rbm = GaussianBinaryRBM(2, 2, log_sigma_v=sigma.log())
                ebm = DebtaEbm(num_y=2, num_h=2, sigma=0.5, transience=False)
                with torch.no_grad():
                    rbm.weights.copy_(torch.tensor([[0.3, 0.1], [-0.2, 0.4]]))
                    rbm.hidden_bias.copy_(torch.tensor([0.2, -0.3]))
                    rbm.visible_bias.copy_(torch.tensor([0.1, -0.2]))
                    ebm.weights(level=1).copy_(rbm.weights.T)
                    ebm.hidden_bias(level=1).copy_(rbm.hidden_bias)
                    ebm.visible_bias(level=1).copy_(rbm.visible_bias)
                    ebm.log_sigma_x.copy_(rbm.log_sigma_v)
                x = torch.tensor([[1., -1.], [0.2, 0.4]])
                h = torch.tensor([[0.2, 0.8], [1., 0.]])
                torch.testing.assert_close(rbm.v2h(x), ebm.encode(x))
                torch.testing.assert_close(rbm.v2h(x[:, None, :]), rbm.v2h(x))
                torch.testing.assert_close(rbm.h2v(h), ebm.decode(h).reshape_as(x))
                torch.testing.assert_close(rbm.free_energy(x), ebm.free_energy(x))

    def test_free_energy_matches_explicit_hidden_marginal(self):
        rbm = GaussianBinaryRBM(2, 2, log_sigma_v=torch.tensor([[0.3, 0.8]]).log())
        x = torch.tensor([[1., -1.], [0.2, 0.4]])
        hidden = torch.tensor(list(itertools.product([0., 1.], repeat=2)))
        visible_energy = ((x - rbm.visible_bias).square() / (2 * rbm.sigma_v().square())).sum(1)
        logits = (x / rbm.sigma_v()) @ rbm.weights + rbm.hidden_bias
        energies = visible_energy[:, None] - logits @ hidden.T
        torch.testing.assert_close(rbm.free_energy(x), -torch.logsumexp(-energies, dim=1))


if __name__ == '__main__':
    unittest.main()
