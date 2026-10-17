from torch import nn
import torch.nn.functional as F
import torch
import pprint
import time as tm
from datetime import timedelta
import numpy as np
from torch.utils.data import DataLoader



class ThreeWayBinaryBinaryRBM (nn.Module):
    """
    Three-way Bernoulli-Bernoulli RBM with binary visible units, binary hidden
    units, and binary auxiliary/context units.

    This model extends a standard BinaryBinaryRBM by adding multiplicative
    three-way interactions between:
    - visible units `v`
    - hidden units `h`
    - auxiliary units `u`

    The three-way interaction tensor has shape:

        (n_visible, n_hidden, n_auxiliary)

    so that the hidden pre-activation receives an additional contribution of the form

        sum_{i,k} v_i * W3[i, j, k] * u_k

    Parameters
    ----------
    n_visible : int
        Number of visible units.
    n_hidden : int
        Number of hidden units.
    n_auxiliary : int
        Number of auxiliary/context units.
    device : str, optional
        Target device. Default is ``"cpu"``.

    Notes
    -----
    This class assumes input samples for training are arranged so that:
    - the visible state is the last slice along the final axis
    - the auxiliary state occupies the remaining slices along the final axis

    Concretely, the current training code expects something like:

        d[:, :, -1]   -> visible
        d[:, :, :-1]  -> auxiliary

    If your dataset uses a different layout, adapt the data parsing logic in
    :meth:`split_visible_auxiliary`.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        n_auxiliary: int,
        device: str = "cpu",
    ):
        """
        Initialize the three-way binary RBM.

        Parameters
        ----------
        n_visible : int
            Number of visible units.
        n_hidden : int
            Number of hidden units.
        n_auxiliary : int
            Number of auxiliary/context units.
        device : str, optional
            Target device. Default is ``"cpu"``.
        """
        super().__init__(n_visible=n_visible, n_hidden=n_hidden, device=device)

        self.n_auxiliary = n_auxiliary
        self.three_way_weights = nn.Parameter(
            torch.randn(n_visible, n_hidden, n_auxiliary) * 0.01
        )
        self.auxiliary_bias = nn.Parameter(torch.zeros(n_auxiliary))

        self.to(device)

    def split_visible_auxiliary(self, d: torch.Tensor):
        """
        Split a training tensor into visible and auxiliary parts.

        Parameters
        ----------
        d : torch.Tensor
            Input tensor containing both visible and auxiliary information.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple ``(v, u)`` where:
            - ``v`` is the visible state tensor
            - ``u`` is the auxiliary/context tensor

        Notes
        -----
        This preserves the layout convention from your original implementation:

        - ``v = d[:, :, -1]``
        - ``u = d[:, :, :-1]``

        Override or modify this helper if your actual data layout differs.
        """
        v = d[:, :, -1]
        u = d[:, :, :-1]
        return v, u

    def vu2h(
        self,
        v: torch.Tensor,
        u: torch.Tensor,
        pre_sigmoid: bool = False,
    ) -> torch.Tensor:
        """
        Compute hidden logits or probabilities from visible and auxiliary units.

        Parameters
        ----------
        v : torch.Tensor
            Visible states of shape ``(batch_size, n_visible)`` or flattenable to
            that shape.
        u : torch.Tensor
            Auxiliary states of shape ``(batch_size, n_auxiliary)`` or flattenable
            to that shape.
        pre_sigmoid : bool, optional
            If ``True``, return hidden logits.
            If ``False``, return hidden Bernoulli probabilities.
            Default is ``False``.

        Returns
        -------
        torch.Tensor
            Hidden logits or probabilities with shape
            ``(batch_size, n_hidden)``.
        """
        v = v.view(v.size(0), -1)
        u = u.view(u.size(0), -1)

        three_way_contrib = torch.einsum(
            "bi,ijk,bk->bj",
            v,
            self.three_way_weights,
            u,
        )
        hidden_logits = v @ self.weights + self.hidden_bias + three_way_contrib

        return hidden_logits if pre_sigmoid else torch.sigmoid(hidden_logits)

    def vh2u(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary-unit Bernoulli probabilities from visible and hidden units.

        Parameters
        ----------
        v : torch.Tensor
            Visible states with shape ``(batch_size, n_visible)``.
        h : torch.Tensor
            Hidden states or activations with shape ``(batch_size, n_hidden)``.

        Returns
        -------
        torch.Tensor
            Auxiliary Bernoulli probabilities with shape
            ``(batch_size, n_auxiliary)``.
        """
        v = v.view(v.size(0), -1)
        h = h.view(h.size(0), -1)

        auxiliary_logits = (
            torch.einsum("bi,bj,ijk->bk", v, h, self.three_way_weights)
            + self.auxiliary_bias
        )
        return torch.sigmoid(auxiliary_logits)

    def sample_u(self, auxiliary_probs: torch.Tensor) -> torch.Tensor:
        """
        Sample binary auxiliary states from auxiliary Bernoulli probabilities.

        Parameters
        ----------
        auxiliary_probs : torch.Tensor
            Auxiliary-unit Bernoulli probabilities.

        Returns
        -------
        torch.Tensor
            Sampled binary auxiliary states.
        """
        auxiliary_probs = torch.clamp(auxiliary_probs, 0.0, 1.0)
        return torch.bernoulli(auxiliary_probs)

    def hu2v(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute visible-unit Bernoulli probabilities from hidden and auxiliary units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden states or activations with shape ``(batch_size, n_hidden)``.
        u : torch.Tensor
            Auxiliary states with shape ``(batch_size, n_auxiliary)``.

        Returns
        -------
        torch.Tensor
            Visible Bernoulli probabilities with shape
            ``(batch_size, n_visible)``.
        """
        h = h.view(h.size(0), -1)
        u = u.view(u.size(0), -1)

        three_way_contrib = torch.einsum(
            "bj,ijk,bk->bi",
            h,
            self.three_way_weights,
            u,
        )
        visible_logits = h @ self.weights.t() + self.visible_bias + three_way_contrib
        return torch.sigmoid(visible_logits)

    def h2v(self, h: torch.Tensor) -> torch.Tensor:
        """
        Standard hidden-to-visible mapping inherited from the binary-binary RBM.

        Parameters
        ----------
        h : torch.Tensor
            Hidden states.

        Returns
        -------
        torch.Tensor
            Visible Bernoulli probabilities.

        Notes
        -----
        For the three-way model, visible reconstruction generally depends on both
        hidden and auxiliary units. This method is kept only to satisfy the
        BinaryBinaryRBM interface. The actual three-way visible conditional is
        implemented in :meth:`hu2v`.
        """
        return super().h2v(h)

    def sample_v(self, visible_probs: torch.Tensor) -> torch.Tensor:
        """
        Sample visible states from visible Bernoulli probabilities.

        Parameters
        ----------
        visible_probs : torch.Tensor
            Visible-unit Bernoulli probabilities.

        Returns
        -------
        torch.Tensor
            Sampled binary visible states.
        """
        visible_probs = torch.clamp(visible_probs, 0.0, 1.0)
        return torch.bernoulli(visible_probs)

    def free_energy(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute free energy for visible and auxiliary states.

        Parameters
        ----------
        v : torch.Tensor
            Visible states.
        u : torch.Tensor
            Auxiliary states.

        Returns
        -------
        torch.Tensor
            Per-sample free energy values with shape ``(batch_size,)``.

        Notes
        -----
        The free energy is computed as

            F(v, u) = -v^T b_v - u^T b_u - sum_j softplus(v^T W[:,j] + b_h[j] + T_j(v, u))

        where `T_j(v, u)` is the three-way contribution to hidden unit `j`.
        """
        v = v.view(v.size(0), -1)
        u = u.view(u.size(0), -1)

        visible_term = torch.sum(v * self.visible_bias, dim=1)
        auxiliary_term = torch.sum(u * self.auxiliary_bias, dim=1)

        three_way_contrib = torch.einsum(
            "bi,ijk,bk->bj",
            v,
            self.three_way_weights,
            u,
        )
        hidden_logits = v @ self.weights + self.hidden_bias + three_way_contrib
        hidden_term = torch.sum(F.softplus(hidden_logits), dim=1)

        return -(visible_term + auxiliary_term + hidden_term)

    def contrastive_divergence(
        self,
        d: torch.Tensor,
        num_gibbs: int = 1,
        use_probability_last_v_update: bool = False,
    ) -> torch.Tensor:
        """
        Compute CD-k loss for the three-way RBM.

        Parameters
        ----------
        d : torch.Tensor
            Input minibatch containing both visible and auxiliary parts.
        num_gibbs : int, optional
            Number of Gibbs steps. Default is ``1``.
        use_probability_last_v_update : bool, optional
            If ``True``, use visible probabilities rather than sampled visible
            states in the final negative step. Default is ``False``.

        Returns
        -------
        torch.Tensor
            Scalar CD-k loss.
        """
        v0, u0 = self.split_visible_auxiliary(d)

        with torch.no_grad():
            vk = v0.detach()
            uk = u0.detach()

            hk = self.sample_h(self.vu2h(vk, uk))

            for step in range(num_gibbs):
                visible_probs = self.hu2v(hk, uk)
                is_last = (step == num_gibbs - 1)

                if is_last and use_probability_last_v_update:
                    vk = visible_probs
                else:
                    vk = self.sample_v(visible_probs)

                uk = self.sample_u(self.vh2u(vk, hk))
                hk = self.sample_h(self.vu2h(vk, uk))

        return torch.mean(self.free_energy(v0, u0) - self.free_energy(vk, uk))

    @torch.no_grad()
    def generate(self, num_examples: int, num_steps: int = 10):
        """
        Generate visible and auxiliary samples by Gibbs sampling.

        Parameters
        ----------
        num_examples : int
            Number of samples to generate.
        num_steps : int, optional
            Number of Gibbs steps. Default is ``10``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple ``(v, u)`` of generated visible and auxiliary binary samples.
        """
        v = torch.randint(0, 2, (num_examples, self.n_visible), device=self.device).float()
        u = torch.randint(0, 2, (num_examples, self.n_auxiliary), device=self.device).float()

        for _ in range(num_steps):
            h = self.sample_h(self.vu2h(v, u))
            u = self.sample_u(self.vh2u(v, h))
            v = self.sample_v(self.hu2v(h, u))

        return v, u

    def _get_progress_rbm(self, d: torch.Tensor):
        """
        Compute monitoring metrics for the three-way RBM.

        Parameters
        ----------
        d : torch.Tensor
            Batch or dataset tensor containing both visible and auxiliary parts.

        Returns
        -------
        dict
            Dictionary containing summary metrics:
            - ``MSE``: mean squared visible reconstruction error
            - ``AuxCE``: mean squared auxiliary reconstruction proxy
            - ``Sparsity``: mean hidden activation
            - ``Energy``: mean free energy
            - ``Weights``: mean absolute pairwise weight magnitude
            - ``ThreeWayWeights``: mean absolute three-way weight magnitude
            - ``VisBias``: mean absolute visible bias magnitude
            - ``HidBias``: mean absolute hidden bias magnitude
            - ``AuxBias``: mean absolute auxiliary bias magnitude

        Notes
        -----
        This method evaluates reconstruction conditioned on the true auxiliary state.
        It also computes an auxiliary reconstruction proxy from ``p(u | v, h)``.
        """
        with torch.no_grad():
            v, u = self.split_visible_auxiliary(d)

            h_prob = self.vu2h(v, u)
            v_recon = self.hu2v(h_prob, u)
            u_recon = self.vh2u(v, h_prob)
            e = self.free_energy(v, u)

            progress = dict(
                MSE=torch.mean((v - v_recon) ** 2).item(),
                AuxCE=torch.mean((u - u_recon) ** 2).item(),
                Sparsity=torch.mean(h_prob).item(),
                Energy=torch.mean(e).item(),
                Weights=torch.mean(torch.abs(self.weights)).item(),
                ThreeWayWeights=torch.mean(torch.abs(self.three_way_weights)).item(),
                VisBias=torch.mean(torch.abs(self.visible_bias)).item(),
                HidBias=torch.mean(torch.abs(self.hidden_bias)).item(),
                AuxBias=torch.mean(torch.abs(self.auxiliary_bias)).item(),
            )

        return progress

