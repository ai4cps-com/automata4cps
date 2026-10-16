import itertools
import math
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from ml4cps.debta.rbms.base import XBinaryRBM


class GaussianBinaryRBM(XBinaryRBM):
    """
    Gaussian-Binary Restricted Boltzmann Machine.

    This RBM uses:
    - Gaussian visible units
    - binary hidden units

    The conditional distributions are:

    .. math::
        p(h_j = 1 \\mid v) = \\sigma\\left(\\sum_i \\frac{v_i}{\\sigma_i} W_{ij} + b_j^h\\right)

    and

    .. math::
        p(v \\mid h) = \\mathcal{N}(\\mu(h), \\operatorname{diag}(\\sigma^2))

    where the conditional visible mean is

    .. math::
        \\mu(h) = \\sigma \\odot (h W^T) + b^v

    Parameters
    ----------
    n_visible : int
        Number of visible units.
    n_hidden : int
        Number of hidden units.
    device : str, optional
        Device on which the module parameters and buffers are stored.
        Default is ``"cpu"``.
    is_sigma_learnable : bool, optional
        Whether the visible standard deviations are learnable.
        If ``True``, ``log_sigma_v`` is an ``nn.Parameter``.
        If ``False``, it is stored as a non-trainable buffer.
        Default is ``False``.
    sigma : float, optional
        Initial value of the visible standard deviation for all visible units.
        Default is ``1.0``.

    Attributes
    ----------
    is_sigma_learnable : bool
        Whether the visible standard deviation is learnable.
    log_sigma_v : torch.Tensor or nn.Parameter
        Log-standard-deviation of the visible units, stored with shape
        ``(1, n_visible)``.

    Notes
    -----
    - The model assumes a diagonal covariance matrix for the visible units.
    - When ``is_sigma_learnable=False``, the visible noise level is fixed.
    - The weight matrix is reinitialized after calling the base constructor to
      preserve the original Gaussian-RBM scaling convention.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        device: str = "cpu",
        sigma: float = 1.0,
        log_sigma_v = None,
        name='',
        is_sigma_learnable: bool = False,
    ):
        """
        Initialize the Gaussian-Binary RBM.

        Parameters
        ----------
        n_visible : int
            Number of visible units.
        n_hidden : int
            Number of hidden units.
        device : str, optional
            Target device. Default is ``"cpu"``.
        sigma : float, optional
            Initial standard deviation used for all visible units.
        is_sigma_learnable : bool, optional
            Whether to train the visible standard deviations. Defaults to False.

        Notes
        -----
        The visible log-standard-deviation is stored in ``log_sigma_v`` rather
        than ``sigma_v`` directly to ensure positivity through exponentiation.
        """
        super().__init__(n_visible, n_hidden, name=name, device=device)
        self.is_sigma_learnable = is_sigma_learnable

        if log_sigma_v is not None:
            log_sigma_init = log_sigma_v.detach().clone()
            sigma_value = torch.exp(log_sigma_init).mean().item()
        else:
            sigma_value = float(sigma)
            log_sigma_init = torch.full((1, n_visible), math.log(sigma_value), dtype=torch.float32)

        if self.is_sigma_learnable:
            self.log_sigma_v = nn.Parameter(log_sigma_init)
        else:
            self.register_buffer("log_sigma_v", log_sigma_init)

        # Reinitialize weights using the original Gaussian-RBM scaling.
        self.weights = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01 * sigma_value)
        self.to(device)

    def _init_args(self) -> dict:
        init_args = super()._init_args()
        init_args.update(
            {
                "is_sigma_learnable": self.is_sigma_learnable,
                "sigma": float(torch.exp(self.log_sigma_v.detach()).mean().cpu()),
            }
        )
        return init_args

    def sigma_v(self, visible: torch.Tensor = None) -> torch.Tensor:
        """
        Return the visible-unit standard deviations.

        Parameters
        ----------
        visible : torch.Tensor, optional
            Optional visible tensor used only to adapt the returned tensor shape
            for broadcasting. If ``visible`` is 3-dimensional, the returned
            standard deviations are unsqueezed along the last dimension.

        Returns
        -------
        torch.Tensor
            Tensor containing the visible-unit standard deviations.

        Notes
        -----
        This method computes

        .. math::
            \\sigma_v = \\exp(\\log\\sigma_v)

        so that standard deviations remain strictly positive.
        """
        sigma_v = torch.exp(self.log_sigma_v)

        if visible is not None and visible.dim() == 3:
            sigma_v = sigma_v.unsqueeze(2)

        return sigma_v

    def h2v(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the conditional visible mean given hidden units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden activations or sampled hidden states with shape
            ``(batch_size, n_hidden)``.

        Returns
        -------
        torch.Tensor
            Conditional mean of the visible Gaussian distribution with shape
            ``(batch_size, n_visible)``.

        Notes
        -----
        The conditional mean is

        .. math::
            \\mu(h) = \\sigma \\odot (h W^T) + b^v

        This method returns the mean of ``p(v | h)``, not a sampled visible state.
        """
        visible_mean = (h @ self.weights.t()) * self.sigma_v() + self.visible_bias
        return visible_mean

    def v2h(self, v: torch.Tensor, pre_sigmoid: bool = False) -> torch.Tensor:
        """
        Compute hidden logits or hidden activation probabilities given visible units.

        Parameters
        ----------
        v : torch.Tensor
            Visible samples.
        pre_sigmoid : bool, optional
            If ``True``, return pre-sigmoid hidden logits.
            If ``False``, return hidden Bernoulli probabilities.
            Default is ``False``.

        Returns
        -------
        torch.Tensor
            Hidden logits or hidden probabilities with shape
            ``(batch_size, n_hidden)``.

        Notes
        -----
        For Gaussian visible units, the visible input is first rescaled by the
        inverse standard deviation:

        .. math::
            \\tilde{v}_i = \\frac{v_i}{\\sigma_i}

        and the hidden conditional is then computed using the base binary-hidden
        implementation.
        """
        v = v.reshape(v.size(0), -1)
        sigma_v = self.sigma_v(v)
        scaled_v = v / sigma_v
        return super().v2h(scaled_v, pre_sigmoid=pre_sigmoid)

    def sample_v(self, visible_mean: torch.Tensor) -> torch.Tensor:
        """
        Sample visible states from the Gaussian visible conditional distribution.

        Parameters
        ----------
        visible_mean : torch.Tensor
            Mean of the visible Gaussian conditional distribution, typically the
            output of :meth:`h2v`.

        Returns
        -------
        torch.Tensor
            Sampled visible states with the same shape as ``visible_mean``.

        Notes
        -----
        Sampling is performed as

        .. math::
            v = \\mu + \\sigma \\odot \\epsilon

        where ``epsilon ~ N(0, I)``.
        """
        sigma_v = self.sigma_v(visible_mean)
        return visible_mean + sigma_v * torch.randn_like(visible_mean)

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy of a batch of visible samples.

        Parameters
        ----------
        v : torch.Tensor
            Visible samples of shape ``(batch_size, n_visible)`` or any tensor
            flattenable to that shape.

        Returns
        -------
        torch.Tensor
            Per-sample free-energy values with shape ``(batch_size,)``.

        Notes
        -----
        This implementation computes

        .. math::
            F(v) =
            \\sum_i \\frac{(v_i - b_i^v)^2}{2 \\sigma_i^2}
            -
             \\sum_j
            \\log\\left(1 + \\exp\\left(\\sum_i \\frac{v_i}{\\sigma_i} W_{ij} + b_j^h\\right)\\right)

        using ``softplus(z) = log(1 + exp(z))`` for numerical stability.

        The returned tensor contains one free-energy value per sample.
        """
        v = v.view(v.size(0), -1)
        sigma_v = self.sigma_v(v)

        visible_term = torch.sum(
            ((v - self.visible_bias) ** 2) / (2.0 * sigma_v.pow(2)),
            dim=1,
        )
        hidden_logits = self.v2h(v, pre_sigmoid=True)
        hidden_term = torch.sum(F.softplus(hidden_logits), dim=1)

        return visible_term - hidden_term

    @torch.no_grad()
    def gmm_model(self):
        """
        Convert the model marginal ``p(v)`` into an explicit Gaussian mixture model.

        Returns
        -------
        tuple
            A 5-tuple

            ``(weights, means, gmm_sigmas, hidden_states, Z)``

            where:

            - ``weights`` is an array of normalized mixture weights with shape
              ``(2**n_hidden, 1)``
            - ``means`` is an array of component means with shape
              ``(2**n_hidden, n_visible)``
            - ``gmm_sigmas`` is an array of component standard deviations with
              shape ``(2**n_hidden, n_visible)``
            - ``hidden_states`` is an array containing the binary hidden
              configuration corresponding to each mixture component, with shape
              ``(2**n_hidden, n_hidden)``
            - ``Z`` is the unnormalized partition-like sum used before weight
              normalization

        Notes
        -----
        This method explicitly enumerates all hidden configurations, so it is
        only feasible for very small values of ``n_hidden``.

        The method preserves the logic of the original implementation by using

        .. math::
            \\exp(-F(\\mu(h)))

        as the unnormalized component weight associated with hidden state ``h``.
        """
        num_components = 2 ** self.n_hidden

        sigma = self.sigma_v().detach().cpu().numpy()
        gmm_sigmas = np.repeat(sigma, num_components, axis=0)

        weights = np.zeros((num_components, 1), dtype=np.float64)
        means = np.zeros((num_components, self.n_visible), dtype=np.float64)
        hidden_states = np.zeros((num_components, self.n_hidden), dtype=np.float64)

        for i, bits in enumerate(itertools.product([0.0, 1.0], repeat=self.n_hidden)):
            h = torch.tensor(bits, dtype=torch.float32, device=self.device).unsqueeze(0)
            hidden_states[i, :] = np.array(bits, dtype=np.float64)

            mean = self.h2v(h).squeeze(0)
            means[i, :] = mean.detach().cpu().numpy()

            weights[i, 0] = torch.exp(-self.free_energy(mean.unsqueeze(0))).item()

        Z = float(np.sum(weights))
        if Z > 0:
            weights = weights / Z

        return weights, means, gmm_sigmas, hidden_states, Z
