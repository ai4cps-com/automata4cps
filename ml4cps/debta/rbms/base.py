"""
Base classes and shared functionality for Restricted Boltzmann Machines.

This module defines :class:`XBinaryRBM`, an abstract base class for RBM
variants with Bernoulli-distributed hidden units. The visible-unit model is
left to subclasses.

Author
------
Nemanja Hranisavljevic
hranisan@hsu.hamburg
nemanja@ai4cps.com
"""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
from ml4cps.tools import compute_purity


class XBinaryRBM(nn.Module, ABC):
    """
    Abstract base class for Restricted Boltzmann Machines with binary hidden units.

    This class provides shared functionality for RBM variants whose hidden units
    are Bernoulli-distributed. Subclasses define the visible-unit conditional
    distribution and the corresponding free-energy function.

    The base class includes:

    - hidden conditional computation :meth:`v2h`
    - Bernoulli hidden sampling via :meth:`sample_h`
    - deterministic reconstruction via :meth:`recon`
    - Gibbs sampling utilities
    - generic loss dispatch for CD, PCD, score matching, and reconstruction loss
    - simple sample generation from a Gibbs chain

    Subclasses must implement:

    - :meth:`h2v`
    - :meth:`sample_v`
    - :meth:`free_energy`

    Subclasses may additionally override:

    - :meth:`score_matching_loss`
    - :meth:`generate`
    - :meth:`prepare_input`
    - :meth:`decode_input`

    Notes
    -----
    This base class assumes a standard RBM interface over visible samples
    ``x``. Models whose energy depends on additional state beyond visible units
    may require a different abstraction rather than direct inheritance from
    :class:`XBinaryRBM`.
    """

    def __init__(self, n_visible: int, n_hidden: int, device: str = "cpu", name: str = "") -> None:
        """
        Initialize the RBM base module.

        Parameters
        ----------
        n_visible : int
            Number of visible units.
        n_hidden : int
            Number of hidden units.
        device : str, optional
            Target device on which the module parameters are placed.
            Default is ``"cpu"``.

        Raises
        ------
        ValueError
            If ``n_visible`` or ``n_hidden`` is not positive.
        """
        super().__init__()

        if n_visible <= 0:
            raise ValueError(f"n_visible must be positive, got {n_visible}.")
        if n_hidden <= 0:
            raise ValueError(f"n_hidden must be positive, got {n_hidden}.")

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.weights = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))

        # Training metadata may be populated by external trainer utilities.
        self.learning_curve: list[dict[str, float]] = []
        self.valid_curve: list[dict[str, float]] = []
        self.hidden_activation_curve: list[dict[str, list[float]]] = []
        self.valid_hidden_activation_curve: list[dict[str, list[float]]] = []
        self.gradient_curve: list[dict[str, float]] = []
        self.num_epoch: int = 0

        # Persistent fantasy particles used by Persistent Contrastive Divergence.
        # This is intentionally stored as runtime state rather than as a buffer.
        self.persistent_visible: torch.Tensor | None = None
        self.name = name

        self.to(device)

    @property
    def device(self) -> torch.device:
        """
        Return the device on which the model parameters currently live.

        Returns
        -------
        torch.device
            Device of the parameter tensors.
        """
        return self.weights.device

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optionally preprocess input data before it is used by the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Preprocessed tensor.

        Notes
        -----
        The default implementation is the identity function.
        Subclasses may override this when the visible representation used
        internally differs from the external data representation.
        """
        return x

    def decode_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optionally decode a tensor from model-visible representation.

        Parameters
        ----------
        x : torch.Tensor
            Tensor expressed in the model's visible representation.

        Returns
        -------
        torch.Tensor
            Decoded tensor.

        Notes
        -----
        The default implementation is the identity function.
        """
        return x

    def v2h(self, visible: torch.Tensor, pre_sigmoid: bool = False) -> torch.Tensor:
        """
        Compute hidden logits or activation probabilities from visible units.

        Parameters
        ----------
        visible : torch.Tensor
            Visible samples. The tensor is flattened to shape
            ``(batch_size, n_visible)`` before applying the affine transform.
        pre_sigmoid : bool, optional
            If ``True``, return hidden logits. Otherwise, return hidden
            Bernoulli probabilities obtained by applying the logistic sigmoid.

        Returns
        -------
        torch.Tensor
            Hidden logits or probabilities with shape
            ``(batch_size, n_hidden)``.
        """
        visible = visible.reshape(visible.size(0), -1)
        logits = visible @ self.weights + self.hidden_bias
        return logits if pre_sigmoid else torch.sigmoid(logits)

    def sample_h(self, hidden_probs: torch.Tensor) -> torch.Tensor:
        """
        Sample binary hidden states from Bernoulli probabilities.

        Parameters
        ----------
        hidden_probs : torch.Tensor
            Hidden-unit activation probabilities.

        Returns
        -------
        torch.Tensor
            Binary hidden samples with the same shape as ``hidden_probs``.
        """
        return torch.bernoulli(hidden_probs)

    @abstractmethod
    def h2v(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute visible distribution parameters from hidden units.

        Parameters
        ----------
        hidden : torch.Tensor
            Hidden activations or sampled hidden states.

        Returns
        -------
        torch.Tensor
            Parameters of the visible conditional distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_v(self, visible_params: torch.Tensor) -> torch.Tensor:
        """
        Sample visible states from visible distribution parameters.

        Parameters
        ----------
        visible_params : torch.Tensor
            Parameters returned by :meth:`h2v`.

        Returns
        -------
        torch.Tensor
            Sampled visible states.
        """
        raise NotImplementedError

    @abstractmethod
    def free_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy for a batch of visible samples.

        Parameters
        ----------
        x : torch.Tensor
            Visible samples.

        Returns
        -------
        torch.Tensor
            Per-sample free energies with shape ``(batch_size,)``.
        """
        raise NotImplementedError

    def score_matching_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the score matching loss.

        Parameters
        ----------
        x : torch.Tensor
            Input visible samples.

        Returns
        -------
        torch.Tensor
            Scalar score matching loss.

        Raises
        ------
        NotImplementedError
            Always raised by the base implementation, because score matching
            is model-specific for many RBM variants.

        Notes
        -----
        Subclasses should override this method if score matching training is
        supported.
        """
        raise NotImplementedError(
            "score_matching_loss(...) must be implemented by subclasses "
            "that support score matching."
        )

    def forward(self, visible: torch.Tensor) -> torch.Tensor:
        """
        Perform a deterministic reconstruction pass.

        Parameters
        ----------
        visible : torch.Tensor
            Input visible samples.

        Returns
        -------
        torch.Tensor
            Reconstructed visible distribution parameters.
        """
        reconstructed_visible, _ = self.recon(visible)
        return reconstructed_visible

    def recon(
        self,
        v: torch.Tensor,
        round_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct visible units from visible input.

        Parameters
        ----------
        v : torch.Tensor
            Input visible samples.
        round_hidden : bool, optional
            If ``True``, round hidden activation probabilities before returning
            them. Otherwise return the probabilities directly.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A pair ``(v_recon, h_out)`` where:

            - ``v_recon`` is the visible reconstruction parameter returned by
              :meth:`h2v`
            - ``h_out`` is the hidden probability tensor or its rounded version
        """
        h_prob = self.v2h(v)
        v_param = self.h2v(h_prob)
        h_out = torch.round(h_prob) if round_hidden else h_prob
        return v_param, h_out

    def recon_error(
        self,
        data: torch.Tensor,
        model_input: torch.Tensor | None = None,
        per_point: bool = False,
        round_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Compute mean squared reconstruction error.

        Parameters
        ----------
        data : torch.Tensor
            Reconstruction target.
        model_input : torch.Tensor | None, optional
            Input passed into the model. If ``None``, ``data`` is used.
        per_point : bool, optional
            If ``True``, return one reconstruction error value per sample.
            Otherwise return a scalar mean over all samples and dimensions.
        round_hidden : bool, optional
            Whether to round hidden probabilities during reconstruction.

        Returns
        -------
        torch.Tensor
            Scalar reconstruction error or a vector of per-sample errors.

        Notes
        -----
        This method always uses mean squared error, regardless of the visible
        distribution assumed by the subclass.
        """
        if model_input is None:
            model_input = data

        recon, _ = self.recon(model_input, round_hidden=round_hidden)
        dim = tuple(range(1, data.dim())) if per_point else None
        return torch.mean((data - recon) ** 2, dim=dim)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the deterministic reconstruction loss.

        Parameters
        ----------
        x : torch.Tensor
            Input minibatch.

        Returns
        -------
        torch.Tensor
            Scalar reconstruction loss.

        Notes
        -----
        The reconstruction loss is the mean squared error between ``x`` and
        the deterministic reconstruction produced by :meth:`recon`.
        """
        recon, _ = self.recon(x)
        return torch.mean((recon - x) ** 2)

    def gibbs_step(
        self,
        visible: torch.Tensor,
        sample_visible: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one full Gibbs update ``v -> h -> v``.

        Parameters
        ----------
        visible : torch.Tensor
            Current visible state.
        sample_visible : bool, optional
            If ``True``, sample visible states using :meth:`sample_v`.
            Otherwise return the visible distribution parameters directly.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple ``(v_state, v_param, h_state, h_prob)`` containing:

            - the next visible state
            - the visible conditional parameters
            - the sampled hidden state
            - the hidden activation probabilities
        """
        h_prob = self.v2h(visible)
        h_state = self.sample_h(h_prob)

        v_param = self.h2v(h_state)
        v_state = self.sample_v(v_param) if sample_visible else v_param

        return v_state, v_param, h_state, h_prob

    def gibbs_chain(
        self,
        v_init: torch.Tensor,
        num_gibbs: int = 1,
        use_probability_last_v_update: bool = False,
    ) -> torch.Tensor:
        """
        Run a Gibbs chain for a fixed number of steps.

        Parameters
        ----------
        v_init : torch.Tensor
            Initial visible state.
        num_gibbs : int, optional
            Number of Gibbs steps to run. Must be at least 1.
        use_probability_last_v_update : bool, optional
            If ``True``, use visible distribution parameters instead of sampled
            visible states in the final visible update.

        Returns
        -------
        torch.Tensor
            Final visible state after the Gibbs chain.

        Raises
        ------
        ValueError
            If ``num_gibbs`` is less than 1.
        """
        if num_gibbs < 1:
            raise ValueError(f"num_gibbs must be at least 1, got {num_gibbs}.")

        vk = v_init

        for step in range(num_gibbs):
            h_prob = self.v2h(vk)
            h_state = self.sample_h(h_prob)

            v_param = self.h2v(h_state)
            is_last = step == num_gibbs - 1

            if is_last and use_probability_last_v_update:
                vk = v_param
            else:
                vk = self.sample_v(v_param)

        return vk

    def cd_loss(
        self,
        v0: torch.Tensor,
        num_gibbs: int = 1,
        use_probability_last_v_update: bool = False,
    ) -> torch.Tensor:
        """
        Compute standard Contrastive Divergence loss.

        Parameters
        ----------
        v0 : torch.Tensor
            Data minibatch.
        num_gibbs : int, optional
            Number of Gibbs steps used to construct the negative sample.
        use_probability_last_v_update : bool, optional
            Whether to use visible distribution parameters instead of sampled
            visible states in the final negative update.

        Returns
        -------
        torch.Tensor
            Scalar CD-k loss.

        Notes
        -----
        The negative chain is initialized from the current minibatch.
        """
        with torch.no_grad():
            vk = self.gibbs_chain(
                v_init=v0.detach(),
                num_gibbs=num_gibbs,
                use_probability_last_v_update=use_probability_last_v_update,
            )

        return torch.mean(self.free_energy(v0) - self.free_energy(vk))

    def reset_persistent_chain(self) -> None:
        """
        Reset the persistent fantasy particles used by PCD.

        Notes
        -----
        Call this before starting a new independent training run with
        Persistent Contrastive Divergence.
        """
        self.persistent_visible = None

    def _init_persistent_chain(
        self,
        batch_size: int,
        like: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize the persistent fantasy particles used by PCD.

        Parameters
        ----------
        batch_size : int
            Number of fantasy particles.
        like : torch.Tensor | None, optional
            Reference tensor whose shape and dtype are used when available.

        Raises
        ------
        ValueError
            If ``batch_size`` is less than 1.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}.")

        if like is not None:
            self.persistent_visible = torch.rand_like(like, device=self.device)
        else:
            self.persistent_visible = torch.rand(
                batch_size,
                self.n_visible,
                device=self.device,
            )

    def pcd_loss(
        self,
        v0: torch.Tensor,
        num_gibbs: int = 1,
        use_probability_last_v_update: bool = False,
    ) -> torch.Tensor:
        """
        Compute Persistent Contrastive Divergence loss.

        Parameters
        ----------
        v0 : torch.Tensor
            Data minibatch.
        num_gibbs : int, optional
            Number of Gibbs updates applied to the persistent chain.
        use_probability_last_v_update : bool, optional
            Whether to use visible distribution parameters instead of sampled
            visible states in the final negative update.

        Returns
        -------
        torch.Tensor
            Scalar PCD-k loss.

        Notes
        -----
        Unlike standard Contrastive Divergence, the negative chain is not
        reinitialized from the current minibatch. Instead, a persistent fantasy
        buffer is reused across minibatches.

        The persistent chain is reinitialized if it does not yet exist or if its
        batch dimension does not match the current minibatch size. In practice,
        trainers often avoid this mismatch by using ``drop_last=True`` when
        training with PCD.
        """
        batch_size = v0.size(0)

        if (
            self.persistent_visible is None
            or self.persistent_visible.size(0) != batch_size
        ):
            self._init_persistent_chain(batch_size=batch_size, like=v0)

        with torch.no_grad():
            vk = self.gibbs_chain(
                v_init=self.persistent_visible.detach(),
                num_gibbs=num_gibbs,
                use_probability_last_v_update=use_probability_last_v_update,
            )
            self.persistent_visible = vk.detach()

        return torch.mean(self.free_energy(v0) - self.free_energy(vk))

    def compute_loss(
        self,
        x: torch.Tensor,
        loss_name: str = "cd",
        num_gibbs: int = 1,
        use_probability_last_v_update: bool = False,
        sparsity_target: float = 0.25,
        sparsity_weight: float = 0.0,
        temporal_persistence_weight: float = 0.0,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Compute one of the supported training losses.

        Parameters
        ----------
        x : torch.Tensor
            Input minibatch shaped (batch, visible) or (batch, time, visible).
            The base loss includes every time step of a supplied window.
        loss_name : str, optional
            Name of the loss to compute. Supported values are:

            - ``"cd"``
            - ``"pcd"``
            - ``"sm"``
            - ``"recon"``
        num_gibbs : int, optional
            Number of Gibbs steps used for ``"cd"`` and ``"pcd"``.
        use_probability_last_v_update : bool, optional
            Whether to use visible distribution parameters instead of sampled
            visible states in the final negative update for ``"cd"`` and
            ``"pcd"``.
        sparsity_target : float, optional
            Target mean activation of each hidden unit, strictly between 0 and 1.
        sparsity_weight : float, optional
            Nonnegative weight of the per-unit KL sparsity penalty.
        temporal_persistence_weight : float, optional
            Nonnegative weight of BCE between adjacent hidden activations within
            each input window. Has no effect on 2D inputs or one-step windows.
        return_components : bool, optional
            Return the base loss, raw penalties, weighted penalties and total
            from the same computation, without resampling the negative chain.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            Scalar training loss, or its components when return_components=True.

        Raises
        ------
        ValueError
            If ``loss_name`` is not supported.
        """
        if x.ndim not in (2, 3):
            raise ValueError("RBM input must have shape (batch, visible) or (batch, time, visible)")
        samples = x.reshape(-1, x.shape[-1])
        if loss_name == "cd":
            loss = self.cd_loss(
                samples,
                num_gibbs=num_gibbs,
                use_probability_last_v_update=use_probability_last_v_update,
            )
        elif loss_name == "pcd":
            loss = self.pcd_loss(
                samples,
                num_gibbs=num_gibbs,
                use_probability_last_v_update=use_probability_last_v_update,
            )
        elif loss_name == "sm":
            loss = self.score_matching_loss(samples)
        elif loss_name == "recon":
            loss = self.reconstruction_loss(samples)
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        regularization = self.regularization_loss(
            x, sparsity_target=sparsity_target, sparsity_weight=sparsity_weight,
            temporal_persistence_weight=temporal_persistence_weight,
            return_components=return_components,
        )
        if not return_components:
            return loss + regularization
        base_name = {"cd": "CD", "pcd": "PCD", "sm": "ScoreMatchingLoss",
                     "recon": "ReconstructionLoss"}[loss_name]
        return {base_name: loss, **regularization,
                "Loss": loss + regularization["WeightedSparsityKL"]
                        + regularization["WeightedTemporalBCE"]}

    def regularization_loss(
        self, x, sparsity_target=0.25, sparsity_weight=0.0,
        temporal_persistence_weight=0.0,
        return_components=False,
    ):
        """Log-space per-unit KL sparsity and adjacent-step BCE within windows."""
        if not 0 < sparsity_target < 1:
            raise ValueError("sparsity_target must be strictly between 0 and 1")
        if sparsity_weight < 0 or temporal_persistence_weight < 0:
            raise ValueError("Regularization weights must be nonnegative")
        kl = x.new_zeros(())
        temporal = x.new_zeros(())
        if sparsity_weight == 0 and temporal_persistence_weight == 0 and not return_components:
            return kl

        logits = self.v2h(x.reshape(-1, x.shape[-1]), pre_sigmoid=True)
        if sparsity_weight or return_components:
            # log(mean(sigmoid(z))) and log(mean(1 - sigmoid(z))) stay
            # differentiable even when float32 sigmoid rounds to zero or one.
            log_count = math.log(logits.shape[0])
            log_active = torch.logsumexp(F.logsigmoid(logits), dim=0) - log_count
            log_inactive = torch.logsumexp(F.logsigmoid(-logits), dim=0) - log_count
            target = logits.new_tensor(sparsity_target)
            kl = (target * (target.log() - log_active) +
                  (1 - target) * ((1 - target).log() - log_inactive)).mean()

        if (temporal_persistence_weight or return_components) and x.ndim == 3 and x.shape[1] > 1:
            window_logits = logits.reshape(x.shape[0], x.shape[1], -1)
            temporal = F.binary_cross_entropy_with_logits(
                window_logits[:, 1:, :], window_logits[:, :-1, :].sigmoid().detach(),
            )
        weighted_kl = sparsity_weight * kl
        weighted_temporal = temporal_persistence_weight * temporal
        if return_components:
            return dict(SparsityKL=kl, TemporalBCE=temporal,
                        WeightedSparsityKL=weighted_kl, WeightedTemporalBCE=weighted_temporal)
        return weighted_kl + weighted_temporal

    @torch.no_grad()
    def generate(self, num_examples: int, num_steps: int = 100) -> torch.Tensor:
        """
        Generate visible samples by running a Gibbs chain.

        Parameters
        ----------
        num_examples : int
            Number of samples to generate.
        num_steps : int, optional
            Number of Gibbs steps to run.

        Returns
        -------
        torch.Tensor
            Generated visible samples with shape
            ``(num_examples, n_visible)``.

        Raises
        ------
        ValueError
            If ``num_examples`` or ``num_steps`` is less than 1.
        """
        if num_examples < 1:
            raise ValueError(
                f"num_examples must be at least 1, got {num_examples}."
            )
        if num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}.")

        x = torch.bernoulli(
            torch.full((num_examples, self.n_visible), 0.1, device=self.device)
        )

        for step in range(num_steps):
            h = self.sample_h(self.v2h(x))
            if step < num_steps - 1:
                x = self.sample_v(self.h2v(h))
            else:
                x = self.h2v(h)
        return x

    def _init_args(self) -> dict:
        return {
            "n_visible": self.n_visible,
            "n_hidden": self.n_hidden,
            "device": "cpu",
        }

    def save(self, filename=None, filename_sufix: str = "") -> None:
        if filename is None:
            filename = f"{self.__class__.__name__.lower()}_{filename_sufix}.pt"
        torch.save(
            {
                "class_name": self.__class__.__name__,
                "module_name": self.__class__.__module__,
                "init_args": self._init_args(),
                "model_state_dict": self.state_dict(),
                "n_visible": self.n_visible,
                "n_hidden": self.n_hidden,
                "num_epoch": self.num_epoch,
                "learning_curve": self.learning_curve,
                "valid_curve": self.valid_curve,
            },
            filename,
        )
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename):
        checkpoint = torch.load(filename, map_location="cpu")

        module = importlib.import_module(checkpoint["module_name"])
        ModelClass = getattr(module, checkpoint["class_name"])

        model = ModelClass(**checkpoint["init_args"])
        model.load_state_dict(checkpoint["model_state_dict"])

        model.num_epoch = checkpoint.get("num_epoch", 0)
        model.learning_curve = checkpoint.get("learning_curve", [])
        model.valid_curve = checkpoint.get("valid_curve", [])
        model.eval()
        model.to(model.device)
        return model

