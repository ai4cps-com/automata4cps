"""Learning timed automata from hybrid CPPS data using the DEBTA algorithm."""
from pandas import DataFrame

from ml4cps.debta import debta_tools
from ml4cps.debta.rbms.training import train_rbm
from ml4cps.debta.rbms.bb import BinaryBinaryRBM
from ml4cps.debta.rbms.gb import GaussianBinaryRBM
from ml4cps.debta.vis import plot_learning_curve

ALGORITHM_NAME = "DEBTA"

__author__ = "Nemanja Hranisavljevic"
__email__ = ("hranisan@hsu-hh.de", "nemanja@ai4cps.com")

__citation__ = (
    "Hranisavljevic, N., Maier, A., & Niggemann, O. (2020). "
    "Discretization of hybrid CPPS data into timed automaton using "
    "restricted Boltzmann machines. Engineering Applications of Artificial "
    "Intelligence, 95, 103826."
)
__doi__ = "10.1016/j.engappai.2020.103826"
__paper_url__ = "https://www.sciencedirect.com/science/article/abs/pii/S0952197620301986"

import ast
import importlib
import pprint
import time
import pandas as pd
from plotly import graph_objects as go
import numpy as np
from datetime import timedelta
from ml4cps._optional_mlflow import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ml4cps import automata, tools, vis
from ml4cps.automata import learn as automata_learn
from ml4cps.debta.rbms import three_way


class WindowedSequenceDataset(Dataset):
    def __init__(self, sequences, window_size=1, window_stride=1):
        """
        Initializes an object for handling and processing sequences with specified
        window size and stride. Transforms the input sequences into smaller, overlapping
        windows and flattens them into a single tensor for future processing.

        Parameters:
        sequences: An iterable containing input sequences to be processed.
        window_size: An integer specifying the size of each window. Defaults to 1.
        window_stride: An integer specifying the step size between consecutive windows.
            Defaults to 1.

        Attributes:
        sequences: A tensor containing all processed sequences concatenated together.
        sequence_start: A numpy array indicating the starting indices of original
            sequences in the concatenated tensor.
        sequence_finish: A numpy array indicating the ending indices of original
            sequences in the concatenated tensor.
        window_size: The window size used for processing the sequences.
        window_stride: The step size used between consecutive windows.
        """
        if isinstance(sequences, (list, tuple)):
            source_sequences = list(sequences)
        else:
            source_sequences = [sequences]

        source_sequences_as_t = [torch.as_tensor(np.asarray(s)).float() if not isinstance(s, torch.Tensor) else s
                            for s in source_sequences]
        windowed_sequences = tools.window(source_sequences_as_t, window_size, window_stride)
        self.sequence_start = np.cumsum([0] + [s.shape[0] for s in windowed_sequences])[:-1]
        self.sequence_finish = np.cumsum([0] + [s.shape[0] for s in windowed_sequences])[1:]
        self.sequences = torch.cat(windowed_sequences, dim=0)

        # Keep windowed time indices aligned with self.sequences.
        # For DataFrame inputs this uses the original index values.
        windowed_time_indices = []
        for seq, seq_windows in zip(source_sequences, windowed_sequences):
            if isinstance(seq, DataFrame):
                seq_index = seq.index.to_numpy()
            else:
                if hasattr(seq, "shape") and len(seq.shape) > 0:
                    source_length = int(seq.shape[0])
                else:
                    source_length = len(seq)
                seq_index = np.arange(source_length)

            if len(seq_index) < window_size:
                empty_shape = (0, window_size)
                windowed_index = np.empty(empty_shape, dtype=seq_index.dtype if hasattr(seq_index, "dtype") else object)
            else:
                windowed_index = np.asarray(
                    [
                        seq_index[i: i + window_size]
                        for i in range(0, len(seq_index) - window_size + 1, window_stride)
                    ]
                )
            windowed_time_indices.append(windowed_index)

        if len(windowed_time_indices) > 0:
            self.time_index_windows = np.concatenate(windowed_time_indices, axis=0)
        else:
            self.time_index_windows = np.empty((0, window_size))
        self.window_size = window_size
        self.window_stride = window_stride


    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        d = self.sequences[idx, :, :]
        return d

    def windowed_sequences(self):
        return [self.sequences[start:finish, :]
                for start, finish in zip(self.sequence_start, self.sequence_finish)]

    def windowed_time_indices(self):
        return [self.time_index_windows[start:finish, :]
                for start, finish in zip(self.sequence_start, self.sequence_finish)]

    def get_with_time(self, idx):
        return self.sequences[idx, :, :], self.time_index_windows[idx]

    def get_random_sample(self, num_examples=10000):
        return self[torch.randint(0, len(self), (num_examples,))]


    def normalize(self, variables=None, mean=None, std=None):
        """
        Normalizes selected variables across all sequences by subtracting their mean
        and dividing by their standard deviation. If no variables, mean, or standard
        deviation are provided, defaults are computed automatically.

        Parameters:
        variables : array-like, optional
            The indices of the variables to normalize. Defaults to all variables
            in the sequences.
        mean : array-like, optional
            Mean values to use for normalization. Must match the shape of the
            selected variables. If not provided, the mean is computed across
            all sequences and time steps for the selected variables.
        std : array-like, optional
            Standard deviation values to use for normalization. Must match the
            shape of the selected variables. If not provided, the standard deviation
            is computed across all sequences and time steps for the selected variables.

        Returns:
        tuple
            A tuple containing the mean and standard deviation used for normalization.
        """
        if variables is None:
            variables = np.arange(0, self.sequences[0].shape[1])
        if mean is None:
            mean = self.sequences[:,:,variables].mean(dim=[0, 1], keepdim=True)
        if std is None:
            std = self.sequences[:,:,variables].std(dim=[0, 1], keepdim=True)

        self.sequences[:, :, variables] = (self.sequences[:, :, variables] - mean) / std
        return mean, std


class DebtaEbm(nn.Module):
    """
    Handles the construction and functioning of the DebtaEbm model.

    This class implements a neural network model based on energy-based architectures. It leverages
    discretization layers, sparsity mechanisms, transience properties, and optional temporal modeling
    for various unsupervised or supervised learning tasks. The flexible architecture allows for
    customizations such as adjustable hidden layers, sparsity tuning, activation functions, and more.

    Attributes:
        split_on_quantile (int): Determines whether to split based on quantiles of data.
        sparsity_weight (float): Weight applied to sparsity loss for regularization.
        sparsity_target (float): Target sparsity level to enforce during training.
        transience (bool): Indicates whether transience is enabled.
        temporal_persistence_weight (float): Weight applied to temporal persistence loss.
        use_derivatives (list): Specifies derivatives to include as input features. Converted to a list.
        window_size (int): Size of the input window for sequential data.
        window_step (int): Step size for sliding the input window.
        device: Device on which the model operates ('cpu' or 'cuda').
        num_d (int): Dimensionality of additional input state variables.
        num_y (int): Dimensionality of target outputs.
        max_epoch_size (int): Maximum size of epoch data processed.
        log_sigma_x: Logarithm of the sigma value used by the model, represented as a tensor.
        timed_model: Optional temporal model used for sequential or time-based layers.
        threshold: Placeholder for any learned or preset activation thresholds.
        learning_curve (list): Tracks the progression of the training loss over epochs.
        valid_curve (list): Tracks the progression of the validation loss over epochs.
        num_epoch (int): Counts the number of completed training epochs.

    Methods:
        num_h(level):
            Return the number of hidden units at a given hidden layer.

        num_v(level):
            Return the number of visible/input units for a given RBM layer.

        num_hidden_layers():
            Return the total number of hidden layers in the model.
    """
    def __init__(self, num_y, num_h, num_d=0, first_hidden_size=None, sigma=1.,
                 sparsity_weight=0.01, transience=True, temporal_persistence_weight=0, num_sigm_layers=1, split_on_quantile=1,
                 window_size=1, window_step=1, sparsity_target=0.1, use_derivatives=0, device='cpu', log_mlflow=False,
                 n_factors=10, timed_model="automaton", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_on_quantile = split_on_quantile
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        self.transience = transience
        self.temporal_persistence_weight = temporal_persistence_weight
        self.use_derivatives = np.asarray(use_derivatives).reshape(-1).tolist()
        self.window_size = window_size
        self.window_step = window_step
        self.device = device
        self._mean = None
        self._std = None
        self.num_d = num_d
        self.num_y = num_y
        self.max_epoch_size = 10**6

        if use_derivatives:
            input_size = num_y * len(self.use_derivatives)
        else:
            input_size = num_y

        self.log_sigma_x = np.log(sigma) * torch.ones(1, input_size, requires_grad=False).to(device)

        self._discretization_layers = nn.Sequential().to(device)
        self._decoder_biases = nn.ParameterList().to(device)

        if timed_model in ['rbm']:
            self.timed_model = three_way.ThreeWayBinaryBinaryRBM(num_h, num_h, round(num_h * 1.1), n_factors).to(device)
        else:
            self.timed_model = None

        enc_layers = []
        if first_hidden_size is None:
            if num_sigm_layers == 1:
                first_hidden_size = num_h
            else:
                first_hidden_size = int(input_size * 1.5)

        layer_sizes = ([input_size] +
                       np.round(np.linspace(first_hidden_size, num_h, num_sigm_layers)).astype(int).tolist())
        for i in range(num_sigm_layers):
            enc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], device=device))
            self._discretization_layers.add_module(f'linear_v2h{i}', enc_layers[i])
            self._discretization_layers.add_module(f'sigmoid_v2h{i}', nn.Sigmoid())
            self._decoder_biases.append(nn.Parameter(torch.zeros(layer_sizes[i], device=device)))

        self.threshold = None
        self.learning_curve = []
        self.valid_curve = []
        self.num_epoch = 0

        if log_mlflow:
            mlflow.log_param("model", "DEBTA")
            mlflow.log_param("num_signals", num_y)
            mlflow.log_param("input_size", input_size)
            mlflow.log_param("num_sigm_layers", num_sigm_layers)
            mlflow.log_param("num_hidden_units", num_h)
            mlflow.log_param("sparsity_target", self.sparsity_target)
            mlflow.log_param("sparsity_weight", self.sparsity_weight)
            mlflow.log_param("sigma", sigma)
            mlflow.log_param("use_derivatives", self.use_derivatives)
            mlflow.log_param("window_size", self.window_size)
            mlflow.log_param("first_hidden_size", first_hidden_size)
            mlflow.log_param('temporal_persistence_weight', temporal_persistence_weight)

    def num_h(self, level=-1):
        """
        Return the number of hidden units at a given hidden layer.

        This method looks up the output dimensionality of the encoder linear layer
        corresponding to the requested hidden level.

        Args:
            level (int, optional):
                Hidden layer index for which the number of hidden units is returned.
                Positive indices are 1-based:
                - ``1`` refers to the first hidden layer
                - ``2`` refers to the second hidden layer
                - ...

                Negative indices count backward from the top hidden layer:
                - ``-1`` refers to the last hidden layer
                - ``-2`` refers to the second-to-last hidden layer
                Defaults to ``-1``.

        Returns:
            int:
                Number of hidden units (output features) in the selected layer.

        Raises:
            ValueError:
                If ``level`` is outside the valid range
                ``[1, num_hidden_layers()]`` or below ``-num_hidden_layers()``.

        Notes:
            - The hidden-layer size is obtained from the corresponding linear layer
              inside ``self._discretization_layers``.
            - Hidden levels are mapped to linear layers by skipping the interleaved
              sigmoid activations.

        Example:
            >>> model.num_h(1)
            64
            >>> model.num_h(-1)
            16
        """
        n = self.num_hidden_layers()

        if level < 0:
            level = n + level + 1

        if level < 1 or level > n:
            raise ValueError(f"level must be between 1 and {n}, or negative down to {-n}")

        l = self._discretization_layers[(level - 1) * 2]
        return l.out_features

    def num_v(self, level=1):
        """
        Return the number of visible/input units for a given RBM layer.

        This method looks up the input dimensionality of the encoder linear layer
        corresponding to the requested hidden level. In the RBM interpretation,
        this is the number of visible units connected to that hidden layer.

        Args:
            level (int, optional):
                Hidden layer index whose visible/input dimensionality should be
                returned. Positive indices are 1-based:
                - ``1`` refers to the first RBM visible layer
                - ``2`` refers to the visible side of the second RBM
                - ...

                Negative indices count backward from the top hidden layer:
                - ``-1`` refers to the topmost RBM
                - ``-2`` refers to the one below it
                Defaults to ``1``.

        Returns:
            int:
                Number of visible/input units (input features) for the selected
                layer.

        Raises:
            ValueError:
                If ``level`` is outside the valid range
                ``[1, num_hidden_layers()]`` or below ``-num_hidden_layers()``.

        Notes:
            - The visible-layer size is obtained from the corresponding linear layer
              inside ``self._discretization_layers``.
            - For the first hidden layer, this usually matches the model input size.
            - For deeper layers, this corresponds to the size of the previous hidden
              layer.

        Example:
            >>> model.num_v(1)
            32
            >>> model.num_v(-1)
            64
        """
        n = self.num_hidden_layers()

        if level < 0:
            level = n + level + 1

        if level < 1 or level > n:
            raise ValueError(f"level must be between 1 and {n}, or negative down to {-n}")

        l = self._discretization_layers[(level - 1) * 2]
        return l.in_features

    def num_hidden_layers(self):
        """
        Return the total number of hidden layers in the model.

        This count is derived from the encoder stack stored in
        ``self._discretization_layers``, where each hidden layer consists of a
        linear transformation followed by a sigmoid activation. If an additional
        temporal top model is present in ``self.timed_model``, it is counted as one
        extra hidden layer.

        Returns:
            int:
                Total number of hidden layers, including the optional temporal top
                layer if present.

        Notes:
            - ``self._discretization_layers`` stores modules in alternating order:
              linear, sigmoid, linear, sigmoid, ...
            - Therefore, the number of hidden layers in the encoder stack is
              ``len(self._discretization_layers) // 2``.
            - If ``self.timed_model is not None``, the returned value includes that
              additional model as one more layer.

        Example:
            >>> model.num_hidden_layers()
            3
        """
        L = len(self._discretization_layers) // 2
        if self.timed_model is not None:
            L += 1
        return L

    @property
    def sigma_x(self):
        """
        Return the visible-unit standard deviation tensor.

        This property exposes the standard deviation of the Gaussian visible layer
        by exponentiating the internally stored log-standard-deviation parameter
        ``log_sigma_x``. It is useful for inspecting the current input noise scale
        during training and inference without manually calling ``torch.exp``.

        Returns:
            torch.Tensor:
                A tensor of shape ``(1, input_size)`` containing the standard
                deviation value for each visible/input dimension. If a single
                shared sigma was initialized, all entries will have the same value.

        Example:
            >>> model.sigma_x
            tensor([[0.3000, 0.3000, 0.3000]], device='cuda:0')
        """
        return torch.exp(self.log_sigma_x)

    def encode(self, x, rounding=False, x_level=0, h_level=None, pre_sigmoid=False, scale_input=True):
        """
        Encode input data into a latent representation using the discretization stack.

        This method propagates the input through the encoder layers from a specified
        visible/input level ``x_level`` up to a target hidden level ``h_level``.
        It can optionally return pre-sigmoid activations, round the resulting latent
        representation to binary values, and scale the input by the visible-unit
        standard deviation ``sigma_x`` at the lowest input level.

        If a list of tensors is provided, the method applies the same encoding
        recursively to each element and returns a list of encoded outputs.

        Args:
            x (torch.Tensor or list[torch.Tensor]):
                Input tensor to encode, or a list of input tensors. The tensor is
                expected to have a feature dimension compatible with the selected
                ``x_level``.
            rounding (bool, optional):
                If True, rounds the final encoded activations to 0/1 using
                ``torch.round``. Useful for obtaining discrete latent states.
                Defaults to False.
            x_level (int, optional):
                Index of the starting level in the encoder stack.
                - ``0`` means raw visible/input space.
                - Higher values indicate that ``x`` is already represented in an
                  intermediate hidden space and encoding should continue from there.
                Defaults to 0.
            h_level (int or None, optional):
                Target hidden level up to which the input should be encoded.
                If None, encoding proceeds through all available hidden layers.
                Defaults to None.
            pre_sigmoid (bool, optional):
                If True, stops one module earlier so that the returned output is the
                linear pre-sigmoid activation of the target layer instead of the
                post-sigmoid activation. Defaults to False.
            scale_input (bool, optional):
                If True and ``x_level == 0``, scales the input by dividing it by
                ``self.sigma_x`` before applying the encoder. This is typically used
                for the Gaussian visible layer in the first RBM. Defaults to True.

        Returns:
            torch.Tensor or list[torch.Tensor]:
                The encoded latent representation at the requested level. If
                ``rounding=True``, the output is rounded element-wise to binary-like
                values. If the input was a list, returns a list of encoded tensors.

        Notes:
            - The encoder is stored in ``self._discretization_layers`` as an
              alternating sequence of linear and sigmoid modules.
            - When ``pre_sigmoid=True``, the returned tensor corresponds to the
              affine transformation output before the final sigmoid activation.
            - If ``h_level`` exceeds ``self.num_hidden_layers()``, the additional
              transformation is delegated to ``self.timed_model``.

        Example:
            >>> h = model.encode(x)
            >>> h_bin = model.encode(x, rounding=True)
            >>> logits = model.encode(x, h_level=1, pre_sigmoid=True)
        """
        if h_level is None:
            h_level = self.num_hidden_layers()
        if isinstance(x, list):
            return [self.encode(xx, rounding=rounding, x_level=x_level, h_level=h_level,
                                pre_sigmoid=pre_sigmoid, scale_input=scale_input) for xx in x]

        if x_level == 0 and scale_input:
            x = x / self.sigma_x

        discr_layer_end = h_level * 2
        if pre_sigmoid:
            discr_layer_end -= 1
        h = self._discretization_layers[x_level * 2:discr_layer_end](x)
        if h_level > self.num_hidden_layers():
            h = self.timed_model.forward(h)
        if rounding:
            h = torch.round(h)
        return h

    def decode(self, h, x_level=0, h_level=None):
        """
        Decode a latent representation back to a lower-level or visible-space representation.

        This method performs the downward pass of the model by traversing the
        encoder weights in reverse order, using tied weights for reconstruction.
        Starting from a hidden representation ``h`` at level ``h_level``, it
        reconstructs the representation at level ``x_level``. If decoding reaches
        the visible layer (``x_level == 0``), the reconstruction is scaled by
        ``sigma_x`` and shifted by the corresponding decoder bias.

        For models with an additional temporal top layer, inputs above the standard
        hidden stack are first decoded through ``self.timed_model``.

        Args:
            h (torch.Tensor):
                Latent representation to decode. It is assumed to correspond to
                level ``h_level`` of the model.
            x_level (int, optional):
                Target level to decode to.
                - ``0`` means decode all the way to visible/input space.
                - Higher values decode only to an intermediate hidden level.
                Defaults to 0.
            h_level (int or None, optional):
                The level at which ``h`` is defined. If None, ``h`` is assumed to
                belong to the top hidden layer of the model. Defaults to None.

        Returns:
            torch.Tensor:
                The reconstructed representation at level ``x_level``.
                - If ``x_level == 0``, the output is reshaped to
                  ``(batch_size, -1, self.num_y)``.
                - Otherwise, the output remains in the hidden-space shape produced
                  by the decoder path.

        Notes:
            - Reconstruction uses tied weights from ``self._discretization_layers``:
              the transpose of each encoder weight matrix is used for decoding.
            - For intermediate hidden layers, a sigmoid activation is applied after
              each affine transformation.
            - For the visible layer (``l == 1``), no sigmoid is applied; instead,
              the reconstruction is scaled by ``self.sigma_x`` and shifted by the
              corresponding decoder bias.
            - If ``h_level`` exceeds ``self.num_hidden_layers()``, decoding begins
              by applying ``self.timed_model.decode``.

        Example:
            >>> h = model.encode(x)
            >>> x_recon = model.decode(h)
            >>> h1 = model.decode(h, x_level=1, h_level=2)
        """
        if h_level is None:
            h_level = self.num_hidden_layers()
        if h_level > self.num_hidden_layers():
            h = self.timed_model.decode(h)

        y = h
        for l in range(h_level, x_level, -1):
            layer = self._discretization_layers[(l - 1) * 2]
            W = layer.weight
            b = self._decoder_biases[l - 1]

            if l == 1:
                pre = F.linear(y, W.t(), None)

                if x_level == 0:
                    y = pre * self.sigma_x + b
                    y = y.reshape(y.shape[0], -1, self.num_y)
                else:
                    y = pre + b
            else:
                y = F.linear(y, W.t(), b)
                y = torch.sigmoid(y)
        return y

    def prepare_data(self, x):
        """
        Split an input sequence tensor into prediction data and past context windows.

        This method converts the input sequence into overlapping windows of length 2
        with step size 1 using ``tools.window``. Each resulting window is then split
        along the temporal window dimension into:
        - ``context``: the first half of the window, detached from the computation graph
        - ``data``: the second half of the window, kept for downstream learning

        The method is intended for temporal modeling setups where the model uses
        past observations as context for reconstructing or predicting the next part
        of the sequence.

        Args:
            x (torch.Tensor):
                Input sequence tensor to window and split. The expected shape depends
                on ``tools.window``, but it should contain a temporal dimension that
                can be segmented into overlapping windows.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                A tuple ``(data, context)`` where:
                - ``data`` contains the second half of each window
                - ``context`` contains the first half of each window and is detached
                  from the gradient graph

        Notes:
            - The window length is fixed to 2 and the step size to 1.
            - ``context`` is detached via ``.detach()`` so gradients are not
              propagated through the past context branch.
            - The split is performed at ``windowed.shape[-2] // 2``, i.e. halfway
              along the second-to-last dimension of the windowed tensor.

        Example:
            >>> data, context = model.prepare_data(x)
        """
        windowed = tools.window(x, 2, 1)
        context = windowed[:, :, 0:windowed.shape[-2] // 2, :].detach()
        data = windowed[:, :, windowed.shape[-2] // 2:, :]
        return data, context

    def predict_discrete_mode(self, data):
        """
        Predict the discrete mode representation for the input data. This method can process
        either a single data instance or a list of data instances. If a list is passed, the
        method recursively predicts the mode for each element in the list. The function utilizes
        an encoding mechanism with optional rounding and converts the resulting binary vector
        into a mode representation.

        Parameters
        ----------
        data : object, list of object
            The input data or list of data instances for which the discrete mode representation
            is to be predicted.

        Returns
        -------
        numpy.ndarray, list of numpy.ndarray
            Returns the predicted discrete mode representation. If input is a single data instance,
            the output is a single numpy array. If input is a list, the output is a list of numpy
            arrays corresponding to each data instance in the input.

        """
        if type(data) is list:
            results = [self.predict_discrete_mode(d) for d in data]
            return [pd.Series(r, name='mode') for r in results]
        else:
            h = self.encode(data, rounding=True)
            h = debta_tools.binary_vector_to_mode(h)
            mode_array = np.concatenate([np.full((self.window_size - 1,), np.nan), h], axis=0)
            return pd.Series(mode_array, name='mode')

    def forward(self, x):
        x = self.prepare_data(x)
        return self.decode(*self.encode(x))

    def energy(self, x, h, level=1):
        """
        Compute the joint energy E(x, h) for one RBM layer.

        Supports:
        - level=1: Gaussian-Bernoulli RBM between visible input x and hidden h
        - level>1: Binary-Binary RBM between lower hidden activations x and upper hidden h

        Args:
            x (torch.Tensor):
                Visible state for the selected RBM.
                - For level 1: input/visible variables
                - For level > 1: hidden activations from level-1
            h (torch.Tensor):
                Hidden state of the selected RBM.
            level (int, optional):
                RBM level to evaluate. Defaults to 1.

        Returns:
            torch.Tensor:
                Energy per sample, shape ``(batch_size,)``.

        Notes:
            - This implementation is consistent with ``free_energy()``.
            - For level 1, the visible term is Gaussian and uses ``self.sigma_x``.
            - For level > 1, both visible and hidden units are treated as binary.
        """
        if level < 1 or level > self.num_hidden_layers():
            raise ValueError(f"level must be between 1 and {self.num_hidden_layers()}")

        if x.ndim == 2:
            x = x.unsqueeze(1)
        else:
            x = x[:, [-1], :]

        if h.ndim == 2:
            h = h.unsqueeze(1)
        else:
            h = h[:, [-1], :]

        layer = self._discretization_layers[(level - 1) * 2]
        W = layer.weight
        bh = layer.bias
        bx = self.visible_bias(level)

        if level == 1:
            bx = bx.reshape(-1, self.num_y)
            sigma_x = self.sigma_x.reshape(-1, self.num_y)

            vis = torch.sum((x - bx) ** 2 / (2 * sigma_x ** 2), dim=[1, 2])
            hid = torch.sum(h * bh, dim=[1, 2])
            x_scaled = x / sigma_x
            xWh = torch.sum(torch.matmul(x_scaled, W.t()) * h, dim=[1, 2])

            return vis - hid - xWh

        else:
            vis = torch.sum(x * bx, dim=list(range(1, x.ndim)))
            hid = torch.sum(h * bh, dim=list(range(1, h.ndim)))
            xWh = torch.sum(torch.matmul(x, W.t()) * h, dim=list(range(1, h.ndim)))

            return -vis - hid - xWh

    def free_energy(self, x, level=1):
        """
        Compute the free energy of a visible configuration for one RBM level.

        This method evaluates the free energy ``F(x)`` obtained by marginalizing out
        the hidden units of a single RBM layer in the stack.

        Supported cases:
        - ``level == 1``: Gaussian-Bernoulli RBM (GBRBM), where the visible/input
          units are Gaussian and the hidden units are Bernoulli.
        - ``level > 1``: Binary-Binary RBM (BBRBM), where both the visible units
          (coming from the previous hidden layer) and the hidden units are treated
          as Bernoulli.

        The method accepts either a 2D tensor of shape ``(batch_size, features)``
        or a 3D tensor with a temporal/window dimension. In the 3D case, only the
        last time step is used for the free-energy computation.

        Args:
            x (torch.Tensor):
                Input visible configuration for the selected RBM level.
                - For ``level == 1``, this should be data in visible/input space.
                - For ``level > 1``, this should be the activations of the previous
                  hidden layer.
            level (int, optional):
                RBM level for which the free energy is computed.
                Defaults to 1.

        Returns:
            torch.Tensor:
                A 1D tensor of shape ``(batch_size,)`` containing the free-energy
                value for each sample.

        Notes:
            - If ``x.ndim == 2``, the tensor is temporarily expanded to 3D by adding
              a singleton time/window dimension.
            - If ``x.ndim > 2``, only the last slice along the second dimension is
              used: ``x = x[:, [-1], :]``.
            - For ``level == 1``, the visible term uses the Gaussian energy
              involving ``self.sigma_x`` and the visible bias.
            - For ``level > 1``, the visible term is the standard binary-visible
              linear term with the decoder bias of the selected level.
            - The hidden contribution is computed via the standard
              ``sum(softplus(Wx + b_h))`` term.

        Formula:
            For ``level == 1`` (Gaussian-Bernoulli RBM):

            ``F(x) = sum_i ((x_i - b_i)^2 / (2 * sigma_i^2)) - sum_j softplus(((x / sigma)W^T + c)_j)``

            For ``level > 1`` (Binary-Binary RBM):

            ``F(x) = -x·b - sum_j softplus((xW^T + c)_j)``

        Example:
            >>> fe = model.free_energy(x, level=1)
            >>> fe.shape
            torch.Size([batch_size])
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        else:
            x = x[:, [-1], :]

        if level == 1:
            bx = self.visible_bias(level).reshape(1, 1, -1)
            sigma_x = self.sigma_x.reshape(1, 1, -1)

            vis = torch.sum((x - bx) ** 2 / (2 * sigma_x ** 2), dim=[1, 2])
            wx_b = self.encode(x, h_level=level, pre_sigmoid=True, scale_input=True)
            fec = F.softplus(wx_b).sum(dim=[1, 2])
            return vis - fec
        else:
            bx = self.visible_bias(level).reshape(1, 1, -1)
            wx_b = self.encode(x, x_level=level - 1, h_level=level, pre_sigmoid=True)

            vis = -(x * bx).sum(dim=[1, 2])
            fec = F.softplus(wx_b).sum(dim=[1, 2])
            return vis - fec

    def score(self, x, level=1):
        """
        Compute the score function ∇_x log p(x) for a given RBM level.

        The score is defined as the gradient of the log-density with respect to the
        input. For an energy-based model with free energy F(x), this is

            score(x) = ∇_x log p(x) = -∇_x F(x).

        Args:
            x (torch.Tensor):
                Input tensor for which the score is computed.
            level (int, optional):
                RBM level whose free energy is used. Defaults to 1.

        Returns:
            torch.Tensor:
                Score tensor with the same shape as ``x``.
        """
        x = x.requires_grad_(True)
        logp = -self.free_energy(x, level=level).sum()
        grad = torch.autograd.grad(logp, x, create_graph=True)[0]
        return grad

    def dsm_loss(self, x, v, sigma=0.1):
        """
        Compute the denoising score matching (DSM) loss.

        This loss trains the model to estimate the score function
        ``∇_x log p(x)`` by matching it to the score of a Gaussian-corrupted
        version of the data. Given clean samples ``x`` and sampled Gaussian noise
        ``v``, noisy samples are formed as

            x_ = x + sigma * v

        and the model score is evaluated at ``x_``. For Gaussian perturbations,
        the target score of the conditional noise distribution is

            -(x_ - x) / sigma^2 = -v / sigma

        so the DSM objective becomes

            1/2 * || s(x_) + (x_ - x) / sigma^2 ||^2
          = 1/2 * || s(x_) + v / sigma^2 ||^2

        where ``s(x_) = ∇_{x_} log p(x_)`` is the score predicted by the model.

        Args:
            x (torch.Tensor):
                Clean input samples.
            v (torch.Tensor):
                Noise samples, typically drawn from a standard normal distribution
                with the same shape as ``x``.
            sigma (float, optional):
                Standard deviation of the Gaussian corruption noise.
                Defaults to 0.1.

        Returns:
            torch.Tensor:
                Scalar DSM loss averaged over the batch.

        Notes:
            - ``x`` is marked with ``requires_grad=True`` so that the score can be
              computed through automatic differentiation.
            - ``v`` is scaled by ``sigma`` before being added to ``x``.
            - The score is computed using ``self.score(x_)``, which should return
              ``∇_x log p(x)``.
            - ``create_graph=True`` must be enabled inside ``self.score`` for this
              loss to remain differentiable with respect to model parameters.

        Example:
            >>> noise = torch.randn_like(x)
            >>> loss = model.dsm_loss(x, noise, sigma=0.1)
        """
        x = x.requires_grad_()
        v = v * sigma
        x_ = x + v
        s = self.score(x_)
        loss = torch.norm(s + v / (sigma ** 2), dim=-1) ** 2
        loss = loss.mean() / 2.
        return loss

    def sample_h(self, h):
        """
        Sample binary hidden states from hidden-unit activation probabilities.

        This method treats the input tensor ``h`` as Bernoulli probabilities and
        draws one binary sample per element. It is typically used during Gibbs
        sampling or contrastive divergence to obtain stochastic hidden states from
        the encoder's sigmoid outputs.

        Args:
            h (torch.Tensor):
                Tensor of hidden-unit activation probabilities. Each element should
                lie in the interval ``[0, 1]``.

        Returns:
            torch.Tensor:
                A tensor of the same shape as ``h`` containing Bernoulli samples,
                i.e. values in ``{0, 1}``.

        Notes:
            - This method assumes that ``h`` already contains probabilities, not
              logits.
            - Sampling is performed element-wise using ``torch.bernoulli``.

        Example:
            >>> p_h = model.encode(x)
            >>> h_sample = model.sample_h(p_h)
        """
        return torch.bernoulli(h)

    def sample_x(self, x, level):
        """
        Sample visible states from the reconstruction distribution of a given layer.

        This method draws stochastic samples for the visible side of an RBM during
        Gibbs sampling or contrastive divergence.

        Sampling behavior depends on the layer type:
        - ``level == 0``: the visible layer is treated as Gaussian, so samples are
          drawn from a normal distribution with mean ``x`` and standard deviation
          ``self.sigma_x``.
        - ``level > 0``: the visible layer is treated as binary, so samples are
          drawn element-wise from Bernoulli distributions parameterized by ``x``.

        Args:
            x (torch.Tensor):
                Tensor of reconstruction parameters.
                - For ``level == 0``, ``x`` is interpreted as the Gaussian mean.
                - For ``level > 0``, ``x`` is interpreted as Bernoulli
                  probabilities.
            level (int):
                Visible level to sample from.
                - ``0`` corresponds to the original Gaussian visible/input layer.
                - ``> 0`` corresponds to binary visible layers in deeper RBMs.

        Returns:
            torch.Tensor:
                A sampled tensor with the same shape as ``x``.
                - Gaussian samples for ``level == 0``
                - Bernoulli samples for ``level > 0``

        Notes:
            - For Gaussian visible units, the sampling variance is determined by
              ``self.sigma_x``.
            - For binary visible units, this method assumes that ``x`` already
              contains probabilities in ``[0, 1]``.
            - This design matches the model structure where the first RBM is
              Gaussian-Bernoulli and deeper RBMs are Binary-Binary.

        Example:
            >>> p_x = model.decode(h, x_level=0, h_level=1)
            >>> x_sample = model.sample_x(p_x, level=0)

            >>> p_v = model.decode(h_top, x_level=1, h_level=2)
            >>> v_sample = model.sample_x(p_v, level=1)
        """
        if level == 0:
            return torch.normal(x, std=self.sigma_x)
        else:
            return torch.bernoulli(x)

    def generate(self, num_examples, num_steps=10):
        """
        Generate new samples from the model by Gibbs sampling in the top RBM.

        This method draws synthetic samples from the generative model defined by the
        stacked RBM architecture.

        Behavior depends on the number of hidden layers:
        - If the model has a single hidden layer, generation is performed directly
          in the first Gaussian-Bernoulli RBM by alternating between hidden-state
          sampling and visible-state sampling.
        - If the model has multiple hidden layers, Gibbs sampling is performed only
          in the topmost RBM between the last two hidden layers. The resulting
          sample is then propagated deterministically downward through the lower
          decoder stack to obtain a sample in visible space.

        Args:
            num_examples (int):
                Number of samples to generate.
            num_steps (int, optional):
                Number of Gibbs sampling steps to perform in the active RBM.
                Defaults to 10.

        Returns:
            torch.Tensor:
                Generated samples in visible space.
                - For a single-layer model, the shape is typically
                  ``(num_examples, num_visible)``.
                - For deeper models, the returned shape follows the output of
                  ``decode(..., x_level=0, ...)``, typically including the visible
                  feature dimension and any reshaping applied there.

        Notes:
            - For ``L == 1``, the method initializes visible samples from a standard
              normal distribution and alternates:
                  ``x -> p(h|x) -> h -> p(x|h) -> x``.
            - For ``L > 1``, the method initializes the visible side of the top RBM
              from Bernoulli(0.5) samples and alternates:
                  ``h^(L-1) -> p(h^L|h^(L-1)) -> h^L -> p(h^(L-1)|h^L) -> h^(L-1)``.
            - After Gibbs sampling in the top RBM, the sample is decoded
              deterministically through the lower layers to produce a visible-space
              sample.
            - The first RBM uses Gaussian visible sampling via ``sample_x(..., level=0)``.
            - Deeper RBMs use Bernoulli visible sampling via ``sample_x(..., level>0)``.

        Example:
            >>> samples = model.generate(num_examples=32, num_steps=20)
            >>> samples.shape
        """
        L = self.num_hidden_layers()

        if L == 1:
            # Gibbs sampling in the first (visible-hidden) RBM
            x = torch.randn(num_examples, self.num_v(1), device=self.device)

            for _ in range(num_steps):
                ph = self.encode(x, x_level=0, h_level=1, scale_input=True)
                h = self.sample_h(ph)

                px = self.decode(h, x_level=0, h_level=1)
                x = self.sample_x(px, level=0)

            return x

        else:
            # Start from the layer below the top hidden layer
            h_lower = torch.full(
                (num_examples, self.num_v(-1)),
                0.5,
                device=self.device
            )
            h_lower = torch.bernoulli(h_lower)

            # Gibbs sampling in the top RBM: h^{L-1} <-> h^L
            for _ in range(num_steps):
                ph_top = self.encode(h_lower, x_level=L - 1, h_level=L)
                h_top = self.sample_h(ph_top)

                p_lower = self.decode(h_top, x_level=L - 1, h_level=L)
                h_lower = self.sample_x(p_lower, level=L - 1)

            # Deterministic downward pass through lower layers
            x = self.decode(h_lower, x_level=0, h_level=L - 1)
            return x

    def recon(self, v, round=False, x_level=0, h_level=None):
        """
        Reconstruct an input by encoding it to a latent representation and decoding it back.

        This method performs a full encode-decode pass between a specified input
        level ``x_level`` and hidden level ``h_level``. It first encodes the input
        ``v`` into its latent representation ``h``, optionally rounds the latent
        activations to binary values, and then decodes the latent state back to the
        reconstruction ``r``.

        Args:
            v (torch.Tensor):
                Input tensor to reconstruct. It is assumed to be represented at
                level ``x_level`` of the model.
            round (bool, optional):
                If True, rounds the latent representation ``h`` element-wise using
                ``torch.round`` before decoding. This is useful for reconstructing
                from discrete latent states. Defaults to False.
            x_level (int, optional):
                Starting level of the input representation.
                - ``0`` refers to the visible/input space.
                - Higher values indicate that ``v`` is already given in an
                  intermediate hidden representation.
                Defaults to 0.
            h_level (int or None, optional):
                Target hidden level used for encoding before reconstruction.
                If None, the top hidden layer of the model is used.
                Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                A tuple ``(r, h)`` where:
                - ``r`` is the reconstructed input at level ``x_level``
                - ``h`` is the latent representation obtained during encoding

        Notes:
            - This method is a convenience wrapper around ``encode`` followed by
              ``decode``.
            - If ``round=True``, reconstruction is performed from a discretized
              latent code rather than from continuous activation probabilities.
            - The reconstruction shape depends on ``x_level`` and the behavior of
              ``decode``.

        Example:
            >>> r, h = model.recon(x)
            >>> r_bin, h_bin = model.recon(x, round=True)
        """
        if h_level is None:
            h_level = self.num_hidden_layers()
        h = self.encode(v, x_level=x_level, h_level=h_level)
        if round:
            h = torch.round(h)
        r = self.decode(h, x_level, h_level)
        return r, h

    def pretrain_layers(self, train_data, valid_data, **kwargs):
        """
        Perform greedy layer-wise pretraining of the stacked RBM encoder.

        This method trains each hidden layer of the model sequentially as an RBM,
        starting from the first hidden layer and proceeding upward through the
        stack. For each layer, the provided keyword arguments are forwarded to
        ``train_rbm(...)``. If a keyword argument is given as a list, tuple, or
        NumPy array for selected hyperparameters such as ``lr``, the value
        corresponding to the current layer is used.

        After each layer is trained, the method displays the learning curve for
        that layer.

        Args:
            train_data:
                Training dataset used for RBM pretraining. Its expected interface is
                the one required by ``train_rbm(...)``.
            valid_data:
                Validation dataset used to monitor pretraining progress. It may be
                ``None`` if no validation monitoring is desired.
            **kwargs:
                Additional keyword arguments passed to ``train_rbm(...)`` for each
                layer. For certain hyperparameters such as ``lr``, a per-layer
                sequence may be provided. In that case, the element at index
                ``layer - 1`` is used for the current layer.
                Sparsity target and sparsity/temporal persistence weights default
                to this model's settings. Persistence compares steps within each window.

        Returns:
            None

        Notes:
            - Layers are pretrained in ascending order:
              ``1, 2, ..., self.num_hidden_layers()``.
            - For deeper layers, ``train_rbm(...)`` is responsible for encoding the
              data through the already pretrained lower layers before fitting the
              current RBM.
            - If ``kwargs["lr"]`` is a sequence, a different learning rate can be
              used for each layer.
            - The comment ``+1 for three-way RBM on top`` is not reflected in the
              current loop bounds; the loop runs exactly over
              ``self.num_hidden_layers()``.
            - The learning curve shown after each layer corresponds to the training
              history collected by ``train_rbm(...)``.

        Example:
            >>> model.pretrain_layers(
            ...     train_data,
            ...     valid_data,
            ...     lr=[0.01, 0.005, 0.001],
            ...     max_epoch=20,
            ...     batch_size=128
            ... )
        """
        train_data = train_data.sequences.float().to(self.device)
        valid_data = valid_data.sequences.float().to(self.device)
        for layer in range(1, self.num_hidden_layers() + 1):  # +1 for three-way RBM on top
            print('Training layer {}'.format(layer))


            rbm_kwargs = {}
            for k, w in kwargs.items():
                if k in ['lr'] and isinstance(w, (list, tuple, np.ndarray)):
                    rbm_kwargs[k] = w[min(layer-1, len(w)-1)]
                else:
                    rbm_kwargs[k] = w

            # Initialize RBM from current model parameters
            if layer == 1:
                rbm = GaussianBinaryRBM(
                    n_visible=self.num_v(layer),
                    n_hidden=self.num_h(layer),
                    log_sigma_v=self.log_sigma_x.detach().clone(),
                    device=self.device, name=f'layer {layer}'
                )
            else:
                rbm = BinaryBinaryRBM(
                    n_visible=self.num_v(layer),
                    n_hidden=self.num_h(layer),
                    device=self.device,
                    name=f'layer {layer}'
                )

            rbm_kwargs.setdefault("log_mlflow", False)
            rbm_kwargs.setdefault("sparsity_target", self.sparsity_target)
            rbm_kwargs.setdefault("sparsity_weight", self.sparsity_weight)
            rbm_kwargs.setdefault("temporal_persistence_weight", self.temporal_persistence_weight)
            train_rbm(rbm, train_data, valid_data, **rbm_kwargs)

            # Update model parameters from trained RBM
            self.weights(level=layer).data = rbm.weights.data.t()
            self.hidden_bias(level=layer).data = rbm.hidden_bias.data
            self.visible_bias(level=layer).data = rbm.visible_bias.data
            if layer == 1:
                self.log_sigma_x.data = rbm.log_sigma_v.detach().clone()

            learning_curve_fig = plot_learning_curve(rbm)
            if rbm_kwargs["log_mlflow"]:
                tools.log_plotly_figure_to_mlflow(learning_curve_fig, "learning_curve_layer_{}.html".format(layer))
            else:
                learning_curve_fig.show('browser')
            print('Finished training layer {}'.format(layer))

            with torch.no_grad():
                train_data = self.encode(train_data, x_level=layer - 1, h_level=layer).detach()
                valid_data = self.encode(valid_data, x_level=layer - 1, h_level=layer).detach()

    def _get_progress_learn(self, past_source, past_dest, source, dest):
        with torch.no_grad():
            past_source = past_source[:,:,:self.num_y].to(self.device)
            past_dest = past_dest[:,:,:self.num_y].to(self.device)

            source = source[:, :, :self.num_y].to(self.device)
            dest = dest[:, :, :self.num_y].to(self.device)

            past_r, past_h = self.recon(past_source, past_dest)
            r, h = self.recon(source, dest)

        progress = dict(MSE=self.mse(dest, r).item(),
                        Transience=self.transience_bce(past_h, h).item(),
                        Sparsity=self.sparsity(h).item())
        if False:
            valid_energy = torch.mean(self.free_energy(valid_data)).item()
            progress['Energy'] = valid_energy
        return progress



    def _is_set_mean_std(self):
        return self._std is not None and self._mean is not None

    def finetune(self, train_data, valid_data, max_epoch=10, min_epoch=0, estimate_switches_every=5, lr=0.01,
                            weight_decay=0., batch_size=128, shuffle=True, verbose=True, early_stopping=False,
                            early_stopping_patience=3, round_latent_during_learning=False, log_mlflow=False):
        if log_mlflow:
            mlflow.log_param("min_epoch_ft", min_epoch)
            mlflow.log_param("max_epoch_ft", max_epoch)
            mlflow.log_param("batch_size_ft", batch_size)
            mlflow.log_param("shuffle_ft", shuffle)
            mlflow.log_param("weight_decay_ft", weight_decay)
            mlflow.log_param("early_stopping_patience_ft", early_stopping_patience)
            mlflow.log_param("early_stopping_ft", early_stopping)
            mlflow.log_param('round_latent_during_learning_ft', round_latent_during_learning)
            mlflow.log_param('estimate_switches_every_ft', estimate_switches_every)

        train_data.pairs = True
        valid_data.pairs = True
        if valid_data is not None:
            dest = valid_data.get_random_sample()
            self.valid_curve.append(self._get_progress_learn(dest))

        dest = train_data.get_random_sample()
        self.learning_curve.append(self._get_progress_learn(dest))

        data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, num_workers=1,
                                 pin_memory=True)

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # opt = torch.optim.RMSprop(self.parameters(), weight_decay=weight_decay, lr=lr)
        t_start = time.time()
        epoch = 0
        for epoch in range(1, max_epoch + 1):
            if epoch!=0 and estimate_switches_every and epoch % estimate_switches_every == 0:
                pass
            # train_data = self.prepare_data(train_data)
            # valid_data = self.prepare_data(valid_data)
            start_time = time.time()
            for i, (past_source, past_dest, source, dest) in enumerate(data_loader):
                if i % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {i} batches in {elapsed:.2f} seconds")
                # xk, context_k = self.prepare_data(d[:, :, :self.num_y])
                context_k = dest.to(self.device)
                xk = source.to(self.device)

                x0 = xk.clone()
                if False: # self.variant in ['dsebm', 'dae']:
                    xk = torch.normal(mean=xk, std=torch.exp(self.log_sigma_x))

                r, h = self.recon(xk, context_k, round=round_latent_during_learning)
                with torch.no_grad():
                    h_past, _ = self.encode(past_dest.to(self.device), past_source.to(self.device))
                loss = self.mse(x0, r) + self.temporal_persistence_weight * self.transience_bce(h_past, h) + self.sparsity_weight * self.sparsity_loss(h)
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                past_source, past_dest, source, dest = train_data.get_random_sample()
                progress = self._get_progress_learn(past_source, past_dest, source, dest)
                if verbose:
                    print(f'\n############### Epoch {epoch} ###############')
                    print('Train: ')
                    pprint.pp(progress)
                self.learning_curve.append(progress)
            if valid_data is not None:
                with torch.no_grad():
                    past_source, past_dest, source, dest = valid_data.get_random_sample()
                    progress = self._get_progress_learn(past_source, past_dest, source, dest)
                    self.valid_curve.append(progress)
                if verbose:
                    print('Valid: ')
                    pprint.pp(progress)

                if early_stopping and epoch > min_epoch and epoch > early_stopping_patience:
                    valid_metrics = np.array([v['MSE'] for v in self.valid_curve[-early_stopping_patience-1:]])
                    if np.all(valid_metrics[1:] > valid_metrics[0]):
                        print('Early stop after valid metrics: ', valid_metrics)
                        break

        self.eval()
        self.num_epoch = epoch

        if log_mlflow:
            with torch.no_grad():
                mlflow.log_metrics(self.valid_curve[-1])
                mlflow.log_metric("num_epoch", self.num_epoch)
                run_id = mlflow.active_run().info.run_id
                mlflow.log_figure(self.plot_learning_curve(), f'figures/learning_curve_DEBTA_{run_id}.html')

                past_source, past_dest, source, dest = valid_data.get_random_sample()
                mlflow.log_figure(self.plot_activation_probabilities(dest, source),
                                  f'figures/activations_hist_DEBTA_{run_id}.html')
                mlflow.log_figure(self.plot_frequency_of_latent_combinations(dest, source),
                                  f'figures/latent_frequencies_DEBTA_{run_id}.html')

        if verbose:
            print('Training finished after ', timedelta(seconds=time.time() - t_start))

    # def sparsity_loss(self, h):
    #     sparsity_penalty = self.sparsity_weight * torch.sum((self.sparsity_target - h) ** 2)
    #     return sparsity_penalty
    def sparsity(self, activations, per_point=False):
        if per_point:
            return torch.mean(activations, dim=1)
        else:
            return torch.mean(activations)

    def sparsity_loss(self, activations):
        # Mean activation per neuron
        mean_activations = self.sparsity(activations, per_point=True)

        # KL divergence between the target sparsity level and the means
        kl_div = self.sparsity_target * torch.log(self.sparsity_target / mean_activations) + \
                 (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - mean_activations))
        kl_div = torch.mean(kl_div)  # Sum over all neurons

        return kl_div

    def recon_error(self, data, input=None, per_point=False, round=False):
        if input is None:
            input = data
        recon, _ = self.recon(input, round=round)
        if per_point:
            dim = (1, 2)
        else:
            dim = None
        squared_error = torch.mean(torch.square(data - recon), dim=dim)
        return squared_error

    def anomaly_score(self, s, use_context=False, return_numpy=True):
        """
        Compute anomaly scores from reconstruction error.

        This method evaluates how well the model reconstructs the input and uses the
        per-sample reconstruction error as the anomaly score. Higher scores indicate
        that the sample is less well explained by the model and is therefore more
        likely to be anomalous.

        By default, the method applies ``prepare_data(...)`` and computes the score
        only on the returned ``data`` part. If ``use_context=True``, the method
        reconstructs ``data`` conditioned on ``context`` and measures the error
        against ``data``.

        Args:
            s (torch.Tensor):
                Input sequence or batch of samples from which anomaly scores should
                be computed.
            use_context (bool, optional):
                If True, use both outputs of ``prepare_data(s)`` and compute the
                reconstruction error of ``data`` given ``context``.
                If False, compute the reconstruction error only on ``data``.
                Defaults to False.
            return_numpy (bool, optional):
                If True, return the scores as a NumPy array on CPU.
                If False, return a torch tensor.
                Defaults to True.

        Returns:
            numpy.ndarray or torch.Tensor:
                Per-sample anomaly scores. Larger values correspond to larger
                reconstruction errors.

        Notes:
            - This method uses mean squared reconstruction error via
              ``self.recon_error(..., per_point=True)``.
            - ``prepare_data(...)`` returns ``(data, context)``, so the tuple must
              be unpacked before passing values to reconstruction functions.
            - If ``use_context=False``, only the ``data`` part is scored.

        Example:
            >>> scores = model.anomaly_score(x)
            >>> scores.shape
        """
        data, context = self.prepare_data(s)

        if use_context:
            recon, _ = self.recon(context, data)
            score_in_time = torch.mean(torch.square(data - recon), dim=(1, 2))
        else:
            score_in_time = self.recon_error(data, per_point=True)

        if return_numpy:
            return score_in_time.detach().cpu().numpy()
        return score_in_time

    def plot_activation_probabilities(self, d):
        d = d.to(self.device)
        h = self.encode(d)#
        h = h.cpu().detach().numpy()
        fig = go.Figure(data=[go.Histogram(x=h.reshape(-1), name="Probability", histnorm="percent", nbinsx=20)],
                        layout_title="Histogram of activation probabilities")
        return fig



    def anomaly_detection(self, s):
        s = self.prepare_data(s)
        score_in_time = self.recon_error(s, per_point=True).cpu().detach().numpy()
        return (score_in_time > self.threshold).astype(float), score_in_time



    def visible_bias(self, level=1, reshape=False):
        if level < 1 or level > self.num_hidden_layers():
            raise ValueError(f"level must be between 1 and {self.num_hidden_layers()}")

        b = self._decoder_biases[level - 1]

        if reshape and level == 1:
            b = b.reshape(-1, self.num_y)

        return b

    def hidden_bias(self, level=1):
        if level < 1 or level > self.num_hidden_layers():
            raise ValueError(f"level must be between 1 and {self.num_hidden_layers()}")

        layer = self._discretization_layers[(level - 1) * 2]
        return layer.bias

    def weights(self, level=1, transpose=False):
        if level < 1 or level > self.num_hidden_layers():
            raise ValueError(f"level must be between 1 and {self.num_hidden_layers()}")

        W = self._discretization_layers[(level - 1) * 2].weight
        return W.t() if transpose else W


class DEBTA(automata.Automaton):
    """
    DEBTA: DEep Network Timed Automata.

    This class represents an automaton model built from a latent representation
    learned by a neural energy-based model (EBM). It integrates deep learning (via `DebtaEbm`)
    with automata theory (via `automata.Automaton`).

    Attributes:
        ebm (DebtaEbm): The underlying energy-based model for learning latent representations.
        _mean (float or None): Placeholder for mean normalization (not initialized).
        _std (float or None): Placeholder for standard deviation normalization (not initialized).
        log_mlflow (bool): Whether to log metrics and figures to MLflow.
    """

    def __init__(self, log_mlflow=False, *args, **kwargs):
        """
        Initialize the DEBTA automaton.

        Args:
            log_mlflow (bool): If True, enables logging metrics and figures to MLflow.
            *args: Additional positional arguments passed to DebtaEbm constructor.
            **kwargs: Additional keyword arguments passed to DebtaEbm constructor.
        """
        automata.Automaton.__init__(self)
        self.ebm = DebtaEbm(*args, **kwargs, log_mlflow=log_mlflow)
        self._mean = None
        self._std = None
        self.log_mlflow = log_mlflow

    def predict(self, context, model_input: np.ndarray, params=None):
        """

        Parameters
        ----------
        context
        model_input : torch.Tensor | numpy.ndarray | list
            Input batch, moved to the underlying model's device before prediction.
            MLflow schema inference uses an input example supplied when logging.
        params : dict, optional
            Reserved for MLflow prediction parameters; currently unused.
        """
        model_input = torch.as_tensor(model_input, dtype=torch.float32, device=self.ebm.device)
        with torch.no_grad():
            return self.ebm.predict_discrete_mode(model_input)

    @staticmethod
    def _configuration_tuple(config):
        """Decode tuple labels, ``Config[v1,...]`` or legacy ``Config1`` labels."""
        if isinstance(config, str):
            if config.startswith('Config[') and config.endswith(']'):
                config = ast.literal_eval(config[6:])
            elif config.startswith('Config'):
                config = (ast.literal_eval(config[6:]),)
            else:
                raise ValueError(f'Cannot decode configuration label: {config!r}')
        if not isinstance(config, (tuple, list, np.ndarray)):
            raise ValueError('A configuration must be a tuple/list or a Config label')
        # Match the numeric precision used by configuration labels in training.
        return tuple(float(format(float(v), '.9g')) if isinstance(v, (float, np.floating))
                     else v for v in config)

    def predict_reconfig(self, model_input, current_config, n=1, allowed_values=None):
        """Return up to ``n`` alternative configuration tuples, most frequent first.

        ``model_input`` uses the same sensor input format and preprocessing as
        :meth:`predict`. For a sequence, only its final predicted mode is used.
        Candidates are ranked by the total number of timing observations on
        outgoing transitions under each configuration, excluding ``current_config``.
        All configuration-labelled edges are considered, independently of the
        automaton's active configuration. This does not change model state.

        ``current_config`` is a tuple/list or a ``Config[v1,...]`` label.
        ``allowed_values`` is an optional list of tuples, one per configuration
        variable, e.g. ``[(0, 1), (2,)]`` allows either value for variable 1 and
        fixes variable 2 to 2. An empty tuple allows no values for that variable.
        Only historically observed configurations are recommended; combinations
        are not synthesized. Legacy ``Config1`` labels are decoded as ``(1,)``.

        Ties retain graph/configuration insertion order. Unknown modes or missing
        configuration-labelled observations return an empty list. Scores measure
        historical support, not optimality or switching feasibility.
        """
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a non-negative integer")
        if n == 0:
            return []

        current_config = self._configuration_tuple(current_config)
        if allowed_values is not None:
            if (not isinstance(allowed_values, list)
                    or len(allowed_values) != len(current_config)
                    or not all(isinstance(values, tuple) for values in allowed_values)):
                raise ValueError('allowed_values must be a list of tuples, one per configuration variable')
            allowed_values = [self._configuration_tuple(values) for values in allowed_values]

        modes = np.asarray(self.predict(None, model_input)).reshape(-1)
        if modes.size == 0:
            return []
        state = modes[-1]
        if state not in self._G:
            return []

        counts = {}
        for _, _, _, data in self._G.out_edges(state, data=True, keys=True):
            timing = data.get('timing')
            if not isinstance(timing, dict):
                continue
            for config, observations in timing.items():
                if config is None:
                    continue
                config = self._configuration_tuple(config)
                if len(config) != len(current_config):
                    raise ValueError('Configuration labels and current_config have different dimensions')
                if config == current_config:
                    continue
                if allowed_values is not None and any(
                        value not in allowed for value, allowed in zip(config, allowed_values)):
                    continue
                count = len(observations)
                if count:
                    counts[config] = counts.get(config, 0) + count
        return sorted(counts, key=counts.get, reverse=True)[:n]

    def evaluate_purity(self, data, mode_data, metric_name=None):
        """
        Evaluate latent mode purity on a dataset.

        Args:
            data: Dataset object with a `sequences` tensor attribute.
            mode_data: Ground-truth mode/class assignments as either a list/tuple
                of pandas objects (concatenated) or a single array-like object.
            metric_name: Optional MLflow metric name. If provided and
                `self.log_mlflow` is True, the metric is logged.

        Returns:
            float: Purity score.
        """
        x = data.sequences.squeeze(1).detach().to(self.ebm.device)

        with torch.no_grad():
            mode_prediction = self.predict(None, x)

        if isinstance(mode_data, (list, tuple)):
            class_assignments = pd.concat(mode_data, ignore_index=True).to_numpy()
        else:
            class_assignments = np.asarray(mode_data)

        purity = float(
            tools.compute_purity(
                cluster_assignments=mode_prediction,
                class_assignments=class_assignments
            )
        )

        if metric_name is not None and self.log_mlflow:
            mlflow.log_metric(metric_name, purity)

        return purity

    def pretrain_layers(self, train_data, valid_data, *args, **kwargs):
        self.ebm.pretrain_layers(train_data, valid_data, log_mlflow=self.log_mlflow, **kwargs)

    def finetune_DEBTA_network(self, train_dataset, valid_dataset, **kwargs):
        """
        Train the underlying energy-based model (EBM) using provided data.

        Args:
            *args: Positional arguments passed to `self.ebm.learn`.
            **kwargs: Keyword arguments passed to `self.ebm.learn`.
        """
        self.ebm.finetune(train_dataset, valid_dataset, log_mlflow=self.log_mlflow, **kwargs)

    def recon(self, dest):
        dest = dest.to(self.ebm.device)
        r, h = self.ebm.recon(dest, round=True)
        return r, h

    def learn_latent_automaton(self, train_dataset: object, train_config=None, valid_dataset: object = None,
                               valid_config=None, verbose: int = False) -> None:
        """
        Learn the latent automaton by discretizing latent states and extracting transitions.

        This function:
        1. Encodes training data into latent representations using the EBM.
        2. Discretizes the latent representations by rounding.
        3. Learns an automaton structure from the discretized latent vectors.

        Args:
            train_dataset: The training dataset object with `windowed_sequences()` method.
            valid_dataset: Validation dataset (not used in current implementation).
            verbose (bool): If True, prints progress messages during discretization.

        Side Effects:
            - Updates `self._G` with the learned automaton graph.
            - If `log_mlflow` is True, logs:
                - Metric "num_modes" (number of discrete modes/states)
                - Visualization figure (Plotly HTML) of the automaton
        """
        sig_names = [f'h{i + 1}' for i in range(self.ebm.num_h())]
        h = []
        all_sequences = train_dataset.windowed_sequences()
        train_time_windows = train_dataset.windowed_time_indices()

        for seq_ind, (d, time_windows) in enumerate(zip(all_sequences, train_time_windows)):
            if verbose:
                print(f"Discretizing sequence {seq_ind+1}/{len(all_sequences)} of length {d.shape[0]}")

            # Encode the data and discretize latent states
            hh = self.ebm.encode(d.to(self.ebm.device), rounding=True)
            hh = hh.cpu().detach().numpy()

            # Convert latent states to DataFrame with symbolic column names
            hh_values = hh[:, -1, :]

            time_windows = np.asarray(time_windows)
            if time_windows.ndim > 1:
                latent_time_index = time_windows[:, -1]
            else:
                latent_time_index = time_windows
            hh = pd.DataFrame(hh_values, columns=sig_names, index=latent_time_index)
            h.append(hh)

        # Learn automaton from discretized latent signal vectors
        a = automata_learn.simple_learn_from_signal_vectors(h, config=train_config, drop_no_changes=True, verbose=verbose)
        self._G = a._G
        self.q0 = a.q0
        self.initial_r = a.initial_r
        self.final_q = a.final_q
        if hasattr(a, '_Automaton__super_states'):
            self._Automaton__super_states = a._Automaton__super_states
        if hasattr(a, 'decision_states'):
            self.decision_states = a.decision_states
        if hasattr(a, 'previous_node_positions'):
            self.previous_node_positions = a.previous_node_positions
        if hasattr(a, 'id'):
            self.id = a.id
        if hasattr(a, 'dt'):
            self.dt = a.dt
        if hasattr(a, 'discrete_state_names'):
            self.discrete_state_names = a.discrete_state_names
        if hasattr(a, 'discrete_output_names'):
            self.discrete_output_names = a.discrete_output_names
        if hasattr(a, 'cont_state_names'):
            self.cont_state_names = a.cont_state_names
        if hasattr(a, 'cont_output_names'):
            self.cont_output_names = a.cont_output_names
        if hasattr(a, 'UNKNOWN_STATE'):
            self.UNKNOWN_STATE = a.UNKNOWN_STATE

        # Count non-existing transitions in validation data
        if valid_dataset is not None:
            h_valid = []
            all_sequences_valid = valid_dataset.windowed_sequences()
            valid_time_windows = valid_dataset.windowed_time_indices()

            for seq_ind, (d, time_windows) in enumerate(zip(all_sequences_valid, valid_time_windows)):
                if verbose:
                    print(
                        f"Discretizing validation sequence {seq_ind + 1}/{len(all_sequences_valid)} of length {d.shape[0]}")

                hh = self.ebm.encode(d.to(self.ebm.device), rounding=True)
                hh = hh.cpu().detach().numpy()
                hh_values = hh[:, -1, :]
                time_windows = np.asarray(time_windows)
                if time_windows.ndim > 1:
                    latent_time_index = time_windows[:, -1]
                else:
                    latent_time_index = time_windows
                hh = pd.DataFrame(hh_values, columns=sig_names, index=latent_time_index)
                hh = tools.remove_timestamps_without_change(hh)
                h_valid.append(hh)

            # Extract transitions from validation data, we are not checking time here
            valid_transitions = []
            for seq_ind, seq in enumerate(h_valid):
                sequence_config = valid_config[seq_ind][seq.index]
                time_v = np.asarray(sequence_config.index)
                sequence_config = sequence_config.values
                for i in range(len(seq) - 1):
                    m1 = pprint.pformat(tuple(seq.iloc[i,:]), compact=True, width=10000).replace(".0", "")
                    m2 = pprint.pformat(tuple(seq.iloc[i+1,:]), compact=True, width=10000).replace(".0", "")

                    valid_transitions.append((m1, m2, sequence_config[i+1], time_v[i+1] - time_v[i]))

            # Count non-existing transitions
            non_existing_count = 0
            for source, target, config, t in valid_transitions:
                self.set_configuration(config)
                transition_exists = any(dest == target for _, dest, _, _ in self.out_transitions(source))
                if not transition_exists:
                    non_existing_count += 1

            self.set_configuration(None)

            total_valid_transitions = len(valid_transitions)
            non_existing_percentage = (
                        non_existing_count / total_valid_transitions * 100) if total_valid_transitions > 0 else 0.0

            if verbose:
                print(
                    f"Non-existing transitions in validation data: {non_existing_count}/{total_valid_transitions} ({non_existing_percentage:.2f}%)")

            if self.log_mlflow:
                mlflow.log_metric("valid_non_existing_transitions_count", non_existing_count)
                mlflow.log_metric("valid_non_existing_transitions_percentage", non_existing_percentage)

        if self.log_mlflow:
            # Log the number of discovered modes (states)
            mlflow.log_metric("num_modes", self.num_modes)

            # Log an interactive visualization of the automaton as an HTML figure
            mlflow.log_figure(
                vis.plot_cps_plotly(self, show_num_occur=True),
                f'figures/DEBTA_automaton_{mlflow.active_run().info.run_id}.html'
            )
        self.set_configuration(None)

    def save(self, filename=None, filename_suffix=''):
        if filename is None:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            filename = f"{self.__class__.__name__.lower()}{suffix}.pt"

        checkpoint = {
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__,
            "ebm": self.ebm,
            "graph": self._G,
            "q0": self.q0,
            "initial_r": self.initial_r,
            "final_q": self.final_q,
            "super_states": getattr(self, "_Automaton__super_states", ()),
            "decision_states": getattr(self, "decision_states", ()),
            "previous_node_positions": self.previous_node_positions,
            "state": {
                "t": self._t,
                "q": self._q,
                "xt": self._xt,
                "xk": self._xk,
            },
            "mean": self._mean,
            "std": self._std,
            "log_mlflow": self.log_mlflow,
            "id": self.id,
            "dt": self.dt,
            "discrete_state_names": self.discrete_state_names,
            "discrete_output_names": self.discrete_output_names,
            "cont_state_names": self.cont_state_names,
            "cont_output_names": self.cont_output_names,
            "unknown_state": self.UNKNOWN_STATE,
        }

        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")


    @staticmethod
    def load(filename):
        try:
            checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(filename, map_location="cpu")

        module = importlib.import_module(checkpoint["module_name"])
        ModelClass = getattr(module, checkpoint["class_name"])

        model = ModelClass.__new__(ModelClass)
        automata.Automaton.__init__(
            model,
            id=checkpoint.get("id", ""),
            dt=checkpoint.get("dt", -1.0),
            initial_q=list(checkpoint.get("q0", {}).keys()),
            initial_r=checkpoint.get("initial_r"),
            final_q=list(checkpoint.get("final_q", {}).keys()),
            super_states=checkpoint.get("super_states", ()),
            decision_states=checkpoint.get("decision_states", ()),
            unknown_state=checkpoint.get("unknown_state", "raise"),
            discr_state_names=checkpoint.get("discrete_state_names"),
            discr_output_names=checkpoint.get("discrete_output_names"),
            cont_state_names=checkpoint.get("cont_state_names"),
            cont_output_names=checkpoint.get("cont_output_names"),
        )

        model.ebm = checkpoint["ebm"]
        model._G = checkpoint["graph"]
        model.q0 = checkpoint.get("q0", model.q0)
        model.initial_r = checkpoint.get("initial_r", model.initial_r)
        model.final_q = checkpoint.get("final_q", model.final_q)
        model.previous_node_positions = checkpoint.get("previous_node_positions")
        model._mean = checkpoint.get("mean")
        model._std = checkpoint.get("std")
        model.log_mlflow = checkpoint.get("log_mlflow", False)
        model.UNKNOWN_STATE = checkpoint.get("unknown_state", model.UNKNOWN_STATE)

        state = checkpoint.get("state", {})
        model._t = state.get("t", model._t)
        model._q = state.get("q", model._q)
        model._xt = state.get("xt", model._xt)
        model._xk = state.get("xk", model._xk)

        if hasattr(model.ebm, "device"):
            model.ebm.device = "cpu"
        if hasattr(model.ebm, "eval"):
            model.ebm.eval()

        return model



