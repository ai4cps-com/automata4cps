import time as tm
import mlflow
import numpy as np
from datetime import timedelta
import pprint
import warnings
import functools
import inspect
import io
import re
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import torch
from torch.utils.data import DataLoader, Dataset

from ml4cps.debta.debta_tools import mse, sparsity, bce
from ml4cps.debta.rbms.base import XBinaryRBM


class _LearningLogTee:
    """Copy console output into a per-training log while preserving the console."""

    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, text):
        self.log.write(text)
        return self.console.write(text)

    def flush(self):
        self.console.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def _save_learning_log(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        if not arguments.arguments['log_mlflow']:
            return function(*args, **kwargs)
        rbm = arguments.arguments['rbm']
        name = rbm.name or type(rbm).__name__
        artifact_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
        log = io.StringIO()
        log.write(f"Training {name}\n")
        try:
            with redirect_stdout(_LearningLogTee(sys.stdout, log)), \
                 redirect_stderr(_LearningLogTee(sys.stderr, log)):
                return function(*args, **kwargs)
        except BaseException:
            log.write(traceback.format_exc())
            raise
        finally:
            mlflow.log_text(log.getvalue(), f"logs/{artifact_name}_learning.log")

    return wrapped


@_save_learning_log
def train_rbm(
        rbm,
        train_data,
        valid_data=None,
        loss_name="cd",
        min_epoch=0,
        max_epoch=10,
        weight_decay=0.0,
        lr=0.01,
        batch_size=128,
        shuffle=True,
        num_gibbs=1,
        verbose=True,
        early_stopping=False,
        early_stopping_patience=3,
        use_probability_last_v_update=True,
        log_mlflow=False,
        optimizer="RMSprop",
        batch_log_interval=1,
        sparsity_target=0.25,
        sparsity_weight=0.0,
        temporal_persistence_weight=0.0,
):
    """
    Train the RBM using one of the supported losses.

    The optimized objective adds ``sparsity_weight * KL`` and
    ``temporal_persistence_weight * BCE`` to the selected base loss. Sparsity uses
    per-unit means over all batch/window steps; temporal persistence compares adjacent
    steps within each supplied window, independently of batch ordering.

    Parameters
    ----------
    rbm: XBinaryRBM
    train_data : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] | Dataset | DataLoader
        Training data, including WindowedSequenceDataset. Windows have shape
        (batch, time, visible). Lists/tuples of tensors are vertically stacked.
    valid_data : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] | None, optional
        Validation data, also accepting a DataLoader. Evaluation visits the
        entire underlying dataset, including its final partial batch, without
        using the supplied loader's sampler or drop_last setting.
    loss_name : str, optional
        Training objective. Supported values:
        - ``"cd"``
        - ``"pcd"``
        - ``"sm"``
        - ``"recon"``
    min_epoch : int, optional
        Minimum epoch before early stopping is allowed.
    max_epoch : int, optional
        Maximum number of epochs.
    weight_decay : float, optional
        Weight decay used by the optimizer.
    lr : float, optional
        Learning rate.
    batch_size : int, optional
        Minibatch size.
    shuffle : bool, optional
        Whether to shuffle training data.
    num_gibbs : int, optional
        Number of Gibbs steps used by CD/PCD.
    verbose : bool | int, optional
        Verbosity level. If > 1, prints batch-level stats.
    early_stopping : bool, optional
        Whether to use early stopping on validation MSE.
    early_stopping_patience : int, optional
        Patience window for early stopping.
    use_probability_last_v_update : bool, optional
        Whether to use visible probabilities in the final negative step
        for CD/PCD.
    log_mlflow : bool, optional
        Whether to log hyperparameters, epoch metrics, figures and a text
        learning log (console output, warnings and any traceback) to MLflow.
    optimizer : str, optional
        Optimizer type. Supported values:
        - ``"SGD"``
        - ``"RMSprop"``
    batch_log_interval : int, optional
        Print batch progress every this many batches when verbose > 1.
    sparsity_target : float, optional
        Desired mean activation per hidden unit (strictly between 0 and 1).
    sparsity_weight : float, optional
        Nonnegative coefficient of KL sparsity regularization.
    temporal_persistence_weight : float, optional
        Nonnegative coefficient of adjacent-step BCE within input windows.
        It has no effect on 2D inputs or one-step windows.

    Notes
    -----
    Validation and progress tracking are based on reconstruction metrics and
    free energy, regardless of the training loss.
    Epoch-end reconstruction and activation metrics cover the entire training
    dataset at the final epoch parameters, including the final partial batch.
    UpdateMean* loss components average the actual pre-update minibatch losses
    during the epoch, weighted by the number of windows/samples in each batch.
    Batch progress labels reference-minibatch metrics and current update losses
    separately. Epoch metrics and loss components are also logged to MLflow.
    Whole-dataset per-unit means and saturation fractions (p <= .01 / p >= .99)
    are stored in the activation curves and logged separately from scalar curves.
    Weighted per-term gradient norms are measured before updates at batch report
    intervals and at the last batch, including norms restricted to hidden biases.
    Warns at training startup when temporal persistence is requested without
    multi-step windows, or multi-step windows have temporal persistence disabled.
    """
    if batch_log_interval < 1:
        raise ValueError("batch_log_interval must be at least 1")
    if not 0 < sparsity_target < 1:
        raise ValueError("sparsity_target must be strictly between 0 and 1")
    if sparsity_weight < 0 or temporal_persistence_weight < 0:
        raise ValueError("Regularization weights must be nonnegative")

    if log_mlflow:
        mlflow.log_param(rbm.name + "min_epoch", min_epoch)
        mlflow.log_param(rbm.name + "max_epoch", max_epoch)
        mlflow.log_param(rbm.name + "batch_size", batch_size)
        mlflow.log_param(rbm.name + "shuffle", shuffle)
        mlflow.log_param(rbm.name + "weight_decay", weight_decay)
        mlflow.log_param(rbm.name + "early_stopping_patience", early_stopping_patience)
        mlflow.log_param(rbm.name + "early_stopping", early_stopping)
        mlflow.log_param(rbm.name + "lr", lr)
        mlflow.log_param(rbm.name + "num_gibbs", num_gibbs)
        mlflow.log_param(rbm.name + "optimizer", optimizer)
        mlflow.log_param(rbm.name + "loss_name", loss_name)
        mlflow.log_param(rbm.name + "sparsity_target", sparsity_target)
        mlflow.log_param(rbm.name + "sparsity_weight", sparsity_weight)
        mlflow.log_param(rbm.name + "temporal_persistence_weight", temporal_persistence_weight)

    if isinstance(train_data, DataLoader):
        data_loader = train_data
    elif isinstance(train_data, Dataset):
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle,
                                 drop_last=loss_name == "pcd")
    else:
        if isinstance(train_data, (list, tuple)):
            train_data = torch.vstack(train_data)

        train_data = rbm.prepare_input(train_data).float().to(rbm.device)
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, drop_last=loss_name == "pcd")

    train_eval_loader = DataLoader(
        data_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=data_loader.collate_fn,
    )

    valid_loader = None
    if valid_data is not None:
        if isinstance(valid_data, DataLoader):
            valid_loader = DataLoader(
                valid_data.dataset,
                batch_size=valid_data.batch_size or batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=valid_data.collate_fn,
            )
        else:
            if isinstance(valid_data, (list, tuple)):
                valid_data = torch.vstack(valid_data)
            valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    train_batch = next(iter(data_loader))
    if isinstance(train_batch, (tuple, list)):
        train_batch = train_batch[0]
    train_batch = rbm.prepare_input(train_batch).float().to(rbm.device)

    window_size = train_batch.shape[1] if train_batch.ndim == 3 else 1
    if temporal_persistence_weight > 0 and window_size <= 1:
        warnings.warn(
            f"temporal_persistence_weight={temporal_persistence_weight} has no effect with window_size={window_size} "
            f"(training batch shape {tuple(train_batch.shape)}). "
            "Pass WindowedSequenceDataset windows with window_size > 1, "
            "or set temporal_persistence_weight=0.",
            UserWarning, stacklevel=2,
        )
    elif window_size > 1 and temporal_persistence_weight == 0:
        warnings.warn(
            f"window_size={window_size} but temporal_persistence_weight=0: all window steps "
            "are used in the base loss, but temporal regularization is disabled. "
            "Set temporal_persistence_weight > 0 to enable it.",
            UserWarning, stacklevel=2,
        )

    if optimizer == "SGD":
        opt = torch.optim.SGD(rbm.parameters(), weight_decay=weight_decay, lr=lr)
    elif optimizer == "RMSprop":
        opt = torch.optim.RMSprop(rbm.parameters(), weight_decay=weight_decay, lr=lr)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {optimizer}")

    if loss_name == "pcd":
        rbm.reset_persistent_chain()

    t_start = tm.time()
    epoch = 0
    for epoch in range(1, max_epoch + 1):
        if verbose:
            print(f"Epoch {epoch}/{max_epoch} started...")

        rbm.train()
        loss_totals = {}
        loss_sample_count = 0
        final_gradient_norms = None
        for i, batch in enumerate(data_loader):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            x0 = batch.to(rbm.device)

            opt.zero_grad()
            components = rbm.compute_loss(
                x0,
                loss_name=loss_name,
                num_gibbs=num_gibbs,
                use_probability_last_v_update=use_probability_last_v_update,
                sparsity_target=sparsity_target,
                sparsity_weight=sparsity_weight,
                temporal_persistence_weight=temporal_persistence_weight,
                return_components=True,
            )
            interval_hit = (i + 1) % batch_log_interval == 0
            report_batch = bool(verbose and verbose > 1 and interval_hit)
            gradient_norms = None
            if report_batch or (log_mlflow and interval_hit) or i + 1 == len(data_loader):
                gradient_norms = _loss_gradient_norms(rbm, components, weight_decay)
                final_gradient_norms = gradient_norms
                step = (epoch - 1) * len(data_loader) + i + 1
                rbm.gradient_curve.append(dict(epoch=epoch, batch=i + 1, step=step, **gradient_norms))
                if log_mlflow:
                    mlflow.log_metrics({f"{rbm.name}/gradients/{key}": value
                                        for key, value in gradient_norms.items()}, step=step)
            components["Loss"].backward()
            opt.step()
            count = x0.shape[0]
            for key, value in components.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + value.detach() * count
            loss_sample_count += count

            if report_batch:
                with torch.no_grad():
                    progress = _get_progress_rbm(rbm, train_batch)
                    print(f"\n############### Epoch {epoch}/{max_epoch} | BATCH {i + 1}/{len(data_loader)} ###############")
                    print("Train (reference minibatch):")
                    pprint.pp(progress)
                    print("Loss components (current training batch, before update):")
                    pprint.pp({key: value.detach().item() for key, value in components.items()})
                    print("Gradient L2 norms (weighted loss terms, before update):")
                    pprint.pp(gradient_norms)
                    _print_hidden_bias(rbm)

        rbm.eval()

        with torch.no_grad():
            train_progress, hidden_stats = _get_dataset_progress_rbm(
                rbm, train_eval_loader, return_hidden_stats=True,
            )
            rbm.hidden_activation_curve.append(hidden_stats)
            train_progress.update({f"UpdateMean{key}": (value / loss_sample_count).item()
                                   for key, value in loss_totals.items()})
            rbm.learning_curve.append(train_progress)
            if log_mlflow:
                metrics = {f"{rbm.name}/train/{key}": value
                           for key, value in train_progress.items() if value is not None}
                metrics.update(_hidden_stat_metrics(rbm.name, 'train', hidden_stats))
                mlflow.log_metrics(metrics, step=epoch)

            if verbose:
                print(f"\n############### Epoch {epoch}/{max_epoch} ###############")
                print("Train (whole dataset; UpdateMean* averaged over epoch updates):")
                pprint.pp(train_progress)
                _print_hidden_bias(rbm)
                _print_hidden_stats(hidden_stats)
                print("Gradient L2 norms (last training batch, before update):")
                pprint.pp(final_gradient_norms)

        if valid_data is not None:
            with torch.no_grad():
                valid_progress, valid_hidden_stats = _get_dataset_progress_rbm(
                    rbm, valid_loader, return_hidden_stats=True,
                )
                rbm.valid_hidden_activation_curve.append(valid_hidden_stats)
                rbm.valid_curve.append(valid_progress)
                if log_mlflow:
                    metrics = {f"{rbm.name}/valid/{key}": value
                               for key, value in valid_progress.items() if value is not None}
                    metrics.update(_hidden_stat_metrics(rbm.name, 'valid', valid_hidden_stats))
                    mlflow.log_metrics(metrics, step=epoch)

            if verbose:
                print("Valid:")
                pprint.pp(valid_progress)
                _print_hidden_stats(valid_hidden_stats)

            if (
                    early_stopping
                    and epoch > min_epoch
                    and epoch > early_stopping_patience
            ):
                valid_metrics = np.array(
                    [v["MSE"] for v in rbm.valid_curve[-early_stopping_patience - 1:]]
                )
                if np.all(valid_metrics[1:] > valid_metrics[0]):
                    print("Early stop after valid metrics:", valid_metrics)
                    break

    rbm.eval()
    rbm.num_epoch = epoch
    print("Training finished after", timedelta(seconds=tm.time() - t_start))
    figure, histogram = _plot_hidden_unit_diagnostics(rbm, train_eval_loader)
    if log_mlflow:
        mlflow.log_figure(figure, f"figures/{rbm.name}_hidden_units.html")
        mlflow.log_figure(histogram, f"figures/{rbm.name}_hidden_unit_histograms.html")
    figure.show("browser")
    histogram.show("browser")
    
    
@_save_learning_log
def train_3way_rbm(
        rbm,
        train_data,
        valid_data=None,
        min_epoch: int = 0,
        max_epoch: int = 10,
        weight_decay: float = 0.0,
        lr: float = 0.01,
        batch_size: int = 128,
        shuffle: bool = True,
        num_gibbs: int = 1,
        verbose=True,
        early_stopping: bool = False,
        early_stopping_patience: int = 3,
        use_probability_last_v_update: bool = True,
        log_mlflow: bool = False,
        optimizer: str = "RMSprop",
):
    """
    Train the Three-Way Binary-Binary RBM.

    Parameters
    ----------
    train_data : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        Training data containing both visible and auxiliary parts.
        If a list or tuple is given, tensors are stacked vertically.
    valid_data : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] | None, optional
        Validation data in the same format as ``train_data``.
    min_epoch : int, optional
        Minimum number of epochs before early stopping is allowed.
    max_epoch : int, optional
        Maximum number of epochs.
    weight_decay : float, optional
        Weight decay coefficient for the optimizer.
    lr : float, optional
        Learning rate.
    batch_size : int, optional
        Mini-batch size.
    shuffle : bool, optional
        Whether to shuffle the training data.
    num_gibbs : int, optional
        Number of Gibbs steps used in Contrastive Divergence.
    verbose : bool | int, optional
        If truthy, prints epoch-level progress. If greater than 1, also prints
        batch-level progress.
    early_stopping : bool, optional
        Whether to enable validation-based early stopping.
    early_stopping_patience : int, optional
        Number of epochs used in the early-stopping window.
    use_probability_last_v_update : bool, optional
        If ``True``, the final visible negative sample in CD uses visible
        probabilities instead of sampled states.
    log_mlflow : bool, optional
        Whether to log selected hyperparameters to MLflow.
    optimizer : str, optional
        Optimizer name. Supported values are ``"SGD"`` and ``"RMSprop"``.

    Returns
    -------
    None

    Notes
    -----
    This trainer is specialized for the three-way RBM and does not rely on the
    generic :class:`XBinaryRBM` training logic, because this model has a joint
    visible/auxiliary state and free energy of the form ``F(v, u)``.

    Early stopping is based on the validation visible reconstruction MSE.
    """
    if log_mlflow:
        mlflow.log_param(rbm.name + "min_epoch", min_epoch)
        mlflow.log_param(rbm.name + "max_epoch", max_epoch)
        mlflow.log_param(rbm.name + "batch_size", batch_size)
        mlflow.log_param(rbm.name + "shuffle", shuffle)
        mlflow.log_param(rbm.name + "weight_decay", weight_decay)
        mlflow.log_param(rbm.name + "early_stopping_patience", early_stopping_patience)
        mlflow.log_param(rbm.name + "early_stopping", early_stopping)
        mlflow.log_param(rbm.name + "lr", lr)
        mlflow.log_param(rbm.name + "num_gibbs", num_gibbs)
        mlflow.log_param(rbm.name + "optimizer", optimizer)

    if isinstance(train_data, (list, tuple)):
        train_data = torch.vstack(train_data)
    train_data = train_data.float().to(rbm.device)

    if valid_data is not None:
        if isinstance(valid_data, (list, tuple)):
            valid_data = torch.vstack(valid_data)
        valid_data = valid_data.float().to(rbm.device)

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

    if optimizer == "SGD":
        opt = torch.optim.SGD(rbm.parameters(), weight_decay=weight_decay, lr=lr)
    elif optimizer == "RMSprop":
        opt = torch.optim.RMSprop(rbm.parameters(), weight_decay=weight_decay, lr=lr)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {optimizer}")

    t_start = tm.time()

    for epoch in range(1, max_epoch + 1):
        if verbose:
            print(f"Epoch {epoch}/{max_epoch} started...")

        rbm.train()
        for i, batch in enumerate(data_loader):
            batch = batch.to(rbm.device)

            opt.zero_grad()
            loss = rbm.contrastive_divergence(
                batch,
                num_gibbs=num_gibbs,
                use_probability_last_v_update=use_probability_last_v_update,
            )
            loss.backward()
            opt.step()

            if verbose and verbose > 1:
                with torch.no_grad():
                    progress = _get_progress_rbm(rbm, train_data)
                    print(f"\n############### Epoch {epoch}/{max_epoch} | BATCH {i + 1}/{len(data_loader)} ###############")
                    print("Train:")
                    pprint.pp(progress)
                    _print_hidden_bias(rbm)

        rbm.eval()

        with torch.no_grad():
            train_progress = _get_progress_rbm(rbm, train_data)
            rbm.learning_curve.append(train_progress)

            if verbose:
                print(f"\n############### Epoch {epoch}/{max_epoch} ###############")
                print("Train:")
                pprint.pp(train_progress)
                _print_hidden_bias(rbm)

        if valid_data is not None:
            with torch.no_grad():
                valid_progress = _get_progress_rbm(rbm, valid_data)
                rbm.valid_curve.append(valid_progress)

            if verbose:
                print("Valid:")
                pprint.pp(valid_progress)

            if (
                    early_stopping
                    and epoch > min_epoch
                    and epoch > early_stopping_patience
            ):
                valid_metrics = np.array(
                    [v["MSE"] for v in rbm.valid_curve[-early_stopping_patience - 1:]]
                )
                if np.all(valid_metrics[1:] > valid_metrics[0]):
                    print("Early stop after valid metrics:", valid_metrics)
                    break

    rbm.eval()
    rbm.num_epoch = epoch
    print("Training finished after", timedelta(seconds=tm.time() - t_start))
    
    
@torch.no_grad()
def _get_dataset_progress_rbm(rbm, data_loader, return_hidden_stats=False):
    """Aggregate full-dataset metrics and optional per-unit activation statistics."""
    totals = {}
    num_samples = 0
    hidden_totals = {}
    num_hidden_samples = 0
    for batch in data_loader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        inputs = rbm.prepare_input(batch).float().to(rbm.device)
        count = inputs.shape[0]
        if count == 0:
            continue
        if return_hidden_stats:
            progress, batch_stats = _get_progress_rbm(rbm, inputs, return_hidden_stats=True)
            num_hidden_samples += batch_stats.pop('count')
            for key, value in batch_stats.items():
                hidden_totals[key] = hidden_totals.get(key, 0.0) + value
        else:
            progress = _get_progress_rbm(rbm, inputs)
        for key, value in progress.items():
            if value is None or (key in totals and totals[key] is None):
                totals[key] = None
            else:
                totals[key] = totals.get(key, 0.0) + value * count
        num_samples += count
    if num_samples == 0:
        raise ValueError("Cannot compute validation metrics for an empty dataset")
    progress = {key: value / num_samples if value is not None else None
                for key, value in totals.items()}
    if return_hidden_stats:
        stats = {key: (value / num_hidden_samples).cpu().tolist()
                 for key, value in hidden_totals.items()}
        return progress, stats
    return progress


def _loss_gradient_norms(rbm, components, weight_decay=0.0):
    """Measure weighted contributions without modifying .grad or drawing samples."""
    parameters = [parameter for parameter in rbm.parameters() if parameter.requires_grad]
    terms = ('CD', 'PCD', 'ScoreMatchingLoss', 'ReconstructionLoss',
             'WeightedSparsityKL', 'WeightedTemporalBCE')
    result = {}
    for name in terms:
        if name not in components:
            continue
        value = components[name]
        gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True) \
            if value.requires_grad else [None] * len(parameters)
        squared_norm = rbm.weights.new_zeros(())
        bias_squared_norm = rbm.weights.new_zeros(())
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                squared_norm = squared_norm + gradient.detach().square().sum()
                if parameter is rbm.hidden_bias:
                    bias_squared_norm = gradient.detach().square().sum()
        result[f'{name}GradNorm'] = squared_norm.sqrt().item()
        result[f'{name}HiddenBiasGradNorm'] = bias_squared_norm.sqrt().item()
    # RMSprop/SGD add L2 weight decay to gradients outside the explicit loss.
    result['WeightDecayGradNorm'] = (sum(parameter.detach().square().sum()
                                         for parameter in parameters).sqrt() * weight_decay).item()
    result['WeightDecayHiddenBiasGradNorm'] = (rbm.hidden_bias.detach().norm() * weight_decay).item()
    return result


def _hidden_stat_metrics(name, partition, stats):
    return {f'{name}/{partition}/hidden/h{unit + 1}/{key}': value
            for key, values in stats.items() for unit, value in enumerate(values)}


def _print_hidden_stats(stats):
    labels = {'mean': 'Hidden activation means', 'low_fraction': 'Hidden fraction <= 0.01',
              'high_fraction': 'Hidden fraction >= 0.99'}
    for key, label in labels.items():
        values = ', '.join(f'{value:.4f}' for value in stats[key])
        print(f'{label} (h1..hN): [{values}]')


def _print_hidden_bias(rbm):
    """Print all signed hidden biases on one compact line, in unit order."""
    values = ", ".join(f"{value:.4f}" for value in rbm.hidden_bias.detach().cpu().tolist())
    print(f"Hidden biases (h1..hN): [{values}]")


@torch.no_grad()
def _plot_hidden_unit_diagnostics(rbm, data_loader, trace_limit=2000, num_bins=50):
    """Plot chronological probability traces and full-dataset unit histograms."""
    from plotly.subplots import make_subplots
    from plotly import graph_objects as go

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    counts = None
    traces = []
    trace_count = 0
    for batch in data_loader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        inputs = rbm.prepare_input(batch).float().to(rbm.device)
        probabilities = rbm.v2h(inputs.reshape(-1, inputs.shape[-1])).detach().cpu().numpy()
        if counts is None:
            counts = np.zeros((probabilities.shape[1], num_bins), dtype=np.int64)
        for unit in range(probabilities.shape[1]):
            counts[unit] += np.histogram(probabilities[:, unit], bins=edges)[0]
        if trace_count < trace_limit:
            part = probabilities[:trace_limit - trace_count].copy()
            traces.append(part)
            trace_count += len(part)
    if counts is None:
        raise ValueError("Cannot plot hidden units for an empty dataset")
    traces = np.concatenate(traces, axis=0)
    num_units = counts.shape[0]
    figure = make_subplots(rows=num_units, cols=1,
                          subplot_titles=[f"h{unit + 1}: activation probability" for unit in range(num_units)],
                          vertical_spacing=min(0.02, 0.2 / num_units))
    histogram = make_subplots(rows=num_units, cols=1,
                             subplot_titles=[f"h{unit + 1}: histogram" for unit in range(num_units)],
                             vertical_spacing=min(0.02, 0.2 / num_units))
    centers = (edges[:-1] + edges[1:]) / 2
    for unit in range(num_units):
        figure.add_trace(go.Scattergl(y=traces[:, unit], mode="lines",
                                     name=f"h{unit + 1}"), row=unit + 1, col=1)
        histogram.add_trace(go.Bar(x=centers, y=counts[unit], width=1.0 / num_bins,
                                  name=f"h{unit + 1} histogram"), row=unit + 1, col=1)
        figure.update_yaxes(range=[0, 1], title_text="P(h=1 | v)", row=unit + 1, col=1)
        figure.update_xaxes(title_text="Training window step", row=unit + 1, col=1)
        histogram.update_xaxes(range=[0, 1], title_text="P(h=1 | v)", row=unit + 1, col=1)
        histogram.update_yaxes(title_text="Count", row=unit + 1, col=1)
    figure.update_layout(
        title=f"{rbm.name}: hidden units (first {trace_count} training window steps)",
        height=max(400, 200 * num_units), width=1200, showlegend=False,
    )
    histogram.update_layout(
        title=f"{rbm.name}: hidden-unit histograms (all training window steps)",
        height=max(400, 200 * num_units), width=1000, showlegend=False,
    )
    return figure, histogram


@torch.no_grad()
def _full_dataset_mse(rbm, data_loader):
    """Compute element-weighted reconstruction MSE with bounded batch memory."""
    squared_error = torch.zeros((), dtype=torch.float64, device=rbm.device)
    num_elements = 0
    for batch in data_loader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        inputs = rbm.prepare_input(batch).float().to(rbm.device)
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        reconstruction, _ = rbm.recon(flat_inputs)
        squared_error += (flat_inputs - reconstruction).square().sum(dtype=torch.float64)
        num_elements += inputs.numel()
    if num_elements == 0:
        raise ValueError("Cannot compute training MSE for an empty dataset")
    return (squared_error / num_elements).item()


def _get_progress_rbm(rbm, d: torch.Tensor, return_hidden_stats=False):
    """
    Compute simple progress metrics for logging.

    Parameters
    ----------
    d : torch.Tensor
        Dataset or minibatch.

    Returns
    -------
    dict
        Dictionary of summary metrics.

    Notes
    -----
    Assumes external helper functions ``mse(...)`` and ``sparsity(...)`` are
    available in scope.
    """
    with torch.no_grad():
        if isinstance(rbm, XBinaryRBM):
            d = d.reshape(-1, d.shape[-1])
        r, h = rbm.recon(d)
        e = rbm.free_energy(d)

        d_flat = d.reshape(d.shape[0], -1)
        r_flat = r.reshape(r.shape[0], -1)

        # BCE is only valid for probabilities/targets in [0, 1].
        # On CUDA, out-of-range BCE inputs can trigger device-side asserts.
        d_in_unit = bool(torch.isfinite(d_flat).all() and (d_flat >= 0).all() and (d_flat <= 1).all())
        r_in_unit = bool(torch.isfinite(r_flat).all() and (r_flat >= 0).all() and (r_flat <= 1).all())
        bce_val = bce(d_flat, r_flat).item() if d_in_unit and r_in_unit else None

        mse_val = mse(d_flat, r_flat).item()
        progress = dict(
            MSE=mse_val,
            BCE=bce_val,
            Sparsity=sparsity(h).item(),
            Energy=torch.mean(e).item(),
            Weights=torch.mean(torch.abs(rbm.weights)).item(),
            VisBias=torch.mean(torch.abs(rbm.visible_bias)).item(),
            HidBias=torch.mean(torch.abs(rbm.hidden_bias)).item(),
        )
        if return_hidden_stats:
            hidden = h.reshape(-1, h.shape[-1])
            stats = dict(mean=hidden.sum(dim=0, dtype=torch.float64),
                         low_fraction=(hidden <= 0.01).sum(dim=0, dtype=torch.float64),
                         high_fraction=(hidden >= 0.99).sum(dim=0, dtype=torch.float64),
                         count=hidden.shape[0])
            return progress, stats
    return progress
