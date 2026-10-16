import torch
from denta.rbms.bb import BinaryBinaryRBM
from denta.rbms.gb import GaussianBinaryRBM
from denta.rbms.training import train_rbm
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def test_bbrbm(
    seed: int = 0,
    n_visible: int = 6,
    n_hidden: int = 3,
    n_train: int = 2000,
    n_valid: int = 400,
    epochs: int = 150,
    lr: float = 1e-2,
    batch_size: int = 128,
    verbose: bool = False,
):
    """
    Train a BinaryBinaryRBM on a simple clustered binary dataset and verify
    that it learns a meaningful model.

    The test checks:
    1. Reconstruction error decreases after training.
    2. Free energy on true data is lower than on random binary noise.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_visible : int
        Number of visible units.
    n_hidden : int
        Number of hidden units.
    n_train : int
        Number of training samples.
    n_valid : int
        Number of validation samples.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Batch size.
    verbose : bool
        Whether to print progress from `train_rbm`.

    Returns
    -------
    dict
        Summary statistics from the test.
    """
    torch.manual_seed(seed)

    # Three simple binary prototype patterns.
    prototypes = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    assert prototypes.size(1) == n_visible, "Adjust prototypes or n_visible."

    def sample_binary_mixture(n):
        idx = torch.randint(0, prototypes.size(0), (n,))
        x = prototypes[idx].clone()

        # Small bit-flip noise to make the task non-trivial.
        flip_mask = torch.bernoulli(torch.full_like(x, 0.05))
        x = torch.logical_xor(x.bool(), flip_mask.bool()).float()
        return x

    train_x = sample_binary_mixture(n_train)
    valid_x = sample_binary_mixture(n_valid)
    random_x = torch.bernoulli(torch.full((n_valid, n_visible), 0.5))

    model = BinaryBinaryRBM(n_visible=n_visible, n_hidden=n_hidden)

    with torch.no_grad():
        recon_before = model.reconstruction_loss(valid_x).item()
        fe_data_before = model.free_energy(valid_x).mean().item()
        fe_noise_before = model.free_energy(random_x).mean().item()

    train_rbm(model,
        train_data=[train_x],
        valid_data=[valid_x],
        loss_name="cd",
        max_epoch=epochs,
        lr=lr,
        batch_size=batch_size,
        num_gibbs=1,
        optimizer="RMSprop",
        verbose=verbose,
    )

    with torch.no_grad():
        recon_after = model.reconstruction_loss(valid_x).item()
        fe_data_after = model.free_energy(valid_x).mean().item()
        fe_noise_after = model.free_energy(random_x).mean().item()

    # Sanity checks.
    assert recon_after < recon_before, (
        f"BBRBM reconstruction did not improve: "
        f"{recon_before:.6f} -> {recon_after:.6f}"
    )
    assert fe_data_after < fe_noise_after, (
        f"BBRBM free energy does not prefer real data: "
        f"FE(data)={fe_data_after:.6f}, FE(noise)={fe_noise_after:.6f}"
    )

    result = {
        "recon_before": recon_before,
        "recon_after": recon_after,
        "fe_data_before": fe_data_before,
        "fe_noise_before": fe_noise_before,
        "fe_data_after": fe_data_after,
        "fe_noise_after": fe_noise_after,
    }

    print("test_bbrbm passed")
    for k, v in result.items():
        print(f"{k}: {v:.6f}")

    return result


def test_gbrbm(
    seed: int = 0,
    n_visible: int = 2,
    n_hidden: int = 4,
    n_train: int = 3000,
    n_valid: int = 600,
    epochs: int = 200,
    lr: float = 5e-3,
    batch_size: int = 128,
    sigma: float = 0.25,
    verbose: bool = False,
):
    """
    Train a GaussianBinaryRBM on a simple Gaussian mixture dataset and verify
    that it learns a meaningful model.

    The test checks:
    1. Reconstruction error decreases after training.
    2. Free energy on true mixture samples is lower than on broad out-of-distribution noise.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_visible : int
        Number of visible units. This test expects 2 for the toy dataset.
    n_hidden : int
        Number of hidden units.
    n_train : int
        Number of training samples.
    n_valid : int
        Number of validation samples.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Batch size.
    sigma : float
        Visible standard deviation used to initialize the GBRBM.
    verbose : bool
        Whether to print progress from `train_rbm`.

    Returns
    -------
    dict
        Summary statistics from the test.
    """
    torch.manual_seed(seed)
    assert n_visible == 2, "This simple toy test is written for 2D visible data."

    means = torch.tensor(
        [
            [-2.0, -2.0],
            [-2.0,  2.0],
            [ 2.0, -2.0],
            [ 2.0,  2.0],
        ],
        dtype=torch.float32,
    )

    def sample_gaussian_mixture(n):
        idx = torch.randint(0, means.size(0), (n,))
        x = means[idx] + 0.30 * torch.randn(n, 2)
        return x

    train_x = sample_gaussian_mixture(n_train)
    valid_x = sample_gaussian_mixture(n_valid)

    # OOD reference: much broader noise centered at zero.
    random_x = 4.0 * torch.randn(n_valid, n_visible)

    model = GaussianBinaryRBM(
        n_visible=n_visible,
        n_hidden=n_hidden,
        sigma=sigma,
        is_sigma_learnable=False,
    )

    with torch.no_grad():
        recon_before = model.reconstruction_loss(valid_x).item()
        fe_data_before = model.free_energy(valid_x).mean().item()
        fe_noise_before = model.free_energy(random_x).mean().item()

    train_rbm(model,
        train_data=[train_x],
        valid_data=[valid_x],
        loss_name="cd",
        max_epoch=epochs,
        lr=lr,
        batch_size=batch_size,
        num_gibbs=1,
        optimizer="RMSprop",
        verbose=verbose,
    )

    with torch.no_grad():
        recon_after = model.reconstruction_loss(valid_x).item()
        fe_data_after = model.free_energy(valid_x).mean().item()
        fe_noise_after = model.free_energy(random_x).mean().item()

    assert recon_after < recon_before, (
        f"GBRBM reconstruction did not improve: "
        f"{recon_before:.6f} -> {recon_after:.6f}"
    )
    assert fe_data_after < fe_noise_after, (
        f"GBRBM free energy does not prefer real data: "
        f"FE(data)={fe_data_after:.6f}, FE(noise)={fe_noise_after:.6f}"
    )

    result = {
        "recon_before": recon_before,
        "recon_after": recon_after,
        "fe_data_before": fe_data_before,
        "fe_noise_before": fe_noise_before,
        "fe_data_after": fe_data_after,
        "fe_noise_after": fe_noise_after,
    }

    print("test_gbrbm passed")
    for k, v in result.items():
        print(f"{k}: {v:.6f}")

    return result


if __name__ == "__main__":
    test_bbrbm()
    # test_gbrbm()
    # test_relubrbm()
    # test_3wbbrbm()
