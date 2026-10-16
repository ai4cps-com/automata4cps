import pprint

import numpy as np
import pandas as pd
from ml4cps import tools
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

loss_bce_pp = nn.BCELoss(reduction='none')
loss_mse_pp = nn.MSELoss(reduction='none')
loss_bce = nn.BCELoss()
loss_mse = nn.MSELoss()




def load_data_from_dataset(dataset, sample_size=1000, shufle=True):
    temp_loader = DataLoader(dataset, batch_size=min(sample_size, len(dataset)), shuffle=shufle)
    return next(iter(temp_loader))

def binary_vector_to_mode(h):
    if type(h) is list:
        return [binary_vector_to_mode(x) for x in h]
    else:
        if isinstance(h, torch.Tensor):
            if h.ndim == 3:
                h = h[:,-1,:]
            df = pd.DataFrame(h.cpu().detach().numpy())
        elif isinstance(h, pd.DataFrame):
            df = h
        else:
            df =pd.DataFrame(h)
        df = df.astype(int)
        return np.array([pprint.pformat(row, compact=True, width=10000).replace(".0", "")
                         for row in df.itertuples(index=False, name=None)])

def encode_ordinal(x, columns, order=None):
    """
    Encode ordinal categorical variables into ordered numeric representations.

    This is a thin wrapper around ``tools.encode_ordinal``. It transforms the
    specified ordinal columns of the input data into numeric values that respect
    the ordering of the categories.

    Args:
        x:
            Input data structure containing the columns to encode. Its expected
            type and behavior depend on ``tools.encode_ordinal``.
        columns:
            Column name or collection of column names to encode as ordinal
            variables.
        order (dict or None, optional):
            Optional mapping that defines the category order for each encoded
            column. If omitted, the default behavior of
            ``tools.encode_ordinal`` is used.

    Returns:
        Encoded version of ``x`` with the specified ordinal columns converted
        to ordered numeric form.

    Notes:
        - Use this function when categories have a meaningful order, such as
          ``low < medium < high``.
        - The exact output format depends on ``tools.encode_ordinal``.

    Example:
        >>> x_enc = encode_ordinal(x, columns=["priority"], order={"priority": ["low", "medium", "high"]})
    """
    return tools.encode_ordinal(x, columns, order)


def encode_nominal(x, columns=None):
    """
    Encode nominal categorical variables into a machine-readable representation.

    This is a thin wrapper around ``tools.encode_nominal``. It transforms the
    specified nominal columns of the input data into an encoded representation
    suitable for downstream modeling.

    Args:
        x:
            Input data structure containing the columns to encode. Its expected
            type and behavior depend on ``tools.encode_nominal``.
        columns (None, str, or collection, optional):
            Column name or collection of column names to encode as nominal
            variables. If None, the default behavior of ``tools.encode_nominal``
            is used.

    Returns:
        Encoded version of ``x`` with the selected nominal columns transformed
        into a machine-readable representation.

    Notes:
        - Use this function for unordered categories, such as colors, IDs,
          or labels with no intrinsic ranking.
        - The exact encoding scheme and output format depend on
          ``tools.encode_nominal``.

    Example:
        >>> x_enc = encode_nominal(x, columns=["color", "machine_type"])
    """
    return tools.encode_nominal(x, columns)


def mse(x, r, per_point=False):
    if per_point:
        return loss_mse_pp(x, r)
    else:
        return loss_mse(x, r)

def bce(x, r, per_point=False):
    if per_point:
        return loss_bce_pp(x, r)
    else:
        return loss_bce(x, r)

def transience_bce(past_h, h, per_point=False):
    if per_point:
        return loss_bce_pp(past_h, h)
    else:
        return loss_bce(past_h, h)

def sparsity(activations, per_point=False):
    if per_point:
        return torch.mean(activations, dim=1)
    else:
        return torch.mean(activations)