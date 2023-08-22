from __future__ import annotations

__all__ = [
    'ccs_loss',
    'make_probes',
    'normalize',
    'train_test_split',
]

import torch

_DimT = 'int | str | tuple[int | str, ...] | None'


def ccs_loss(probabilities: torch.Tensor,
             answer_dim: _DimT = 'answer',
             mean_dim: _DimT = None) -> torch.Tensor:
    """Calculate the CCS loss for a tensor of probabilities.

    The loss is equal to the sum of the consistency loss and the
    confidence loss.
    Consistency loss = (sum of answer probabilites - 1)^2
    Confidence loss = (minimum answer probability)^2

    Parameters
    ----------
    probabilities : torch.Tensor
        The tensor of probabilites.
    answer_dim : int or str or tuple[int | str], default: 'answer'
        The dimension corresponding to the answers.
    mean_dim : int or str or tuple[int | str], optional
        The dimension(s) to average the loss over. By default, all
        values are averaged to return a single loss value.

    Returns
    -------
    torch.Tensor
        The loss.

    """
    consistency_loss = (probabilities.sum(answer_dim) - 1) ** 2
    confidence_loss = probabilities.min(answer_dim).values ** 2
    return torch.mean(consistency_loss + confidence_loss, mean_dim)


def make_probes(input_size: int, output_size: int = 1) -> torch.nn.Module:
    """Create a set of CCS probes.

    Represented by a linear layer followed by a sigmoid.

    Parameters
    ----------
    input_size : int
        The number of features in the input.
    output_size : int, default: 1
        The number of probes.

    Returns
    -------
    torch.nn.Module
        A PyTorch model representing the probes.

    """
    return torch.nn.Sequential(torch.nn.Linear(input_size, output_size),
                               torch.nn.Sigmoid())


def normalize(representations: torch.Tensor,
              dim: _DimT = 0) -> torch.Tensor:
    """Normalize a tensor of representations over a given dimension.

    Parameters
    ----------
    representations : torch.Tensor
        The tensor of representations.
    dim : int or str or tuple[int | str], default: 0
        The dimension(s) to normalize over.
    """
    mean = representations.mean(dim)
    std_dev = representations.std(dim, correction=0)
    return (representations - mean) / std_dev


def train_test_split(
    *tensors: torch.Tensor,
    train_size: float | int | None = None,
    test_size: float | int | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Create a train-test split of data tensors.

    At least one of `train_size` and `test_size` must be specified.

    Parameters
    ----------
    *tensors : torch.Tensor
        The tensors to split. They must all be the same length (in the
        first dimension).
    train_size : float or int, optional
        The size of the training set. If given as a float, it is
        interpreted as a fraction of the total size. If given as an int,
        it is interpreted as the number of examples.
    test_size : float or int, optional
        The size of the test set. If given as a float, it is interpreted
        as a fraction of the total size. If given as an int, it is
        interpreted as the number of examples.

    """
    assert len(set(map(len, tensors))) == 1
    size = len(tensors[0])

    if test_size is None and train_size is None:
        raise ValueError('One of `test_size` and `train_size` must be '
                         'specified.')

    try:
        if 0 < train_size < 1:
            train_size = round(size * train_size)
    except TypeError:
        pass

    try:
        if 0 < test_size < 1:
            test_size = round(size * test_size)
    except TypeError:
        pass

    if train_size is None:
        train_size = size - test_size
    if test_size is None:
        test_size = size - train_size

    assert train_size > 0
    assert test_size > 0
    assert train_size + test_size == size

    return tuple((tensor[:train_size], tensor[train_size:])
                 for tensor in tensors)
