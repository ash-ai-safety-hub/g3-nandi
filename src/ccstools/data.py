from __future__ import annotations

__all__ = ['balance_filter_map_dataset']

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import datasets


def balance_filter_map_dataset(
    examples_per_label: int,
    dataset: datasets.Dataset,
    function: Callable[[dict[str, Any]], dict[str, Any] | None],
    **kwargs: Any
) -> datasets.Dataset:
    """Balance a dataset to a specific size while also mapping over it.

    Takes the examples in order, so if a random sample is desired, the
    dataset should be pre-shuffled.

    Parameters
    ----------
    examples_per_label : int
        The number of examples expected per label. If this exceeds the
        number of examples in the dataset, then the output may still be
        inbalanced.
    dataset : datasets.Dataset
        The dataset to process.
    function : Callable[[dict[str, Any]], dict[str, Any] | None]
        The mapping function to call for each example.
    **kwargs : Any
        Any other arguments to pass to `Dataset.map`.
        This should not contain: `batched` or `batch_size`.
    """
    for illegal_key in ('batched', 'batch_size'):
        if illegal_key in kwargs:
            raise ValueError(f'Illegal keyword argument: `{illegal_key}`')

    # TODO can this be parallalised?
    count_per_label = [0] * dataset.features['label'].num_classes

    def f(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # unbatch example
        example = {key: value[0] for key, value in batch.items()}

        if count_per_label[example['label']] < examples_per_label:
            mapped_example = function(example)
            if mapped_example is not None:
                count_per_label[example['label']] += 1

                # rebatch mapped example
                return {k: [v] for k, v in mapped_example.items()}

        return {key: [] for key in example}  # empty batch

    return dataset.map(f, batched=True, batch_size=1, **kwargs)
