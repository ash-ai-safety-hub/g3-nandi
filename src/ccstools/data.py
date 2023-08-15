from __future__ import annotations

__all__ = ['balance_filter_map_dataset']

from collections.abc import Callable
from tqdm.auto import tqdm
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import datasets


def balance_filter_map_dataset(
    examples_per_label: int,
    dataset: datasets.Dataset,
    function: Callable[[dict[str, Any]], dict[str, Any] | None],
    *,
    show_progress: bool = True,
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
    show_progress : bool, default: True
        Whether to display a tqdm progress bar showing how many examples
        have been generated.
    **kwargs : Any
        Any other arguments to pass to `Dataset.map`.
        This should not contain: `batched`, `batch_size` or `num_proc`.

    """
    for illegal_key in ('batched', 'batch_size', 'num_proc'):
        if illegal_key in kwargs:
            raise ValueError(f'Illegal keyword argument: `{illegal_key}`')

    # TODO can this be parallalised?
    count_per_label = [0] * dataset.features['label'].num_classes

    with tqdm(total=examples_per_label * len(count_per_label),
              desc='Generating dataset',
              disable=not show_progress) as progress_bar:
        # hack so that `Dataset.map` can generate a fingerprint
        class TqdmWrapper:
            def __init__(self, bar: tqdm): self.bar = bar
            def __getstate__(self): return {}
        wrapped_progress_bar = TqdmWrapper(progress_bar)

        def apply_function(batch: dict[str, list[Any]],
                           *args: Any,
                           **kwargs: Any) -> dict[str, list[Any]]:
            # unbatch example
            example = {key: value[0] for key, value in batch.items()}

            if count_per_label[example['label']] < examples_per_label:
                mapped_example = function(example, *args, **kwargs)
                if mapped_example is not None:
                    count_per_label[example['label']] += 1
                    wrapped_progress_bar.bar.update(1)

                    # rebatch mapped example
                    return {k: [v] for k, v in mapped_example.items()}

            return {key: [] for key in example}  # empty batch

        mapped_dataset = dataset.map(apply_function,
                                     batched=True,
                                     batch_size=1,
                                     **kwargs)

        # useful for cached datasets
        progress_bar.update(len(mapped_dataset) - progress_bar.n)

    return mapped_dataset
