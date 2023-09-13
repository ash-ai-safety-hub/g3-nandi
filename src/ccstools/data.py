from __future__ import annotations

__all__ = ['balance_filter_map_dataset', 'add_prompts', 'make_binary']

from collections.abc import Callable, Iterable
from typing import Any, TYPE_CHECKING

from datasets import ClassLabel
import numpy as np
from tqdm.auto import tqdm


if TYPE_CHECKING:
    from typing import Sequence, TypeVar

    import datasets
    import transformers

    from ccstools.prompts import Template

    _T = TypeVar('_T')


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
        This should not contain: `num_proc`.

    """
    # TODO can this be parallalised?
    ILLEGAL_KEYS = ('num_proc',)
    for illegal_key in ILLEGAL_KEYS:
        if illegal_key in kwargs:
            raise ValueError(f'Illegal keyword argument: `{illegal_key}`')

    batched = kwargs.pop('batched', False)
    batch_size = kwargs.pop('batch_size', 1000)

    num_classes = dataset.features['label'].num_classes
    remaining_per_label = [examples_per_label] * num_classes

    with tqdm(total=examples_per_label * num_classes,
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
            if any(remaining_per_label):
                # unbatch example, if required
                if not batched:
                    example = {key: value[0] for key, value in batch.items()}

                mapped_example = function(example, *args, **kwargs)
                if mapped_example is not None:
                    bar = wrapped_progress_bar.bar
                    if batched:
                        removed = False
                        to_keep = []
                        for i, label in enumerate(mapped_example['label']):
                            if remaining_per_label[label]:
                                remaining_per_label[label] -= 1
                                to_keep.append(i)
                            else:
                                removed = True
                        bar.update(len(to_keep))
                        if removed:
                            return {k: [v[i] for i in to_keep]
                                    for k, v in mapped_example.items()}
                        else:
                            return mapped_example
                    elif remaining_per_label[mapped_example['label']]:
                        remaining_per_label[mapped_example['label']] -= 1
                        bar.update(1)
                        # rebatch mapped example
                        return {k: [v] for k, v in mapped_example.items()}

            return {key: [] for key in batch}  # empty batch

        mapped_dataset = dataset.map(apply_function,
                                     batched=True,
                                     batch_size=batch_size if batched else 1,
                                     **kwargs)

        # useful for cached datasets
        progress_bar.update(len(mapped_dataset) - progress_bar.n)

    return mapped_dataset


def add_prompts(example: dict[str, Any],
                *,
                tokenizer: transformers.PreTrainedTokenizer,
                templates: Iterable[Template],
                max_length: int,
                prefix: str = '',
                suffix: str = '') -> dict[str, Any] | None:
    """Add tokenized prompts to a given dataset example.

    Given a tokenizer and a set of prompts, this applies all of the
    templates to the input example, and tokenizes them into a single
    batched tensor.

    In particular, this adds the following entries to the input example:
    - 'prompts' : list[dict[str, torch.Tensor]]
        A list of each prompt tokenized.
    - 'answered_prompts' : list[dict[str, torch.Tensor]]
        A list of tokenized batches, where each batch corresponds to a
        single prompt with each possible answer appended
    - 'answers' : list[dict[str, torch.Tensor]]
        A list of tokenized batches, where each batch corresponds to all
        possible answers to a single prompt (without the prompt at the
        beginning).

    A maximum token length can be specified, such that if any prompt is
    longer than this maximum length, the input is rejected and the
    function returns `None`.
    Thus, this can be used with `balance_filter_map_dataset`.

    Parameters
    ----------
    example : dict[str, Any]
        The example to add prompts to.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use to tokenize the prompts.
    templates : Iterable[Template]
        A list of templates that generate the prompts to be added.
    max_length : int
        The maximum length a generated prompt can be (number of tokens).
    prefix : str, optional
        An optional prefix to add to each prompt.
    suffix : str, optional
        An optional suffix to add to each prompt.

    Returns
    -------
    dict[str, Any] or None
        The example with extra entries for the added prompts, unless any
        prompt is too long, in which case `None` is returned.

    """
    answered_prompts = example['answered_prompts'] = []
    prompts = example['prompts'] = []  # without answers
    choices = example['choices'] = []
    for template in templates:
        template_output = template(example, prefix=prefix, suffix=suffix)
        assert template_output.choices

        answered_prompt = tokenizer([f'{template_output.prompt} {choice}'
                                     for choice in template_output.choices],
                                    padding=True,
                                    return_length=True)

        length = answered_prompt.pop('length')
        if max(length) > max_length:
            return None

        answered_prompts.append(answered_prompt)
        prompts.append(tokenizer(template_output.prompt))
        choices.append(tokenizer(template_output.choices, padding=True))

    return example


def make_binary_example(example: dict[str, Any],
                        class_label: ClassLabel,
                        new_label_columns: Sequence[str],
                        rng: np.random.Generator) -> dict[str, Any]:
    label = example['label']

    other = rng.integers(class_label.num_classes - 1, dtype=int)
    if other >= label:
        other += 1

    new_label = rng.integers(2, dtype=int)
    example['label'] = new_label
    example[new_label_columns[new_label]] = class_label.int2str(label)
    example[new_label_columns[1-new_label]] = class_label.int2str(other)

    return example


def make_binary(dataset: datasets.Dataset,
                new_label_columns: Sequence[str] = ('label0', 'label1'),
                seed: int | None = None) -> datasets.Dataset:
    assert len(new_label_columns) == 2

    rng = np.random.default_rng(seed)
    dataset = dataset.map(make_binary_example,
                          fn_kwargs={'class_label': dataset.features['label'],
                                     'new_label_columns': new_label_columns,
                                     'rng': rng})

    features = dataset.features.copy()
    features['label'] = ClassLabel(names=new_label_columns)
    return dataset.cast(features)
