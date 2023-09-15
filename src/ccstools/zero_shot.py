"""
Zero-shot
---------
Functions for running zero-shot experiments.

"""

from __future__ import annotations

__all__ = ['get_logprobs', 'add_logprobs', 'evaluate_logprobs']

from typing import Any, TYPE_CHECKING

import torch

from ccstools.models import get_model_type

if TYPE_CHECKING:
    import transformers

_DimT = 'int | str | tuple[int | str, ...] | None'


def get_logprobs_lm(
    model: transformers.PreTrainedModel,
    prompts: dict[str, torch.Tensor],
    choices: dict[str, torch.Tensor],
    ignore_final_token: bool,
    use_decoder_input: bool,
    return_hidden_states: bool
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor]]:
    """Calculate log-probabilities of given prompt/answer pairs.

    Optionally, also return the hidden states of the final two tokens.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model used to generate the output logits.
    prompts : dict[str, torch.Tensor]
        A tokenized prompt or batch of prompts.
    choices : dict[str, torch.Tensor]
        A batch of tokenized choices.
    ignore_final_token : bool
        Whether to ignore the log-probability of outputting the final
        token (typically a special end-of-text token).
    use_decoder_input : bool
        Whether a separate decoder input is used for the choices, used
        for encoder-decoder models.
    return_hidden_states : bool
        Whether to also return the hidden states of the final two
        tokens. For encoder-decoder models, if `use_decoder_input` is
        set to `True`, then this returns the decoder hidden states,
        otherwise, it returns the encoder hidden states.

    Returns
    -------
    torch.Tensor
        The log-probability of each answer to the prompt.
    tuple[torch.Tensor]
        If `return_hidden_states` is `True`, then this second argument
        is returned. Each entry contains the hidden state of the final
        two tokens at that layer.

    """
    labels = choices['input_ids']

    with torch.no_grad():
        if use_decoder_input:
            n_choices = labels.size(0)
            output = model(
                **{k: v.repeat(n_choices, 1) for k, v in prompts.items()},
                labels=labels,
                output_hidden_states=True
            )
        else:
            output = model(**prompts, output_hidden_states=True)

    all_token_logprobs = output.logits.log_softmax(-1)

    # get the tokens for the choices by remove the padding tokens
    # and optionally the final token
    choice_tokens = [tokens[:attention_mask.sum() - int(ignore_final_token)]
                     for tokens, attention_mask
                     in zip(labels, choices['attention_mask'])]

    start = -labels.size(1)  # works whether or not the labels are separate
    logprobs = torch.stack(tuple(
        token_logprobs[range(start, start + len(tokens)), tokens].sum()
        for tokens, token_logprobs in zip(choice_tokens, all_token_logprobs)
    ))

    if return_hidden_states:
        if use_decoder_input:
            hidden_states = output.decoder_hidden_states
            lengths = choices['attention_mask'].sum(1)
        else:
            try:
                hidden_states = output.encoder_hidden_states
            except AttributeError:
                hidden_states = output.hidden_states
            lengths = prompts['attention_mask'].sum(1)

        final_hidden_states = [
            state[[[i, i] for i in range(len(lengths))],
                  [[length - 2, length - 1] for length in lengths]]
            for state in hidden_states
        ]
        return logprobs, final_hidden_states
    else:
        return logprobs


def get_logprobs_nli(
    model: transformers.PreTrainedModel,
    prompts: dict[str, torch.Tensor],
    return_hidden_states: bool
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor]]:
    """Calculate log-probabilities of given prompt/answer pairs, in NLI.

    Optionally, also return the hidden states of the final two tokens.

    This is used in encoder-only models that have been fine-tuned on an
    NLI task.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model used to generate the output logits.
    prompts : dict[str, torch.Tensor]
        A tokenized batch of a prompt followed by each possible answer.
    return_hidden_states : bool
        Whether to also return the hidden states of the final two
        tokens.

    Returns
    -------
    torch.Tensor
        The log-probability of the NLI labels for each prompt/answer
        pair.
    tuple[torch.Tensor]
        If `return_hidden_states` is `True`, then this second argument
        is returned. Each entry contains the hidden state of the final
        two tokens at that layer.

    """
    with torch.no_grad():
        output = model(**prompts, output_hidden_states=True)

    assert output.logits.shape[1] == 3
    logprobs = output.logits.log_softmax(-1)

    if return_hidden_states:
        hidden_states = output.hidden_states
        lengths = prompts['attention_mask'].sum(1)

        final_hidden_states = [
            state[[[i, i] for i in range(len(lengths))],
                  [[length - 2, length - 1] for length in lengths]]
            for state in hidden_states
        ]
        return logprobs, final_hidden_states
    else:
        return logprobs


def get_logprobs_single(
    model: transformers.PreTrainedModel,
    prompts: dict[str, torch.Tensor],
    answered_prompts: dict[str, torch.Tensor],
    choices: dict[str, torch.Tensor],
    return_hidden_states: bool
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor]]:
    """Get log-probabilities for a single batch of data.

    Optionally, also return the hidden states of the final two tokens.

    This delegates depending on the type of the model:
    - If encoder-only, uses `get_logprobs_nli`
    - Otherwise, uses `get_logprobs_lm`

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model used to generate the output logits.
    prompts : dict[str, torch.Tensor]
        A tokenized prompt.
    answered_prompts : dict[str, torch.Tensor]
        A tokenized batch of a prompt followed by each possible answer.
    return_hidden_states : bool
        Whether to also return the hidden states of the final two
        tokens.

    Returns
    -------
    torch.Tensor
        The log-probability for each prompt/answer pair.
    tuple[torch.Tensor]
        If `return_hidden_states` is `True`, then this second argument
        is returned. Each entry contains the hidden state of the final
        two tokens at that layer.

    """
    model_type = get_model_type(model)
    if model_type == 'encoder':
        return get_logprobs_nli(model=model,
                                prompts=answered_prompts,
                                return_hidden_states=return_hidden_states)
    elif model_type == 'encoder_decoder':
        return get_logprobs_lm(model=model,
                               prompts=prompts,
                               choices=choices,
                               ignore_final_token=True,
                               use_decoder_input=True,
                               return_hidden_states=return_hidden_states)
    elif model_type == 'decoder':
        return get_logprobs_lm(
            model=model,
            prompts=answered_prompts,
            choices=choices,
            ignore_final_token=False,  # in GPT models, there is no EOS token
            use_decoder_input=False,
            return_hidden_states=return_hidden_states
        )
    raise AssertionError('unreachable')


def get_logprobs(
    model: transformers.PreTrainedModel,
    example: dict[str, Any],
    return_hidden_states: bool,
    prompts_column: str = 'prompts',
    answered_prompts_column: str = 'answered_prompts',
    choices_column: str = 'choices'
) -> list[torch.Tensor] | list[tuple[torch.Tensor, tuple[torch.Tensor]]]:
    """Get log-probabilities for a single example in a dataset.

    Optionally, also return the hidden states of the final two tokens.

    This calls `get_logprobs_single` for each prompt.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model used to generate the output logits.
    example : dict[str, Any]
        The example for which to generate the logits.
    return_hidden_states : bool
        Whether to also return the hidden states of the final two
        tokens.
    prompts_column : str, default: 'prompts'
        The name of the column containing the prompts.
    answered_prompts_column : str, default 'answered_prompts'
        The name of the column containing the answered prompts.
    choices_column : str, default: 'choices'
        The name of the column containing the choices.

    Returns
    -------
    list[torch.Tensor] | list[tuple[torch.Tensor, tuple[torch.Tensor]]]
        A list containing either just the log-probabibilities (if
        `return_hidden_states` is False) or a list containing tuples of
        the log-probabilities and the hidden states (if
        `return_hidden_states` is True).

    """
    return [*map(
        get_logprobs_single,
        iter(lambda: model, None),
        example[prompts_column],
        example[answered_prompts_column],
        example[choices_column],
        iter(lambda: return_hidden_states, None)
    )]


def add_logprobs(example: dict[str, Any],
                 *,
                 model: transformers.PreTrainedModel,
                 add_hidden_states: bool,
                 logprobs_column: str = 'logprobs',
                 hidden_states_column: str = 'hidden_states',
                 prompts_column: str = 'prompts',
                 answered_prompts_column: str = 'answered_prompts',
                 choices_column: str = 'choices') -> dict[str, Any]:
    """Add log-probabilities to a dataset.

    Optionally, also add the hidden states of the final two tokens.

    This can be used with `Dataset.map`.

    Parameters
    ----------
    example : dict[str, Any]
        The example for which to generate the logits.
    model : transformers.PreTrainedModel
        The model used to generate the output logits.
    add_hidden_states : bool
        Whether to also add the hidden states of the final two tokens.
    logprobs_column : str, default: 'logprobs'
        The name of the column in which to add the log-probabilities.
    hidden_states_column : str, default: 'hidden_states'
        The name of the column in which to add the hidden states.
    prompts_column : str, default: 'prompts'
        The name of the column containing the prompts.
    answered_prompts_column : str, default 'answered_prompts'
        The name of the column containing the answered prompts.
    choices_column : str, default: 'choices'
        The name of the column containing the choices.

    Returns
    -------
    dict[str, Any]
        The example updated to contain the log-probabilities, and
        optionally the hidden states.

    """
    output = get_logprobs(model=model,
                          example=example,
                          return_hidden_states=add_hidden_states,
                          prompts_column=prompts_column,
                          answered_prompts_column=answered_prompts_column,
                          choices_column=choices_column)
    if add_hidden_states:
        example[logprobs_column], example[hidden_states_column] = zip(*output)
    else:
        example[logprobs_column] = output
    return example


def evaluate_logprobs(logprobs: torch.Tensor,
                      labels: torch.Tensor,
                      calibrate: bool = True) -> float:
    """Evaluate log-probabilities against target labels.

    We assume that the log-probabilites contains at least 3 dimensions,
    the first being the index, the second corresponding to the prompt,
    and the third corresponding to the answer pairs.
    If the tensor has four dimensions, then we assume that the extra
    dimension contains the log-probabilites of the labels in an NLI
    task. We calculate the log-probability of each answer by taking the
    difference in log-probability between the final NLI label and the
    first NLI label, assumed to denote entailment and contradiction
    respectively.
    Otherwise, we assume the final third dimension are the direct
    log-probabilities of the answer pairs.

    The predicted label is determined by the difference in
    log-probability between the answer at index 1 and the answer at
    index 0. The differences are then averaged over the prompts, and
    (if calibrating) de-biased so that the ratio of labels is 50:50.

    Parameters
    ----------
    logprobs : torch.Tensor
        A tensor of log-probabilities.
    labels : torch.Tensor
        A tensor of the target labels.
    prompt_dim : int or str, default: 'prompt'
        The log-probabilities are first averaged over all of the prompts
        used. This parameter specifies the dimension over which this
        averaging takes place.
    answer_dim : int or str, default: 'answer'
        The dimension corresponding to the two different answers.
    calibrate : bool, default: True
        Whether to calibrate the predicted labels to be 50:50.

    Returns
    -------
    float
        The accuracy of the log-probabilities, according to the labels.

    """
    PROMPT_DIM, ANSWER_DIM, NLI_LABEL_DIM = 1, 2, 3
    CONTRADICTION, ENTAILMENT = 0, -1

    assert len(logprobs) == len(labels)
    assert logprobs.ndim in (3, 4)
    assert logprobs.size(2) == 2

    if logprobs.ndim == 4:
        assert logprobs.size(3) == 3
        logprobs = (logprobs.select(NLI_LABEL_DIM, ENTAILMENT)
                    - logprobs.select(NLI_LABEL_DIM, CONTRADICTION))

    differences = (logprobs.select(ANSWER_DIM, 1)
                   - logprobs.select(ANSWER_DIM, 0))
    mean_differences = differences.mean(PROMPT_DIM)
    calibration_correction = mean_differences.median() if calibrate else 0
    predictions = mean_differences - calibration_correction > 0
    return (predictions == labels).sum().item() / len(labels)
