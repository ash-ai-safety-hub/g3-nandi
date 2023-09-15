"""
Utilities for working with models.

"""
from __future__ import annotations

__all__ = ['get_model_type', 'load_model']

from collections.abc import Iterable
from typing import Any, TYPE_CHECKING

from transformers import (AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification)

if TYPE_CHECKING:
    import transformers


CAUSAL_MODELS = (
    'gpt',
)
ENCODER_DECODER_MODELS = (
    'T0',
    't5',
    'unifiedqa',
)
ENCODER_ONLY_MODELS = (
    'DeBERTa',
    'roberta',
)


def _prefix_search(prefixes: Iterable[str], name: str) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def get_model_type(model: str | transformers.PreTrainedModel) -> str:
    try:
        model = model.name_or_path
    except AttributeError:
        pass
    model = model.rpartition('/')[2]

    if _prefix_search(CAUSAL_MODELS, model):
        return 'decoder'
    elif _prefix_search(ENCODER_DECODER_MODELS, model):
        return 'encoder_decoder'
    elif _prefix_search(ENCODER_ONLY_MODELS, model):
        return 'encoder'
    else:
        raise ValueError(f'Unknown model: {model}')


def load_model(model_name: str,
               *args: Any,
               **kwargs: Any) -> transformers.PreTrainedModel:
    CLASSES = {
        'decoder': AutoModelForCausalLM,
        'encoder_decoder': AutoModelForSeq2SeqLM,
        'encoder': AutoModelForSequenceClassification
    }
    model_type = get_model_type(model_name)
    return CLASSES[model_type].from_pretrained(model_name, *args, **kwargs)
