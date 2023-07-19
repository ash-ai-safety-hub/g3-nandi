from typing import Any

from datasets import load_dataset
import torch
from transformers import AutoTokenizer

from simplepromptsource import load_templates
from util import balance_filter_map_dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = 't5-base'

DATASET = 'super_glue'
SUB_DATASET = 'boolq'
TEMPLATES_ROOT = 'converted-templates'
EXAMPLES_PER_LABEL = 500
TEST_SPLIT = 0.4

EPOCHS = 1000
LEARNING_RATE = 0.01

SEED = 0

templates = load_templates(DATASET, SUB_DATASET, root=TEMPLATES_ROOT)
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)


def add_prompts(datapoint: dict[str, Any]) -> dict[str, Any] | None:
    max_length = tokenizer.model_max_length - 2  # as per the paper
    datapoint['prompts'] = []
    for template in templates.templates.values():
        prompt = template.apply(datapoint)[0]

        pair = []
        for answer in template.get_fixed_answer_choices_list():
            tokens = tokenizer(f'{prompt} {answer}')['input_ids']
            if tokens > max_length:
                return None

        datapoint['prompts'].append(tuple(pair))
    return datapoint


# As per the paper, the dataset is sampled from the validation split
raw_dataset = load_dataset(DATASET, SUB_DATASET, split='validation')
tokenized_dataset = balance_filter_map_dataset(
    EXAMPLES_PER_LABEL,
    raw_dataset.shuffle(seed=SEED).flatten_indices(),
    add_prompts
)
