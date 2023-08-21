from datasets import load_dataset
import torch
from transformers import AutoTokenizer

from ccstools import (
    add_logprobs,
    add_prompts,
    balance_filter_map_dataset,
    evaluate_logprobs,
    load_ccs_templates,
    load_model,
)

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = 'eleutherai/gpt-neo-125m'
MAX_LENGTH = 512

DATASET = 'imdb'
SUB_DATASET = None
EXAMPLES_PER_LABEL = 10

PREFIX = 'What is the incorrect answer?\n'
SUFFIX = '\nThe incorrect answer is'

SEED = 0

# Load model and tokenizer
model = load_model(MODEL).eval().to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512)
if 'gpt' in MODEL:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
ccs_templates = load_ccs_templates()
templates = ccs_templates[DATASET, SUB_DATASET]

raw_dataset = load_dataset(DATASET, SUB_DATASET, split='train')
shuffled_dataset = raw_dataset.shuffle(seed=SEED)

# Balance and tokenize dataset
dataset = balance_filter_map_dataset(
    EXAMPLES_PER_LABEL,
    shuffled_dataset,
    add_prompts,
    fn_kwargs={
        'tokenizer': tokenizer,
        'templates': templates.values(),
        'max_length': MAX_LENGTH,
        'prefix': PREFIX,
        'suffix': SUFFIX,
    }
).with_format('torch', device=DEVICE)

with torch.no_grad():
    dataset = dataset.map(
        add_logprobs,
        fn_kwargs={
            'model': model,

            # When also running CCS experiments, setting this to `True`
            # will be useful to save passing the prompts through the
            # model a second time. However, it should not be set for
            # encoder-decoder models where you want to use the encoder
            # hidden states, since in the zero-shot experiments, the
            # choices are passed to the decoder, whereas in CCS, the
            # choices are passed to the encoder. In those cases, you
            # will have to make a separate pass through the model.
            'add_hidden_states': False
        }
    )

zero_shot_accuracy = evaluate_logprobs(dataset['logprobs'], dataset['label'])
print(f'Zero-shot accuracy: {zero_shot_accuracy:%}')
