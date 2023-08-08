"""
Convert promptsource templates from YAML to JSON.

Also change Jinja templates to Python format strings, as best as that
can be easily done.

"""
import json
from pathlib import Path

import jinja2
from promptsource.templates import TemplateCollection
from tqdm.auto import tqdm


ROOT = Path('converted-templates')

env = jinja2.Environment()

converted = 0
skipped = 0
all_templates = TemplateCollection().datasets_templates.items()
for (dataset, sub_dataset), templates in tqdm(all_templates):
    new_templates = {}
    for template in templates.templates.values():
        # `template.jinja` has the format `<prompt>|||<answer>`
        new_templates[template.name] = d = {'template': template.jinja}

        choices = template.get_fixed_answer_choices_list()
        if not choices:
            if template.answer_choices:
                choices = [answer.strip()
                           for answer in template.answer_choices.split('|||')]
                try:
                    for choice in choices:
                        env.parse(choice)
                except jinja2.TemplateSyntaxError:
                    skipped += 1
                    break
            else:  # template is not multiple-choice
                choices = None
        d['choices'] = choices
    else:  # i.e. not skipped
        # save to disk
        if sub_dataset:
            path = f'{dataset}/{sub_dataset}.json'
        else:
            path = f'{dataset}.json'

        file = ROOT / path
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'w') as f:
            json.dump(new_templates, f, indent=1)

        converted += 1

print(f'Converted {converted}, skipped {skipped}')
