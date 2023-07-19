"""
Convert promptsource templates from YAML to JSON.

Also change Jinja templates to Python format strings, as best as that
can be easily done.

"""
import json
from pathlib import Path

from promptsource.templates import TemplateCollection
from tqdm.auto import tqdm


def unjinja(template: str) -> str:
    """Change a Jinja template to a format string."""
    return (
        template
        .replace('{{ ', '{')
        .replace('{{', '{')
        .replace(' }}', '}')
        .replace('}}', '}')
    )


ROOT = Path('converted-templates')

converted = 0
skipped = 0
to_check = []
all_templates = TemplateCollection().datasets_templates.items()
for (dataset, sub_dataset), templates in tqdm(all_templates):
    # if the Jinja templates could not be fully converted
    needs_checking = False

    new_templates = {}
    for template in templates.templates.values():
        # `template.jinja` has the format `<prompt>|||<answer>`
        tpl = unjinja(template.jinja.partition('|||')[0].strip())
        if '{%' in tpl:
            needs_checking = True

        new_templates[template.name] = d = {'template': tpl}

        choices = template.get_fixed_answer_choices_list()
        if choices:
            d['choices'] = choices
        elif template.answer_choices:
            if '{%' in template.answer_choices:
                needs_checking = True
            d['choices'] = [*map(
                str.strip,
                unjinja(template.answer_choices).split('|||'))
            ]
        else:  # template is not multiple-choice  (TODO support this)
            skipped += 1
            break
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
        if needs_checking:
            to_check.append(file)

print(f'Converted {converted}, skipped {skipped}')
print(f'{len(to_check)} files should be checked:')
for file in to_check:
    print(f'- {file}')
