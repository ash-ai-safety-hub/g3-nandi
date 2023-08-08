"""
A stripped-down adaptation of promptsource.

It supports reading a set of templates from a JSON file, and then
applying them to dataset examples.
Only multiple-choice templates are supported.
Custom templates can be defined by the user, but it does not natively
support being written to disk.

"""
from __future__ import annotations

__all__ = ['PromptTemplate', 'load_templates']

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files

_StrPathT = 'str | os.PathLike[str]'

ENV = Environment()
TEMPLATES_ROOTS = [files('ccstools.templates').joinpath('converted'),
                   files('ccstools.templates').joinpath('additional')]


def _iter_default_roots() -> Iterator[Path]:
    for root in TEMPLATES_ROOTS:
        with as_file(root) as path:
            yield path


@dataclass
class PromptTemplateOutput:
    prompt: str
    answer: str
    choices: list[str] | None


@dataclass
class PromptTemplate:
    """A template for generating prompts.

    Attributes
    ----------
    name : str
        The name of the template.
    template : str
        The Jinja string for generating the prompt.
    choices : list[str]
        A list of Jinja strings for generating the choices for the
        answers.
    source : str, optional
        A string denoting the source of the template, usually a file
        path.

    """
    name: str
    template: str
    choices: list[str] | None

    source: str | None = None

    def __call__(self,
                 example: dict[str, Any],
                 prefix: str = '') -> PromptTemplateOutput:
        """Generate the prompt for a given example.

        Parameters
        ----------
        example : dict[str, Any]
            The example to generate the prompt for.
        prefix : str, optional
            An additional prefix to add to the start of the prompt, if
            provided.

        Returns
        -------
        PromptTemplateOutput
            A dataclass containing the following attributes:
            - prompt : str
                The input prompt for the example.
            - answer : str
                The answer for the example.
            - choices : list[str] | None
                A list of all the choices for the example, if available.
                The answer is located at the index given by
                `example['label']`.

        """
        if self.choices:
            choices = tuple(choice.format(**example)
                            for choice in self.choices)
        else:
            choices = None

        jinja_template = ENV.from_string(self.template)
        if choices:
            full_prompt = jinja_template.render(**example,
                                                answer_choices=choices)
        else:
            full_prompt = jinja_template.render(**example)
        prompt, *rest = map(str.strip, full_prompt.split('|||'))

        try:
            answer = rest[0]
        except IndexError:
            answer = choices[example['label']]

        return PromptTemplateOutput(
            prompt=prefix+prompt,
            answer=answer,
            choices=choices
        )


class TemplateLoadingError(Exception):
    pass


def load_templates(dataset: str,
                   sub_dataset: str = '',
                   custom_root: _StrPathT = '') -> dict[str, PromptTemplate]:
    """Load templates from a JSON file.

    Parameters
    ----------
    dataset : str
        The name of the dataset, e.g. 'super_glue'.
    sub_dataset : str, optional
        The name of the sub dataset, if needed, e.g. 'boolq'.
    custom_root : str, optional
        The path to the root directory containing the templates.
        The template file is located at `<root>/<dataset>.json` or
        `<root>/<dataset>/<sub_dataset>.json` depending on whether
        `sub_dataset` is defined.

    Returns
    -------
    dict[str, PromptTemplate]
        A mapping of names of templates to the templates themselves.

    """
    templates = {}
    dataset_filenames = []
    roots = [Path(custom_root)] if custom_root else _iter_default_roots()
    for root in roots:
        if sub_dataset:
            dataset_file = root / dataset / f'{sub_dataset}.json'
        else:
            dataset_file = root / f'{dataset}.json'
        dataset_filename = str(dataset_file.resolve())
        dataset_filenames.append(dataset_filename)

        try:
            with open(dataset_file) as f:
                tpls = json.load(f)
        except FileNotFoundError:
            pass
        else:
            for name, template in tpls.items():
                templates[name] = PromptTemplate(name=name,
                                                 source=dataset_filename,
                                                 **template)

    if not templates:
        raise TemplateLoadingError(
            'No templates loaded, tried looking in:\n'
            + '\n'.join(f'- {filename}' for filename in dataset_filenames)
            + '\nbut they either do not exist or are empty.')

    return templates
