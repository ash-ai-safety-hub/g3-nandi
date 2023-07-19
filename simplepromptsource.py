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

from dataclasses import dataclass
import json
from typing import Any

# TODO work out what the best variable names should be

TEMPLATES_ROOT = 'templates'  # TODO make this more robust


@dataclass
class PromptTemplateOutput:
    prompt: str
    answer: str
    choices: list[str]


@dataclass
class PromptTemplate:
    """A template for generating prompts.

    Attributes
    ----------
    name : str
        The name of the template.
    template : str
        The format string for generating the prompt.
    choices : list[str]
        A list of format strings for generating the choices for the
        answers.

    """
    name: str
    template: str
    choices: list[str]

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
            - choices : list[str]
                A list of all the choices for the example. The answer
                is located at the index given by `example['label']`.

        """
        choices = tuple(choice.format(**example) for choice in self.choices)
        return PromptTemplateOutput(
            prompt=prefix+self.template.format(**example,
                                               answer_choices=choices),
            answer=choices[example['label']],
            choices=choices
        )


def load_templates(dataset: str,
                   sub_dataset: str = '',
                   root: str = TEMPLATES_ROOT) -> dict[str, PromptTemplate]:
    """Load templates from a JSON file.

    Parameters
    ----------
    dataset : str
        The name of the dataset, e.g. 'super_glue'.
    sub_dataset : str, optional
        The name of the sub dataset, if needed, e.g. 'boolq'.
    root : str, optional
        The path to the root directory containing the templates.
        The template file is located at `<root>/<dataset>.json` or
        `<root>/<dataset>/<sub_dataset>.json` depending on whether
        `sub_dataset` is defined.

    Returns
    -------
    dict[str, PromptTemplate]
        A dictionary mapping names of templates to the templates
        themselves.

    """
    full_dataset = dataset + (f'/{sub_dataset}' if sub_dataset else '')
    with open(f'{root}/{full_dataset}.json') as f:
        templates = json.load(f)
    return {name: PromptTemplate(name=name, **template)
            for name, template in templates.items()}
