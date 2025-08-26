from omegaconf import DictConfig
import random
from vlmx.agent.agent import Agent


class InContextExampleModel(Agent):

    def _make_prompt_parts(self, *args, **kwargs):
        prompt_parts = [
            "Here are some examples. Study each of these examples carefully:\n"
        ]
        prompt_parts.extend(self._get_in_context_examples())  # examples
        prompt_parts.append("## Your task\n")
        prompt_parts.extend(self._format_content(*args, **kwargs))  # your task
        return prompt_parts

    def _get_in_context_examples(self):
        examples = []
        example_paths = self.get_example_paths()
        selected_examples = self._select_examples(example_paths)

        if self.cfg.in_context.shuffle_examples:
            random.shuffle(selected_examples)

        for i, example_path in enumerate(selected_examples):
            example_kwargs = self._extract_example_kwargs(example_path)
            if example_kwargs:
                formatted_example = self._format_content(**example_kwargs)
                examples.append(f"## Example {i + 1}:\n")
                examples.extend(formatted_example)
        return examples

    def _select_examples(self, example_paths):
        if isinstance(self.cfg.in_context.num_examples, int):
            assert self.cfg.in_context.num_examples >= 0, "num_examples must be a non-negative integer"
            return random.sample(example_paths, min(self.cfg.in_context.num_examples, len(example_paths)))
        else:
            return example_paths

    def get_example_paths(self):
        """
        Returns a list of paths to example files or directories.
        Subclasses should implement this method based on their specific needs.
        """
        raise NotImplementedError(
            "Subclasses must implement get_example_paths")

    def _make_system_instruction(self):
        raise NotImplementedError(
            "Subclasses must implement _make_system_instruction")

    def _format_content(self, *args, **kwargs):
        """
        This should format content the same way as _make_prompt_parts in the `basic` prompting
        agent
        """
        raise NotImplementedError("Subclasses must implement _format_content")

    def _extract_example_kwargs(self, example_path):
        """
        Returns the kwargs to pass into _format_content based on the example_path.
        """
        raise NotImplementedError(
            "Subclasses must implement _extract_example_kwargs")