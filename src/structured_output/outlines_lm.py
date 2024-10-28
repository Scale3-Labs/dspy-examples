import dspy
import outlines.models as models
from outlines import generate
from typing import Literal
from typing import Union, Callable

# enum for generate_fn
GenerateFn = Literal["json", "choice", "text", "regex"]


class OutlinesLM(dspy.LM):
    """
    OutlinesLM is a class that uses Outlines to generate structured outputs.
    """
    def __init__(self,
                 model,
                 generate_fn: GenerateFn,
                 schema_object: Union[str, object, Callable],
                 **kwargs):
        self.history = []
        super().__init__(model, **kwargs)
        self.model = models.openai(model)  # type: ignore
        self.generate_fn = generate_fn
        self.schema_object = schema_object

    def __call__(self,
                 prompt: str | None = None,
                 messages: list[dict] | None = None,
                 **kwargs):
        # extract prompt and system prompt from messages
        system_prompt = None
        if messages:
            for message in messages:
                if message["role"] == "system":
                    system_prompt = message["content"]
                else:
                    prompt = message["content"]
        if self.generate_fn == "json":
            generator = generate.json(  # type: ignore
                self.model, self.schema_object)
        elif self.generate_fn == "choice":
            generator = generate.choice(  # type: ignore
                self.model, self.schema_object)
        elif self.generate_fn == "text":
            generator = generate.text(self.model)  # type: ignore
        elif self.generate_fn == "regex":
            generator = generate.regex(  # type: ignore
                self.model, self.schema_object)
        else:
            raise ValueError(f"Invalid generate_fn: {self.generate_fn}")
        completion = generator(
            prompt, system_prompt=system_prompt)  # type: ignore
        self.history.append({"prompt": prompt, "completion": completion})

        # Must return a list of strings
        return completion

    def inspect_history(self):
        for interaction in self.history:
            print(f"Prompt: {interaction['prompt']} -> "
                  f"Completion: {interaction['completion']}")
