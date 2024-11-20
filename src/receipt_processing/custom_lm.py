import dspy
import litellm
import numpy as np


class CustomLM(dspy.LM):
    """
    CustomLM is a class that also attaches logprobs to the completion.
    """
    def __init__(self,
                 model,
                 **kwargs):
        self.history = []
        self.accuracy = 0
        self.model = model
        super().__init__(model, **kwargs)

    def __call__(self,
                 prompt: str | None = None,
                 messages: list[dict] | None = None,
                 **kwargs):
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        response = litellm.completion(
            model=self.model,
            messages=messages,
            logprobs=True,
            stream=False,
            **kwargs,
        )

        total_logprob = 0
        sum_logprobs = 0
        logprobs = response.choices[0].logprobs
        if logprobs and 'content' in logprobs:
            for token_data in logprobs['content']:
                # Extract the logprob value
                logprob = token_data.get('logprob')
                total_logprob += 1
                sum_logprobs += np.round(np.exp(logprob)*100, 2)

        # compute the average logprob
        self.accuracy = sum_logprobs / total_logprob
        completion = response.choices[0].message.content
        self.history.append({"prompt": prompt, "completion": completion})

        # Must return a list of strings
        return [completion]
    
    def get_accuracy(self):
        return self.accuracy

    def inspect_history(self):
        for interaction in self.history:
            print(f"Prompt: {interaction['prompt']} -> "
                  f"Completion: {interaction['completion']}")
