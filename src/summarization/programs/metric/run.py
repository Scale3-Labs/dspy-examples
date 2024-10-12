import json

import dspy
from dotenv import find_dotenv, load_dotenv

from src.summarization.programs.metric.program import Metric

_ = load_dotenv(find_dotenv())

lm = dspy.LM("openai/gpt-4o-mini", max_tokens=250)
dspy.settings.configure(lm=lm)

# load data
dataset = []
with open('src/summarization/programs/metric/dataset.jsonl', 'r',
          encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)

        passage = data.get("passage", "")
        summary = data.get("summary", "")

        example = dspy.Example(passage=passage, summary=summary)
        pred = dspy.Example(passage=passage, summary=summary)
        score = data.get("score", "0")
        dataset.append(dspy.Example(example=example, pred=pred, score=score))

# create program
program = Metric()

res = program(example=dataset[0].example, pred=dataset[0].pred)
print(res)
