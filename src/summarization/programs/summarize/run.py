import json

import dspy
from dotenv import find_dotenv, load_dotenv
from dspy.evaluate import Evaluate

from src.summarization.programs.metric.program import Metric
from src.summarization.programs.summarize.program import Summarize

_ = load_dotenv(find_dotenv())

lm = dspy.LM("openai/gpt-4o-mini", max_tokens=250)
dspy.settings.configure(lm=lm)

# load data
dataset = []
with open('src/summarization/programs/summarize/dataset.jsonl', 'r',
          encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)

        passage = data.get("passage", "")
        summary = data.get("summary", "")
        score = data.get("score", 0)

        example = dspy.Example(passage=passage, summary=summary, score=score)
        dataset.append(example)

trainset = [x.with_inputs("passage") for x in dataset]

# create program
program = Summarize()


def metric(gold, pred, trace=None):
    metric_program = Metric()
    examp = dspy.Example(passage=gold.passage)
    predicted = dspy.Example(summary=pred)
    pred_score = metric_program(example=examp, pred=predicted)
    gold_score = gold.score
    print("gold_score:", gold_score)
    print("pred_score:", pred_score)
    # check if they are almost equal
    return abs(float(gold_score) - float(pred_score)) < 0.2


evaluate = Evaluate(devset=trainset, metric=metric,
                    display_progress=True,
                    display_table=True, provide_traceback=True)

res = evaluate(program)
print(res)
