# Observed failures of a small local slow tier

`observed_failures.py` prompts Qwen3-1.7B (thinking off, greedy, max 40 new tokens) to route the first 200 Meridian test messages with free-form JSON `{"route": "<route>"}` and no constrained decoding. Each output is classified as valid, label_outside_set, malformed_json_but_label_found, malformed_json or empty.

Result (`of200.jsonl`, `of200.log`): 195 valid (97.5%, Wilson 94.3% to 98.9%), 5 label_outside_set (2.5%, 1.1% to 5.7%), no malformed or empty outputs. Of the 195 valid, 125 matched the gold route (64%, 57% to 71%). One model, one prompt, one run; says nothing about hosted models.

Run from the repo root: `PYTHONPATH=. .venv/bin/python data/results/12_observed_failures/observed_failures.py 200 out.jsonl Qwen/Qwen3-1.7B`. Seconds per message in the log (26 to 75 s) were taken on a heavily loaded machine and are not timings.
