# Deterministic AI on CPUs: workshop notebooks

Companion code for the Packt live workshop *Deterministic AI on CPUs: Build low-latency LLM workflows* (30 October 2026) and for the book *Deterministic by Design*.
You build one claims-and-triage router for **Meridian**, a fictional travel company, on a laptop CPU: a small encoder decides, a conformal gate checks, a LangGraph state machine
acts, and an LLM sees only the messages the gate refuses.

## Setup

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name deterministic-ai
.venv/bin/python -m pytest -q test_meridian.py        # 4 tests: splits, Wilson, cluster bootstrap
```

The pins in `requirements.txt` are the last versions with wheels for an Intel Mac (torch 2.2.2, numpy 1.26, pyarrow 17). On Apple silicon or Linux, newer versions should work but are untested here.
Run each notebook from this directory, top to bottom. The first run downloads about 1.3 GB of models (bge-small, ModernBERT-base, LFM2.5-Encoder-230M).

## Run order

| Notebook | What it does | Needs |
| :---- | :---- | :---- |
| `00_setup_and_data` | Environment check, model download, the data, and a check that the LFM2.5 encoder's weights actually loaded | network |
| `01_encoders_and_baseline` | Frozen embeddings plus logistic regression for three encoders, with intervals; CPU latency; cost of long inputs | `00` |
| `02_zero_shot_and_setfit` | Route descriptions as a zero-shot router; learning curve of a scoring head; SetFit against a prototype baseline over five seeds | `01` |
| `03_calibration_and_conformal` | Reliability and ECE, temperature scaling, split conformal sets, the coverage-versus-automation curve, where the guarantee breaks | `01` |
| `04_statistics_and_hard_negatives` | Exact coverage of Wilson, Wald and bootstrap intervals; McNemar; LLM-written hard negatives tested on held-out data | `01` |
| `05_langgraph_cascade` | The state machine: classify, gate, escalate, review, idempotent execute; scorecard against an LLM-only router | `01`, `03` |
| `06_quantization_and_release` | int8 quantization with a paired comparison, reproducibility check, and the release gate | `05` |

Each notebook writes its numbers to `data/results/<notebook>.json`. The workshop slides quote those files.

## What is real, what is recorded

- **Meridian messages** (`data/meridian_triage.jsonl`) were written by Claude Haiku from route definitions. Labels are the generator's intent. **No person has verified the test split**, so every accuracy here is a measurement on synthetic text.
- **Slow-tier answers** (`data/llm/test_haiku_answers.jsonl`) are recorded Claude Haiku replies to the 200 test messages, produced once from the route descriptions without labels. They agree with the generator's labels on 190 of 200; the ten disagreements are two boundary scenarios (a message that asks to change the date before cancelling; a complaint that also asks for the price difference back), where either label can be defended. Notebook 05 replays them by message id; it makes no API call and measures no LLM latency or price.
- **Hard negatives** (`data/llm/hard_negatives.jsonl`) are 80 recorded Haiku messages. Their labels are unchecked.
- **Latency numbers** were taken on one Intel Core i7-7567U laptop (4 logical cores) that was running other work; notebook 06 stores the load average beside its numbers and the others record only the hardware. Re-run on your machine before quoting them.

## Layout

`meridian.py` holds the non-teaching helpers (loader, Wilson and cluster-bootstrap intervals, embeddings cache, timing). Everything the workshop teaches lives in notebook cells.
`data/cache/` (embeddings) is regenerated on first run and is not committed.

## License

MIT, see `LICENSE`. The model weights the notebooks download carry their own licenses (LFM2.5-Encoder-230M uses the LFM Open License v1.0).
