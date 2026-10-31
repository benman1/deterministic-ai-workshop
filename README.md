# Deterministic AI on CPUs: workshop notebooks

Companion code for the Packt live workshop *Deterministic AI on CPUs: Build low-latency LLM workflows* (30 October 2026) and for the book *Deterministic by Design*.
You build one claims-and-triage router for **Meridian**, a fictional travel company, on a laptop CPU: a small encoder decides, a conformal gate checks, a LangGraph state machine
acts, and an LLM sees only the messages the gate refuses.

## Setup

```bash
git clone https://github.com/benman1/deterministic-ai-workshop && cd deterministic-ai-workshop
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name deterministic-ai
.venv/bin/python -m pytest -q test_meridian.py        # 4 tests: splits, Wilson, cluster bootstrap
```

On Linux, add the PyTorch CPU index, or `uv` installs the CUDA build of torch and about 2.5 GB of NVIDIA packages that a CPU run never uses:

```bash
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

To run the notebooks, start JupyterLab from the repository directory and open `00_setup_and_data.ipynb`:

```bash
source .venv/bin/activate
jupyter lab
```

Pick the `deterministic-ai` kernel (Kernel, then Change Kernel). Run each notebook from the first cell to the last. In VS Code or Cursor, open the notebook file, choose the same kernel from the kernel picker, and use Run All.

Notebook 01's long-context cell needs about 4 GB of free memory at 8,192 tokens; on a machine with less, run `LONG_CONTEXT_MAX=1024` before starting Jupyter (a 7.7 GB shared server lost the kernel to the out-of-memory killer there).

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
| `07_policy_check` | The refund policy as named rules; a solver (Z3) checks a proposed action and names the rules in the way, and checks the rule list for conflicts and gaps; no model, runs in seconds | none |
| `08_guardrail_comparison` | An open injection classifier (archived repository) against the typed cascade on 96 injected messages; what each catches, what each costs, and the weak point of the slow tier | `01`, `03`; downloads one extra model (about 0.7 GB) |
| `09_word_substitution_attack` | A TextFooler-style word-swap attack (written from the paper's algorithm) on the encoder and the gate: how often swapping a few words makes the gate act on a wrong route, with a random-swap control and a model's read of whether the message keeps its meaning | `01`, `03`; downloads counter-fitted word vectors; needs NLTK data first (`.venv/bin/python -m nltk.downloader averaged_perceptron_tagger_eng stopwords`, which this laptop already had; the notebook does not fetch it); the full run took about 40 minutes on a shared laptop |
| `10_consensus_and_agreement` | Do voters that agree know more than one voter? Pairwise agreement and shared errors of three encoder heads and the recorded slow tier, then a consensus gate (act when the voters agree, send on when they do not) against the best single encoder and the conformal gate | `01`, `05` (cached embeddings and the recorded answers); about 5 seconds, no model is called |
| `11_decision_heads_compared` | Why logistic regression? Six heads (logistic regression, linear SVM, neural net, prototype, gradient-boosted trees, nearest neighbors) on the same frozen embeddings: accuracy with Wilson intervals, the paired test against logistic regression, head time, and calibration before and after temperature scaling | `01`; about 30 seconds |
| `12_fallback_on_failure` | What happens when a tier fails: simulated slow-tier failures at four rates for four designs (a language model alone, with a retry, a cascade with no failure rule, a cascade that falls back to a person), the cost of falling back when the encoder is down, and a test that every message ends with a typed outcome | `05`, `06`; about 20 seconds |
| `13_human_review_stage` | Who a reviewer should read first (errors found per messages read), how many acted messages a random audit needs, which messages a shadow run asks people to label, and a review tool that hides the model's pick until the reviewer commits | `05`, `10`; about 10 seconds |
| `14_perplexity_gate` | Whether a small language model's perplexity can gate unusual input: a threshold set from benign messages, then the share of fluent injections, word-swap attacks and out-of-scope messages above it, with AUROC, and the effect of message length | `08`, `09`; about 10 minutes on a loaded laptop (0.5B model, CPU) |
| `15_similarity_vs_relevance` | Which of BM25, a sentence transformer, their fusion and a cross-encoder puts the right Meridian document first on 20 questions; what happens when a superseded document is the nearer one; whether the best-document score can gate "no answer" | about 3 minutes on a loaded laptop (bge-small and bge-reranker-base, CPU); uses the policy and destination documents in `data/rag/` |
| `16_same_question_different_answer` | Why the same prompt at temperature 0 can give different answers: the Thinking Machines matrix test on a CPU, the same text scored alone and inside a batch in float32 and bfloat16, 48 "pick a random number" requests served alone and in batches, a closed choice with an escalation margin, a decision cache, and a graph of temperature against the number of distinct answers | `05` (the closed choice); about 2 hours on a loaded laptop (0.5B model, CPU); a bfloat16 pass takes tens of seconds, so every long cell prints per step |

Each notebook writes its numbers to `data/results/<notebook>.json`. The workshop slides quote those files.

## What is real, what is recorded

- **Meridian messages** (`data/meridian_triage.jsonl`) were written by Claude Haiku from route definitions. Labels are the generator's intent. **No person has verified the test split**, so every accuracy here is a measurement on synthetic text. A model read all 200 test messages on 27 October (`data/review/test_review.csv`, annotator column says so): it agrees with the generator on 195, differs on the five cancel-or-move-the-date messages (it labels the first thing asked, a date change), and flags 15 rows as ambiguous (three scenarios of five variants).
- **Slow-tier answers** (`data/llm/test_haiku_answers.jsonl`) are recorded Claude Haiku replies to the 200 test messages, produced once from the route descriptions without labels. They agree with the generator's labels on 190 of 200; the ten disagreements are two boundary scenarios (a message that asks to change the date before cancelling; a complaint that also asks for the price difference back), where either label can be defended. Notebook 05 replays them by message id; it makes no API call and measures no LLM latency or price.
- **Hard negatives** (`data/llm/hard_negatives.jsonl`) are 80 recorded Haiku messages. Their labels are unchecked.
- **Reproduction on another machine.** A fresh clone ran notebooks 00 to 06 on x86 Linux (4 vCPU, CPU-only torch 2.2.2). Every figure matched except ModernBERT's frozen-encoder accuracy: 184 of 200 there, against 185 of 200 on the Mac (0.920 against 0.925). Expect a one-message difference from floating-point details when you re-run.
- **Latency numbers** were taken on one Intel Core i7-7567U laptop (4 logical cores) that was running other work; notebook 06 stores the load average beside its numbers and the others record only the hardware. Re-run on your machine before quoting them.

## Layout

`meridian.py` holds the non-teaching helpers (loader, Wilson and cluster-bootstrap intervals, embeddings cache, timing). Everything the workshop teaches lives in notebook cells.
`data/cache/` (embeddings) is regenerated on first run and is not committed.

## License

MIT, see `LICENSE`. The model weights the notebooks download carry their own licenses (LFM2.5-Encoder-230M uses the LFM Open License v1.0).
