# Notebook 17 raw record (3 November 2026)

Per-example scores behind `17_grounding_checkers.ipynb`.

- `scores_300.jsonl`, `scores_300.log`: 300 VitaminC flip pairs (600 examples), four checkers, one JSON line per example. Pairs are chosen with a fixed seed over sorted claim ids, so the draw reproduces. This is the record the notebook analyzes.
- `run_checkers.py`: the scorer that produced it (absolute scratchpad paths; rewrite before running elsewhere).
- `analyze.py`, `analyze.out`: 200 random 50/50 splits by pair, split conformal at 90% and 95% targets, ECE with 10 equal-width bins.
- `probe.*`, `mc_probe.*`: first probes (10 pairs through four checkers; MiniCheck from the safetensors conversion pull request `refs/pr/1`).
- `step1.*`: an earlier 50-pair run.
- `superseded_unsorted_draw/`: an earlier 300-pair run whose pairs were chosen by Python set order and cannot be reproduced. Kept for audit only; its figures (295 of 300 for MiniCheck, 198 for the cross-encoder) are not used anywhere.

Data: VitaminC test split (Schuster et al.), CC BY-SA 3.0, not stored here. CPU timings (`timing_60_load5.jsonl`, 4-core Intel Mac, load average about 5.5, 60 examples each, one example per forward pass), median / p95 in ms: MiniCheck 353 / 663, HHEM 144 / 253, NLI model 218 / 386, cross-encoder 18 / 39. Earlier runs at load 13 or higher gave several times these figures (`timing_60.jsonl`) and are kept only for audit. Your numbers will differ with hardware and load.
