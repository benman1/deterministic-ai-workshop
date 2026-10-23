# Meridian triage set

`meridian_triage.jsonl`: 1,200 synthetic customer messages for a fictional travel company, 8 routes, 150 per route.

**Provenance.** Written on 17 October 2026 by eight claude-haiku subagents (one per route) in a Claude Code session, from the route definitions in `raw/` prompts: 30 scenarios per route, 5 variants per scenario (five writer personas), scenarios 26 to 30 are boundary cases that mention a neighboring route. No seed (the generator has none); regenerating gives different text. Raw per-route files are in `raw/`.

**Checks run:** valid JSON, 150 lines and 30 scenarios per route, no duplicate texts, no message names its own route. **Not checked:** label correctness by a person. Every label is the generator's intended route. Treat the test split as unverified until a human has read it.

**Splits** are by scenario, so the five variants of one situation never straddle splits: per route 20 scenarios train, 5 calibration, 5 test (seed 20261030), giving 800, 200, 200 messages. Test items are 40 clusters of 5 variants, so the effective sample is smaller than 200; use clustered standard errors.

Messages are short (8 to 51 words), so they do not exercise a long-context encoder.
