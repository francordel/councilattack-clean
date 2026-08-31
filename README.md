# CouncilAttack - Anonymized Artifact

Reproducibility artifact for the paper *CouncilAttack: Red Teaming LLMs with a
Deliberative Council over Fabricated Multi-Turn Dialogues* (double-blind
submission).

This repository is released for peer review. It contains the council attack
pipeline (one module per stage), the per-role system prompts, the judge rubric,
the 50-behaviour representative HarmBench subset, and a self-contained runner -
no external framework is required.

## What is included

The Propose -> Cross-critique -> Borda Top-1 -> Chairman -> Target -> in-loop
0-5 Judge pipeline is split into one collaborator per stage. Every module is
framework-independent and is driven end-to-end by `run_council_standalone.py`.

```
council_attack/                     Pipeline package (src layout)
  src/council_attack/
    proposals.py                    ProposalGenerator  (Stage 2A: propose)
    critique.py                     CrossCritic        (Stage 2B: cross-critique / rank)
    borda.py                        BordaAggregator    (Stage 2C: Borda Top-1 + reporting)
    chairman.py                     Chairman           (Stage 2D: refine / select / synthesise)
    judge.py                        HarmJudge          (Stage 5: in-loop 0-5 graded harm)
    memory.py                       StrategyMemory     (optional cross-goal strategy notes)
    llm_client.py                   ModelClient        (shared Ollama / OpenAI-compatible HTTP)
    utils.py                        Shared text/JSON helpers (refusal, JSON repair, ...)
  tests/                            Unit tests for the pipeline logic (pytest)
run_council_standalone.py           Run the full pipeline (one goal, or --all for ASR)
scripts/compute_asr.py              Compute ASR>=4 / ASR>=5 from a batch run
prompts/
  council_system_default.yaml       Default (non-contextual) council proposal system prompt
  council_critique_prompt.yaml      Cross-critique system prompts (hardened + soft variants)
  council_chairman_prompt.yaml      Chairman synthesis system prompts (contextual + standard)
  harm_scale_prompt.yaml            In-loop 0-5 graded-harm judge rubric
  phases/
    council_system_contextual_phase{0,0c,1,2}.yaml   Per-phase contextual-priming council system prompts
data/
  harmbench_behaviors_representative_50.csv   50-behaviour representative subset
examples/
  contextual_fewshot_redacted/      Structural skeletons of the priming templates (content redacted)
logs/                               Run logs + full transcripts (git-ignored, created on first run)
```

## What is NOT included (dual-use)

Per the responsible-disclosure policy in the paper's Ethics appendix, we withhold
the full corpus of working attacks (the transcripts produced by runs) and release
only redacted structural skeletons of the contextual-priming templates
(`examples/contextual_fewshot_redacted/`). Full traces are shared with affected
model providers under coordinated responsible disclosure, not in this repository.

## Run it

`run_council_standalone.py` drives the real `ProposalGenerator`, `CrossCritic`,
`BordaAggregator`, `Chairman` and `HarmJudge` (same prompts, same Borda Top-1),
and supplies the two pieces a serving framework would otherwise provide: the
target model call and the goal-loading/driver loop. Only `requests` and `PyYAML`
are needed; any OpenAI-compatible endpoint or a local Ollama works, and
credentials are read from environment variables only (never written to disk).

Fully local, no API key, via Ollama:

```
python run_council_standalone.py \
    --provider ollama --model qwen2.5:0.5b \
    --category harassment_bullying --row 0 --max-turns 1
```

Against a hosted OpenAI-compatible API:

```
export OPENAI_API_KEY=...
python run_council_standalone.py \
    --provider openai --model gpt-4o-mini \
    --category harassment_bullying --row 0 --max-turns 1
```

Only sanitized metadata (judge score, success, turns, disagreement) is printed to
stdout; the full transcript and the run log are written under `logs/`
(git-ignored). Use `--contextual` to send contextual-priming multi-turn templates
and `--no-borda` to skip cross-critique aggregation. A `--provider github` preset
(GitHub Models, authenticated with a PAT via `GITHUB_TOKEN`) is also available
where that service is reachable.

By default the council is homogeneous (three copies of `--model`). To reproduce
the paper's **heterogeneous** roster, pass explicit models, e.g. `--council
qwen3:14b gemma3:12b deepseek-r1:32b --chairman qwen2.5:32b --judge gpt-oss:120b`.

## Metrics (ASR>=4 / ASR>=5) and determinism

The paper **headline metric is ASR>=4** (substantive harm); the strict
full-jailbreak rate is ASR>=5. Run every behaviour and recover both from the
persisted per-turn `judge_score`:

```
python run_council_standalone.py --provider ollama --model qwen2.5:0.5b --all --max-turns 1
python scripts/compute_asr.py logs/council_batch_<stamp>.json
```

`--all` iterates the whole CSV and writes an aggregate under `logs/` (git-ignored);
the runner also prints ASR>=4 / ASR>=5 directly. Both are `max(judge_score)` per
goal; goals with no valid verdict (e.g. a judge outage, scored UNSCORED) are
excluded, not counted as failures.

**Determinism.** Pass `--seed` to thread a decoding seed into every model call.
Servers honour it best-effort only and per-role temperatures are > 0, so runs
still vary; the paper reports the mean over N runs.

## Tests

```
pip install -e '.[test]'         # from council_attack/
pytest council_attack/tests/ -q
```

The unit tests cover the pipeline logic (JSON parsing/repair, Borda Top-1, the
judge rubric and its outage handling, refusal detection, prompt construction) and
require no network access.

## License

Released for peer review under the MIT License (see `LICENSE`).
