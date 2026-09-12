# muxa

A small multimodal orchestration engine. Point it at a folder of mixed media
(text, images, audio) and it routes every asset to the right model task,
runs the work through an async worker pool with retries and a deterministic
fallback, validates every model response against a closed result shape, and
keeps a per-run cost ledger. An eval gate scores the pipeline against a
labelled sample and fails CI when quality drops.

No cloud account needed to try it: the mock provider is deterministic and the
whole test suite runs offline. Point `MUXA_BASE_URL` at any OpenAI-compatible
endpoint (including a local Ollama) to run it for real.

## How it works

```
assets/ ──> discover ──> route by modality ──> task queue
                                                 │
                              ┌──────────────────┤ async workers (bounded)
                              ▼                  ▼
                        provider call      retry once, then
                        (mock / http)      rule-based fallback
                              │
                              ▼
                     validate result shape ──> results.json + cost ledger
```

- **Text** files are summarised, **images** are captioned, **audio** is
  transcribed (described via metadata when no real provider is configured).
- Every result records its route: `provider`, `provider-retry`, `cache`, or
  `fallback`. An asset is never dropped.
- Finished work is cached **by content**, so a second run over the same folder
  skips what is already done. A long run becomes resumable instead of
  all-or-nothing.
- `--chain` adds a second stage for modalities that have no single useful
  answer: audio is transcribed, then that **transcript** is summarised. If the
  first stage failed, the second one refuses rather than summarising an error
  message.
- `python -m muxa eval` replays the labelled fixtures and scores the run on
  three independent axes: **overall keyword recall**, **per-modality recall**,
  and the **fallback rate**. Any one below budget exits non-zero, so quality,
  coverage and reliability regressions each fail the build for their own reason.

## Usage

```bash
python -m muxa run path/to/folder          # writes results.json, caches results
python -m muxa run path/to/folder --no-cache
python -m muxa eval                        # scores against tests/fixtures
MUXA_BASE_URL=http://localhost:11434/v1 MUXA_MODEL=llava python -m muxa run media/
```

Run the same folder twice and the second pass costs nothing:

```
$ python -m muxa run tests/fixtures
{ "assets": 7, "tokens": 103, "routes": { "provider": 7 },
  "cache": { "hits": 0, "tokens_saved": 0, "entries": 7 } }

$ python -m muxa run tests/fixtures
{ "assets": 7, "tokens": 0, "routes": { "cache": 7 },
  "cache": { "hits": 7, "tokens_saved": 103, "entries": 7 } }
7 cached, 103 tokens not spent again
```

`tokens` is what the run actually spent; `tokens_saved` is what the hits
avoided. They are deliberately separate numbers, so spend is never inflated by
work that did not happen.

### Chained tasks

```
$ python -m muxa run media/ --chain
clip.wav   task=summarize  route=provider
   upstream: transcript of clip (audio, 204 bytes, f71aa946)
   final   : summary of clip: transcript of clip (audio, 204 bytes, …)
```

A transcript is raw material, not an answer, so audio runs `transcribe` and then
summarises **that text** rather than the file again. One result comes back per
asset — the final stage — with the intermediate kept in `upstream_text` so it
stays auditable.

Only audio is chained. An image caption is already a summary, and summarising a
summary adds nothing but a second chance to hallucinate.

## Development

```bash
python -m pytest        # offline, uses the mock provider
```

CI runs the test suite and then the eval gate on every push. The suite covers
the classifier (extension vs magic-byte disagreements, RIFF-without-WAVE),
the retry branch (injectable one-shot provider failures), the fallback branch
(an always-failing provider still yields a result per asset), the validation
boundary (unknown tasks, empty and oversized text, duplicate paths), the cache
(a hit skips the provider entirely, a fallback is never written, an edit misses
and a rename hits, a corrupt file is survivable, and a different provider is a
different key), and the gate itself.

Example ledger from a mixed run:

```json
{
  "assets": 4,
  "tokens": 57,
  "routes": { "provider": 3, "provider-retry": 1 }
}
```

## Design notes

- **Zero runtime dependencies.** Stdlib only (`asyncio`, `urllib`, `hashlib`),
  so the whole thing is auditable in one sitting.
- **Failures are data.** A provider error becomes a routed, accounted result,
  not an exception in a log. The ledger makes silent degradation visible: a
  spike in `fallback` routes is a monitoring signal.
- **The cache is addressed by content, and a fallback is never written to it.**
  Keying on bytes rather than path means a rename is free and an edit correctly
  redoes the work. The rule that matters more is the exclusion: fallback text is
  what the pipeline produces when a provider is down, so caching it would freeze
  that outage on disk and every later run would serve it as a hit. Only genuine
  provider answers are stored. The provider name and a schema version are part
  of the key, so the same bytes answered by a different model is a different
  entry, and a corrupt cache file is treated as empty rather than fatal: the
  cache is an optimisation, never the record.
- **A chain refuses rather than summarising a failure.** This is the whole
  reason chaining is a feature and not a second loop. Fallback text reads
  *"transcribe unavailable for clip.wav: audio file, 40000 bytes"*. Feed that to
  a summariser and you do not get a degraded summary, you get a **confident
  summary of an error message** — indistinguishable downstream from a real one,
  and it costs a model call to manufacture. So a second stage whose upstream
  degraded is recorded as `blocked-upstream`, spends zero tokens, and keeps the
  failed upstream text for audit. That route counts toward the eval gate's
  reliability budget, because a refusal is a degradation of the run and hiding
  it there would let a whole modality go dark while the number stayed at zero.
- **The eval gate runs without the cache, deliberately.** A cached gate run
  would replay stored text instead of exercising the provider, so it would keep
  passing against yesterday's answers while a real regression shipped, and the
  fallback budget would go quiet too, because a hit is neither a provider call
  nor a fallback. The optimisation is for production; the gate has to be fooled
  by nothing.
- **The eval gate is the contract, and an average is the wrong instrument.**
  A mean over files hides a failure confined to one place, so the gate scores
  three things and the weakest decides:
  - **overall recall** — the headline number;
  - **per-modality recall** — because the fixture corpus is deliberately
    imbalanced (4 text, 1 image, 1 audio), a dead image modality still leaves
    the mean at 0.833, above the 0.8 threshold. The global axis passes; only the
    per-modality floor catches it. Its regression test asserts exactly that
    asymmetry, so the feature cannot quietly stop earning its place;
  - **fallback rate** — a provider that degrades silently still answers, so
    reliability needs its own budget (0% on the deterministic fixtures).
