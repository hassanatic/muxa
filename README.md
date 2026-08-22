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
- Every result records its route: `provider`, `provider-retry`, or `fallback`.
  An asset is never dropped.
- `python -m muxa eval` replays the labelled fixtures and reports
  keyword-recall per modality with a non-zero exit code below the threshold,
  so quality regressions fail the build.

## Usage

```bash
python -m muxa run path/to/folder          # writes results.json
python -m muxa eval                        # scores against tests/fixtures
MUXA_BASE_URL=http://localhost:11434/v1 MUXA_MODEL=llava python -m muxa run media/
```

## Development

```bash
python -m pytest        # offline, uses the mock provider
```
