# Native AIBOOK multi-turn route trace

A standard-library client for **an already running, capture-enabled AIBOOK SGLang server**, plus a model-free chronological replay. The five prompts in `prompts.json` develop a route-capture harness, test its cached-prefix boundary, then change topic to reporting. This is one real coding conversation, not independent fragments: the client appends the actual assistant text between user messages. Text is rendered again through the server's chat template (`POST /v1/tokenize`, thinking disabled), then sent as those exact `input_ids` to native `POST /generate`. No OpenAI chat route extension, local tokenizer install, server-side chat session, flush, pin, retries, or model startup is needed.

## Run

Offline verification (no network, no ML dependencies):

```sh
python3 -B -m unittest discover -s trace -p 'test_*.py' -v
```

**Opt-in future capture**, not part of offline verification. Use a reachable, already-authorized endpoint for this model, e.g. on the host or through an existing tunnel. This sends real generation requests which consume compute and may evict other users' KV entries. A salt isolates keys, not physical resources. Do not start/restart a service or change its memory settings on behalf of this script. The size limits below are not proof that a full-resident server fits its memory budget.

```sh
python3 trace/capture.py --base-url http://127.0.0.1:30140 \
    --output trace/run-001 --turns 5 --max-new 96 && \
python3 trace/analyze.py trace/run-001/routes.jsonl --capacity 32 \
    --decay 0.95 > trace/run-001/report-32.jsonl
```

The output directory must not exist. Inspect the private transcript in `turns.jsonl` (actual user/assistant text and input/output IDs), `routes.jsonl` (two phase records per successful turn), and `complete.json` (manifest, written **only after success**). On failure, preserve the directory for diagnosis but **do not analyze it as a complete run**. The analyzer accepts streams/generic JSONL and cannot tell whether complete whole turns were omitted at EOF. Do not publish transcripts without reviewing them. CLI limits are 1–5 bundled turns, ≤2048 rendered prompt tokens, ≤128 sampled tokens per request, and ≤8192 cumulative prompt + requested completion tokens; count the full prompt against the cap even on a hit. Responses are byte-capped before parsing; socket-operation timeout ≤180 seconds, not a whole-run deadline. No automatic truncation, retry or resume.

## Interpreting the capture

`P` is the submitted templated prompt length, `C` the positive completion token count through stop, `K` the observed `cached_tokens` (required). At route offset **zero** the wire tensor must have exactly `(P+C−1) × 40 × 8` little-endian int32 values, returned as base64. The 256-ID vocabulary here is for routed experts only. There are no gate weights or shared-expert columns. All retained per-layer top-8 IDs must be distinct and in `[0,256)`.

| Absolute forward positions | Meaning |
|---|---|
| `[0,K)` | Reused KV prefix. Discard; a returned host-slot value is not evidence of work on this turn. |
| `[K,P)` | Newly computed prompt positions, **prefill**. Recomputed common-history tokens still count. |
| `[P,P+C−1)` | Newly computed generated-token inputs, **decode**. The final sampled token has no forward row; `C=1` gives zero rows, not one. |

Each run uses a fresh `cache_salt` fixed across its turns. `K` must not exceed the longest token prefix shared with **any** earlier forwarded sequence in that salt. The check uses actually submitted prompt IDs and all but the last output ID, never prompt characters or a presumed stable rendering. Re-rendering can alter the shared prefix; cache eviction and cache alignment can reduce `K` further. Zero reuse on a follow-up is a valid measurement, not evidence of warm KV. The client never subtracts an estimated common-prefix length. It rejects missing/inconsistent metadata, wrong model path, speculative decoding, retractions and wrong payload length instead of assigning guessed phases. It cannot detect every server bug that returns plausible but incorrect IDs.

## Offline report

`analyze.py [INPUT|-] --capacity K [--decay F]` streams one JSON object per turn and phase, including empty phases. Slots are **per layer**, 8–256. Re-run the analyzer with other budgets on the *same* capture. `turn_class` distinguishes the first turn, same topic, and first turn after a topic change. Both phases share that classification; `cold` denotes session start, not a claim that every decode access missed. `policies.*.global` sums assignment hits/misses over layers, while `per_layer` lists all 40 layers in order. Empty denominators give `null`, not 100%. Static means the preceding whole turn's most frequent observed experts, frozen before the current turn. Adaptive LRU and optional resident decayed frequency probe the full top-8 before updating; they persist across phases and topic changes and reset at the next session. Input sessions must be contiguous, start at turn 1, and contain exactly one prefill then one decode per turn. Discard partial analyzer output on a nonzero exit.

These are **counterfactual slot hits/misses**, not measured NVMe reads, disk bytes, transfer time, hardware cache residency, TPOT, speedup, or evidence of a 24 GB peak. Repeated runs/studies need independent conversations and held-out topics for broader conclusions.

## Vendor evidence (read-only SSH, 2026-11-24)

Installed source root on AIBOOK: `/home/lf/miniforge3/envs/sglang-py310/lib/python3.10/site-packages/sglang/`, vendor package `sglang-0.5.18+20261101.torch2.11`. The model's `config.json` under `/home/lf/models/gptq-Qwen3.6-35B-A3B-4bit-channel-lmhead` has `text_config.num_hidden_layers=40`, `num_experts_per_tok=8`, `num_experts=256`. The client is intentionally specific to this model path/layout.

- `srt/entrypoints/openai/serving_tokenize.py`, `_tokenize_chat_request`: uses the chat renderer and returns token IDs. `srt/entrypoints/http_server.py`: `/v1/tokenize`, `/generate`, `/server_info`, `/model_info`.
- `srt/managers/io_struct.py`: `GenerateReqInput.routed_experts_start_len` is an absolute position (default 0), not cached token count. `srt/managers/scheduler_components/batch_result_processor.py:116–158`, `_maybe_collect_routed_experts`: gathers `[start_len,seqlen−1)` using prompt plus output IDs through stop. `output_streamer.py` records completion count through stop.
- `srt/state_capturer/base.py`, `get_topk`: gathers stored host-cache rows via request-to-token slot indices. `srt/state_capturer/routed_experts.py`, `_get_local_slice`: truncates any fused shared-expert columns to routed top-k. Native encoder/decoder use raw int32 bytes, no shape header (AIBOOK is little-endian).
- `srt/managers/schedule_batch.py`: radix lookup uses token IDs and `cache_salt`; `prepare_for_extend` accumulates `pre_len - already_computed` for non-retracted requests. Native `tokenizer_manager.py` reports `cached_tokens`, `num_retractions`, `prompt_tokens`, `completion_tokens`, `weight_version`, and base64 `routed_experts`. Nonstream finished responses carry `text` and `output_ids`.

Verification here was **source inspection and local synthetic replay only**, not a live trace or performance measurement.
