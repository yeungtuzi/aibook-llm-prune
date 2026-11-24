# Bounded expert slots: M1000 eager prototype

**Opt-in by Python import path**, not an installed wheel patch. Without the private overlay the vendor code is unchanged. This is a single-model prototype, not a tested server release or a memory/performance guarantee. No fallback to full routed residency. Do not try it alongside the baseline server.

### Interception points (inspected AIBOOK wheel `sglang-0.5.18+20261101.torch2.11`)

| Vendor location | What the prototype changes |
| --- | --- |
| `srt/model_loader/loader.py:get_model_loader` | Validate actual resolved args, FP16 Qwen3_5MoeForConditionalGeneration + GPTQ `moe_wna16`, original checkpoint index/config and completed pack receipt. Only serial DefaultModelLoader, TP/EP/PP/DP=1, eager prefill **and** decode; refuse graphs, speculative decoding, EPLB, offload, LoRA, prefetch, etc. |
| `srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock.__init__` | Both Qwen3.5 decoder types use this block. Replace the **construction call**, not already allocated parameters. `FusedMoE` and its `MoeWNA16Method.create_weights` never run for routed experts. Require shared fusion off (vendor Qwen3.6 launcher normally turns it on). Leave the standalone `Qwen2MoeMLP`, sigmoid shared gate, router, capture and add in vendor code. All 40 blocks are checked before load. |
| `DefaultModelLoader._get_weights_iterator` | Keep the vendor's shard resolution; open the original eight immutable files via CPU `safe_open`. Ignore 122,880 indexed routed qweight/scales/qzeros/g_idx keys **before `get_tensor`**, as well as MTP (including dense expert tensors) and visual keys. No whole-shard `disable_mmap` reads, buffered iterator, routed GPTQ transpose/cast, or post-load repack. The other 1,546 tensors, including all 480 standalone shared projection tensors and 40 shared gates, follow normal vendor loading. |
| `BoundedExperts.forward` | Vendor TopK runs over 256 logical IDs; only the GEMVs get remapped int32 slot IDs. Pack v1 bytes are already converted to the MATE **legacy** layout (not coalesced). The same physical IDs are used for W13 and W2, with weighting on W2 only. Return a reduced FP16 activation; deferred-sum epilogue must be off. |

The loader does not read all source weights to authenticate them during startup. Use the pinned original, read-only checkpoint and packs. The receipt was pinned after all forty layers completed; a new model or regenerated receipt requires review, not relaxing the guard. Original tensor contents can optionally be SHA-checked offline. `backbone-v1` is deliberately not accepted by this version.

The slot budget is per layer: 8 × 1,579,008 × 40 = **505,282,560 bytes** of routed device buffers at the smoke setting. Shared experts and the rest of the model remain resident. Staging is a single active 1,579,008-byte CPU record, not one prefetch queue per layer. Each row pins its whole route, favors empty slots, then evicts LRU among unpinned slots. Each logical expert is SHA-checked on first fill; subsequent reads rely on immutable files and check file metadata. Prefill rows are processed serially, reusing the same slots across rows; no requirement for the union of a batch's experts to fit. Device fencing before reuse and Python I/O make this a correctness-oriented baseline, not a throughput design.

### Read-only preflight (no SGLang imports or service required)

In a checkout on the target with the original files available:

```bash
site=$HOME/miniforge3/envs/sglang-py310/lib/python3.10/site-packages
model=$HOME/models/gptq-Qwen3.6-35B-A3B-4bit-channel-lmhead
packs=$HOME/sglang-m1000/routed-packs-v1
python3 -B integration/overlay.py "$site"
python3 -B integration/check_inputs.py "$model" "$packs"
```

The first command checks five exact vendor-source hashes, transforms two sources in memory and compiles them. The second reads checkpoint headers, checks their key sets against the pinned index, and checks receipt, manifest hashes and pack lengths. Add `--hash-sources` **offline** to re-read the entire original checkpoint; the default avoids that load. The runtime checks per-expert payloads as consumed. Both default checks passed against deployed files via read-only SSH; no worker was launched. A header-only mock of the actual iterator additionally counted 1,546 retained and 122,880 skipped without touching tensor payloads.

### Manual isolated trial (future; not executed here)

Only after approval, baseline shutdown by its operator, idle memory measurement and a plan to monitor total physical use:

```bash
export AIBOOK_OVERLAY=$HOME/sglang-m1000/slot-overlay-EXPERIMENT1  # NEW directory
python3 -B integration/overlay.py "$site" --out "$AIBOOK_OVERLAY"
AIBOOK_OFFLINE_SMOKE=YES bash integration/smoke.sh
```

Materialization copies the installed SGLang package into a new persistent directory (no hardlinks). Do not add that directory to login/service environment. The smoke wrapper scopes `PYTHONPATH`, checks package resolution, uses the vendor Qwen3.6 script in eager mode, binds loopback port 30141, 1 request, max 1,024 tokens, context 1,024, prefill chunks 32. The 0.55 memory fraction is a *trial setting*, not an enforcement mechanism. A shape/cache miss under the vendor's compile-forbid defaults may require separately approved offline MATE/GDN kernel preparation. Do not increase memory limits or enable online compilation merely to get a green startup. Rollback = use the original launcher/environment again in a separately authorized window. This wrapper never stops another service.

For a first request POST `/generate` with a short text or known `input_ids`, `sampling_params: {"temperature":0,"max_new_tokens":1}`; then use several decode tokens. Require both AIBOOK count logs, no missing shared weights, and compare router IDs, logits/logprobs and output against a saved eager full-resident reference (also shared fusion off). Repeat with >32 prompt tokens to cover chunking. Verify startup peak, steady-state memory and swap, device accounting, correctness, hit/miss and physical read latency before any budget/speed claim. Graphs, distributed scheduling, hot reload, shared fusion, multi-request overlap, performant batched prefill and automatic cache tuning are out of scope. Hardware validation of this integrated forward and these launch settings is outstanding.
