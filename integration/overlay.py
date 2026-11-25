"""Preflight the deployed vendor wheel; optionally make a private patched copy."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys

# Paths are relative to site-packages. Drift is an error, never fuzzy patch.
BLOCK = 'sglang/srt/models/qwen2_moe.py'
LOADER = 'sglang/srt/model_loader/loader.py'
SPEC_HOOK = 'sglang/srt/arg_groups/speculative_hook.py'
EAGLE_UTILS = 'sglang/srt/speculative/eagle_utils.py'
NGRAM_WORKER = 'sglang/srt/speculative/ngram_worker.py'
PINS = {
    BLOCK: '458c5ee4eb425310ce9eb2f664cc6f1e608d36f3056bf306a07025fc7f259b1e',
    LOADER: '543e5a49a45fa738ff7115820dc07e5d1074387d6583b4204e58e09e1e25ded9',
    'sglang/srt/models/qwen3_5.py': '4f5b6311857e3f5fc0045c06797bbc50138d8185658147e4f311f0112bfd38cb',
    'sglang/srt/layers/quantization/moe_wna16.py': '23146f3f51a00247d5e78f5771bff354bf98587d22de36094c6dae928fa6bb9d',
    'mate/batched_gemv/tilelang/batched_gemv_w4a16_moe.py': '7654aacc136978cb3c12f348794cc3203f285521afc295588cce368d163d9750',
    SPEC_HOOK: '0817c9e460a94b62abc08702d51e3dc1ba47a739ab9dd6ef2227ffd6787a4f8f',
    EAGLE_UTILS: '32d9035c9c240f65876cdcbfd8f7d64ec5c7f6acba710b1978cb5f1af2a775ab',
    NGRAM_WORKER: '74e8e20515dd0eb5c7e165c11480150f755a4bdf7c3089f953d3898fbc508b69',
}


def replace_once(src, old, new):
    if src.count(old) != 1:
        raise ValueError('vendor anchor missing/ambiguous: ' + old)
    return src.replace(old, new)


def transform(site):
    sources = {}
    for name, sha in PINS.items():
        data = (site/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError('vendor source drift: ' + name)
        if name in (LOADER, BLOCK, SPEC_HOOK, EAGLE_UTILS, NGRAM_WORKER):
            sources[name] = data.decode()
    src = sources[BLOCK]
    start = '        self.experts = get_moe_impl_class(quant_config)('
    end = '\n        self.gate = ReplicatedLinear('
    original = src[src.index(start):src.index(end, src.index(start))]
    indented = ''.join(('    ' + line) if line.strip() else line for line in original.splitlines(True))
    replacement = (
        '        if config.num_hidden_layers == 1:\n'
        + indented
        + '        else:\n'
        '            from aibook_loader import make_experts\n'
        '            self.experts = make_experts(self, config, quant_config, get_moe_a2a_backend(), _musa_fused_moe_epilogue)\n')
    sources[BLOCK] = replace_once(src, original, replacement)
    gate_anchor = '        router_logits, _ = self.gate(hidden_states)'
    if sources[BLOCK].count(gate_anchor) != 2:
        raise ValueError('gate anchor missing/ambiguous')
    sources[BLOCK] = sources[BLOCK].replace(gate_anchor, gate_anchor + '\n'
        '        _pm = getattr(self.experts, "prune_mask", None)\n'
        '        if _pm is not None:\n'
        '            router_logits = router_logits + _pm')
    src = sources[LOADER]
    anchor = '    """Get a model loader based on the load format."""'
    src = replace_once(src, anchor, anchor + '\n    from aibook_loader import validate\n'
        '    validate(get_server_args(), model_config, load_config)\n    return DefaultModelLoader(load_config)')
    anchor = '            self.load_weights_and_postprocess(\n                model, self._get_all_weights(model_config, model), target_device'
    src = replace_once(src, anchor, '            from aibook_loader import check_model\n            check_model(model)\n' + anchor)
    anchor = '        if self.load_config.load_format == LoadFormat.NPCACHE:'
    start = src.index(anchor)
    end = src.index('    @classmethod\n    def _filter_mtp_weights', start)
    src = src[:start] + ('        from aibook_loader import weights\n'
        '        self.counter_before_loading_weights = time.perf_counter()\n'
        '        return weights(hf_weights_files, source, use_safetensors)\n\n') + src[end:]
    sources[LOADER] = src
    src = sources[SPEC_HOOK]
    src = replace_once(src, 'if server_args.device not in ("cuda", "cpu"):',
                       'if server_args.device not in ("cuda", "cpu", "musa"):')
    src = replace_once(src, 'Ngram speculative decoding only supports CUDA or CPU devices.',
                       'Ngram speculative decoding only supports CUDA, CPU or MUSA devices.')
    sources[SPEC_HOOK] = src
    src = sources[EAGLE_UTILS]
    src = replace_once(src, '''    elif _is_musa:
        if topk != 1:
            raise NotImplementedError(
                "MUSA tree verify currently supports topk=1 only"
            )''', '''    elif _is_musa:
        if topk != 1:
            import os as _os
            if _os.environ.get('AIBOOK_NGRAM_CPU_VERIFY') == '1':
                from aibook_loader import verify_tree_greedy_musa
                verify_tree_greedy_musa(
                    predicts, accept_index, accept_token_num, candidates,
                    retrieve_index, retrieve_next_token, retrieve_next_sibling,
                    target_predict, topk)
            elif _os.environ.get('AIBOOK_NGRAM_TORCH_VERIFY') == '1':
                from aibook_loader import verify_tree_greedy_musa_torch
                verify_tree_greedy_musa_torch(
                    predicts, accept_index, accept_token_num, candidates,
                    retrieve_index, retrieve_next_token, retrieve_next_sibling,
                    target_predict, topk)
            else:
                from aibook_loader import verify_tree_greedy_musa_native
                verify_tree_greedy_musa_native(
                    predicts, accept_index, accept_token_num, candidates,
                    retrieve_index, retrieve_next_token, retrieve_next_sibling,
                    target_predict, topk)''')
    sources[EAGLE_UTILS] = src
    src = sources[NGRAM_WORKER]
    src = replace_once(src, '        self.adaptive_controller = None\n',
        '        self.adaptive_controller = None\n'
        '        self._aibook_gate_enabled = __import__("os").environ.get("AIBOOK_NGRAM_GATE") == "1"\n'
        '        self._aibook_gate_off = set()\n'
        '        self._aibook_low = {}\n')
    src = replace_once(src, '''    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        fwd_stream''', '''    def _aibook_gate_bypass(self, batch):
        live = {r.rid for r in batch.reqs}
        self._aibook_gate_off &= live
        self._aibook_low = {k: v for k, v in self._aibook_low.items() if k in live}
        return bool(live) and all(rid in self._aibook_gate_off for rid in live)

    def _aibook_observe_gate(self, batch, accept_lens):
        for req, val in zip(batch.reqs, accept_lens.tolist()):
            rid = req.rid
            if rid in self._aibook_gate_off:
                continue
            low = self._aibook_low.get(rid, 0) + 1 if val <= 1 else 0
            if low >= 4:
                self._aibook_gate_off.add(rid)
                self._aibook_low.pop(rid, None)
                logging.warning("AIBOOK: NGRAM gate off rid=%s accept_len=%s", rid, val)
            else:
                self._aibook_low[rid] = low

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        if self._aibook_gate_enabled and self._aibook_gate_bypass(batch):
            logging.warning("AIBOOK: NGRAM bypass to target decode")
            last = []
            for req in batch.reqs:
                ids = req.output_ids if req.output_ids else req.origin_input_ids
                last.append(int(ids[-1]))
            batch.input_ids = torch.tensor(last, dtype=torch.int64, device=self.device)
            if getattr(batch, "out_cache_loc", None) is not None:
                batch.out_cache_loc = batch.out_cache_loc[: len(batch.reqs)]
            batch.forward_mode = ForwardMode.DECODE
            batch.spec_info = None
            res = self.target_worker.forward_batch_generation(batch)
            res.next_token_ids = res.next_token_ids.cpu()
            res.accept_lens = torch.ones(len(batch.reqs), dtype=torch.int32, device="cpu")
            res.speculative_num_draft_tokens = 1
            res.num_correct_drafts = 0
            return res
        fwd_stream''')
    src = replace_once(src, '            new_seq_lens = batch.seq_lens + accept_lens\n',
        '            new_seq_lens = batch.seq_lens + accept_lens\n'
        '            if self._aibook_gate_enabled:\n'
        '                self._aibook_observe_gate(batch, accept_lens)\n')
    sources[NGRAM_WORKER] = src
    for path, src in sources.items():
        compile(src, path, 'exec')
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', type=Path, help='pinned site-packages directory')
    parser.add_argument('--out', type=Path, help='NEW private directory; omit for read-only preflight')
    args = parser.parse_args()
    site = args.site.resolve(strict=True)
    sources = transform(site)
    if args.out:
        out = args.out.resolve()
        if site == out or site in out.parents:
            sys.exit('refusing to write inside vendor environment')
        # No hardlinks/symlinks to installed sources. Never overwrite a release.
        out.mkdir(exist_ok=False)
        shutil.copytree(site/'sglang', out/'sglang', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name, src in sources.items():
            (out/name).write_text(src)
        for name in ('aibook_loader.py', 'aibook_slots.py', 'module_timing.py', 'routed_graph.py', 'projection_graph.py', 'mapped_banks.py', 'graph_compat.py', 'prune-160.json', 'prune-128.json', 'prune-96.json'):
            shutil.copyfile(Path(__file__).with_name(name), out/name)
        (out/'AIBOOK_READY').write_text('pinned eager prototype\n')
    print('PASS: 6 source pins, constructor and loader hooks compiled' + ('; overlay ready' if args.out else '; read-only'))


if __name__ == '__main__':
    main()
