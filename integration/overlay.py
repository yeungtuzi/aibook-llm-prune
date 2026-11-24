"""Preflight the deployed vendor wheel; optionally make a private patched copy."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys

# Paths are relative to site-packages. Drift is an error, never fuzzy patch.
BLOCK = 'sglang/srt/models/qwen2_moe.py'
LOADER = 'sglang/srt/model_loader/loader.py'
PINS = {
    BLOCK: '458c5ee4eb425310ce9eb2f664cc6f1e608d36f3056bf306a07025fc7f259b1e',
    LOADER: '543e5a49a45fa738ff7115820dc07e5d1074387d6583b4204e58e09e1e25ded9',
    'sglang/srt/models/qwen3_5.py': '4f5b6311857e3f5fc0045c06797bbc50138d8185658147e4f311f0112bfd38cb',
    'sglang/srt/layers/quantization/moe_wna16.py': '23146f3f51a00247d5e78f5771bff354bf98587d22de36094c6dae928fa6bb9d',
    'mate/batched_gemv/tilelang/batched_gemv_w4a16_moe.py': '7654aacc136978cb3c12f348794cc3203f285521afc295588cce368d163d9750',
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
        if name in (LOADER, BLOCK):
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
        for name in ('aibook_loader.py', 'aibook_slots.py', 'module_timing.py', 'routed_graph.py', 'projection_graph.py', 'mapped_banks.py', 'graph_compat.py', 'prune-160.json', 'prune-128.json'):
            shutil.copyfile(Path(__file__).with_name(name), out/name)
        (out/'AIBOOK_READY').write_text('pinned eager prototype\n')
    print('PASS: 5 source pins, constructor and loader hooks compiled' + ('; overlay ready' if args.out else '; read-only'))


if __name__ == '__main__':
    main()
