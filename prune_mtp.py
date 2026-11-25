"""Build a pruned MTP-head draft checkpoint (256 -> keep experts)."""
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=Path.home()/'models/gptq-Qwen3.6-35B-A3B-4bit-channel-lmhead')
    ap.add_argument('--table', type=Path, default=Path('integration/prune-160.json'))
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--keep', type=int, default=160)
    ap.add_argument('--out', type=Path, default=Path.home()/'models/mtp-160')
    args = ap.parse_args()
    keep = sorted(json.loads(args.table.read_text())['layers'][args.layer])
    if len(keep) != args.keep:
        raise SystemExit(f'table has {len(keep)} experts, expected {args.keep}')
    idx = torch.tensor(keep, dtype=torch.long)
    index = json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    mtp = {k: v for k, v in index.items() if k.startswith('mtp.')}
    tensors = {}
    for key, shard in sorted(mtp.items()):
        with safe_open(args.model/shard, framework='pt', device='cpu') as f:
            t = f.get_tensor(key)
        if key == 'mtp.layers.0.mlp.gate.weight' or key in (
                'mtp.layers.0.mlp.experts.gate_up_proj', 'mtp.layers.0.mlp.experts.down_proj'):
            t = t.index_select(0, idx).contiguous()
        tensors[key] = t
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.out/'model.safetensors')
    (args.out/'model.safetensors.index.json').write_text(json.dumps(
        {'metadata': {}, 'weight_map': {k: 'model.safetensors' for k in tensors}}, indent=2))
    cfg = json.loads((args.model/'config.json').read_text())
    tc = cfg.get('text_config', cfg)
    tc['num_experts'] = args.keep
    if 'num_experts' in cfg: cfg['num_experts'] = args.keep
    (args.out/'config.json').write_text(json.dumps(cfg, indent=2))
    for name in ('tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt'):
        src = args.model/name
        if src.exists(): shutil.copy2(src, args.out/name)
    print(json.dumps({'out': str(args.out), 'experts': args.keep, 'tensors': len(tensors),
                      'gate': list(tensors['mtp.layers.0.mlp.gate.weight'].shape),
                      'gate_up': list(tensors['mtp.layers.0.mlp.experts.gate_up_proj'].shape),
                      'bytes': sum(t.numel()*t.element_size() for t in tensors.values())}))

if __name__ == '__main__':
    main()
