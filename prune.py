"""Build a top-N routed-expert prune table from captured decode routes."""
import argparse, base64, collections, json, struct
from pathlib import Path

LAYERS, TOPK = 40, 8


def counts(paths):
    per_layer = [collections.Counter() for _ in range(LAYERS)]
    tokens = 0
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row.get('phase') != 'decode':
                continue
            raw = base64.b64decode(row['routed_b64'])
            n = row['tokens']
            tokens += n
            for t in range(n):
                for layer in range(LAYERS):
                    per_layer[layer].update(struct.unpack_from('<8i', raw, (t * LAYERS + layer) * 32))
    return per_layer, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', type=int, default=160)
    ap.add_argument('--out', type=Path, default=Path('integration/prune-160.json'))
    ap.add_argument('routes', nargs='+', help='JSONL route captures (decode rows are used)')
    args = ap.parse_args()
    per_layer, tokens = counts(args.routes)
    layers, stats = [], []
    for layer in range(LAYERS):
        ranked = [e for e, _ in sorted(per_layer[layer].items(), key=lambda kv: (-kv[1], kv[0]))]
        kept = ranked[:args.keep]
        if len(kept) < args.keep:
            kept += [e for e in range(256) if e not in per_layer[layer]][:args.keep - len(kept)]
        if len(kept) != args.keep or len(set(kept)) != args.keep:
            raise ValueError(f'layer {layer}: cannot build {args.keep} unique experts')
        layers.append(kept)
        total = sum(per_layer[layer].values())
        mass = sum(per_layer[layer][e] for e in kept)
        stats.append({'layer': layer, 'unique': len(per_layer[layer]), 'mass_loss': round(1 - mass / total, 4)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'keep': args.keep, 'tokens': tokens, 'method': 'top-N by decode routing frequency',
                                    'layers': layers}, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'tokens': tokens, 'keep': args.keep,
                      'worst': sorted(stats, key=lambda s: -s['mass_loss'])[:5]}))
    # A compact copy for quick checks.
    (args.out.parent / 'prune-stats.json').write_text(json.dumps(stats, indent=2) + '\n')


if __name__ == '__main__':
    main()
