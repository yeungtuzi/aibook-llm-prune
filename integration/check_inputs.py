"""Read-only offline preflight: stdlib only, no model, kernel or shard payload reads by default."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from aibook_loader import CONFIG, INDEX, RECEIPT, ROUTED, digest, require


def check(model, packs, deep=False):
    require(digest(model/'config.json') == CONFIG and digest(model/'model.safetensors.index.json') == INDEX,
            'original checkpoint metadata mismatch')
    require(digest(packs/'verified.json') == RECEIPT, 'pack completion digest mismatch')
    receipt = json.loads((packs/'verified.json').read_bytes())
    sources = receipt['source_sha256']
    require(json.loads((packs/'source-sha256.json').read_bytes()) == sources, 'pack provenance mismatch')
    index = json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map']
    files = sorted(model.glob('*.safetensors'))
    require({p.name for p in files} == set(sources)-{'config.json'}, 'unexpected checkpoint files')
    for p in files:
        with p.open('rb') as f:
            n, = struct.unpack('<Q', f.read(8))
            require(n < 32 << 20, 'unexpected shard header size')
            header = json.loads(f.read(n)); header.pop('__metadata__', None)
            require(set(header) == {k for k,v in index.items() if v == p.name}, 'shard header/index mismatch')
            if deep:
                f.seek(0); h = hashlib.sha256()
                for chunk in iter(lambda: f.read(4 << 20), b''):
                    h.update(chunk)
                require(h.hexdigest() == sources[p.name], 'checkpoint SHA mismatch: ' + p.name)
    count = sum(bool(ROUTED.fullmatch(k)) for k in index)
    require(count == 122880, 'routed key count mismatch')
    for k, sha in receipt['manifest_sha256'].items():
        require(digest(packs/k/'manifest.json') == sha, 'manifest mismatch: ' + k)
        require((packs/k/'weights.bin').stat().st_size == 255*1581056 + 1579008, 'pack length mismatch: '+k)
    print(f'PASS: {len(files)} shard headers, {count} routed keys, 40 manifests and pack lengths; '
          + ('source payloads SHA checked' if deep else 'source payloads not rehashed'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path); p.add_argument('packs', type=Path)
    p.add_argument('--hash-sources', action='store_true', help='read all original shard bytes (offline only)')
    args = p.parse_args(); check(args.model, args.packs, args.hash_sources)
