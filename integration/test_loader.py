"""Regression test: filtered keys must never reach get_tensor (stdlib only)."""
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from aibook_loader import weights


class SelectiveRead(unittest.TestCase):
    def test_filter_before_read(self):
        reads = []
        keep = ['model.language_model.layers.0.mlp.shared_expert.gate_proj.qweight',
                'model.language_model.layers.0.mlp.shared_expert_gate.weight',
                'model.language_model.layers.0.mlp.gate.weight']

        class Shard:
            def __enter__(self): return self
            def __exit__(self, *exc): pass
            def keys(self):
                yield from keep
                yield 'mtp.layers.0.mlp.experts.down_proj'
                yield 'model.visual.patch_embed.proj.weight'
                for layer in range(40):
                    for expert in range(256):
                        for proj in ('gate', 'up', 'down'):
                            for kind in ('qweight', 'scales', 'qzeros', 'g_idx'):
                                yield f'model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.{kind}'
            def get_tensor(self, k):
                if k not in keep: raise AssertionError(k)
                reads.append(k)
                return k

        root = Path.cwd(); f = 'example.safetensors'
        receipt = b'{"source_sha256":{"config.json":"config","example.safetensors":"payload"}}'
        with patch.dict('sys.modules', safetensors=NS(safe_open=lambda *a, **kw: Shard())), \
             patch('aibook_loader.pack_root', return_value=root), \
             patch.object(Path, 'read_bytes', return_value=receipt):
            self.assertEqual(list(weights([root/f], NS(model_or_path=root, prefix=''), True)), [(k,k) for k in keep])
        self.assertEqual(reads, keep)


if __name__ == '__main__':
    unittest.main()
