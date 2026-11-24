"""Synthetic tests; run: python3 -B -m unittest discover -s trace -p test_analyze.py"""
import base64
import io
import json
import struct
import unittest

from analyze import analyze


def phase(turn=1, kind='prefill', starts=(), topic='x', sid='s'):
    payload = b''.join(struct.pack('<8i', *range(start + ((layer % 2) * 32),
                                                start + ((layer % 2) * 32) + 8))
                       for start in starts for layer in range(40))
    return dict(session=sid, turn=turn, topic=topic, phase=kind, tokens=len(starts),
                routed_b64=base64.b64encode(payload).decode())


def replay(*rows, **kwargs):
    data = '\n'.join(json.dumps(r) for r in rows).encode()
    return list(analyze(io.BytesIO(data), **kwargs))


class AnalyzeTests(unittest.TestCase):
    def test_chronology(self):
        rows = [phase(starts=[0, 0]), phase(kind='decode', starts=[8]),
                phase(2, starts=[8]), phase(2, 'decode', [0]),
                phase(3, starts=[8], topic='y'), phase(3, 'decode', topic='y'),
                phase(starts=[8], sid='other'), phase(kind='decode', sid='other')]
        reports = replay(*rows, capacity=8, decay=.5)
        self.assertEqual([r['turn_class'] for r in reports],
                         ['cold'] * 2 + ['same-topic'] * 2 + ['shift'] * 2 + ['cold'] * 2)
        for report, fixed, online in zip(reports, [0, 0, 0, 1, 0, None, 0, None],
                                          [.5, 0, 1, 0, 0, None, 0, None]):
            for name, rate in [('static_prior_turn', fixed), ('adaptive_lru', online),
                               ('adaptive_decay', online)]:
                p = report['policies'][name]
                self.assertEqual(len(p['per_layer']), 40)
                self.assertEqual(p['global']['slot_hit_fraction'], rate)
                self.assertEqual(p['global']['assignments'], report['tokens'] * 320)
                self.assertEqual(p['global']['slot_hits'], sum(x['slot_hits'] for x in p['per_layer']))

    def test_layer_specific_and_atomic_top8(self):
        row = phase(kind='decode', starts=[0])
        raw = bytearray(base64.b64decode(row['routed_b64']))
        # With sequential eviction, touching 8 first would evict 0 prematurely.
        struct.pack_into('<8i', raw, 0, 8, 0, 1, 2, 3, 4, 5, 6)
        row['routed_b64'] = base64.b64encode(raw).decode()
        p = replay(phase(starts=[0]), row, capacity=8)[1]['policies']['adaptive_lru']
        self.assertEqual(p['global']['slot_hits'], 319)
        self.assertEqual(p['per_layer'][0]['slot_hits'], 7)
        self.assertEqual(p['per_layer'][1]['slot_hits'], 8)

    def test_decay_retains_frequent_experts(self):
        p = replay(phase(starts=[0] * 4 + [8, 16]), phase(kind='decode', starts=[0]),
                   capacity=16, decay=.9)[1]['policies']
        self.assertEqual(p['adaptive_lru']['global']['slot_hits'], 0)
        self.assertEqual(p['adaptive_decay']['global']['slot_hits'], 320)

    def test_schema_shape_ranges_and_duplicates(self):
        base = phase(starts=[0])
        for changes in [{'tokens': True}, {'turn': True}, {'tokens': 2}, {'session': 3},
                        {'topic': None}, {'phase': 'other'}, {'routed_b64': '!'},
                        {'routed_b64': '/' * len(base['routed_b64'])},
                        {'routed_b64': '!' + base['routed_b64'][1:]}]:
            with self.subTest(changes=list(changes)), self.assertRaisesRegex(ValueError, 'line 1:'):
                replay(dict(base, **changes))
        for bad in (1, -1, 256):
            raw = bytearray(base64.b64decode(base['routed_b64']))
            struct.pack_into('<i', raw, 0, bad)
            with self.assertRaisesRegex(ValueError, 'invalid routed experts'):
                replay(dict(base, routed_b64=base64.b64encode(raw).decode()))
        with self.assertRaisesRegex(ValueError, 'duplicate JSON field'):
            list(analyze(io.BytesIO(b'{"turn":1,"turn":1}')))

    def test_order(self):
        p, d = phase(), phase(kind='decode')
        for rows in [[p], [d], [p, p], [p, d, d], [p, dict(d, topic='y')], [dict(p, turn=0)],
                     [dict(p, turn=2)], [p, d, dict(p, turn=3)], [p, dict(p, session='t')],
                     [p, d, dict(p, session='t'), dict(d, session='t'), p]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                replay(*rows)

    def test_resource_limits(self):
        with self.assertRaisesRegex(ValueError, 'token count'):
            replay(phase(starts=[0]), max_tokens=0)
        with self.assertRaisesRegex(ValueError, 'session limit'):
            replay(phase(), phase(kind='decode'), phase(sid='t'), max_sessions=1)
        stream = io.BytesIO(b' ' * 65536)
        with self.assertRaisesRegex(ValueError, 'byte limit'):
            list(analyze(stream, max_tokens=0))
        self.assertLess(stream.tell(), len(stream.getvalue()))
        for opts in [dict(capacity=7), dict(capacity=257), dict(decay=float('nan')), dict(decay=1)]:
            with self.assertRaises(ValueError):
                replay(**opts)


if __name__ == '__main__':
    unittest.main()
