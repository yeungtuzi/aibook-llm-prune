import base64
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import analyze
import capture


def answer(p=4, c=2, k=0):
    # Garbage in reused slots is intentional: it must never enter replay.
    payload = b''.join((struct.pack('<8i', *([-1]*8 if i < k else range(i, i+8))) * 40)
                       for i in range(p+c-1))
    return dict(text='The real previous answer.', output_ids=list(range(20, 20+c)),
                meta_info=dict(prompt_tokens=p, completion_tokens=c, cached_tokens=k, num_retractions=0,
                               weight_version='v1', finish_reason={'type':'length'},
                               routed_experts=base64.b64encode(payload).decode()))


class FakeServer:
    def __init__(self):
        self.histories = []
        self.generated = []

    def __call__(self, path, payload=None, limit=None):
        if path == '/server_info':
            return {'enable_return_routed_experts': True, 'speculative_algorithm': None, 'context_length': 100}
        if path == '/model_info':
            return {'model_path': capture.MODEL, 'is_generation': True, 'weight_version': 'v1'}
        if path == '/v1/tokenize':
            self.histories.append(list(payload['messages']))
            # Second turn shares the first prompt + one forwarded generated token.
            ids = [1,2,3,4] if len(self.histories) == 1 else [1,2,3,4,20,25,1]
            return {'tokens': ids, 'count': len(ids)}
        if path == '/generate':
            self.generated.append(payload)
            return answer(len(payload['input_ids']), 2, 0 if len(self.generated) == 1 else 5)
        raise AssertionError(path)


class CaptureTest(unittest.TestCase):
    def test_phase_positions(self):
        for k in (0, 1, 4):
            for c in (1, 3):
                with self.subTest(k=k, c=c):
                    rows = capture.phase_rows(answer(c=c, k=k)['meta_info'], 4, 8)
                    for row, (lo, hi) in zip(rows, [(k,4),(4,4+c-1)]):
                        self.assertEqual((row['start'], row['stop'], row['tokens']), (lo,hi,hi-lo))
                        data = base64.b64decode(row['routed_b64'])
                        self.assertEqual([struct.unpack_from('<i', data, i)[0] for i in range(0,len(data), capture.WIDTH)], list(range(lo,hi)))

    def test_reject_bad_payloads(self):
        meta = answer()['meta_info']
        for override in [dict(cached_tokens=-1), dict(cached_tokens=5), dict(cached_tokens=None),
                         dict(prompt_tokens=5), dict(completion_tokens=0), dict(completion_tokens=9),
                         dict(num_retractions=1), dict(finish_reason={'type':'abort'}),
                         dict(routed_experts=meta['routed_experts'][4:]),
                         dict(routed_experts='!' + meta['routed_experts'][1:]),
                         dict(routed_experts=base64.b64encode(b'\x00'*(5*capture.WIDTH)).decode())]:
            with self.subTest(keys=list(override)), self.assertRaises(ValueError):
                capture.phase_rows(dict(meta, **override), 4, 8)

    @contextlib.contextmanager
    def directory(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
            yield Path(d)/'new'

    def test_warm_conversation_replays(self):
        server = FakeServer()
        with self.directory() as output:
            prompts = json.loads(capture.PROMPTS.read_text())[:2]
            capture.run(server, prompts, output, max_new=2)
            self.assertEqual(server.histories[1][2], {'role':'assistant', 'content':'The real previous answer.'})
            self.assertEqual([m['role'] for m in server.histories[1]], ['system','user','assistant','user'])
            salts = {p['cache_salt'] for p in server.generated}
            self.assertEqual(len(salts), 1)
            self.assertTrue(all(p['routed_experts_start_len'] == 0 and p['return_routed_experts'] for p in server.generated))
            data = (output/'routes.jsonl').read_bytes()
            stats = list(analyze.analyze(io.BytesIO(data), capacity=8))
            self.assertEqual([s['tokens'] for s in stats], [4,1,2,1])
            self.assertEqual(json.loads((output/'complete.json').read_text())['turns'], 2)
            self.assertEqual(len((output/'turns.jsonl').read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                capture.run(server, prompts, output)

    def test_fail_closed(self):
        prompts = json.loads(capture.PROMPTS.read_text())[:1]
        for mode in ('disabled','spec','wrong-model','oversize','alien-cache','context'):
            fake = FakeServer()
            def request(path, *args):
                obj = fake(path, *args)
                if path == '/server_info' and mode == 'disabled': obj['enable_return_routed_experts'] = False
                if path == '/server_info' and mode == 'spec': obj['speculative_algorithm'] = 'EAGLE'
                if path == '/model_info' and mode == 'wrong-model': obj['model_path'] = 'wrong'
                if path == '/server_info' and mode == 'context': obj['context_length'] = 5
                if path == '/generate' and mode == 'alien-cache': obj['meta_info']['cached_tokens'] = 1
                return obj
            with self.subTest(mode=mode), self.directory() as output:
                with self.assertRaises(ValueError):
                    capture.run(request, prompts, output, max_new=2, max_prompt=3 if mode=='oversize' else 4)
                self.assertFalse((output/'complete.json').exists())
                self.assertEqual(len(fake.generated), int(mode=='alien-cache'))


if __name__ == '__main__':
    unittest.main()
