"""Offline chronological slot simulation. Input and output are JSON Lines.

Only newly routed positions belong in the input. Sessions are contiguous; each
turn is prefill then decode, even when empty. Caches start empty per session.
Static freezes the immediately preceding whole turn's most frequent observed
experts per layer (ties: lower ID); unused slots are not padded. Adaptive
caches persist across both phases and turns. Score the full top-8 before any
updates; all eight remain resident after the update. LRU breaks simultaneous
recency ties by route order (last ID newest). Optional resident frequency
multiplies scores by --decay once per token, adds one to chosen IDs, and evicts
lowest unchosen scores (lower ID retained on ties); evictions forget scores.

Global means assignment-weighted across 40 independent layers, NOT a shared
cache. Misses are not measurements of SSD reads, bytes, latency or throughput.
Each output row classifies the turn relative to its immediate predecessor:
cold (first turn), same-topic (equal topic strings), or shift (unequal).
per_layer is ordered by zero-based layer ID; zero assignments give a null rate.
Session/topic labels are limited to 256 characters. The JSON line limit is
16 KiB plus the encoded size of --max-tokens positions. Additional metadata
is ignored. One result per validated phase; on error discard partial output.
"""
import argparse
import base64
import binascii
import json
import struct
import sys
from collections import OrderedDict
from contextlib import nullcontext

LAYERS, TOP, EXPERTS = 40, 8, 256
ROUTE = struct.Struct('<8i')
MAX_TOKENS = 8192
MAX_SESSIONS = 4096
MAX_LABEL = 256


def unique_fields(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON field: ' + key[:40])
        obj[key] = value
    return obj


def records(stream, max_tokens):
    # Limit before reading, parsing JSON, or allocating decoded bytes. Reserve
    # 16 KiB beyond payload for metadata (including escaped labels/whitespace).
    limit = 4 * ((max_tokens * LAYERS * ROUTE.size + 2) // 3) + 16384
    while True:
        line = stream.readline(limit + 1)
        if not line:
            return
        if len(line) > limit:
            raise ValueError('record exceeds byte limit')
        row = json.loads(line.decode('utf-8'), object_pairs_hook=unique_fields)
        if not isinstance(row, dict):
            raise ValueError('expected an object')
        for key in ('session', 'topic'):
            if not isinstance(row.get(key), str) or len(row[key]) > MAX_LABEL:
                raise ValueError(f'{key} must be a string of at most {MAX_LABEL} characters')
        if type(row.get('turn')) is not int or row['turn'] < 1:
            raise ValueError('turn must be a positive integer')
        n = row.get('tokens')
        if type(n) is not int or not 0 <= n <= max_tokens:
            raise ValueError('invalid token count')
        if row.get('phase') not in ('prefill', 'decode'):
            raise ValueError('invalid phase')
        encoded = row.get('routed_b64')
        size = n * LAYERS * ROUTE.size
        if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
            raise ValueError('encoded length does not match tokens')
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != size:
            raise ValueError('decoded shape does not match tokens')
        yield row['session'], row['turn'], row['topic'], row['phase'], n, raw


class Session:
    def __init__(self, capacity, decay):
        self.capacity = capacity
        self.decay = decay
        self.counts = [[0] * EXPERTS for _ in range(LAYERS)]
        self.lru = [OrderedDict() for _ in range(LAYERS)]
        self.scores = [{} for _ in range(LAYERS)] if decay is not None else None

    def start_turn(self):
        self.static = [set(sorted((e for e, count in enumerate(layer) if count),
                                  key=lambda e: (-layer[e], e))[:self.capacity])
                       for layer in self.counts]
        self.counts = [[0] * EXPERTS for _ in range(LAYERS)]

    def score(self, raw, tokens):
        hits = {'static_prior_turn': [0] * LAYERS, 'adaptive_lru': [0] * LAYERS}
        if self.scores is not None:
            hits['adaptive_decay'] = [0] * LAYERS
        for i, route in enumerate(ROUTE.iter_unpack(raw)):
            layer = i % LAYERS
            chosen = set(route)
            if len(chosen) != TOP or min(route) < 0 or max(route) >= EXPERTS:
                raise ValueError(f'invalid routed experts at token {i // LAYERS}, layer {layer}')
            hits['static_prior_turn'][layer] += len(chosen & self.static[layer])
            cache = self.lru[layer]
            hits['adaptive_lru'][layer] += sum(e in cache for e in route)
            for e in route:
                self.counts[layer][e] += 1
                cache[e] = None
                cache.move_to_end(e)
            while len(cache) > self.capacity:
                cache.popitem(last=False)
            if self.scores is not None:
                scores = self.scores[layer]
                hits['adaptive_decay'][layer] += sum(e in scores for e in route)
                for e in scores:
                    scores[e] *= self.decay
                for e in route:
                    scores[e] = scores.get(e, 0.) + 1.
                excess = len(scores) - self.capacity
                if excess > 0:
                    victims = sorted(scores.keys() - chosen, key=lambda e: (scores[e], -e))
                    for e in victims[:excess]:
                        del scores[e]
        return {name: {'global': metric(sum(values), tokens * TOP * LAYERS),
                       'per_layer': [metric(h, tokens * TOP) for h in values]}
                for name, values in hits.items()}


def metric(hits, total):
    return {'assignments': total, 'slot_hits': hits, 'slot_misses': total - hits,
            'slot_hit_fraction': hits / total if total else None}


def analyze(stream, capacity=32, decay=None, max_tokens=MAX_TOKENS, max_sessions=MAX_SESSIONS):
    """Yield reports from a binary stream. Bounds are per phase and per run."""
    if not TOP <= capacity <= EXPERTS:
        raise ValueError('capacity must be 8..256 per layer')
    if decay is not None and not 0 < decay < 1:
        raise ValueError('decay must be strictly between zero and one')
    if max_tokens < 0 or max_sessions < 1:
        raise ValueError('invalid resource limits')
    seen = set()
    active = None
    expected_phase = 'prefill'
    line = 0
    try:
        for sid, turn, topic, phase, n, raw in records(stream, max_tokens):
            if sid != active:
                if expected_phase == 'decode':
                    raise ValueError('missing decode before session change')
                if sid in seen or len(seen) >= max_sessions:
                    raise ValueError('noncontiguous session or session limit exceeded')
                seen.add(sid)
                active = sid
                state = Session(capacity, decay)
                expected_turn, previous_topic = 1, None
            if turn != expected_turn or phase != expected_phase:
                raise ValueError(f'expected turn {expected_turn} {expected_phase}')
            if phase == 'prefill':
                turn_class = ('cold' if turn == 1 else
                              'same-topic' if topic == previous_topic else 'shift')
                state.start_turn()
                previous_topic = topic
                expected_phase = 'decode'
            else:
                if topic != previous_topic:
                    raise ValueError('topic changed within turn')
                expected_turn += 1
                expected_phase = 'prefill'
            policies = state.score(raw, n)
            line += 1
            yield dict(session=sid, turn=turn, topic=topic, phase=phase, tokens=n,
                       turn_class=turn_class, capacity_per_layer=capacity, decay=decay,
                       policies=policies)
        if expected_phase == 'decode':
            raise ValueError('missing decode at EOF')
    except (ValueError, RecursionError, binascii.Error) as exc:
        raise ValueError(f'line {line + 1}: {exc}') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', nargs='?', default='-', help='JSONL path, or - for stdin (default)')
    parser.add_argument('--capacity', type=int, default=32, help='slots per layer, 8..256 (default 32)')
    parser.add_argument('--decay', type=float, help='enable adaptive resident decay (0 < factor < 1)')
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS, help='max tokens per phase (default 8192)')
    parser.add_argument('--max-sessions', type=int, default=MAX_SESSIONS, help='max distinct sessions (default 4096)')
    args = parser.parse_args()
    try:
        with (nullcontext(sys.stdin.buffer) if args.input == '-' else open(args.input, 'rb')) as stream:
            for report in analyze(stream, args.capacity, args.decay, args.max_tokens, args.max_sessions):
                print(json.dumps(report, separators=(',', ':'), allow_nan=False))
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(1, f'analysis failed: {exc}\n')


if __name__ == '__main__':
    main()
