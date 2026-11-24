"""Bounded native SGLang route collection. No startup, cache flush, or retries."""
import argparse
import base64
import itertools
import json
from pathlib import Path
import struct
import urllib.request
import uuid

PROMPTS = Path(__file__).with_name('prompts.json')
MODEL = '/home/lf/models/gptq-Qwen3.6-35B-A3B-4bit-channel-lmhead'
SYSTEM = 'You are pair-programming a measurement harness. Use concise, concrete answers; stay within the scope of the current turn.'
WIDTH = 40 * 8 * 4
MAX_PROMPT = 2048
MAX_NEW = 128
MAX_TOTAL = 8192  # total full prompt + requested completions, not merely cache misses


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, lo, hi, name):
    require(type(value) is int and lo <= value <= hi, f'invalid {name}: expected {lo}..{hi}')
    return value


def phase_rows(meta, prompt, max_new):
    """Absolute route offset is zero. Validate wire shape before discarding old KV rows."""
    p = integer(meta.get('prompt_tokens'), 1, MAX_PROMPT, 'prompt_tokens')
    require(p == prompt, 'server prompt_tokens differs from submitted input_ids length')
    c = integer(meta.get('completion_tokens'), 1, max_new, 'completion_tokens')
    k = integer(meta.get('cached_tokens'), 0, p, 'cached_tokens')
    integer(meta.get('num_retractions'), 0, 0, 'num_retractions')
    require((meta.get('finish_reason') or {}).get('type') in ('stop', 'length'), 'not a successful finished generation')
    end = p + c - 1
    size = WIDTH * end
    encoded = meta.get('routed_experts')
    require(isinstance(encoded, str) and len(encoded) == 4 * ((size + 2) // 3), 'unexpected full route payload size')
    data = base64.b64decode(encoded, validate=True)
    require(len(data) == size, 'incorrect full route byte count')
    rows = []
    for phase, start, stop in [('prefill', k, p), ('decode', p, end)]:
        block = memoryview(data)[start * WIDTH:stop * WIDTH]
        for experts in struct.iter_unpack('<8i', block):
            require(min(experts) >= 0 and max(experts) < 256 and len(set(experts)) == 8, 'invalid newly computed route IDs')
        rows.append(dict(phase=phase, start=start, stop=stop, tokens=stop-start,
                         routed_b64=base64.b64encode(block).decode('ascii')))
    return rows


def http_client(base_url, timeout):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(NoRedirect)

    def request(path, payload=None, limit=1024*1024):
        req = urllib.request.Request(base_url.rstrip('/') + path,
                                     data=None if payload is None else json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=timeout) as res:
            raw = res.read(limit + 1)
        require(len(raw) <= limit, f'{path} exceeded byte cap')
        obj = json.loads(raw)
        require(isinstance(obj, dict), f'{path} did not return an object')
        return obj
    return request


def run(request, prompts, output, max_new=96, max_prompt=MAX_PROMPT):
    integer(max_new, 1, MAX_NEW, 'max_new')
    integer(max_prompt, 1, MAX_PROMPT, 'max_prompt')
    require(isinstance(prompts, list) and 1 <= len(prompts) <= 8, 'expected 1..8 turns')
    for entry in prompts:
        for key, cap in [('topic', 80), ('user', 4096)]:
            require(isinstance(entry.get(key), str) and 0 < len(entry[key].strip()) <= cap and len(entry[key]) <= cap, f'invalid prompt {key}')
    server = request('/server_info')
    require(server.get('enable_return_routed_experts') is True, 'capture disabled; nothing submitted')
    require('speculative_algorithm' in server and server['speculative_algorithm'] is None, 'unsupported or unknown speculative mode')
    model = request('/model_info')
    require(model.get('model_path') == MODEL and model.get('is_generation') is True, 'unexpected model; this layout is AIBOOK Qwen-specific')
    sid = uuid.uuid4().hex
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    messages = [{'role': 'system', 'content': SYSTEM}]
    total = 0
    prior_forwarded = []
    weight_version = model.get('weight_version')
    with (output/'routes.jsonl').open('x', encoding='utf-8') as routes, (output/'turns.jsonl').open('x', encoding='utf-8') as turns:
        for turn, entry in enumerate(prompts, 1):
            messages.append({'role': 'user', 'content': entry['user']})
            tok = request('/v1/tokenize', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}})
            ids = tok.get('tokens')
            require(isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids), 'invalid chat tokenization')
            p = integer(tok.get('count'), 1, max_prompt, 'rendered prompt length')
            require(len(ids) == p, 'tokenizer count mismatch')
            total += p + max_new
            require(total <= MAX_TOTAL, 'run token budget exceeded before generation')
            if server.get('context_length') is not None:
                require(p + max_new <= server['context_length'], 'context window exceeded')
            payload = dict(input_ids=ids, cache_salt=sid, return_routed_experts=True, routed_experts_start_len=0,
                           stream=False, sampling_params={'temperature': 0, 'max_new_tokens': max_new})
            # Full base64 [P+C-1,40,8] plus bounded text/metadata overhead.
            answer = request('/generate', payload, 4 * (((p + max_new - 1)*WIDTH + 2)//3) + 131072)
            meta = answer['meta_info']
            rows = phase_rows(meta, p, max_new)
            require('weight_version' in meta and meta['weight_version'] == weight_version, 'model weight version changed')
            text = answer.get('text')
            out_ids = answer.get('output_ids')
            require(isinstance(text, str) and len(text.encode()) <= 16384, 'invalid assistant text')
            require(isinstance(out_ids, list) and len(out_ids) == meta['completion_tokens'] and
                    all(type(i) is int and i >= 0 for i in out_ids), 'output_ids disagree with completion count')
            # A salt isolates this conversation. Matching might fall back to ANY earlier
            # turn (templates need not be prefix-stable); don't assume reuse of turn N-1.
            longest = max((sum(1 for _ in itertools.takewhile(lambda pair: pair[0] == pair[1], zip(old, ids)))
                           for old in prior_forwarded), default=0)
            require(meta['cached_tokens'] <= longest, 'reported cache reuse outside this isolated conversation')
            prior_forwarded.append(ids + out_ids[:-1])
            info = dict(session=sid, turn=turn, topic=entry['topic'], prompt_tokens=p, completion_tokens=len(out_ids),
                        cached_tokens=meta['cached_tokens'], longest_prior_prefix=longest, request_id=meta.get('id'))
            turns.write(json.dumps(dict(info, user=entry['user'], assistant=text, input_ids=ids, output_ids=out_ids,
                                        finish_reason=meta['finish_reason']), ensure_ascii=False) + '\n')
            for row in rows:
                routes.write(json.dumps(dict(info, **row), separators=(',', ':')) + '\n')
            messages.append({'role': 'assistant', 'content': text})
    # Written last: never analyze a partial capture as a completed conversation.
    manifest = dict(schema=1, session=sid, system=SYSTEM, model=model['model_path'], weight_version=weight_version,
                    server_version=server.get('version'), turns=len(prompts), max_new_tokens=max_new,
                    max_prompt_tokens=max_prompt, requested_token_budget_used=total, layout='little-endian int32 [tokens,40,8]')
    (output/'complete.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:30140')
    parser.add_argument('--output', required=True, type=Path, help='new directory (never overwrite/resume)')
    parser.add_argument('--turns', type=int, default=5, help='take first N bundled turns')
    parser.add_argument('--max-new', type=int, default=96)
    parser.add_argument('--max-prompt', type=int, default=MAX_PROMPT)
    parser.add_argument('--timeout', type=int, default=90, help='per socket operation in seconds; no retries')
    args = parser.parse_args()
    prompts = json.loads(PROMPTS.read_text(encoding='utf-8'))
    integer(args.turns, 1, len(prompts), 'turns')
    integer(args.timeout, 1, 180, 'timeout')
    print(json.dumps(run(http_client(args.base_url, args.timeout), prompts[:args.turns], args.output,
                         max_new=args.max_new, max_prompt=args.max_prompt)))


if __name__ == '__main__':
    main()
