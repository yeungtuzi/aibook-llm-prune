"""Minimal HumanEval pass@1 harness for the local SGLang OpenAI endpoint."""
import argparse, json, re, subprocess, sys, time, urllib.request
from pathlib import Path

URL = 'http://127.0.0.1:30141/v1/completions'
MODEL = '/home/lf/models/gptq-Qwen3.6-35B-A3B-4bit-channel-lmhead'
STOP = ['\nclass ', '\ndef ', '\n#', '\nif __name__', '\nprint(', '\nassert ']

def complete(prompt, max_tokens):
    body = json.dumps({'model': MODEL, 'prompt': prompt, 'max_tokens': max_tokens,
                       'temperature': 0.0, 'stop': STOP}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())['choices'][0]['text']

def clean(code):
    code = re.sub(r'^```(?:python)?\s*', '', code.strip())
    code = re.sub(r'\s*```$', '', code)
    return code

def run_one(task, out_dir, max_tokens):
    code = task['prompt'] + clean(complete(task['prompt'], max_tokens))
    code += '\n\n' + task['test'] + f"\ncheck({task['entry_point']})\n"
    path = out_dir / f"{task['task_id'].replace('/', '_')}.py"
    path.write_text(code)
    try:
        p = subprocess.run([sys.executable, str(path)], capture_output=True, timeout=15, text=True)
        return p.returncode == 0, p.stderr[-300:]
    except subprocess.TimeoutExpired:
        return False, 'timeout'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path.home()/'aibook-bench/HumanEval.jsonl')
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--out', type=Path, default=Path.home()/'aibook-bench/work')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in args.data.read_text().splitlines()][:args.limit]
    passed = 0
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        ok, err = run_one(task, args.out, args.max_tokens)
        passed += ok
        print(json.dumps({'i': i, 'id': task['task_id'], 'ok': ok, 'pass_at_1': passed / i,
                          'err': None if ok else err.replace('\n', ' ')[:160]}), flush=True)
    print(json.dumps({'summary': True, 'passed': passed, 'total': len(tasks),
                      'pass_at_1': passed / len(tasks), 'seconds': time.time() - t0}))

if __name__ == '__main__':
    main()
