"""Independent two-turn chat; collect each streaming interval and worker physical reads."""
import json,time,sys,urllib.request
from pathlib import Path
B='http://127.0.0.1:30141';cg=Path('/sys/fs/cgroup')/sys.argv[1].lstrip('/');model=json.load(urllib.request.urlopen(B+'/v1/models'))['data'][0]['id'];messages=[]
def req(path,data):return urllib.request.urlopen(urllib.request.Request(B+path,json.dumps(data).encode(),{'Content-Type':'application/json'}),timeout=180)
def readbytes():return sum(int(next(l.split()[1] for l in Path('/proc',p,'io').read_text().splitlines() if l.startswith('read_bytes:'))) for p in (cg/'cgroup.procs').read_text().split())
for prompt in ['Write a compact Python parser for elapsed time counters in a compiler test harness. Explain one unit test.', 'Extend that parser to accept nanoseconds and reject negative counts. Show the updated test.']:
    messages.append(dict(role='user',content=prompt));ids=json.load(req('/v1/tokenize',dict(model=model,messages=messages)))['tokens'];b=readbytes();t=time.monotonic();stamps=[];counts=[]
    with req('/generate',dict(input_ids=ids,sampling_params=dict(temperature=0,max_new_tokens=48,ignore_eos=True),stream=True)) as s:
        for line in s:
            if not line.startswith(b'data:') or line.strip().endswith(b'[DONE]'):continue
            d=json.loads(line[5:]);c=d['meta_info']['completion_tokens'];
            if not counts or counts[-1]!=c:counts.append(c);stamps.append(time.monotonic())
    assert counts==list(range(1,49));g=[(y-x)*1000 for x,y in zip(stamps,stamps[1:])];messages.append(dict(role='assistant',content=d['text']));print(json.dumps(dict(prompt_ids=ids,output_ids=d['output_ids'],cached=d['meta_info']['cached_tokens'],read_bytes=readbytes()-b,ttft=stamps[0]-t,tps=47/(stamps[-1]-stamps[0]),p95=sorted(g)[44],gap_ms=g)),flush=True)
