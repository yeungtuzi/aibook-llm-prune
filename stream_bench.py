"""Matched b=1 native SSE probe; records client inter-token gaps and cgroup disk reads."""
import json,math,statistics,sys,time,urllib.request
from pathlib import Path
cg=Path('/sys/fs/cgroup')/sys.argv[1].lstrip('/');b='http://127.0.0.1:30141'
def reads():
    # User cgroup lacks io controller; sum live worker read_bytes, not system diskstats.
    total=0
    for pid in (cg/'cgroup.procs').read_text().splitlines():
        try:total+=int(next(l.split()[1] for l in Path('/proc',pid,'io').read_text().splitlines() if l.startswith('read_bytes:')))
        except FileNotFoundError:pass
    return {'live_worker_processes':total}
payload=dict(input_ids=[int(v) for v in sys.argv[2].split(',')],sampling_params=dict(temperature=0,max_new_tokens=40,ignore_eos=True),stream=True)
for trial in range(2):
    r0=reads();start=time.monotonic();ts=[];counts=[];tokens=[]
    req=urllib.request.Request(b+'/generate',json.dumps(payload).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as stream:
        for line in stream:
            if not line.startswith(b'data:') or line.strip().endswith(b'[DONE]'):continue
            d=json.loads(line[5:]);count=d['meta_info']['completion_tokens'];ids=d.get('output_ids',[])
            if counts and count==counts[-1]:continue
            ts.append(time.monotonic());counts.append(count);tokens=ids
    assert counts==list(range(1,41)),counts
    gap=[1000*(z-a) for a,z in zip(ts,ts[1:])];r1=reads();print(json.dumps(dict(trial=trial,ttft_ms=(ts[0]-start)*1000,decode_tps=39/(ts[-1]-ts[0]),p95_gap_ms=sorted(gap)[math.ceil(.95*len(gap))-1],ms=gap,output_ids=tokens,cgroup_read_bytes_by_device={k:r1[k]-r0.get(k,0) for k in r1},cached=d['meta_info'].get('cached_tokens'))),flush=True)
