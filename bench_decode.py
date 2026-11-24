import json,time,urllib.request
URL='http://127.0.0.1:30141/generate'
prompt='用 Python 写一个快速排序，并解释复杂度。'
body=json.dumps({'text':prompt,'sampling_params':{'max_new_tokens':64,'temperature':0.0},'stream':True}).encode()
t0=time.perf_counter(); stamps=[]; toks=[]; ttft=None
req=urllib.request.Request(URL,data=body,headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=300) as r:
    for line in r:
        line=line.strip()
        if not line.startswith(b'data:'): continue
        data=line[5:].strip()
        if data==b'[DONE]': break
        try: d=json.loads(data)
        except Exception: continue
        if 'text' in d:
            now=time.perf_counter(); stamps.append(now); toks.append(d.get('meta_info',{}).get('completion_tokens',0))
            if ttft is None: ttft=now-t0
g=[(b-a)*1000 for a,b in zip(stamps,stamps[1:])]
print(json.dumps({'ttft':ttft,'chunks':len(stamps),'tps':(len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 else 0,'p95':sorted(g)[int(.95*len(g))-1] if g else None,'final_tokens':toks[-1] if toks else None}))
