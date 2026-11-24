"""Fit fixed per-layer LRU capacities to training decode ONLY; replay untouched continuation."""
import json,sys
from collections import OrderedDict
from analyze import records,ROUTE

def score(rows,caps):
    caches=[OrderedDict() for _ in caps];out=[]
    for _,turn,_,phase,n,raw in rows:
        misses=[0]*40
        for i,route in enumerate(ROUTE.iter_unpack(raw)):
            j=i%40;c=caches[j];misses[j]+=sum(e not in c for e in route)
            for e in route:c[e]=None;c.move_to_end(e)
            while len(c)>caps[j]:c.popitem(last=False)
        if phase=='decode':out.append((turn,n,misses))
    return out

def main(path):
    with open(path,'rb') as f:rows=list(records(f,8192))
    assert len({r[0] for r in rows})==1 and {r[1] for r in rows}==set(range(1,6)); train=[r for r in rows if r[1]<=2]; grid=range(64,193,16);loss={}
    for k in grid:loss[k]=list(map(sum,zip(*(m for _,_,m in score(train,[k]*40)))))
    # Exact dynamic program; secondary objective favors the uniform baseline on ties.
    dp={0:(0,0,[])}
    for j in range(40):
        nxt={}
        for used,(cost,disp,path0) in dp.items():
            for k in grid:
                v=(cost+loss[k][j],disp+abs(k-128),path0+[k]);u=used+k
                if u<=5120 and (u not in nxt or v<nxt[u]):nxt[u]=v
        dp=nxt
    caps=dp[5120][2];a=score(rows,[128]*40);b=score(rows,caps);print(json.dumps(dict(train_turns=[1,2],held_out=[3,4,5],allocated=caps,total_slots=sum(caps),decode=[dict(turn=t,tokens=n,uniform_misses=sum(x),fitted_misses=sum(y)) for (t,n,x),(_,_,y) in zip(a,b)])))
if __name__=='__main__':main(sys.argv[1])
