"""One reusable MUSA graph per routed block. No host routing, storage or eviction in capture."""
import torch
from torch.nn import functional as F

class RoutedGraph:
    def __init__(self, owner, x, wt, ix):
        self.x=x.clone();self.wt=wt.clone();self.ix=ix.clone();self.g=x.new_empty((8,1024));self.d=x.new_empty((8,2048));self.graph=torch.musa.MUSAGraph()
        def run():
            owner._gemv(self.x,owner.w13,self.g,None,owner.s13,None,self.wt,self.ix,False,8,True,False,weight_layout='legacy');gate,up=self.g.chunk(2,dim=1);h=(F.silu(gate)*up).contiguous()
            owner._gemv(h,owner.w2,self.d,None,owner.s2,None,self.wt,self.ix,True,1,True,False,weight_layout='legacy');return self.d.float().sum(0).half()
        reference=run();torch.musa.synchronize()
        with torch.musa.graph(self.graph):self.out=run()
        self.graph.replay();torch.musa.synchronize()
        if not torch.equal(reference,self.out):raise RuntimeError('captured routed output differs from eager')
    def replay(self,x,wt,ix):
        self.x.copy_(x);self.wt.copy_(wt);self.ix.copy_(ix);self.graph.replay();return self.out
