"""Opt-in synchronized, nested decode-only accounting. Profiling adds barriers."""
from collections import defaultdict
from time import perf_counter_ns
import json
import torch

def install(model):
    active=False; totals=defaultdict(int); stack=[]; count=0
    def start(mod,args,kwargs):
        nonlocal active
        ids=kwargs.get('input_ids',args[0] if args else None)
        active=isinstance(ids,torch.Tensor) and ids.numel()==1
        if active:torch.musa.synchronize();stack.append(('model_other',perf_counter_ns(),0))
    def end(mod,args,out):
        nonlocal active,count
        if active:
            pop('model_other');active=False;count+=1
            if count==10:
                print('AIBOOK_MODEL_TIMING '+json.dumps(dict(totals)),flush=True);count=0;totals.clear()
    def pop(k):
        torch.musa.synchronize();tag,t,child=stack.pop();assert tag==k;ns=perf_counter_ns()-t;totals[k]+=ns-child
        if stack:
            p,a,b=stack[-1];stack[-1]=(p,a,b+ns)
    def watch(mod,k):
        def pre(m,a):
            if active:torch.musa.synchronize();stack.append((k,perf_counter_ns(),0))
        def post(m,a,o):
            if active:pop(k)
        mod.register_forward_pre_hook(pre);mod.register_forward_hook(post)
    # Runner calls outer model.forward directly, so hook its invoked inner module.
    core=model.model;core.register_forward_pre_hook(start,with_kwargs=True);core.register_forward_hook(end)
    for layer in core.layers:
        watch(layer,'decoder_other');watch(layer.mlp,'mlp_other');watch(layer.mlp.experts,'routed');watch(layer.mlp.shared_expert,'shared');watch(layer.mlp.shared_expert_gate,'shared_gate');watch(layer.mlp.gate,'router');
        if hasattr(layer,'linear_attn'):watch(layer.linear_attn,'linear_attention')
        else:
            for attr in ('qkv_proj','attn','o_proj'):watch(getattr(layer,attr),'full_attention_parts')
