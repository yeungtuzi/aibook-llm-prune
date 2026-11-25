"""Loader hooks used only by the explicitly selected AIBOOK overlay."""
import hashlib
import json
import logging
import os
from pathlib import Path
import re

CONFIG = '5e3eb35e998dbe5fab29832c221a43624588fde595279b3a7553003d8b739172'
INDEX = 'a0166e050e2ee06b31a55c0d068f971501fe4412e6deb8063ac53928de956df6'
RECEIPT = '795749b6c076e20d8f8b359d7df869772040636b9c88db8feff3905bed3aecd4'
ROUTED = re.compile(r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|down|up)_proj\.(qweight|scales|qzeros|g_idx)')
log = logging.getLogger(__name__)


def require(ok, message):
    if not ok:
        raise RuntimeError('AIBOOK slots: ' + message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_root():
    return Path(os.environ['AIBOOK_PACK_ROOT']).resolve(strict=True)


def validate(args, config, load):
    """Called at loader selection, before the vendor allocates any model layers."""
    mtp = os.environ.get('AIBOOK_MTP') == '1'
    dflash = os.environ.get('AIBOOK_DFLASH') == '1'
    ngram = os.environ.get('AIBOOK_NGRAM') == '1'
    if getattr(config, 'is_draft_model', False) or getattr(load, 'draft_model_idx', None) is not None:
        require(mtp or dflash, 'draft model load requires AIBOOK_MTP=1 or AIBOOK_DFLASH=1')
        os.environ['AIBOOK_LOADING_DRAFT'] = '1'
        return
    require(config is not None and config.hf_config.architectures == ['Qwen3_5MoeForConditionalGeneration']
            and config.quantization == 'moe_wna16' and str(config.dtype) == 'torch.float16',
            'only the pinned FP16 moe_wna16 Qwen model is supported')
    root = Path(config.model_path)
    require(digest(root/'config.json') == CONFIG and digest(root/'model.safetensors.index.json') == INDEX,
            'wrong config or index; use the original immutable checkpoint')
    require(digest(pack_root()/'verified.json') == RECEIPT, 'wrong/incomplete pack receipt')
    require(8 <= int(os.environ['AIBOOK_SLOTS']) < 256, 'slot count must be 8..255')
    require(args.device == 'musa' and args.language_model_only and args.disable_shared_experts_fusion
            and args.tp_size == args.ep_size == args.pp_size == args.dp_size == args.max_running_requests == 1,
            'require MUSA language-only, one worker/request, separate shared expert')
    require(not (args.enable_torch_compile or args.enable_lora or args.enable_eplb or args.ep_num_redundant_experts
                 or (args.speculative_algorithm and not (mtp or dflash or ngram)) or args.enable_dp_attention or args.enable_two_batch_overlap
                 or args.cpu_offload_gb or args.enable_weights_cpu_backup or args.weight_loader_prefetch_checkpoints
                 or args.custom_weight_loader) and args.weight_cache_mode == 'off'
            and args.init_expert_location == 'trivial' and args.startup_weight_load_mode == 'serial',
            'unsupported routing/parallel/loader/compile mode')
    resident = os.environ.get('AIBOOK_RESIDENT') == '1'
    if not resident and os.environ.get('AIBOOK_GRAPH_PROBE') != '1':
        require(all(getattr(args.cuda_graph_config, p).backend == 'disabled' for p in ('decode', 'prefill')),
                'both resolved graph backends must be disabled')
    require(load.load_format in ('auto', 'safetensors') and (load.draft_model_idx is None or mtp),
            'require default serial safetensors loader')


def make_experts(block, config, quant, a2a, epilogue):
    from aibook_slots import BoundedExperts
    require(not block.enable_shared_expert_fusion and block.num_shared_experts == 1 and not epilogue
            and a2a.is_none() and config.shared_expert_intermediate_size == 512,
            'require standalone shared MLP, immediate MoE reduction, no all-to-all')
    require((config.num_experts, config.num_experts_per_tok, config.hidden_size, config.moe_intermediate_size,
             quant.get_name(), quant.linear_quant_method, quant.weight_bits, quant.group_size, quant.has_zp)
            == (256, 8, 2048, 512, 'moe_wna16', 'gptq', 4, -1, False), 'unsupported expert layout')
    slots = int(os.environ['AIBOOK_SLOTS'])
    if os.environ.get('AIBOOK_LAYER_BUDGET'):
        caps = json.loads(Path(os.environ['AIBOOK_LAYER_BUDGET']).read_bytes())['allocated']
        require(len(caps)==40 and all(type(k) is int and 8<=k<256 for k in caps) and sum(caps)==40*slots, 'invalid per-layer slot budget')
        slots = caps[block.layer_id]
    return BoundedExperts(pack_root(), block.layer_id, slots, CONFIG)


def check_model(model):
    from aibook_slots import BoundedExperts
    if model.__class__.__name__.endswith('MTP') or 'DFlash' in model.__class__.__name__:
        return
    layers = model.model.layers
    require(len(layers) == 40 and all(isinstance(layer.mlp.experts, BoundedExperts)
            and layer.mlp.shared_expert is not None and layer.mlp.shared_expert_gate is not None
            for layer in layers), 'incomplete replacement or missing shared branch')
    if os.environ.get('AIBOOK_PROJECTION_GRAPH') == '1':
        from projection_graph import install
        install(model)
    if os.environ.get('AIBOOK_MODULE_TIMING') == '1':
        from module_timing import install
        install(model)
    if os.environ.get('AIBOOK_GRAPH_PROBE') == '1':
        from graph_compat import install
        install()
    if os.environ.get('AIBOOK_RESIDENT') == '1':
        log.warning('AIBOOK: resident mode, %s experts/layer, pruned router mask', os.environ['AIBOOK_SLOTS'])
    else:
        log.warning('AIBOOK: 40 bounded routed blocks, %s slots/layer, separate shared MLP', os.environ['AIBOOK_SLOTS'])


def weights(files, source, safetensors):
    from safetensors import safe_open
    if os.environ.get('AIBOOK_LOADING_DRAFT') == '1':
        for path in files:
            with safe_open(path, framework='pt', device='cpu') as shard:
                for name in shard.keys():
                    if name.startswith('mtp.'):
                        yield name, shard.get_tensor(name)
        return
    root = Path(source.model_or_path).resolve(strict=True)
    expected_files = set(json.loads((pack_root()/'verified.json').read_bytes())['source_sha256']) - {'config.json'}
    require(safetensors and source.prefix == '' and {Path(f).resolve() for f in files}
            == {root / f for f in expected_files}, 'unexpected weight source')
    # Even the preserved shard contains dense MTP expert weights. Never call
    # get_tensor for MTP/vision (text-only), or for any routed checkpoint key.
    skipped = passed = 0
    allow_mtp = os.environ.get('AIBOOK_LOADING_DRAFT') == '1'
    dflash = os.environ.get('AIBOOK_DFLASH') == '1'
    ngram = os.environ.get('AIBOOK_NGRAM') == '1'
    for path in files:
        with safe_open(path, framework='pt', device='cpu') as shard:
            for name in shard.keys():
                if allow_mtp and dflash:
                    yield name, shard.get_tensor(name)
                    continue
                if allow_mtp and name.startswith('mtp.'):
                    yield name, shard.get_tensor(name)
                    continue
                if name.startswith(('mtp.', 'model.visual.')):
                    continue
                if '.mlp.experts.' in name:
                    m = ROUTED.fullmatch(name)
                    require(m and int(m[1]) < 40 and int(m[2]) < 256, f'unexpected expert key {name}')
                    skipped += 1
                    continue
                passed += 1
                yield name, shard.get_tensor(name)
    if allow_mtp and skipped == 0:
        log.warning('AIBOOK: MTP draft load, %d mtp/non-routed keys', passed)
        return
    require(skipped == 40*256*3*4, 'incomplete original expert checkpoint')
    log.warning('AIBOOK: skipped %d routed keys before get_tensor; yielded %d non-routed keys', skipped, passed)


def verify_tree_greedy_musa(predicts, accept_index, accept_token_num, candidates,
                            retrieve_index, retrieve_next_token,
                            retrieve_next_sibling, target_predict, topk=-1):
    """CPU tree walk for irregular NGRAM trees on MUSA (topk != 1)."""
    import numpy as np
    import torch

    if __import__('os').environ.get('AIBOOK_NGRAM_DEBUG') == '1' and not getattr(verify_tree_greedy_musa, '_dbg', False):
        verify_tree_greedy_musa._dbg = True
        try:
            __import__('numpy').savez('/home/lf/aibook-bench/ngram-dbg.npz',
                cand=candidates.detach().cpu().numpy(),
                ri=retrieve_index.detach().cpu().numpy(),
                nxt=retrieve_next_token.detach().cpu().numpy(),
                sib=retrieve_next_sibling.detach().cpu().numpy(),
                tp=target_predict.detach().cpu().numpy(), topk=topk)
        except Exception:
            pass
    cand = candidates.detach().cpu().numpy()
    bs = cand.shape[0]
    ri = retrieve_index.detach().cpu().numpy().reshape(bs, -1)
    nxt = retrieve_next_token.detach().cpu().numpy().reshape(bs, -1)
    sib = retrieve_next_sibling.detach().cpu().numpy().reshape(bs, -1)
    tp = target_predict.detach().cpu().numpy().reshape(bs, -1)
    pred = predicts.detach().cpu().numpy().reshape(bs, -1).copy()
    acc = accept_index.detach().cpu().numpy().copy()
    num = np.zeros((bs,), dtype=np.int32)
    for b in range(bs):
        last = int(ri[b, 0])
        acc[b, 0] = last
        accepted = 0
        for depth in range(1, acc.shape[1]):
            target = int(tp[b, last])
            child = int(nxt[b, last])
            chosen = -1
            while child != -1:
                if int(cand[b, child]) == target:
                    chosen = child
                    break
                child = int(sib[b, child])
            if chosen < 0:
                break
            pred[b, last] = target
            accepted += 1
            acc[b, depth] = chosen
            last = chosen
        pred[b, last] = int(tp[b, last])
        num[b] = accepted
    predicts.copy_(torch.from_numpy(pred.reshape(-1)).to(predicts.device))
    accept_index.copy_(torch.from_numpy(acc).to(accept_index.device))
    accept_token_num.copy_(torch.from_numpy(num).to(accept_token_num.device))


def verify_tree_greedy_musa_torch(predicts, accept_index, accept_token_num,
                                  candidates, retrieve_index,
                                  retrieve_next_token, retrieve_next_sibling,
                                  target_predict, topk=-1):
    """GPU-only irregular-tree greedy verify using small PyTorch ops.

    Avoids the GPU->CPU sync of the tree walk: build the child adjacency on
    device, then walk one depth at a time with masked gathers.
    """
    import torch

    bs, n = candidates.shape
    dev = candidates.device
    max_depth = accept_index.shape[1]
    idx = torch.arange(n, device=dev, dtype=torch.long)
    bidx = torch.arange(bs, device=dev, dtype=torch.long)
    child_mask = torch.zeros((bs, n, n), dtype=torch.bool, device=dev)
    child = retrieve_next_token.to(torch.long).clone()
    for _ in range(n):
        valid = child >= 0
        c = child.clamp_min(0)
        parent = idx.view(1, n).expand(bs, n)
        child_mask[bidx.view(bs, 1), parent, c] = valid
        nxt = retrieve_next_sibling.to(torch.long).gather(1, c)
        child = torch.where(valid, nxt, torch.full_like(child, -1))
    current = retrieve_index[:, 0].to(torch.long)
    pred = predicts.view(bs, n).clone()
    acc = torch.full((bs, max_depth), -1, dtype=torch.int32, device=dev)
    acc[:, 0] = current
    num = torch.zeros(bs, dtype=torch.int32, device=dev)
    active = torch.ones(bs, dtype=torch.bool, device=dev)
    for depth in range(1, max_depth):
        target = target_predict.gather(1, current.view(bs, 1)).squeeze(1).to(pred.dtype)
        children = child_mask.gather(
            1, current.view(bs, 1, 1).expand(bs, 1, n)
        ).squeeze(1)
        match = children & (candidates == target.view(bs, 1)) & active.view(bs, 1)
        has = match.any(dim=1)
        chosen = match.to(torch.int8).argmax(dim=1).to(torch.long)
        chosen = torch.where(has, chosen, current)
        pred.scatter_(1, current.view(bs, 1), target.view(bs, 1))
        acc[:, depth] = torch.where(has, chosen, torch.full_like(chosen, -1)).to(torch.int32)
        current = chosen
        num += has.to(torch.int32)
        active = has
    final_target = target_predict.gather(1, current.view(bs, 1)).squeeze(1).to(pred.dtype)
    pred.scatter_(1, current.view(bs, 1), final_target.view(bs, 1))
    predicts.copy_(pred.view(-1))
    accept_index.copy_(acc)
    accept_token_num.copy_(num)
