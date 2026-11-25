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
                 or (args.speculative_algorithm and not (mtp or dflash)) or args.enable_dp_attention or args.enable_two_batch_overlap
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
