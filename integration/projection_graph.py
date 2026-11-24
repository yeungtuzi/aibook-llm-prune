"""Bounded experiment: replay ONLY layer 0's GDN fused GPTQ input projection.

In the validated 128-slot, serial BS=1 eager overlay, call install(model) once
from check_model. Never enable the SGLang graph backends for this experiment.
The hook is installed before loading; no tensor is bound until a real DECODE.

Why not attention? In vendor 0.5.18+20261101.torch2.11 the eager FA3 backend
allocates per-step page tables, sequence lengths and prefix sums and chooses a
host length bucket (flashattention_backend.py:1266,1371). The fused MUSA RoPE
path uses per-step positions and out_cache_loc (qwen3_5.py:1965). The hybrid
backend maps req_pool_indices to Mamba state slots, translates tracking slots,
branches on tracking masks and can snapshot/advance the replay-SSM ring once
per step (hybrid_linear_attn_backend.py:204). Capturing those pointers or host
branches from one ForwardBatch cannot stand in for graph-runner metadata prep.
Neither capture nor warmup here invokes any cache/state-writing operator.
"""
import logging
import os
import threading


log = logging.getLogger(__name__)
_INPUT = (1, 2048)
_OUTPUT = (1, 12352)  # Q,K (16*128 each), V,Z (32*128 each), B,A (32 each)


def require(ok, why):
    if not ok:
        raise RuntimeError('AIBOOK projection graph: ' + why)


def install(model):
    """No-op by default; not a general torch.nn.Linear graph wrapper."""
    flag = os.environ.get('AIBOOK_PROJECTION_GRAPH', '0')
    if flag == '0':
        return
    require(flag == '1' and os.environ.get('AIBOOK_SLOTS') == '128',
            'requires explicit 1 and exactly 128 slots')
    require(all(os.environ.get(k) != '1' for k in (
        'AIBOOK_ROUTED_GRAPH', 'AIBOOK_MODULE_TIMING', 'AIBOOK_SLOT_TIMING')),
        'test only one change at a time')
    import torch
    from aibook_slots import BoundedExperts
    from sglang.srt.model_executor.forward_mode_context import get_current_forward_mode

    layers = model.model.layers
    require(len(layers) == 40 and all(isinstance(l.mlp.experts, BoundedExperts)
            and l.mlp.experts.w13.shape[0] == 128 for l in layers), 'wrong slot model')
    gdn = layers[0].linear_attn
    require(gdn.use_qkvzba, 'requires fused GPTQ QKVZBA layout')
    proj = gdn.in_proj_qkvzba
    require(type(proj).__name__ == 'MergedColumnParallelLinear' and
            type(proj.quant_method).__name__ == 'GPTQLinearMethod' and
            proj.bias is None and not proj.gather_output and proj.tp_size == 1,
            'unexpected projection or parallel layout')
    require('forward' not in vars(proj), 'projection forward already wrapped')
    eager = proj.forward
    graph = static = output = owner_stream = owner_thread = None
    verified = 0

    def compare(got, expected):
        require(expected.shape == got.shape == _OUTPUT and expected.dtype == got.dtype == torch.float16,
                'unexpected output layout')
        require(torch.isfinite(expected).all().item() and torch.equal(got, expected),
                'non-finite or non-identical projection replay; stop this trial')

    def decode(x):
        nonlocal graph, static, output, owner_stream, owner_thread, verified
        require(not torch.is_grad_enabled() and isinstance(x, torch.Tensor) and
                x.device.type == 'musa' and x.dtype == torch.float16 and
                x.shape == _INPUT and x.is_contiguous(), 'unsupported decode input')
        require(not torch.musa.is_current_stream_capturing(), 'nested capture unsupported')
        stream = torch.musa.current_stream(x.device)
        if graph is None:
            require(proj.qweight.device == x.device and proj.qweight.shape == (256, _OUTPUT[1]) and
                    hasattr(proj, 'qweight_vllm') and hasattr(proj, 'scales_vllm') and
                    hasattr(proj, 'w4a16_coalesced'), 'weights not processed or incompatible')
            owner_thread, owner_stream = threading.get_ident(), stream
            static = torch.zeros_like(x)
            side = torch.musa.Stream(device=x.device)
            side.wait_stream(stream)
            # Warm the exact decode kernel on its capture stream outside capture.
            with torch.musa.stream(side):
                for _ in range(3):
                    eager(static)
            stream.wait_stream(side)
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph, stream=side):
                output, bias = eager(static)
            require(bias is None, 'unexpected bias return')
            graph.replay()  # capture need not have executed the kernels
            compare(output, torch.zeros_like(output))
            log.warning('AIBOOK projection graph: captured layer 0 QKVZBA only')
        require(threading.get_ident() == owner_thread and stream == owner_stream and x.device == static.device,
                'decode stream/thread/device changed; no cross-stream static-buffer reuse')
        static.copy_(x)
        graph.replay()
        if verified < 3:
            reference, _ = eager(x)
            compare(output, reference)
            verified += 1
            if verified == 3:
                log.warning('AIBOOK projection graph: 3 real decode replays matched eager exactly')
        # Never lend the graph output to a caller; no stale aliases on next token.
        return output.clone(), None

    failed = False

    def forward(x):
        nonlocal failed
        require(not failed, 'previous graph failure; discard this experimental worker')
        mode = get_current_forward_mode()  # same ContextVar read by GPTQLinearKernel.apply
        if mode is None or not mode.is_decode():
            return eager(x)
        try:
            return decode(x)
        except BaseException:
            failed = True  # never retry a partial capture or replay a mismatched graph
            raise

    proj.forward = forward
    log.warning('AIBOOK projection graph: armed one lazy layer-0 capture')
