"""Opt-in compatibility shim for the breakable decode-graph probe.

The vendor breakable backend was written for tensor/prefill outputs and rejects
``LogitsProcessorOutput``. Decode with bs=1 needs no bridge buffer: the final
graph segment writes the output tensors in place, exactly like the full graph
backend, so the capture-time object can be returned on every replay.
This is probe-only; it is installed only when AIBOOK_GRAPH_PROBE=1.
"""


def install():
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
        BreakableCudaGraphBackend,
    )
    import sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph as bcg

    # The vendor helper flattens tuples, which destroys a NamedTuple such as
    # StandardTopKOutput; the routed module needs its fields on replay.
    weak0 = bcg._weak_ref_if_tensor

    def weak_ref(value):
        if isinstance(value, tuple) and hasattr(value, '_fields'):
            return type(value)(*(weak0(item) for item in value))
        return weak0(value)

    bcg._weak_ref_if_tensor = weak_ref

    # Decode attention is graph-safe (the full backend captures it); the
    # vendor only routes it through eager breakable ops because the same flag
    # is used for prefill. Probe runs with prefill graphs disabled, so let
    # decode capture attention and keep breaks only at the routed module.
    import sglang.srt.layers.radix_attention as radix_attention
    import sglang.srt.layers.radix_linear_attention as radix_linear_attention

    radix_attention.is_in_breakable_cuda_graph = lambda: False
    radix_linear_attention.is_in_breakable_cuda_graph = lambda: False

    rows0 = BreakableCudaGraphBackend._output_rows
    alloc0 = BreakableCudaGraphBackend._alloc_full_buffer
    slice0 = BreakableCudaGraphBackend._slice_output
    copy0 = BreakableCudaGraphBackend._copy_output_to_buffer

    def opaque(value):
        return isinstance(value, LogitsProcessorOutput)

    def rows(self, output, cap):
        return cap if opaque(output) else rows0(self, output, cap)

    def alloc(self, output, size):
        return output if opaque(output) else alloc0(self, output, size)

    def slice_(self, output, num_tokens):
        return output if opaque(output) else slice0(self, output, num_tokens)

    def copy(self, output, output_buffer, num_tokens):
        if opaque(output) or opaque(output_buffer):
            if opaque(output) and opaque(output_buffer):
                return
            raise ValueError('BCG logits output structure changed during capture')
        return copy0(self, output, output_buffer, num_tokens)

    BreakableCudaGraphBackend._output_rows = rows
    BreakableCudaGraphBackend._alloc_full_buffer = alloc
    BreakableCudaGraphBackend._slice_output = slice_
    BreakableCudaGraphBackend._copy_output_to_buffer = copy
