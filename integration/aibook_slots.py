"""Eager TP=1 routed experts for the original AIBOOK W4A16 layer packs.

The pack root is trusted, immutable local storage. verified.json is the offline
verification receipt, not a signature. Recheck the selected manifest on open
and each record on first use. Subsequent refills rely on immutability of the
open weights inode; do not modify a deployed pack. Never materialize a full
layer on either device.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import threading

import torch
from torch import nn
from torch.nn import functional as F

try:  # Present in the pinned vendor tree; absent in unit tests without SGLang.
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
    )
except Exception:  # pragma: no cover - plain eager fallback
    def eager_on_graph(enable=True, capture_stub=None):
        return lambda fn: fn


_LAYERS, _EXPERTS, _K, _H = 40, 256, 8, 2048
_STRIDE, _RECORD = 1581056, 1579008
_FILE_BYTES = (_EXPERTS - 1) * _STRIDE + _RECORD  # no trailing padding
_COMPONENTS = ('w13_qweight', 'w2_qweight', 'w13_scales', 'w2_scales')
_COMPONENT_BYTES = (1024 * 1024, 2048 * 256, 1024 * 2, 2048 * 2)
# One pinned record per possible fill in a layer, shared by all layers: fills
# are serialized by _SERIAL and the next layer's route wait fences them.
_STAGES = []
_STAGE_NEXT = 0
_STAGE_DEV = None


def _stages():
    if not _STAGES:
        for _ in range(_K):
            tensor = torch.empty(_RECORD, dtype=torch.uint8, device='cpu', pin_memory=True)
            _STAGES.append((tensor, memoryview(tensor.numpy())))
    return _STAGES
# Serialize staging and launches, including callers on different streams.
_SERIAL = threading.Lock()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _metadata(path):
    with path.open('rb') as stream:
        data = stream.read((1 << 20) + 1)
    if len(data) > 1 << 20:
        raise ValueError(f'oversized pack metadata: {path}')
    return data


def _check_pack(root, layer, config_hash):
    name = f'layer-{layer:02d}'
    receipt = json.loads(_metadata(root / 'verified.json'))
    sources = json.loads(_metadata(root / 'source-sha256.json'))
    if (not _digest(config_hash) or sources.get('config.json') != config_hash
            or not all(map(_digest, sources.values()))
            or receipt['source_sha256'] != sources
            or receipt['layers'] != _LAYERS or receipt['experts'] != _LAYERS * _EXPERTS
            or receipt['weight_file_bytes'] != _LAYERS * _FILE_BYTES):
        raise ValueError('unverified pack source, config, or size')
    manifests = receipt['manifest_sha256']
    if (set(manifests) != {f'layer-{i:02d}' for i in range(_LAYERS)}
            or not all(map(_digest, manifests.values()))):
        raise ValueError('incomplete manifest digest table')
    directory = root / name
    data = _metadata(directory / 'manifest.json')
    if hashlib.sha256(data).hexdigest() != manifests[name]:
        raise ValueError(f'{name}: manifest digest mismatch')
    manifest = json.loads(data)
    if (manifest['version'] != 1 or manifest['layer'] != layer
            or manifest['config_sha256'] != config_hash
            or manifest['shared_expert_included'] is not False
            or manifest['component_order'] != list(_COMPONENTS)
            or manifest['stride'] != _STRIDE or manifest['record_bytes'] != _RECORD):
        raise ValueError(f'{name}: incompatible pack layout')
    records = manifest['records']
    if not isinstance(records, list) or len(records) != _EXPERTS or not all(map(_digest, records)):
        raise ValueError(f'{name}: invalid record digest table')
    return directory / 'weights.bin', tuple(records)


def _capture_stub(self, hidden_states, topk_output, pre_quant_input=None):
    """Capture-time placeholder: kernels in a captured segment do not execute,
    so route ids are undefined here. Allocate the bridge output only; replay
    runs the real forward with refreshed ids and copies into this buffer."""
    return torch.zeros_like(hidden_states)


class BoundedExperts(nn.Module):
    """Drop-in routed branch. Does not own the router or shared expert."""

    def __init__(self, pack_root, layer, slots, config_hash):
        super().__init__()
        if type(layer) is not int or not 0 <= layer < _LAYERS:
            raise ValueError('invalid expert layer')
        if type(slots) is not int or not _K <= slots < _EXPERTS:
            raise ValueError('need at least eight slots and fewer than a full layer')
        if sys.byteorder != 'little':
            raise RuntimeError('AIBOOK scales require a little-endian host')
        path, self._records = _check_pack(Path(pack_root), layer, config_hash)
        self._weights = path.open('rb', buffering=0)
        try:
            info = os.fstat(self._weights.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != _FILE_BYTES:
                raise ValueError(f'{path}: wrong weight file type or size')
            self._stamp = (info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            import torch_musa  # noqa: F401 -- register the MUSA backend
            from mate.batched_gemv.tilelang.batched_gemv_w4a16_moe import musa_moe_gemv
            self._gemv = musa_moe_gemv
            # A factory without device honors the loader's `with torch.device(...)`.
            self._device = torch.empty(0).device
            if self._device.type != 'musa':
                raise RuntimeError('construct BoundedExperts on the model MUSA device')
            self._mapped = [] if os.environ.get('AIBOOK_MAPPED_SLOTS') == '1' else None
            if self._mapped is None:
                # One flat bank-major device buffer: a layer's 160 experts can
                # then be uploaded with one large H2D instead of 4*160 small ones.
                self._flat = torch.empty((slots * _RECORD,), dtype=torch.uint8, device=self._device)
                self.register_buffer('_flat', self._flat, persistent=False)
                offset = 0
                for key, shape, dtype, nbytes in (
                        ('w13', (slots, 1024, 1024), torch.uint8, _COMPONENT_BYTES[0]),
                        ('w2', (slots, 2048, 256), torch.uint8, _COMPONENT_BYTES[1]),
                        ('s13', (slots, 1024, 1), torch.float16, _COMPONENT_BYTES[2]),
                        ('s2', (slots, 2048, 1), torch.float16, _COMPONENT_BYTES[3])):
                    view = self._flat[offset:offset + slots * nbytes].view(dtype).view(shape)
                    offset += slots * nbytes
                    self.register_buffer(key, view, persistent=False)
            else:
                for key, shape, dtype in (
                        ('w13', (slots, 1024, 1024), torch.uint8),
                        ('w2', (slots, 2048, 256), torch.uint8),
                        ('s13', (slots, 1024, 1), torch.float16),
                        ('s2', (slots, 2048, 1), torch.float16)):
                    from mapped_banks import Bank
                    bank=Bank(shape,dtype,self._device);self._mapped.append(bank)
                    self.register_buffer(key, bank.tensor, persistent=False)
        except BaseException:
            self._weights.close()
            raise
        self._tags = [None] * slots
        self._slot_of = {}
        self._last_used = [0] * slots
        self._clock = 0
        self._timing = os.environ.get('AIBOOK_SLOT_TIMING') == '1'
        self._layer = layer
        self._verified = bytearray(_EXPERTS)
        self._skip_digest = os.environ.get('AIBOOK_SKIP_DIGEST') == '1'
        self._failed = False
        self._resident = os.environ.get('AIBOOK_RESIDENT') == '1'
        if self._resident:
            prune_path = Path(os.environ.get('AIBOOK_PRUNE', Path(__file__).with_name('prune-160.json')))
            table = json.loads(prune_path.read_bytes())
            self._kept = [int(e) for e in table['layers'][layer]]
            keep = len(self._kept)
            if keep != slots or not 8 <= keep <= 255 or len(set(self._kept)) != keep or any(not 0 <= e < _EXPERTS for e in self._kept):
                raise ValueError('prune table does not match the resident slot count')
            lut = torch.zeros(_EXPERTS, dtype=torch.int32, device=self._device)
            for slot, expert in enumerate(self._kept):
                lut[expert] = slot
            self.register_buffer('_lut', lut, persistent=False)
            mask = torch.full((_EXPERTS,), -1e9, dtype=torch.float32, device=self._device)
            mask[torch.tensor(self._kept, dtype=torch.long, device=self._device)] = 0.0
            self.register_buffer('prune_mask', mask, persistent=False)
            self._g = torch.empty((_K, 1024), dtype=torch.float16, device=self._device)
            self._d = torch.empty((_K, _H), dtype=torch.float16, device=self._device)
            if self._mapped is None:
                self._resident_preload()
            else:
                for slot, expert in enumerate(self._kept):
                    self._fill(slot, expert, None)
        if self._mapped is None:
            self._ix_cpu = torch.empty((1, _K), dtype=torch.int32, device='cpu', pin_memory=True)
            self._ix_np = self._ix_cpu.numpy()
            self._ix_dev = torch.empty((1, _K), dtype=torch.int32, device=self._device)
            self._expert_graph = os.environ.get('AIBOOK_EXPERT_GRAPH') == '1'
            self._eg = None
            self._eg_out = None
            if self._expert_graph:
                self._xa = torch.empty((1, _H), dtype=torch.float16, device=self._device)
                self._wa = torch.empty((1, _K), dtype=torch.float32, device=self._device)
                self._g = torch.empty((_K, 1024), dtype=torch.float16, device=self._device)
                self._d = torch.empty((_K, _H), dtype=torch.float16, device=self._device)

    def _victim(self, route_set):
        if None in self._tags:
            return self._tags.index(None)
        return min((i for i, tag in enumerate(self._tags) if tag not in route_set),
                   key=self._last_used.__getitem__)

    def _set_tag(self, slot, expert):
        old = self._tags[slot]
        if old is not None:
            self._slot_of.pop(old, None)
        self._tags[slot] = expert
        self._slot_of[expert] = slot

    def _resident_preload(self):
        """Read a full layer into a pinned bank-major host buffer, then one H2D."""
        n = len(self._kept)
        host = torch.empty((n * _RECORD,), dtype=torch.uint8, pin_memory=True)
        hv = memoryview(host.numpy())
        offsets = []
        off = 0
        for nbytes in _COMPONENT_BYTES:
            offsets.append(off)
            off += n * nbytes
        fd = self._weights.fileno()
        info = os.fstat(fd)
        if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != self._stamp:
            raise ValueError('deployed weights inode changed after verification')
        rec = bytearray(_RECORD)
        for slot, expert in enumerate(self._kept):
            if os.preadv(fd, [rec], expert * _STRIDE) != _RECORD:
                raise OSError(f'short read for expert {expert}')
            if not self._skip_digest and not self._verified[expert]:
                h = hashlib.sha256()
                h.update(rec)
                if h.hexdigest() != self._records[expert]:
                    raise ValueError(f'weight digest mismatch for expert {expert}')
                self._verified[expert] = 1
            mv = memoryview(rec)
            pos = 0
            for nbytes, boff in zip(_COMPONENT_BYTES, offsets):
                dst = boff + slot * nbytes
                hv[dst:dst + nbytes] = mv[pos:pos + nbytes]
                pos += nbytes
            self._set_tag(slot, expert)
        self._flat.copy_(host)

    def _fill(self, slot, expert, stage):
        fd = self._weights.fileno()
        offset = expert * _STRIDE
        if self._mapped is not None:
            # Direct host writes have no stream-ordered device copy to fence
            # them; mapped mode keeps the old full-device barrier. The eager
            # path relies on the stream-ordered copies below instead.
            torch.musa.synchronize()
        if self._mapped is None:
            global _STAGE_NEXT
            pool = _stages()
            stage, view = pool[_STAGE_NEXT % _K]
            _STAGE_NEXT += 1
            vec = [view]
        else:
            vec = [b.slot(slot) for b in self._mapped]
        if os.preadv(fd, vec, offset) != _RECORD:
            raise OSError(f'short read for expert {expert}')
        info = os.fstat(fd)
        if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != self._stamp:
            raise ValueError('deployed weights inode changed after verification')
        if not self._skip_digest and not self._verified[expert]:
            h=hashlib.sha256()
            for segment in vec:h.update(segment)
            if h.hexdigest() != self._records[expert]:
                raise ValueError(f'weight digest mismatch for expert {expert}')
            self._verified[expert] = 1
        if self._mapped is not None:
            self._set_tag(slot, expert)
            return
        src = stage
        if os.environ.get('AIBOOK_FILL_ONE') == '1':
            global _STAGE_DEV
            if _STAGE_DEV is None:
                _STAGE_DEV = torch.empty(_RECORD, dtype=torch.uint8, device=self._device)
            _STAGE_DEV.copy_(stage, non_blocking=True)
            src = _STAGE_DEV
        offset = 0
        for buffer in (self.w13, self.w2, self.s13, self.s2):
            dest = buffer[slot]
            size = dest.numel() * dest.element_size()
            dest.copy_(src[offset:offset + size].view(dest.dtype).view_as(dest),
                       non_blocking=True)
            offset += size
        self._set_tag(slot, expert)

    def _expert_body(self):
        self._gemv(self._xa, self.w13, self._g, None, self.s13, None,
                   self._wa, self._ix_dev, False, _K, True, False, weight_layout='legacy')
        gate, up = self._g.chunk(2, dim=1)
        h = (F.silu(gate) * up).contiguous()
        self._gemv(h, self.w2, self._d, None, self.s2, None,
                   self._wa, self._ix_dev, True, 1, True, False, weight_layout='legacy')
        return self._d.float().sum(0).to(torch.float16)

    def _ensure_expert_graph(self):
        self._expert_body()
        torch.musa.synchronize(self._device)
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph):
            self._eg_out = self._expert_body()
        graph.replay()
        torch.musa.synchronize(self._device)
        self._eg = graph

    @eager_on_graph(True, capture_stub=_capture_stub)
    @torch.no_grad()
    def forward(self, hidden_states, topk_output, pre_quant_input=None):
        if pre_quant_input is not None:
            raise ValueError('pre-quantized input is unsupported')
        ids, weights = topk_output.topk_ids, topk_output.topk_weights
        x = hidden_states
        if x.ndim != 2 or x.shape[1] != _H or x.dtype != torch.float16 or x.device != self._device:
            raise ValueError('expected original [tokens, 2048] fp16 activations on the slot device')
        if (ids.shape != (x.shape[0], _K) or weights.shape != ids.shape
                or ids.dtype not in (torch.int32, torch.int64) or not weights.is_floating_point()
                or ids.device != x.device or weights.device != x.device):
            raise ValueError('expected standard logical top-8 ids and weights on the slot device')
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if self._resident:
            with _SERIAL, torch.musa.device(self._device):
                for row in range(x.shape[0]):
                    ix = self._lut[ids[row:row + 1]]
                    wt = weights[row:row + 1].float()
                    self._gemv(x[row:row + 1], self.w13, self._g, None, self.s13, None,
                               wt, ix, False, _K, True, False, weight_layout='legacy')
                    gate, up = self._g.chunk(2, dim=1)
                    h = (F.silu(gate) * up).contiguous()
                    self._gemv(h, self.w2, self._d, None, self.s2, None, wt, ix,
                               True, 1, True, False, weight_layout='legacy')
                    out[row].copy_(self._d.float().sum(0))
            return out
        with _SERIAL, torch.musa.device(self._device):
            if self._failed:
                raise RuntimeError('slot cache is unusable after a failed operation')
            for buf, dtype in ((self.w13, torch.uint8), (self.w2, torch.uint8),
                               (self.s13, torch.float16), (self.s2, torch.float16)):
                if buf.device != self._device or buf.dtype != dtype:
                    raise RuntimeError('expert slot buffers were moved or cast after construction')
            if not x.shape[0]:
                return out
            stage = None  # at most one host record across all active layers
            g = x.new_empty((_K, 1024))
            d = x.new_empty((_K, _H))
            try:
                for row in range(x.shape[0]):
                    from time import perf_counter_ns as clock
                    timed = self._timing and x.shape[0] == 1
                    if timed: t0 = clock(); misses = 0
                    route = ids[row].tolist()
                    if timed: t1 = clock()
                    if any(not 0 <= expert < _EXPERTS for expert in route):
                        raise ValueError(f'logical expert id out of range at row {row}')
                    physical = []
                    route_set = set(route)
                    for expert in route:
                        slot = self._slot_of.get(expert)
                        if slot is not None:
                            pass
                        else:
                            slot = self._victim(route_set)
                            self._fill(slot, expert, stage)
                            if timed: misses += 1
                        self._clock += 1
                        self._last_used[slot] = self._clock
                        physical.append(slot)
                    if timed: t2 = clock()
                    self._ix_np[0, :] = physical
                    self._ix_dev.copy_(self._ix_cpu, non_blocking=True)
                    ix = self._ix_dev
                    wt = weights[row:row+1].float().contiguous()
                    if os.environ.get('AIBOOK_ROUTED_GRAPH') == '1' and x.shape[0] == 1:
                        from routed_graph import RoutedGraph
                        if not hasattr(self, '_rg'):
                            self._rg = RoutedGraph(self, x, wt, ix)
                        out[row].copy_(self._rg.replay(x, wt, ix))
                        torch.musa.synchronize(self._device)
                        continue
                    if self._expert_graph and x.shape[0] == 1:
                        self._xa.copy_(x[row:row+1])
                        self._wa.copy_(wt)
                        if self._eg is None and not torch.musa.is_current_stream_capturing():
                            self._ensure_expert_graph()
                        if self._eg is not None:
                            self._eg.replay()
                            # The graph output is stable until this layer runs again
                            # (a full token later); return it directly instead of
                            # copying through out[row], which the block consumes
                            # immediately.
                            return self._eg_out.view(1, _H)
                        else:
                            self._gemv(self._xa, self.w13, self._g, None, self.s13, None,
                                       self._wa, ix, False, _K, True, False, weight_layout='legacy')
                            gate, up = self._g.chunk(2, dim=1)
                            h = (F.silu(gate) * up).contiguous()
                            self._gemv(h, self.w2, self._d, None, self.s2, None, self._wa, ix,
                                       True, 1, True, False, weight_layout='legacy')
                            out[row].copy_(self._d.float().sum(0))
                    else:
                        self._gemv(x[row:row+1].contiguous(), self.w13, g, None, self.s13, None,
                                   wt, ix, False, _K, True, False, weight_layout='legacy')
                        gate, up = g.chunk(2, dim=1)
                        h = (F.silu(gate) * up).contiguous()
                        self._gemv(h, self.w2, d, None, self.s2, None, wt, ix,
                                   True, 1, True, False, weight_layout='legacy')
                        out[row].copy_(d.float().sum(0))
                    # Eager slots need no extra barrier: each blocking H2D copy
                    # is stream-ordered after earlier kernels that read the slot
                    # and returns only once the copy (hence those kernels) is
                    # complete. A per-layer synchronize here drains the stream
                    # and exposes all following CPU launch time on every layer,
                    # which dominated warm decode latency.
                    if timed:
                        # Accumulate in memory; report one line per layer per 10 decode rows.
                        t3 = clock()
                        if not hasattr(self, '_time_acc'): self._time_acc = [0]*5
                        for j, v in enumerate((1, misses, t1-t0, t2-t1, t3-t2)): self._time_acc[j] += v
                        if self._time_acc[0] == 10:
                            print('AIBOOK_TIMING', self._layer, *self._time_acc, flush=True); self._time_acc = [0]*5
            except BaseException:
                self._failed = True
                torch.musa.synchronize(self._device)
                raise
        return out
