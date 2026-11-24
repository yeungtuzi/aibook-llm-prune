# 测试报告（AIBOOK 槽位专家缓存）

日期：2026-11-25
本地提交：`054c31a` / `7cbd2a2`（报告生成时的 HEAD）
环境：AIBOOK M1000，30.5 GiB 可用内存，KDE + Chrome 常驻；MUSA 5.3.0-M1000；SGLang-MUSA 0.5.18+20261101.torch2.11；conda `sglang-py310`；目标模型 `gptq-Qwen3.6-35B-A3B-4bit-channel-lmhead`；pack `~/sglang-m1000/routed-packs-v1`。

## 1. 单元测试

```bash
cd integration && python3 -m unittest test_loader
# Ran 1 test in 0.298s — OK

cd trace && python3 -m unittest test_analyze test_capture
# Ran 10 tests in 0.033s — OK
```

## 2. 集成测试（真实服务 + 两轮 paired-chat）

所有实验均在 `127.0.0.1:30141`，由 `guard_launch.py` 以 22 GB 可用内存降幅为停止线，cgroup 限制 18 GB、swap 0。

| 配置 | 热态 decode | P95 | 峰值可用内存降幅 | 结果 |
|---|---|---|---|---|
| eager 128 槽（参考） | 2.35 / 3.60 tok/s | — | — | 基线，仅用于逐 token 对比 |
| breakable graph + 128 槽 | 11.42 / 10.91 tok/s | 100.1 / 107.8 ms | 20.64 GB | 通过，输出一致 |
| + 逐层专家 MUSAGraph + 128 槽 | 13.08 / 11.16 tok/s | 90.4 / 121.7 ms | 20.80 GB | 通过，输出一致 |
| + overlap scheduler + 128 槽 | 13.29 / 12.91 tok/s | 89.5 / 90.5 ms | 20.81 GB | 通过，输出一致 |
| + 热路径优化 + 128 槽 | 13.21 / 13.16 tok/s | 91.7 / 91.1 ms | 20.80 GB | 通过，输出一致 |
| **140 槽（稳定档）** | **14.05 / 14.16 tok/s** | 85.7 / 80.1 ms | 21.51 GB | 通过，余量 0.64 GB |
| 144 槽（上限） | 14.95 / 14.35 tok/s | 80.3 / 77.1 ms | 21.82 GB | 通过，余量约 0.05 GB |
| 144 槽 + 专家图直接返回 | 14.79 / 14.63 tok/s | 79.8 / 75.3 ms | 21.81 GB | 通过，输出一致 |

一致性：以上所有跑通的配置，两轮输出均与 eager 128 槽参考**逐 token 相同**。

内存：所有跑通配置的 `peak_available_drop` ≤ 21.82 GB < 22 GB 保护线 < 24 GB 硬上限；cgroup 峰值 8.8–9.9 GB；swap 0；未触发保护停止（除下述被拒实验）。

## 3. 被拒/未通过实验

| 实验 | 结论 |
|---|---|
| mapped host 槽位 | 分配成功但更慢（turn2 2.99 vs 3.60 tok/s，P95 496 vs 352 ms），拒绝 |
| coalesced weight layout | 仅约 5% 提升，不值得重打 16 GB pack |
| 160 槽 / 144 槽 + 单次 H2D staging | 触发 22 GB 保护停止（21.97 GB），不作结论 |
| 256 行 bank 几何 | 与 8 行 bank 相同（0.795 vs 0.798 ms/layer），无收益 |
| MTP / DFLASH | 5-token verify 窗口每层只触达 24.7 个不同专家，但容量受限 LRU 的 miss/token 在 144 槽下 W=1 与 W=5 都是 31.3；MTP 头 1.69 GB 还会越保护线，拒绝 |
| 替换策略 LFU / LRU-pin | 仅比 LRU 好约 2–2.5%，不足以改变结论 |

## 4. 根因与上限

- 40 层专家 pair 的 GPU 时间约 32 ms/token（0.8 ms/层），已等于全驻留基线整 token 时间；8 槽与 256 槽 bank 几何无差异。
- 分页额外暴露约 34 ms/token 的 fill/宿主；缓存是容量受限，miss 约 13–17/token。
- 24 GB 上限把缓存锁在 140–144 槽；144 槽已只剩约 50 MB 保护余量。
- MUSA 设备内存不可 CPU 直写（`ctypes.string_at(device_ptr)` 直接 segfault），无法绕过 4 次 H2D。
- SSD 不是热态瓶颈：O_DIRECT 1.58 MB 记录 QD1 中位 0.92 ms（1.71 GB/s），QD8 聚合 2.72 GB/s；热态每 48 token 只读 16–200 MB，多数为 page-cache 命中。M1000 为 UMA，设备拷贝不走 PCIe。

## 5. 已知限制

- 未达到 25–30 tok/s；当前稳定档约 14.1 tok/s。
- 144 槽虽快，但贴着保护线，不作为稳定默认；默认用 140 槽。
- 逐层专家图与 `graph_compat.py` 仍属实验路径（`AIBOOK_GRAPH_PROBE=1` 守卫），未产品化。
- 长上下文、长时间会话、Chrome/KDE 长时间共存、DSH 端到端仍未系统验证。
- 未测 MTP/DFLASH 的在线路径（只有离线路由量化与内存账）。

## 6. 复现命令

```bash
bash experiment_launch.sh graphoverlap 140 2400
# 就绪后：
python3 trace/paired_chat.py <cgroup-path>
# 清理：
systemctl --user stop aibook-slot-smoke-trial-*-140-graphoverlap.service
```

## 7. 全量驻留 160 + 关闭视觉/多模态预留（2026-11-25）

- 视觉塔：checkpoint 索引里 **0 个** `visual.*`/`vision.*` tensor，服务本来就带 `--language-model-only`；真正占内存的是运行时多模态预留（`SGLANG_VLM_CACHE_SIZE_MB=100` + `SGLANG_MM_FEATURE_CACHE_MB=1024`，约 1.12 GB），本次已设为 0。
- 剪枝：用 281 个 decode token 的路由，每层按频次保留 top-160（剪掉 96 个，router logits 加 `-1e9` mask）。每层实际用到的专家数 119–244；保留路由质量 87.5%–100%，最差是 layer 0（丢 12.5%）。PolyStrata Coder 是保留 128/256（按代码/agentic 数据选），本方案 160/256 更保守。
- 全量驻留：160 × 40 个专家常驻（10.1 GB），无换入；**完整 decode graph 捕获成功**（`backend=full, bs=[1]`），prefill graph 关闭。
- 内存实测：服务 READY 后 idle 可用 **9.07 GB**；首个请求峰值可用 **8.16 GB**（22 GB 保护线触发停止），cgroup 峰值 8.34 GB，swap 0。
- MTP：MTP 头 1.689 GB；全量驻留单独已到 8.16 GB，叠加 MTP 后预计 **6.47–7.38 GB，低于 8 GB**，因此本轮没有启用 MTP。
- 结论：**按 160 专家全量驻留 + MTP，剩余内存不足 8 GB**。若要在 MTP 下守住 8 GB，需要把保留专家降到约 128（PolyStrata Coder 档），或降低 KV/Mamba/上下文预算。
