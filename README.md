# aibook-llm-prune

AIBOOK（M1000，32 GB 统一内存）上的 Qwen3.6-35B-A3B 专家剪枝 / 全量驻留 / N-gram 投机解码项目。

- **工作基线**：MUSA 官方 SGLang-M1000 文档 [version-20261030](https://docs.mthreads.com/sglang-musa-m1000/version-20261030/sglang-musa-m1000-doc-online)。本机实际安装的 wheel 是 `sglang 0.5.18+20261101.torch2.11`；本项目的 overlay、kernel 调用和测量都以这条 MUSA 分支为基线，没有直接升级到上游 v0.5.23。

## 发布配置（2026-11-25）

- **Coder 场景**：256 → **160 专家**（按 coder 模型的路由频次校准），160 全量驻留，64k 上下文，INT8 KV，无投机解码；基线 **20.4 tok/s**。
- **运维场景**：同一套权重用 router mask 到 **96 专家**，用于系统安装/运维、故障定位、改配置、写脚本等任务。
- **运维 N-gram 知识库**：基于 tldr、cheat/cheatsheets、systemd unit、Ubuntu man EXAMPLES、部分 `/etc` 配置构建，**1,554,853 tokens / 8,356 文档**，JSONL external corpus，配合 prompt lookup 做 N-gram 投机解码。
- **性能**：96/32K + N-gram，draft=8、native `sgl_kernel.verify_tree_greedy`：
  - 低匹配/miss：**约 15 tok/s**（accept len≈1）
  - 高匹配/match：稳态 **32.8 tok/s**，683 token 复制任务平均 **36.8 tok/s**
  - 相对低匹配约 **100% 提升**；相对 20.4 tok/s 基线约 **60% 提升**
- **自适应门控（实验，默认关闭）**：连续 4 步 `accept_len <= 1` 时切回 target decode；当前 fallback 走 eager target decode，实测约 **9.6 tok/s**，低于基线，因此**没有按“无性能损失”发布**。需要时显式设置 `AIBOOK_NGRAM_GATE=1` 验证，不建议生产开启。
- **内存**：96/32K 可用约 **8.8–9.2 GB**；128/64K 约 5.8 GB、160/64K 约 4.5 GB，均低于 >6 GB 约束，所以 N-gram 只在 96/32K 实验配置启用。
- **MTP / DFLASH 未开启**：受 32 GB 内存限制，为保证系统剩余约 **8 GB** 基本可用性，MTP draft（约 3.5 GB）和 DFLASH draft 都没有叠加；DFLASH 实测对 160 剪枝 target 接受率为 0。
- **生产服务**：仍是 `qwenai.service` 的 **160 / 64k / INT8 / 无投机**，20.4 tok/s；N-gram 是独立实验路径，不替换生产。

## 三种运行配置（互斥，均占用 30141）

三套配置只能同时运行一套；切换前先停掉当前服务或实验 unit。

### 1. 全 256 专家 + 128 动态热专家（质量优先，约 13 tok/s）

- 路由：完整 256 expert，不剪枝；
- 权重：`routed-packs-v1` 的 256 条原始 expert 记录，128-slot LRU 热缓存，miss 直接 O_DIRECT/pread 换入；
- 16K 上下文，`graphoverlap`（breakable decode graph + expert graph + overlap + pinned fill）；
- 实测 **13.2 tok/s**，命中率约 89%，可用内存约 9.5 GB；
- 适合：不接受剪枝质量损失、能接受 13 tok/s 的运维/通用场景。

```bash
# 在 aibook-llm-prune 仓库根目录执行；先停掉 30141 上的服务
ssh -i ~/.ssh/id_ed25519_aibook lf@192.168.100.193 'sudo systemctl stop qwenai.service'
AIBOOK_CONTEXT=16384 AIBOOK_MAX_TOKENS=16384 \
AIBOOK_MAMBA=12 AIBOOK_FRACTION=0.74 \
bash experiment_launch.sh graphoverlap 128 2400
```

> 这是 2400 秒的实验 unit，不是开机自启服务；到期后需重新拉起。

### 2. 160 专家全驻留（Coder 默认，约 20 tok/s）

- 256 → 160 专家，`prune_mask` 只保留 coder 路由校准后的 160 个；
- 64K，INT8 KV，无投机；
- 实测 **20.4 tok/s**，HumanEval 30 题 **80.0%**，可用内存约 7–8 GB；
- 适合：编码、脚本、日常对话；这是当前生产配置。

```bash
ssh -i ~/.ssh/id_ed25519_aibook lf@192.168.100.193 'sudo systemctl start qwenai.service'
# 等约 110 秒，直到 /v1/models 返回 200
```

### 3. 96 专家全驻留 + N-gram（运维高匹配，15–32 tok/s）

- 96 专家全驻留，32K，FP16 KV（N-gram verify 必需）；
- 运维 N-gram 知识库：`/home/lf/aibook-bench/ngram-corpus.jsonl`（1.55M tokens，JSONL）；
- native `sgl_kernel.verify_tree_greedy`，draft=8，external SAM budget=6；
- 实测：高匹配稳态 **32.8 tok/s**，低匹配 **约 15 tok/s**；
- 可用内存约 **8.8–9.2 GB**；
- 适合：systemd/配置/命令模板等重复性高的运维请求；自由排障问答接近 miss，没有收益。
- **不要开自适应门控**：`AIBOOK_NGRAM_GATE` 默认关闭；当前 fallback 实测 9.6 tok/s，低于基线。

```bash
ssh -i ~/.ssh/id_ed25519_aibook lf@192.168.100.193 'sudo systemctl stop qwenai.service'
AIBOOK_NGRAM_CORPUS=/home/lf/aibook-bench/ngram-corpus.jsonl \
AIBOOK_NGRAM_MAX_TOKENS=7000000 \
AIBOOK_NGRAM_SAM_BUDGET=6 \
AIBOOK_NGRAM_DRAFT=8 \
AIBOOK_CONTEXT=32768 AIBOOK_MAX_TOKENS=32768 \
AIBOOK_MAMBA=12 AIBOOK_FRACTION=0.78 \
AIBOOK_DECODE_LOG_INTERVAL=1 \
bash experiment_launch.sh residentngram 96 2400
```

> 同样是 2400 秒实验 unit；日常 coder 服务请用配置 2。

## 目标

1. 把 routed experts 从 **256 剪到 160**（按真实路由频次，router logits 对剪掉的专家加 `-1e9` mask）。
2. 160 个专家 **全量驻留**，不再从 SSD 换入。
3. 保留 **128k 上下文**。
4. 开启 **MTP**（EAGLE / MTP head）。
5. 用公开编码基准对比剪枝前后的分数，量化损失。
6. 全过程系统剩余内存 **> 6 GB**。

## 当前状态

- 160 专家全量驻留已跑通；完整 decode graph 捕获成功（`backend=full, bs=[1]`）。
- 16k 上下文热态 decode：**21.72 / 22.60 tok/s**，P95 46.3 ms。
- 16k 下首个请求峰值可用内存 7.77 GB，cgroup 峰值 8.72 GB，swap 0。
- 128k 上下文正在实测；MTP 尚未启用（MTP head 1.689 GB，160 + 128k 下预计会压到 6 GB 以下）。
- 剪枝依据：281 个 decode token 的路由；每层保留 top-160，保留路由质量 87.5%–100%（最差 layer 0 丢 12.5%）。

## 快速开始

```bash
# 160 专家，全量驻留，16k
bash experiment_launch.sh resident 160 2400

# 160 专家，128k 上下文（需要更大的 guard cap）
AIBOOK_CONTEXT=131072 AIBOOK_MAX_TOKENS=131072 \
AIBOOK_GUARD_CAP_GB=25.5 AIBOOK_GUARD_MARGIN_GB=0.2 \
bash experiment_launch.sh resident 160 2400
```

启动后：

```bash
python3 trace/paired_chat.py <cgroup-path>
```

## 剪枝表

`integration/prune-160.json`：40 层 × 160 个保留专家 ID。生成脚本：

```bash
python3 prune.py --keep 160 --out integration/prune-160.json
```

## 目录

- `integration/`：`BoundedExperts` 的 resident/prune 实现、overlay、loader、smoke。
- `experiment_launch.sh`：只启动 `127.0.0.1:30141` 的实验 unit。
- `guard_launch.py`：内存保护（cap/margin 可配，但硬上限仍是 24 GB 量级）。
- `measurements/`：每次实测证据。
- `trace/`：paired-chat、路由抓取、离线分析。

## 安全边界

- 不切正式 30140；不改系统 MUSA；不重启 KDE/GDM；不清全机 page cache。
- 实验结束停 unit、确认 30141 关闭；不把实验配置带到生产。

## 128k 实测与约束冲突（2026-11-25）

- **160 专家 + 128k**：KV 池只能分到 **48,736 tokens**（128k 需要 131,072），所以 160 下 128k 不可用。
- **128 专家 + 128k**：KV 成功分到 **131,072 tokens**；idle 可用 **6.44 GB**，首个请求峰值 **4.49 GB**（cgroup 7.77 GB）。
- **MTP head 1.689 GB**：128 + 128k 再加 MTP 预计只剩 **约 2.8 GB**，远低于 6 GB。
- 因此“160 专家 + 128k + MTP + 剩余 >6GB”在 32 GB 上不可同时满足。可行的最近组合是 **约 96 专家 + INT8 KV + MTP**（预计约 6.0 GB），或放宽 128k/6GB/MTP 中的一项。

## 160 + 128k + INT8 KV（2026-11-25）

- 160 专家全量驻留，`--kv-cache-dtype int8`，`mem-fraction-static 0.68`，Mamba cache 12。
- KV 成功分配 **131,072 tokens**（K/V 各 0.63 GB），首次实现 **160 + 真 128k**。
- 请求期间系统可用 **8.19 GB**，峰值下降 23.38 GB，swap 0。
- 128 token decode **20.43 tok/s**，输出正确。
- 这是目前最好的可用配置；MTP 仍未叠加（draft 需 3.5 GB，会把可用内存压到约 4.7 GB，且 MTP 输出本身在约 30 token 后损坏）。

## 128k + MTP-160 可行性（2026-11-25）

- 160 + MTP-160 + 128k + FP16 KV + fraction 0.85（按公式预算）：
  - 图捕获阶段可用内存跌到 **0.55 GB**，guard 停止，服务失败；
  - 峰值内存下降 **29.66 GB**，swap 0。
- 结论：**128k + MTP 在 32 GB 上不可行，更不可能满足 >6 GB**。INT8 KV 虽能救 128k，但明确不支持 speculative decode。

## 最终接受配置：160 + 64k + INT8（2026-11-25）

- 160 专家全量驻留，64k 上下文，INT8 KV（K/V 各 0.32 GB），无投机解码。
- 请求期间系统可用内存约 **7.1–7.5 GB**（>6 GB），峰值内存下降 23.08 GB，swap 0。
- 热态 decode **20.54 / 20.76 tok/s**（首个请求 16.85 tok/s）。
- HumanEval 30 题 **80.0%**（对比 256 的 86.7%）。
- 启动：`./serve-160-64k.sh`（默认 2400 秒 guard 窗口）。

## 启动时间与 systemd（2026-11-25）

- `qwenai.service`：160 + 64k + INT8，系统级 user=lf，已 enable，端口 30141；旧 `vllmd.service` 已删除（备份在 `systemd/vllmd.service.removed`）。
- 启动瓶颈实测：权重加载 **101.8 s → 20.1 s**，靠 `AIBOOK_SKIP_DIGEST=1` 跳过每条 expert 记录的 SHA256；`verified.json` 和每层 manifest 仍校验。
- `AIBOOK_FILL_ONE=1` 实测无改善（103.1 s），保留但非关键。
- KV 分配本身只有 **0.36–0.60 s**。load 结束后那段约 13 s 不是 KV，也**不是 Mamba 大小**：Mamba 12→4 把状态从 0.78 GB 降到 0.30 GB，启动时间仍不变（87.5 s vs 87.8 s）。它是 load 后固定的 GPU 拷贝/同步 drain（最可能是 10 GB resident bank 的 H2D 队列）。
- decode graph 捕获约 12 s；这两项目前都无法跨进程持久化。
- TileLang 编译缓存 `~/.cache/tilelang/sglang_m1000`（约 123 MB）持久化，重启复用。
- 从 load begin 到 ready：**130 s → 47 s**。

## JIT / warmup / prefix cache

- **JIT 持久缓存已启用**：`TILELANG_CACHE_DIR=/home/lf/.cache/tilelang/sglang_m1000`（约 127 MB，3,700+ 文件），`SGLANG_CACHE_DIR=/home/lf/.cache/sglang`。编译结果跨重启复用；graph 捕获和运行时状态不能持久化。
- **启动 warmup**：`AIBOOK_WARMUP=short`（默认，SGLang 内置短请求）或 `AIBOOK_WARMUP=prefill`（自定义 `aibook_prefill`）。默认只预热 **2K / 8K**：
  ```bash
  AIBOOK_WARMUP_SIZES=2048,8192
  ```
  需要 16K/64K 时显式加：
  ```bash
  AIBOOK_WARMUP=prefill AIBOOK_WARMUP_SIZES=16384,65520
  ```
  首次编译 16K/64K 很慢（16K 在本机已超过 5 分钟），但结果写入 TileLang 持久缓存，后续启动复用。
- **HiCache 文件后端（实验，未进生产）**：SGLang 支持 `--enable-hierarchical-cache --hicache-storage-backend file`，路径由 `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` 指定。本机实测：
  - 设备内 radix cache 命中正常（469 token 的重复请求 `cached_tokens=448`）；
  - **file 后端没有写出 SSD 文件**（`~/.cache/sglang-hicache` 为空），在 hybrid Mamba/SSM 路径上尚未验证成功；
  - 因此 `AIBOOK_HICACHE` 默认关闭，不用于生产。
- **14 天清理**：`systemd/hicache-cleanup.cron`，每天 04:17 删除 `~/.cache/sglang-hicache` 下超过 14 天的文件；已安装到 `lf` 的 crontab。
