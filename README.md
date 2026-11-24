# aibook-llm-prune

AIBOOK（M1000，32 GB 统一内存）上的 Qwen3.6-35B-A3B 专家剪枝 / 全量驻留 / 128k 上下文 / MTP 实验项目。

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
