# AIBOOK SSD 专家缓存交接｜2026-11-24

**接手点：单层映射槽已在 M1000 上验证；40 层映射版只通过静态语法检查，尚未部署、启动或测速。** 最后一个有硬件结果的提交 `0d973b3`；待验证实现 `2529528`，已提交，无需恢复旧会话的未提交记录。不要将上一轮的预计用时误读为已经执行。交接检查时实验和正式端口 30141/30140 均无监听，无运行中的实验 unit，MemAvailable 30.25 GB。旧目标已暂停；新会话接管前重新检查现场。

**2026-11-24 23:10 更新：40 层 mapped 已验证并否决；breakable MUSA decode graph 已打通，128 槽热态约 10.9–11.4 tok/s、逐 token 与 eager 基线一致。** 详见文末“图路径”一节。

## 目标与权限

- AIBOOK（lf@192.168.100.193，M1000，32 GB 统一内存），Qwen3.6-35B-A3B FQ GPTQ Int4，SGLang-MUSA。目标是为 KDE/Chrome 留内存并保持 DSH 交互；**服务启动和请求全过程，包括 KV、Mamba、图、CPU 和设备缓冲，目标不超过十进制 24,000,000,000 字节**。原全驻留约 31–33 tok/s，接近原交互速度是优化方向，不能以能启动作为验收。
- 本机 `/root/lvllm/repos/aibook-expert-cache` 是唯一 Git 真源。先本地提交、再按 commit 发布到 AIBOOK；不直接改已部署目录或厂商环境。SSH 用已有 `~/.ssh/id_ed25519_aibook`，不要复制或回显密钥。原部署报告：[部署与对照](</root/lvllm/notes/qwen36-fq-sglang-bench-aibook.md>)；旧设计笔记中的命中率与 group-size 猜测不作依据。
- 正式服务原别名 `qwen3.6-35b-a3b`，DSH web/headless 已配置回环 30140，因原服务已获准退役，目前不可用。**不要为了让端点在线而重启全驻留版本。** 实验仅 127.0.0.1:30141；不换 DSH、不重启 KDE、不改系统 MUSA、不清整机页缓存。原部署基线 tag `baseline-full` 供对照，不是默认回滚启动目标。隔离实验回滚＝停止明确的实验 unit，保留权重包和记录。

## 已取得的证据

| 实验 | 结果与限制 |
|---|---|
| 专家包与加载 | `~/sglang-m1000/routed-packs-v1` 共 40 层 × 256 个路由专家，16.19 GB SSD，逐条校验。单记录 1,579,008 B，步长 1,581,056 B。每层按真实 top-8 换入；共享专家和门控始终保留。已证实加载器在 get_tensor 前跳过 122,880 个路由条目，非先全量分配再卸载。使用原始 checkpoint 的配置和索引，非路由 backbone 副本不是启动所需。|
| 物理内存口径 | [校准](<uma-calibration.json>)：写满 256 MiB MUSA 时 cgroup 仅多记约 0.25 MiB。因此不能只靠 MemoryMax，更不能把 cgroup、设备内存、整机下降相加。[保护器](<guard_launch.py>)在相对启动前可用内存下降约 22 GB 时提前停，保留 2 GB 裕度；这是压力信号，不是硬件级峰值证明。|
| 可用档位 | 均匀 128 槽/层 = 8,084,520,960 B 热权重，16K KV 配置。[五轮](<measurements/guard-128long.json>)可用内存最大下降 21.11 GB，服务 swap=0；不是 16K 满上下文认证。[160 槽](<measurements/guard-511fc13.json>)在长一点的 prefill 触发 22 GB 保护线，**不推荐扩大**。|
| 热度 | [五轮真实 40 层轨迹](<measurements/live-five-128/routes.jsonl>)按层 LRU 回放，decode 命中率 92.29%–95.81%。注意区分逻辑 miss、page-cache hit、物理读量和不可掩盖的耗时。前两轮拟合非均匀容量，后三轮离线 miss 少 25.8%；[独立实跑](<measurements/budget-pair-fitted.jsonl>)无可见 decode 收益，仍选均匀 128。|
| 速度与正确性 | [128 槽](<measurements/stream128.jsonl>)同输入重复请求约 5.01 tok/s，P95 212 ms；[另一段独立追问对照](<measurements/budget-pair-eager.jsonl>)约 3.60 tok/s，P95 352 ms。不能把热态慢全归咎 SSD；分模块计时本身有同步开销。[局部路由图](<measurements/stream128-rgraph-repeat.jsonl>)及[单层 GDN 投影图](<measurements/projection128.jsonl>)均无实质端到端收益，默认关。多档 40 token 一致；与旧全驻留显式核对过一条提示的前 10 个 token，不等于质量认证。|
| 最新：直接换入 | [M1000 单层探针](<mapped_slot_pilot.py>)：`musaHostAlloc(..., mapped)` + 设备别名，一个 `preadv` 分散写入四个算子可见张量。两组各 8 个真实专家覆盖同槽，核对原始记录与两次投影输出，逐元素相同；[单层结果](<measurements/mapped-slot.json>)读入中位数 0.93/1.09 ms，热态计算未见退化。避免应用层 stage + 4 次 copy，不是 O_DIRECT，也不代表绕开 Linux 页缓存。|

## 下一步：只改变槽位的存储方式

1. 先 `pwd; git status --short; git log -4 --oneline`；看 `git show --stat 2529528`。审阅 [映射所有权](<integration/mapped_banks.py>)、[实际换入及同步](<integration/aibook_slots.py>) 和 [发布器](<experiment_launch.sh>)。当前设计有意**按进程生命周期**保存原始映射和张量别名，禁止进程内重载/切换模型；尚未验证 40 层的锁页限制、分配失败回收、完整退出后内存回落。已知 AIBOOK 用户会话 `ulimit -l` 报 3940072 KiB；单层通过不证明 8 GB 分配能通过。不要调高限制来绕过失败。
2. 只在无实验 unit、无 30141 监听、整机 MemAvailable ≥26 GB 时从项目根目录运行 `bash experiment_launch.sh mapped 128 1500`。先按本机规则报预计时长、失败重跑代价和受影响产物。该命令只提交已在 Git HEAD 的版本到**新**目录，不等 readiness。记下启动的 unit 和 release。若拒绝或失败，用 journal 精确定位；不弱化保护器，不碰系统驱动。注意：不同提交/模式使用新目录；相同目录存在时启动器拒绝覆盖，这是预期行为。
3. 等 `/v1/models` 可用后，在对应远端 release 中用 [双轮流式客户端](<trace/paired_chat.py>)，传入 `systemctl --user show <unit> -p ControlGroup --value` 返回的 cgroup 路径。保留逐 token 记录并与[均匀 128 普通槽对照](<measurements/budget-pair-eager.jsonl>)比 prompt IDs、output IDs、cached 数、TTFT、decode tok/s、P95、`/proc/<pid>/io` 读量。优先再做反序或重复 A/B；不清全机页缓存，一次物理读量差不构成因果证明。
4. 保存 guard 的 baseline、最大可用内存降幅、cgroup/整机 swap、ready/错误日志；用 `systemctl --user stop <确切实验unit>`，确认 30141 关闭，等待驱动释放并量化可用内存恢复。上一次 A/B 紧接启动曾因 headroom 未恢复被正确拒绝，稍后自然恢复，不要移除门禁。若映射路径失败，保留证据，回到默认 eager 普通槽；不要自动转到生产。
5. 后续验收仍缺：真正长上下文与 Chrome/KDE 共存、实际完整服务峰值、长会话质量/吞吐及尾延迟、桌面可用性、DSH 端到端。必要时明确取舍，未验证前不切 30140。若换入更快但热态仍慢，按实测拆分，不承诺接近原全驻留吞吐。

## 2026-11-24 23:10：图路径与新的瓶颈

### 已完成的实验

1. **40 层 mapped-128**：`musaHostAlloc(mapped)` 全 40 层 × 128 槽分配成功，两轮流式输出与 eager 128 槽逐 token 相同；峰值可用内存下降 20.88 GB、cgroup 8.88 GB、swap 0、退出后完整回落。但速度比普通槽更差：[turn2 2.99 vs 3.60 tok/s](<measurements/mapped-pair.jsonl>)、P95 496 vs 352 ms。**结论：不采用 mapped；保留普通槽。** 证据：[mapped-128.json](<measurements/mapped-128.json>)。
2. **删除每层同步**：eager 槽的 4 次阻塞 H2D copy 已按流排序，去掉每层 `synchronize` 后 warm 仍在 3.6 tok/s，证明“同步打断流水线”不是主因。
3. **breakable MUSA decode graph（关键）**：用 vendor 的 `BreakableCudaGraphBackend`，把 `BoundedExperts.forward` 标成 `eager_on_graph` 断点，capture 时用 `capture_stub` 只分配桥接输出；replay 时真实换入。两轮输出与 eager 逐 token 完全一致，热态 [10.36/10.11 tok/s](<measurements/graphprobe-attn-pair2.jsonl>)，P95 118–122 ms。把 decode attention 也纳入捕获后差异很小。
4. **pinned 异步换入**：换入改为共享 8 个 pinned staging + 4 次 async copy，`ix` 预分配 + pinned H2D；热态 [11.42/10.91 tok/s](<measurements/pinnedfill-pair2.jsonl>)，P95 100–108 ms。峰值下降 20.64 GB、cgroup 8.79 GB、swap 0。证据：[pinnedfill-128.json](<measurements/pinnedfill-128.json>)。

### 现在的时间账（热态、每 token）

- 路由等待（前一段图 + attention/router GPU）：约 21.6 ms
- 槽换入（17 次逻辑 miss）：约 29.3 ms（pinned 后下降，未重测窗口）
- 专家 GEMV + reduce：约 34.4 ms
- 其余图段/宿主：约 10–15 ms

逻辑 miss 约 17/token（5.3%），物理读量 warm 可低至 19–151 MB/48 token，**SSD 不是主瓶颈**。graph replay 本身实测 0.006 ms/段，也不是瓶颈。

### 已排除/低收益

- **coalesced 权重布局**：单层 8 专家 pair 0.759 → 0.722 ms（约 5%），bit-equal；不值得为此重写 16 GB pack。
- 160 槽、非均匀容量、局部路由图、GDN 投影图：此前均无端到端收益。

### 距离 25–30 tok/s 的结构性差距

专家路径每层两次 GEMV（约 0.5–0.86 ms）+ 路由等待（约 0.54 ms）串行叠加，40 层就是约 55 ms；即使换入降到 0，理论上限也只有约 18–20 tok/s。要再翻倍必须：

1. 把逐层断点合并/减少（或把 expert kernel 做成图中可捕获、用设备端 slot 表 + 预测路由），或
2. 换用更快的 fused expert kernel/布局（当前 coalesced 只省 5%），或
3. 重新审视「热专家全部驻留」的内存取舍（16.2 GB host + 5 GB backbone 会逼近/超过 24 GB 目标，当前不做）。

### 复现命令（隔离 30141）

```bash
bash experiment_launch.sh graphprobe 128 2400   # breakable decode graph，无 slot timing
bash experiment_launch.sh graphtiming 128 2400  # 同上 + AIBOOK_SLOT_TIMING
# readiness 后：
cd ~/sglang-m1000/trial-<rev>-128-graphtiming
python3 trace/paired_chat.py "$(systemctl --user show <unit> -p ControlGroup --value)"
systemctl --user stop <unit>
```

图路径目前是 **probe-only**：`graph_compat.py` 在运行时修了 vendor 的 logits 输出桥接、NamedTuple 弱引用、decode attention 断点，并在 `validate` 里用 `AIBOOK_GRAPH_PROBE=1` 放行 graph backend。正式采用前必须把这些补丁变成 overlay 的固定 transform、加 pin/回归，并重新验证长上下文、Chrome/KDE 共存和正式 30140；未验证前不要切生产。

## 2026-11-24 23:30：逐层专家图与 overlap

- 把每层专家数学（2 次 GEMV + silu + down reduce）单独捕获成 MUSAGraph，在 break 内只 `replay()`。热态从 11.4 升到 [13.08/11.16 tok/s](<measurements/graphexpert-pair2.jsonl>)，P95 90/122 ms；两轮输出仍逐 token 一致。
- 打开 vendor 的 overlap scheduler（`--disable-overlap-schedule` 去掉）后 [13.29/12.91 tok/s](<measurements/graphoverlap-pair2.jsonl>)，P95 89/91 ms；峰值下降 20.81 GB、cgroup 9.0 GB、swap 0。
- 新计时账：route 等待 33.45 ms、fill 20.28 ms、gemv 9.93 ms、逻辑 miss 17/token。`gemv` 里原来的宿主开销已移入 route 等待；GPU 总工作量约 33 ms，已接近全驻留基线的 32 ms，剩下的是暴露在外的 fill/宿主约 30 ms。
- **MUSAGraph 不能存盘**：`torch.musa.MUSAGraph` 只有 `capture_begin/capture_end/replay/instantiate/debug_dump/raw_musa_graph(_exec)`，没有 save/load/serialize；图里的设备指针是进程局部的。能缓存的只有 TileLang kernel cache（已在 SSD），而图捕获本身只有秒级，不是 13 分钟启动的主因。
- 逐层“镜像”已经是每层一张小图（内存中），不是瓶颈；分页单位是权重（1.58 MB/专家）。整层预载 = 404 MB/层，当前只剩约 3 GB headroom，最多约 7 层，收益约 3 ms。IO 与计算重叠受路由依赖限制：下一层的路由要等本层专家输出；本层 route 的 D2H 又会 fence 掉前面的 shared expert，因此不能自然重叠。
- 当前最优实验模式：`graphoverlap`（breakable decode graph + 逐层专家图 + pinned 异步换入 + overlap scheduler），128 槽，热态约 13 tok/s。

## 2026-11-24 23:35：热路径小优化与两块排除性证据

- 去掉每层每 token 的 1.58 MB `bytearray` 分配、把 `expert -> slot` 改成 O(1) dict 后，热态稳定在 [13.21/13.16 tok/s](<measurements/hotpath-pair2.jsonl>)，P95 91.6/91.1 ms；峰值下降 20.80 GB、cgroup 8.59 GB、swap 0。证据：[hotpath-128.json](<measurements/hotpath-128.json>)。
- **SSD 不是 warm 瓶颈**：O_DIRECT 1.58 MB 记录 QD1 中位 0.92 ms（1.71 GB/s），QD8 聚合 2.72 GB/s，顺序 1.32 GB/s；warm decode 每 48 token 只读 16–200 MB（多数是 page-cache 命中），且 M1000 是 UMA，设备拷贝不走 PCIe。换 Optane 主要改善冷启动/换页，不会把 13 tok/s 拉高。[ssd-latency.json](<measurements/ssd-latency.json>)
- **逐层专家图已验证**：单层 8 专家 pair 的 GPU 时间约 0.80 ms；8 槽 bank 与 256 槽 bank 几何无差异（0.798 vs 0.795 ms），所以 GPU 侧 40 层约 32 ms，已经等于全驻留基线的整 token 时间。剩余差距是暴露的 fill/宿主，不是 MoE kernel 布局。
- **MTP/DFLASH 尚未测试**。Qwen3.6 文档只给 DFLASH（29–53 tok/s，随 accept len 波动），Qwen3.5 才给 NEXTN MTP（22–25 vs base 15.4）。本机 checkpoint 确实带 1 层 MTP 头（`mtp.*`，1.69 GB BF16，256 个 dense 专家），但 config 只有 `mtp_num_hidden_layers=1`、文档未验证 3.6 的 NEXTN；DFLASH 的独立 draft 模型没有下载。若启用，1.69 GB 会把峰值推到约 22.5 GB（超过 22 GB 保护线），需减槽；且 verify 每步多 token 会成倍放大专家换入。短期内不作为默认路径。

## 2026-11-24 23:42：144 槽（当前最优）

- `graphoverlap` + 144 槽：[14.95/14.35 tok/s](<measurements/slots144-pair2.jsonl>)，P95 80.3/77.1 ms；逐 token 与 eager 基线一致。峰值可用内存降幅 21.82 GB、cgroup 8.80 GB、swap 0，未触发保护停止（[slots144.json](<measurements/slots144.json>)）。
- 144 槽是 22 GB 保护线内的上限：160 槽会再加约 1.0 GB，直接越线；因此槽位数这条杠杆到头了。
- 全驻留不可作为 MTP 验证路径（用户明确内存不够）。MTP 头 1.69 GB 若加在 144 槽上会把峰值推到约 23.5 GB，同样越过 22 GB 保护线；要装下只能退回约 96–112 槽，hit rate 和 fill 都会变差，而 EAGLE verify 每步 5 token 还会成倍放大专家换入。MTP 不是当前约束下的直接解。

## 2026-11-24 23:58：140 槽稳定档与 MTP 路由量化结论

- 140 槽 `graphoverlap` + 专家图直接返回：[14.05/14.16 tok/s](<measurements/slots140-pair2.jsonl>)，P95 85.7/80.1 ms；峰值下降 21.51 GB，距 22 GB 保护线还有 0.64 GB 可用余量（[slots140.json](<measurements/slots140.json>)）。144 槽能到 14.95，但峰值 21.82 GB，余量只有约 0.05 GB；同一配置加单次 H2D 实验时保护在 21.97 GB 触发停止，因此 144 不作为稳定档。
- 单次 H2D staging（`graphonefill`）在 144 槽被保护停止（21.97 GB），未取得有效速度数据；不作为结论。
- **MTP/DFLASH 对分页瓶颈无效**（离线路由量化）：5-token verify 窗口每层只触达 24.7 个不同专家（naive 40 的 62%），但容量受限的 LRU 在 144 槽下 **W=1 与 W=5 的 miss/token 都是 31.3**。投机解码摊薄的是 GPU pass，不是专家换入；再叠加 1.69 GB MTP 头会越保护线，因此它不是 25–30 的解。[mtp-route-analysis.json](<measurements/mtp-route-analysis.json>)
- 当前稳定最优：140 槽，约 14.1 tok/s；144 槽约 14.9 tok/s 但余量过小。

## 2026-11-25 00:02：替换策略也到顶

- 同一 decode trace 上对比 LRU / LFU / LRU-pin-current：140 槽 miss/token 31.79 / 31.01 / 31.72，144 槽 31.33 / 30.60 / 31.22。LFU 只比 LRU 好约 2–2.5%，pin 当前路由无效。缓存是**容量受限**，不是替换策略问题。[policy-analysis.json](<measurements/policy-analysis.json>)
- 至此三条可调杠杆都已量化：槽位数（22 GB 线内 140–144）、替换策略（约 2%）、投机窗口（miss/token 不变）。热态稳定在 14.1–14.9 tok/s。
