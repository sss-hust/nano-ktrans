---
id: ADR-001
title: PIM + MoE Expert Offloading 研究综述与 nano-ktrans 的可借鉴创新点
status: partially-implemented
created: 2026-04-21
updated: 2026-04-21
tags: [architecture, pim, moe, offloading, related-work]
---

# ADR-001: PIM + MoE Expert Offloading 研究综述与 nano-ktrans 的可借鉴创新点

## 背景

nano-ktrans 当前已经打通：
- Hybrid MoE（GPU + CPU + 真实 UPMEM DPU）
- 动态 GPU ↔ PIM/CPU 专家迁移（hotness EMA + scheduler profile）
- 多级 prepared tier（ready → warmed → activated → applied）
- 后台 background offload worker + 多段 staged commit queue
- GPTQ Int4 的 DPU 量化路径（`kernel_mode=4` int8 定点）

这套系统已经在 `scheduler/migration/prepared tier/commit pipeline` 上具备非常细的控制面，但**关键路径性能仍未超过 CPU grouped baseline**，尤其在 `batch > 1` 时优势丧失。

本 ADR 梳理 2024–2026 年与本项目最相关的 9 篇论文，提炼我们**可以直接或经简单改造借鉴**的创新点，不引入硬件级改造。

---

## 相关工作速览

| 论文 | 会议/年份 | 硬件 | 核心创新（一句话） |
|------|----------|------|------------------|
| **Fiddler** | ICLR'25 | CPU + GPU | 让小 batch 时 expert **就地在 CPU 算**，避免 PCIe 权重搬运 |
| **MoE-Infinity** | 2024 (arXiv 2401.14361) | GPU + CPU | 用 **Expert Activation Matrix (EAM)** 做 request-level 激活追踪 + activation-aware 预取/缓存 |
| **HOBBIT** | 2024 (arXiv 2411.01433) | GPU + CPU/NVMe | **混合精度 offloading**：cache-miss 的"不关键"专家用低精度版本，显著降低加载带宽；三层分层 `token / layer / sequence` |
| **HybriMoE** | 2025 (arXiv 2504.05897) | CPU + GPU | 三条优先级（GPU/CPU/PCIe）+ **MRS score-aware 缓存**（$S = \alpha \cdot \text{TopP}(s) + (1-\alpha) S$）+ impact-driven 预取 |
| **fMoE / FineMoE** | EuroSys'26 (arXiv 2502.05370) | GPU + CPU | **iteration-level** 专家激活跟踪（保留完整 gate 概率向量）+ prompt 语义嵌入驱动预取 + pub-sub 异步搜索 |
| **FloE** | ICML'25 poster | 单 GPU + host | On-the-fly MoE inference on memory-constrained GPU |
| **PIMoE** | DAC'25 | NPU + UPMEM PIM | **Throttle-Aware Task Offloading**：基于硬件 throttle 信号动态决定 expert 放 NPU 还是 PIM |
| **NeuPIMs** | ASPLOS'24 (arXiv 2403.00579) | NPU + HBM-PIM | **GEMM → NPU，GEMV → PIM**；硬件 dual row buffer + 软件 **sub-batch interleaving** 消除 blocked 模式 |
| **AttAcc!** | ASPLOS'24 | xPU + HBM-PIM | KV cache 常驻 PIM，attention 阶段 PIM 加速 |
| **PIM-AI** | 2024 (arXiv 2411.17309) | DDR5/LPDDR5 PIM | Decode（带宽密集）卸载 PIM，无需改 memory controller，TCO/能耗双维度评估 |
| **P3-LLM** | 2025 (arXiv 2511.06838) | NPU + PIM | 混合数值格式 + 算子融合降低运行时反量化开销 |

---

## 可借鉴的创新点（按优先级排序）

### 🥇 P1：引入 **Score-Aware 缓存驱逐**（来自 HybriMoE MRS）

**现状**：nano-ktrans 的 `warm_cache` / `activated_cache` 目前的 victim 选择是 `lifecycle 优先级 + hotness`（EMA），但 hotness 只看"是否被激活"，未利用 gate 输出的**完整 softmax 分数**。

**借鉴**：把 scheduler 观察 hotness 的公式升级为 HybriMoE 的 MRS：

```
S_expert = α · TopP(gate_scores) + (1 − α) · S_expert
```

- `α` 作为滑动平均系数（profile 级可调，默认 0.3）
- `TopP` 只累加每层 top-p 个专家的分数（典型 `p = 2 · top_k`，Mixtral 下 `p = 4`）
- 低分专家对复用概率几乎无差异，被截断后既减少噪声也减少更新开销

**预期收益**：HybriMoE 报告在 25% 缓存时命中率从 30% → 36%（Mixtral）；对 nano-ktrans 中"冷 promotion 占比偏高"的场景尤其适用（我们 `cold_promotion_penalty` 就是为这个问题加的）。

**改造点**：
- `utils/expert_runtime_state.py::update_hotness` 新增 `router_scores` 参数
- `scheduler/profiles.py` 三个 profile 各配一个 `hotness_top_p`
- 对 warm/activated cache 的 `_pick_victim` 改用 MRS 分数

**落地成本**：中（~200 行 + 单测 + profile sweep 对比）

**✅ 实现状态（2026-04-21）**：已落地，默认关闭，需显式启用。

实现要点：
- `utils/expert_runtime_state.py::update_hotness` 新增 `router_scores / mrs_alpha / top_p` 三个 kwargs；
  全 None 时保持旧 bincount 行为（向后兼容）
- `SchedulerConfig.hotness_mrs_alpha / hotness_top_p` 配置字段
- `DynamicExpertScheduler.observe(..., topk_weights=...)` 可选接收 router scores；
  MRS 开启但 weights=None 时 fallback 到 bincount 并记入 `hotness_bincount_observations`
- `HybridMoE.forward` 把 router softmax 后的 `topk_weights` 传给 scheduler
- 诊断新增：`hotness_mrs_observations` / `hotness_bincount_observations`

启用：`LLM(scheduler_hotness_mrs_alpha=0.5, scheduler_hotness_top_p=2*top_k, ...)`

关键踩坑（见 [gotchas.md](../../context/gotchas.md)）：
- **必须按 token 数归一化**：prefill 一次 observe 看到 512 token，若直接 scatter_add
  会把 EMA 推到极值；最终实现 `score_mass / token_count` 让单次贡献 ∈ [0, 1]
- `torch.scatter_add_` 而不是 `bincount`（后者不支持 float weight）

---

### 🥈 P2：引入 **Iteration-Level 激活追踪 + Prompt 语义预取**（来自 fMoE）

**现状**：nano-ktrans 的 `expert_selection.profile_expert_activation` 只做离线校准，生成 `activation_freq[layers, experts]` 后做静态 GPU mask；`DynamicExpertScheduler` 也只看当前迭代的 topk，没有**跨请求**的语义记忆。

**借鉴**：新增 **Expert Map Store**：
- 对每次 decode iteration，把每层 gate 的完整概率向量 `P_l ∈ R^J` 连同 **prompt embedding** 一起存入 store
- 新请求进来时：
  - **冷启动前 d 层**：用 prompt 语义嵌入的余弦相似度匹配最相似的历史 iteration，预取其前 d 层的高概率专家
  - **d 层之后**：用已观测的 gate 轨迹做轨迹相似度匹配
- 动态阈值 `δ_l = Clip(1 − score, 0, 1)` 控制预取激进度：低相似度 → 预取更多
- 容量限制 + **Redundancy score** 去重（`RDY = (d/L)·score_sem + ((L−d)/L)·score_traj`）

**预期收益**：fMoE 报告延迟 −47%、命中率 +39%（vs MoE-Infinity）。

**为什么适配 nano-ktrans**：
- 我们已经有 materialization manager + prepared tier，store 的消费路径现成
- prompt embedding 直接复用 `model.embed_tokens` 输出，**无需额外模型**
- 完美补齐当前"decode 每步只能看当前路由"的盲区

**改造点**：
- 新增 `utils/expert_map_store.py`：LRU dict，容量 1K–50K（约 200MB）
- `scheduler/dynamic_expert_scheduler.py::plan_layer` 增加"预取候选 = 来自 store 搜索结果"分支
- pub-sub：把 store 搜索放到 background worker 上，避免 decode 关键路径阻塞

**落地成本**：高（需要存储 + 搜索线程 + 与 prepared tier 对接），但**与现有后台 worker 架构完美契合**

**✅ 实现状态（2026-04-21）**：已落地最小可用版本，默认关闭，需显式启用。

实现要点：
- 新建 [`nano_ktrans/utils/expert_map_store.py`](../../../nano_ktrans/utils/expert_map_store.py)：
  - `ExpertMap` dataclass（L2-normalized prompt embedding + 每层 raw/unit gate 分布）
  - `ExpertMapStore`：LRU 容量管理、两阶段搜索（semantic / trajectory）、`RLock` 线程安全、诊断
- `HybridMoE` 新增 `expert_map_store / expert_map_prefetch_top_k` 构造参数；
  新增 `attach_expert_map` / `_record_router_probs` / `_request_map_store_prefetch` 方法
- `MixtralModel.forward` 负责 iteration 级 begin/commit；
  用 **token embedding 均值** 作 prompt 语义锚点（fMoE §5.1），无需额外模型
- 与现有 scheduler **完全解耦**：即使 `enable_dynamic_expert_scheduler=False`，
  Store 依然能独立工作，所有建议流经同一个 `_request_prefetch` 漏斗
- 诊断新增：每层 `expert_map_prefetch_submitted / semantic_prefetch / trajectory_prefetch` + `LLM.get_offload_diagnostics()["expert_map_store"]`

启用：
```python
llm = LLM(
    enable_expert_map_store=True,
    expert_map_store_capacity=1024,
    expert_map_store_prefetch_distance=3,
    expert_map_prefetch_top_k=2,
)
```

关键设计决策：
- **prompt 锚点用 mean token embedding**（而不是 BOS token）：
  不同 prompt 的 BOS 几乎相同，mean 才有区分度
- **record_router_probs 在 pipeline_lock 外调用**：只改 per-iteration in-flight ExpertMap（单线程对象），不触及共享 pipeline 状态，避免锁竞争
- **store 建议与 dynamic scheduler 候选合流**：经同一个 `_request_prefetch(expert_idx)`
  入口，保持 cache / budget / lifecycle 单一数据源

目前 **未做**的部分（留给后续迭代）：
- store.search 仍在前台 forward 里同步调用；未接到 background worker
- 未实现 fMoE 的 similarity-aware `δ_l` 动态阈值（目前固定 top_k）
- 未做 redundancy-aware 去重（仅按 LRU 淘汰）

---

### 🥉 P3：借鉴 **Fiddler Cost Model**，把"CPU 就地算 vs 拷贝到 GPU"做成显式决策（对齐 `pim_prefill_policy`）

**现状**：nano-ktrans 当前规则是：
- prefill 大 batch → CPU backend（`pim_prefill_policy="cpu"`）
- decode 小 batch → PIM backend
- 阈值是硬编码常量（`pim_prefill_token_threshold=8`）

**借鉴 Fiddler**：把阈值改成**在线 cost model**：

```
T_cpu_local  = batch × flops_per_token / cpu_flops
T_move_to_gpu = weight_bytes / pcie_bw + batch × flops_per_token / gpu_flops
choose min(T_cpu_local, T_move_to_gpu)
```

在 nano-ktrans 语境下是三路决策：**CPU** vs **GPU (after promotion)** vs **PIM (DPU)**；
每个设备维护一个滑动平均的 `flops/token` 和 `bytes/s`，由 benchmark 填充初值，runtime 动态更新。

**预期收益**：Fiddler 报告 beam search 场景 11.57× 提升（主要因为小 batch 下 CPU 本地算完胜 PCIe 搬运）。nano-ktrans 的 `batch > 1 时优势丢失`的问题（见 `2026-04-19` 日志）本质就是同一现象。

**改造点**：
- 新增 `scheduler/cost_model.py`：简单滑动平均 + 每个 backend 的 `latency_per_token(batch, shape)` 接口
- `HybridMoE.submit_forward` 里的 CPU/PIM/GPU 分流从 static mask 升级为 cost-aware
- 可直接复用现有 `_record_fallback` 埋点做校准

**落地成本**：中（主要是 microbench 数据收集 + 阈值决策）

---

### 🏅 P4：借鉴 **HOBBIT 三层分层 + 混合精度** 思想

**现状**：nano-ktrans 的 `kernel_mode=4` 是**全局 int8 定点**，没有"按专家重要性选精度"。batch > 1 时误差放大恰好是这个问题。

**借鉴**：
- **Token-level**：per-token 动态选：高 gate score 的 token 路由的专家 → 全精度；低分 token → int8 定点
- **Layer-level**：prefetch 预测失败（`prefetch_miss`）的专家优先用低精度版本（减少加载带宽），保证 cache 命中的专家是全精度
- **Sequence-level**：保留现有 cache 策略，但给每个专家保留两份权重引用（fp16 + int8）

**潜在风险**：与 DPU 侧 `kernel_mode=4` 数值语义交叉，需要先做一层封装确认 fallback 链不被破坏。

**落地成本**：高（需要 DPU kernel 双版本 + runtime 动态选择），暂列为中期探索。

---

### 🏅 P5：借鉴 **NeuPIMs Sub-Batch Interleaving**

**现状**：nano-ktrans 目前 "GPU experts forward" 和 "PIM experts submit_forward" 虽然异步，但**同一 batch 内**只有一个阶段活跃。

**借鉴**：把单 decode step 的 batch 拆成 sub-batch A / B：
- A 在 GPU 上做 `attention + dense FFN + gate`
- B 的 `cpu/pim offload experts` 同时在后端进行
- 下一 tick 对调角色

**注意**：我们当前 batch_size=1，需要先解除 batch>1 限制；该优化更适合未来支持 beam search / continuous batching 后再做。**放 roadmap**。

---

### 💡 P6：采用 **PIMoE 的 Throttle-Aware 调度思想**

**现状**：当前 scheduler 只看热度和预算，不感知 DPU 硬件侧的动态状态（如 rank 被同系统其它进程占用）。

**借鉴**：给 `PIMLinearRuntime` / `PIMExpertRuntime` 加 **throttle signal**：
- 读取 `num_active_dpus` 与 `dpu_launch_cycles`
- 当 DPU 利用率已接近饱和时，scheduler 暂时把新 promotion 候选从 PIM 切回 CPU

**实现成本**：低（UPMEM SDK 有这些计数器），价值中。

---

## 不直接借鉴的方向

- **AttAcc / NeuPIMs 的硬件侧 dual row buffer**：需要改硬件，超出项目范围
- **PIM-AI 的 DDR5 指令嵌入**：同上
- **P3-LLM 的混合数值格式 + 协同 PIM 单元**：同上
- **MoE-Infinity 的 EAM**：理念已被 fMoE 超越，不单独引入

---

## 决策 & 落地顺序

1. ✅ **已落地（2026-04-21）**：P1 Score-Aware 缓存（MRS 替换 hotness-only）
2. ✅ **已落地（2026-04-21）**：P2 Expert Map Store + 语义预取（最小可用版本）
3. 🟡 **下一里程碑**：P3 Fiddler cost model（把 `pim_prefill_token_threshold` 升级成 cost-aware）
4. 🟡 **下一里程碑**：P6 Throttle-Aware 调度（与 P3 合并做一个 `BackendCostModel`）
5. 🔵 **中期探索**：P4 Mixed-precision expert（依赖 `kernel_mode=4` 稳定）
6. 🔵 **长期，待 batch>1**：P5 Sub-batch Interleaving

### P1 + P2 落地后的下一步验证清单

- [ ] 宿主机上跑 `cuda_cpu_offload + MRS on/off` 的 profile sweep，比较：
  - `cold_promotion_penalty` 是否下降
  - `pipeline_promotion_source_cold / non_cold_total` 比例变化
  - warm/activated cache `hit / eviction` 分布
- [ ] 宿主机上跑 `enable_expert_map_store` on/off，比较：
  - `decode_prefetch_hits / (hits + misses)` 命中率
  - `expert_map_{semantic, trajectory}_prefetch` 两路贡献
  - 多请求 workload 下 `eviction_count` 是否合理（LRU 容量 1024 是否足够）
- [ ] 结合使用 P1 + P2：MRS 驱动 victim 选择 + Map Store 驱动预取候选，预期协同效应

---

## 参考文献

- [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033) — ICLR 2025
- [MoE-Infinity (arXiv 2401.14361)](https://github.com/EfficientMoE/MoE-Infinity) — 2024
- [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433) — 2024
- [HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management](https://arxiv.org/abs/2504.05897) — 2025
- [fMoE / FineMoE: Taming Latency-Memory Trade-Off in MoE-Based LLM Serving](https://arxiv.org/abs/2502.05370) — EuroSys 2026
- [PIMoE: Throttle-Aware Task Offloading on NPU-PIM System](https://ieeexplore.ieee.org/abstract/document/11132528) — DAC 2025
- [NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing](https://arxiv.org/abs/2403.00579) — ASPLOS 2024
- [AttAcc! Unleashing the Power of PIM for Batched Attention](https://dl.acm.org/doi/abs/10.1145/3620665.3640422) — ASPLOS 2024
- [PIM-AI: A DDR5/LPDDR5 PIM Architecture for LLM Inference](https://arxiv.org/pdf/2411.17309) — 2024
- [P3-LLM: An Integrated NPU-PIM Accelerator for LLM Inference](https://arxiv.org/abs/2511.06838) — 2025
