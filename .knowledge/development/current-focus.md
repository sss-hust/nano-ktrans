---
updated: 2026-04-15 10:58
---

# 🔥 当前工作焦点

## 正在进行

- [x] 建立 CPU-only、非 PIM 的最小可运行路径
- [x] 在真实权重上验证 CPU 路径并补齐 benchmark 入口
- [x] 在宿主机上打通 `cuda_cpu_offload` 并拿到真实 Qwen3 benchmark
- [x] 在真实 UPMEM 硬件上跑通多 rank PIM microbenchmark
- [x] 将 `pim_shadow` 接入主推理链路并记录 PIM 可见性与路由统计
- [x] 为 `PIMMoEBackend` 接入最小真实 DPU 数值执行
- [x] 将真实 DPU linear runtime 扩到多 rank / 多 DPU 分片执行，并在主链路上完成 `cuda_pim` 真机验证
- [x] 为 `PIMMoEBackend` 接入 fused expert DPU kernel 的第一版，并确认主链路可真实触发 fused expert 调用
- [ ] 为 CPU offload 接入 `kt-kernel` 或等价高性能实现
- [ ] 将系统目标从静态专家放置切换到 GPU/PIM 动态专家调度
- [x] 落第一版 prefill 保护策略，避免 prefill 大批量 token 直接打到 PIM
- [x] 落第一版 decode 迁移执行数据面：可在运行时 materialize / demote GPU experts
- [x] 落第一版 prefill expert 预取：prefill 期间可将候选热点专家预热到 CPU staging cache
- [x] decode 迁移已接上 GPU budget 约束和运行时 eviction
- [x] decode 阶段也会为后续 promotion 做预取，不再只有 prefill 才预热
- [x] decode promotion 会优先消费“已预热完成”的 expert
- [x] 预取诊断已区分 request 与真实 enqueue，避免把重复请求误当成有效预热
- [x] scheduler 已接入 access age / residency cooldown 元数据与诊断
- [x] scheduler 已支持逻辑步长和 prefill collect-only 模式
- [x] decode 已支持“只消费 prefetch-ready promotion”的保守模式
- [x] migration queue 已接入按 expert 去重和更细的排队诊断
- [x] scheduler 已支持“无立即迁移也可按热度预取 offloaded experts”
- [x] scheduler 已支持 profile 预设，benchmark 已输出迁移/预取摘要，便于直接比较 overlap 相关策略

## 阻塞项

- `Qwen3-30B-A3B` 在本机 `47.41 GiB` GPU 上走全专家纯 `cuda` 路径仍然 OOM
- 当前 `pim` backend 已支持 fused expert DPU kernel，但 fused 路径仍明显慢于 linear3 路径，默认仍应保留 `linear`
- 当前真实 PIM backend 默认只在小 flattened batch 上启用，prefill 大批次仍会回退 CPU
- 当前 `cuda_pim` 已可端到端运行，但性能明显落后于 `cuda_cpu_offload`，瓶颈在每个 offloaded expert 仍需三次 host↔DPU 线性调用
- fused expert kernel 已修正为“hidden 先落 WRAM 再复用”的数据流，但当前单 expert microbench 仍约比 `linear3` 慢一个数量级，端到端 benchmark 暂不值得继续长时间盲跑
- fused expert host bridge 进一步改成了 `hidden_group x row_group` 的二维分片，能让更多 DPU 参与单 expert 计算，但在 Qwen 级别形状上仍未超过 `linear3`
- 当前系统仍是“静态 GPU expert mask + 其余专家走 offload backend”，还没有专家 promotion / demotion 控制面
- 当前 PIM 路径每次调用仍是“按次提交输入和权重”，还不是“专家权重常驻 PIM、按需迁移到 GPU”的层级存储模型
- 当前动态调度还停留在“观察热度 + 生成迁移计划 + 更新驻留表”，真实 GPU/PIM 权重迁移数据面还没接入
- 当前动态调度已改成“观察热度 + 生成迁移计划 + 迁移计划排队”，不会再在没有真实数据面的前提下直接篡改有效驻留状态
- 当前迁移管理器已能按层记录 `prefill` / `decode` 迁移计划，但仍未真正执行 GPU<->PIM 权重搬运
- 当前 decode 阶段已可消费迁移队列，并把单 expert 从 checkpoint 动态 materialize 到 GPU `ModuleDict`；但这仍是同步、逐 expert 的最小实现，还没有异步预取与 overlap
- 当前已新增 CPU staging cache 和 prefill 预取入口，但 promotion 仍是“CPU staging -> GPU module”的同步 materialize，不是真正的 GPU<->PIM 异步 DMA
- 当前 decode promotion 已不会无限增长 GPU resident experts；超过 GPU budget 时会按层内 hotness 驱逐冷 expert，再为热点 expert 腾位
- 当前 promotion 队列已按“本步活跃优先 + hotness 优先”排序，resident set 更接近真正的热点 cache 语义
- 当前 decode promotion 排序已经额外把“prefetch 已就绪”作为最高优先级，减少关键路径阻塞
- 当前诊断里已经能分辨：
- 当前 scheduler 已维护每个 expert 的：
- 当前 scheduler 已新增两类更接近真实系统的控制旋钮：
- 当前 scheduler / migration 控制面已经有三类关键开关：
- 当前 migration queue 已能输出：
 - 当前 scheduler 现在不必等 migration op 产生，已经能按层直接选出 hot offload experts 做候选预取
- 当前 migration queue 已能输出：
  - `total_enqueued_ops`
  - `total_deduped_ops`
  - `total_drained_ops`
  - 每次 phase 的 `deduped_plan_size`
- 当前 scheduler / migration 控制面已经有三类关键开关：
  - `prefill_collect_only`
  - `step_stride_prefill / step_stride_decode`
  - `decode_require_prefetch_ready`
- 当前 scheduler 已维护每个 expert 的：
  - `last_access_step`
  - `last_residency_change_step`
  为后续真正的 anti-thrashing 策略做准备
- 当前诊断里已经能分辨：
  - `prefetch_requested`: 调度层总共发起了多少次预取请求
  - `prefetch_enqueued`: 真正进入 staging cache 流程的次数
  - `decode_prefetch_hits/misses`: decode promotion 是否命中已预热专家
- Python 侧直接驱动 UPMEM 的尝试仍不稳定，当前更可靠的是独立 C host benchmark
- `HTTP_PROXY` / `HTTPS_PROXY` 指向 `127.0.0.1:7897` 时会阻塞 `pip install`

## 下一步

- 继续优化 fused expert kernel 的计算/访存比，优先压缩 `down_proj` 阶段的 WRAM 读取与 MRAM 访问成本
- 评估是否需要把 fused expert 改成“多 expert 打包提交”而不是单 expert 单次调用
- 在保持 `linear` 默认不变的前提下，继续扩大真实 PIM backend 的 prefill batch 支持范围
- 设计动态专家调度骨架：专家驻留表、热度统计、promotion/demotion 队列和异步迁移接口
- 将 `gpu_experts_mask` 从静态初始化参数改成运行时可更新的驻留状态
- 继续扩展 promotion / demotion 数据面：
  - 将当前 prefill 预取 + CPU staging 升级为真正的异步 promotion
  - 为 PIM resident expert 增加常驻句柄与回写
  - 接上迁移和 decode 计算的 overlap 控制
- 将“按层、按 expert”的迁移粒度继续放大，避免 decode 关键路径上重复的 Python 调度和模块构建
- 让 decode 的已规划但未立刻执行的 promotion 继续在后台预热，为下一步真实 overlap 做准备
- 把“prefetch ready”从统计信号继续推进到真正的异步 overlap 控制条件
- 开始把 migration 控制面指标压实，方便后续用 benchmark 观察 overlap 是否真的发生
- 下一步可以在不破坏现有行为的前提下，逐步把 cooldown / idle-age 从纯诊断指标提升成可调度约束
- 后续 benchmark 可以开始扫描：
- 后续 benchmark 可以开始扫描：
- 后续 benchmark 可以开始扫描：
 - 后续 benchmark 可以开始扫描：
  - prefill 只收集热度 vs prefill 直接发迁移计划
  - 大步长 prefill vs 小步长 decode
  - decode 立刻 promotion vs decode 只消费 prefetch-ready promotion
  - queue 去重前后 migration submit 量的差异
  - 纯“候选预取”对后续 decode promotion 命中率的影响
- 对比 `cpu`、`cuda_cpu_offload`、`pim` 三条链路的 prefill/decode 延迟与 offload 命中分布
- 继续补充架构说明、依赖说明和版本化文档
- 基于 `baseline / overlap_safe / eager` 三组 scheduler profile 跑真实 benchmark，对比：
  - `prefetch_hit_rate`
  - `runtime_deferred_for_prefetch`
  - `dedupe_ratio`
  - 端到端 prefill / decode 延迟

## 本轮对话上下文

> 当前 repo 已完成新阶段：CPU 基线、Qwen3 真实权重适配、`cuda_cpu_offload` 真机验证、`pim_shadow` 主链路接入、真实 UPMEM 多 rank benchmark、最小真实 DPU linear backend、`cuda_pim` 真机端到端验证，以及 fused expert DPU kernel 的第一版。新的主目标已经切换为“GPU 保持主干层，专家在 GPU/PIM 间动态迁移并与传输 overlap”，因此下一阶段重点不再只是单 kernel 优化，而是系统级调度改造。

<!-- updated: 2026-04-15 06:58 -->

## 本轮新增进展

- `HybridMoE` 现在会在 `decode` 阶段优先 drain 本层 migration queue，并按 `decode_promote_k` 应用部分迁移。
- promotion 路径已能通过 `ExpertWeightLoader.load_expert()` 从 safetensors 加载单 expert 权重，动态构建 GPU expert module，并更新运行时 `gpu_experts_mask`。
- demotion 路径已能从运行时 `gpu_experts` 中移除对应 expert，并同步更新 offload backend 的 mask。
- 这意味着系统已经从“只有迁移控制面”推进到“最小可执行 GPU materialization 数据面”，但仍未实现 GPU<->PIM 异步拷贝和 overlap。
- 现在 prefill 阶段会对计划中的 `PIM/CPU -> GPU` promotion 做 expert 级预取，把单 expert 权重提前拉到 CPU staging cache，减少 decode promotion 时的 checkpoint I/O。
- `decode` 阶段现在会在应用 promotion 前检查 GPU budget，并按 hotness 驱逐非活跃的冷 resident expert，保证 GPU resident set 始终受控。
- `decode` 阶段现在也会对计划中的 future promotions 发起预取，并对 promotion 队列做“active first, hottest first”排序。
- `decode` promotion 现在会优先提升 staging cache 已就绪的 expert，并暴露 `decode_prefetch_hits/misses` 诊断。
