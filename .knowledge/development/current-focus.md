---
updated: 2026-04-16 03:22
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
- [x] migration queue 已接入 lifecycle 状态：`queued / prefetching / ready / deferred / applied`
- [x] decode 在 `decode_require_prefetch_ready` 模式下已改成 ready-only 消费，不再先 drain 全队列再回退
- [x] materialization manager 已支持后台 prefetch completion 轮询，ready 状态可在进入层前被主动刷新
- [x] migration manager 已支持 ready-only drain，decode 不再手写 ready subset 过滤逻辑
- [x] materialization 预取完成已接入 completion queue，ready 轮询不再扫描全部 future
- [x] ready 轮询已从 `HybridMoE.forward()` 入口上移到 engine/model 层，避免每层重复轮询
- [x] `SimpleEngine` 已抽出统一 offload refresh hook，prefill full/chunked 与 decode 共用同一入口
- [x] offload refresh 已接入模型级统计，benchmark 可直接看到 refresh 次数与每步 ready 收敛量
- [x] layer 级 refresh 已加空队列短路，避免无 pending prefetch 时做无意义轮询
- [x] benchmark 已支持 scheduler profile sweep，可一轮比较多组调度策略
- [x] token-step 级 offload refresh 已升级为最小 migration pipeline runtime：ready 轮询与 ready promotion 可在进入模型前统一推进
- [x] pending promotion 的预取提交已开始收敛到 pipeline runtime，decode 进入层前可先把 `queued -> prefetching/deferred` 往前推进
- [x] GPU promotion 预热现在可优先直接从 offload resident 权重 staging，不必总是回到 checkpoint/safetensors 扫描
- [x] GPU demotion 现在支持 warm expert cache，短时间内回迁的热点 expert 可避免重复模块构建
- [x] ready expert 现在可以在 token-step pipeline 中提前 prebuild 到 warm cache，进一步压缩 applied 时的 build/load 开销
- [x] warm cache 的 prebuild 现在固定落在 CPU，再在 promotion 时执行单次 device transfer，避免 prebuild 阶段污染 GPU 关键路径
- [x] migration lifecycle 现在显式区分 `ready` 和 `warmed`，可以单独观察“数据 ready”和“模块已预构建”这两个阶段
- [x] token-step pipeline 现在显式区分 `warmed -> activated -> applied`，把 warm cache 上的 device transfer 从最终 promotion 应用里拆出来单独观测
- [x] token-step pipeline 现在为 activated expert 引入单独 device-side cache，`activated -> applied` 可以优先命中已完成 device transfer 的 module
- [x] activated cache 现在会按 hotness 和 lifecycle 优先级保留更热的候选，避免 decode 批量 promotion 时把 device-side 预算浪费在较冷 expert 上
- [x] deferred requeue 现在会保留 `prefetching/ready/warmed/activated` 等中间态，不会把已经准备到一半的 expert 重新打回 `queued/deferred`
- [x] migration 诊断现在会显式统计 `requeue_preserved_states`，可以直接看到流水线阶段在 deferred 重排时被保留了多少次
- [x] decode 的 ready promotion 现在不再把“已 ready 但本步预算不够”的 op 重新 enqueue，一部分 ready expert 会直接保留在原队列中等待下一步消费
- [x] `decode_require_prefetch_ready` 模式下，resident-tier 直接 stage 现在也不会被同一步立即消费，而是等下一次 refresh/pipeline 再推进到 ready
- [x] benchmark run 现在会在每次生成前重置 offload runtime 计数，单次 run 的 scheduler summary 不再混入前序 warmup / 历史 step 噪音
- [x] ready expert 的 prebuild 现在也开始按 hotness 和 decode 预算裁剪，只为更有希望进入 activation/applied 的候选构建 module
- [x] pipeline 现在会按 promotion source 区分 `activated/warm/cold`，并直接统计“非冷路径命中”次数，便于判断 overlap 是否真的开始生效
- [x] decode 层内 migration 消费已改成 `peek + selective consume`，只移除真正 `applied` 的 promotion/demotion，预算不足或未 ready 的 op 会继续留在原 pending queue 中
- [x] layer-forward 在 strict ready-only 模式下现在只消费 lifecycle 已推进到 `ready/warmed/activated` 的 promotion，不再因为同一步 materialization cache 命中而越级 promotion
- [x] ready promotion 现在已先经过同层小批量截断，并开始输出 `pipeline_apply_batches / pipeline_apply_batch_experts`，为后续真正的 layer-batched apply 做诊断基础
- [x] ready promotion 批次现在会先统一计算并执行本批次需要的 eviction，减少 batch 内每个 expert 重复做 GPU budget 检查
- [x] token-step runtime 现在会汇总 apply batch 的批次数、expert 数和批内 eviction 数，便于从 step 级视角观察流水线推进
- [x] benchmark/profile sweep 现在会自动汇总 decode TPS、overlap 命中、promotion source 和 apply batch 指标，便于直接比较哪组策略最接近目标
- [x] profile sweep 现在也会带 step 级 runtime apply batch totals，能同时观察 layer 视角和 token-step 视角的批处理推进情况
- [x] pipeline runtime 现在返回增量 batch 指标，不再把层上的累计 apply batch 计数重复计入每个 tick
- [x] profile sweep 现在会额外输出 `comparison_table`、`best_by_metric`、非冷路径 promotion 比例以及 runtime batch size 平均值，便于直接按 overlap 指标比较 profile
- [x] ready promotion 的批处理现在会额外区分 `activated / warm / cold` 三类 apply 来源，layer/runtime/summary 三层都能直接观察 batch 内部构成

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
- 当前 migration pipeline runtime 仍是前台 token-step hook，不是真正独立线程/事件循环；但 ready promotion 已不必等层内 forward 再逐层处理
- 当前 pipeline runtime 已能在 decode 前主动 prime pending promotions，但 `ready -> applied` 仍在前台 step hook 中完成，还没脱离主线程
- 当前 resident-tier 预热已先在 CPU/PIM backend 上打通 export 接口，但还只是同步导出到 CPU staging cache，不是 PIM->GPU 真异步搬运
- 当前 warm expert cache 还只是 CPU 侧 module 复用层，尚未接入 GPU pinned buffer / CUDA graph / 异步拷贝优化
- 当前 prebuild 仍由前台 pipeline hook 触发，适合做控制面和缓存层验证；要真正赢过 cpu+gpu，还需要把 prebuild 与 PIM resident 传输变成异步后台执行
- 当前 warm cache 的 device transfer 仍是同步 `.to(device)`，还没拆成独立 CUDA stream 或后台 copy worker
- 当前 `activated` 只是“warm module 已搬到目标 device、尚未进入 GPU resident set”的前台状态，不是真正后台 activation worker
- 当前 activated cache 仍由前台 token-step hook 填充，尚未按 layer batch 或独立 CUDA stream 做真正异步 activation
- 当前 activated cache 已开始做预算裁剪，但仍是逐 expert 激活/驱逐，不是按层打包 promotion
- 当前 deferred op 虽然已保留中间 lifecycle，但 migration queue 还没有按“阶段完成度”做真正的分层批处理
- 当前控制面已经能保住流水线阶段进度，但 benchmark 还没有把“requeue 保留率”纳入 profile 对比摘要
- 当前 ready promotion 还没有完全拆成 batch commit；虽然已避免重复 requeue，但 applied/deferred 仍是逐 expert 判定
- 当前 decode 层内 migration 虽已避免 drain/requeue 抖动，但 promotion/apply 仍是逐 expert 提交，尚未做真正的同层批量 apply
- 当前 ready promotion 虽然已有 batch 统计和批次截断，但真正的 apply 路径仍是逐 expert 激活/插入 resident set
- 当前 ready promotion 虽然已经批量算出了 eviction 需求，但 batch 内真正的 warm/activated 命中与 resident set 注入仍是逐 expert 完成
- 当前 runtime 已能汇总 batch 级推进情况，但 benchmark 还没把这些 step 级 batch 指标纳入 profile sweep 排序
- 当前 profile sweep 已能自动汇总 batch 指标，但还没有把这些指标和真实 `cuda_pim` 宿主机结果形成持续对照表
- 当前 profile sweep 已覆盖 runtime batch totals，但还没把这些指标做成跨实验的历史趋势表
- 当前 profile sweep 已能给出自动对比表和 metric 排名，但还没在宿主机真实 `cuda_cpu_offload/cuda_pim` sweep 上收集一轮对照结果
- 当前 batch apply 的来源构成已经可见，但还没把这些来源构成真正用于 batch policy，例如按 `activated` 命中率自适应调整 activation/prebuild 预算
- 当前 runtime batch totals 已是按 tick 增量统计，但还没有接入真实宿主机 benchmark 结果做长期趋势归档
- 当前 strict ready-only 语义已经覆盖 resident staging，但 `prefetching -> ready` 仍然依赖前台 refresh，而不是真 completion event 驱动
- 当前 benchmark 已能稳定观察单次 run 的 pipeline 行为，但还缺少 profile sweep 结果表层面的自动对比汇总
- 当前 prebuild 已做候选裁剪，但 warm cache 还没有独立的“低优先级淘汰”策略，仍然主要依赖容量上限和 LRU
- 当前已经能量化 promotion source，但 benchmark 还缺少跨 profile 的自动排名/对比表
- 将 `activated -> applied` 从当前逐 expert 路径推进到同层小批量提交，继续压 decode 关键路径上的 Python 控制开销
- 将当前“批次截断 + 逐 expert apply”升级成真正的 per-layer batched activation/apply，尽量减少 batch 内重复的 GPU budget 检查与 Python 字典操作
- 将当前“批量预腾位 + 逐 expert apply”继续推进成真正的 per-layer batched activation/apply，把 warm/activated 命中后的 resident set 注入也批处理化
- 把 runtime 的 batch 指标接进 benchmark/profile sweep 排序，优先观察哪些策略真正提高了每步 apply batch 的有效利用率
- 用宿主机真实 `cuda_cpu_offload` / `cuda_pim` 跑一轮 profile sweep，把新加的 batch/overlap 指标真正量出来，再决定优先压 activation 还是 promotion source 的冷路径比例
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
- 继续把 lifecycle 从“诊断状态”推进到真正的异步执行状态机：
  - 让后台预取只推进 `queued -> prefetching -> ready`
  - 让 decode 主路径只消费 `ready`
  - 为后续 GPU<->PIM 异步迁移 worker 预留事件接口
- 把 migration lifecycle 和 materialization worker 真正打通：
  - 让 `ready` 由后台预取完成事件驱动
  - 减少当前 decode 入口的同步 `is_ready()` 轮询
- 将 ready 轮询从“每层 forward 入口”进一步收敛到独立 runtime hook 或 worker，避免后续层数增多时引入额外 Python 开销
- 将当前 `MigrationPipelineRuntime` 从前台 tick 升级为真正后台 worker：
  - 让 `prefetching -> ready` 通过 completion/event 推进
  - 让 `warmed -> activated` 脱离当前 token-step 同步 hook，变成独立 activation queue
  - 让 `ready -> applied` 逐步脱离主线程 token-step hook
  - 让 activated cache 支持按层批量 promotion，减少逐 expert 激活/应用
  - 继续把 activated cache 的预算和 eviction 变成真正 per-layer batch 调度，而不是当前的 opportunistic hotness 裁剪
  - 为真实 GPU<->PIM resident migration 预留单独执行通道

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
