---
updated: 2026-04-17 13:45
updated: 2026-04-17 14:05
updated: 2026-04-17 14:20
updated: 2026-04-17 14:35
updated: 2026-04-17 14:50
updated: 2026-04-17 16:20
updated: 2026-04-17 16:45
updated: 2026-04-17 01:53
updated: 2026-04-17 02:01
updated: 2026-04-17 02:05
updated: 2026-04-17 18:40
updated: 2026-04-17 19:10
updated: 2026-04-17 19:35
updated: 2026-04-17 20:00
updated: 2026-04-17 21:05
updated: 2026-04-17 21:35
updated: 2026-04-17 22:20
updated: 2026-04-17 22:45
updated: 2026-04-17 23:05
updated: 2026-04-17 23:25
updated: 2026-04-17 03:38
updated: 2026-04-17 03:50
updated: 2026-04-17 03:53
updated: 2026-04-17 05:18
updated: 2026-04-17 05:27
updated: 2026-04-17 05:42
updated: 2026-04-17 05:58
updated: 2026-04-19 00:50
updated: 2026-04-19 02:10
updated: 2026-04-19 12:40
---

# 🔥 当前工作焦点

## 正在进行

- [x] 建立 CPU-only、非 PIM 的最小可运行路径
- [x] 在真实权重上验证 CPU 路径并补齐 benchmark 入口
- [x] 在宿主机上打通 `cuda_cpu_offload` 并拿到真实 Qwen3 benchmark
- [x] 在真实 UPMEM 硬件上跑通多 rank PIM microbenchmark
- [x] 真实 `Qwen/Qwen3-30B-A3B-GPTQ-Int4` 权重的 operator-only `gate/up/down` projection 已在真机跑通，并确认当前 `W4A32 + soft-float dequant` 路径显著慢于 CPU grouped baseline
- [x] quantized PIM operator 已补齐 `transfer_only / unpack_only / dequant_only / full` 分项 profiling，确认主瓶颈在 DPU 侧 unpack/dequant 计算本体，不在 host 传输
- [x] quantized DPU kernel 已接入 block-level dequant LUT，真实 GPTQ case 下 PIM operator 时间明显下降，但仍未超过 CPU
- [x] 已新增一个最小整数化原型 `kernel_mode=4`：host 端按 batch 将输入量化为 int8、按组生成 int16 dequant LUT，DPU 侧执行 `int8 x int16 -> int32` accumulate，再在 host 侧按 batch 回标定输出
- [x] `kernel_mode=4` 现已恢复为默认验证的稳定整数化主线；真实 Qwen3 GPTQ `batch=1` 下，`gate` 已达到约 `1.30x` CPU grouped，`down` 已达到约 `2.54x` CPU grouped
- [x] `kernel_mode=4` 的真实 rank sweet spot 已经摸清：`gate batch=1` 在 `8~32 ranks` 都能超过 CPU，`down batch=1` 在 `1~8 ranks` 都能超过 CPU，其中 `4 ranks` 最优；但 `batch=8` 时两类 shape 都会退到 `0.56x~0.63x` CPU grouped，优势无法保持
- [x] `kernel_mode=4` 已补齐 DPU batch-tile 数据流和权重缓存修正：quantized runtime 不再错误复用“同 shape 不同权重”的旧 resident weights，real-DPU 回归新增 `batch=4` 覆盖；当前 `down batch=4` 已接近 CPU grouped 持平，但 `gate batch=4` 仍未翻盘
- [x] 已对 `kernel_mode=4` 做真实 `FIXED_BATCH_TILE=1/2/4/8` sweep：`down batch=4` 因 shape-gated fallback 基本不受影响，而 `gate batch=4/8` 没有出现稳定的 tile sweet spot；当前 `tile=4/8` 最多只带来小幅波动，不足以把 `gate batch>1` 稳定拉回 CPU grouped 之上
- [ ] 继续验证并收敛整数化 quantized kernel，重点比较 `soft-float full` 与 `int8 fixed-point` 的速度/误差权衡，并观察 `batch>1` 时优势是否能保持
- [ ] 评估更激进的 block-aware runtime LUT 路径；当前 `kernel_mode=5` 因 MRAM 容量限制暂不适用于真实 Qwen3 gate/down 形状
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
- [x] prepared tier controller 已显式区分静态 prepared budget 与 `effective_prepared_cache_limit`，当重平衡压力持续偏高且 activation stage bonus 偏低时，会临时收缩 prepared tier 的有效预算
- [x] scheduler summary / profile sweep 已补充 `effective_prepared_cache_limit`、`effective_prepared_cache_utilization` 与 `prepared_cache_rebalance_pressure_avg`，prepared tier 的预算收缩行为已可观测
- [x] prepared tier controller 已新增 `prepared_cache_budget_backoff`，可按重平衡压力分级收缩 effective prepared budget，并在 `cold_promotion_penalty` 偏高时撤销 backoff
- [x] prepared-tier controller 现在会把 `prepared_cache_budget_backoff` 反馈到 `adaptive_activation_limit / adaptive_prebuild_limit`，prepared budget 收缩与候选准备 aggressiveness 已形成更一致的控制面
- [x] prepared-cache rebalance pressure 现已按 `pipeline_ticks` 归一，而不是只按静态 prepared budget 归一，长运行时下的回退压力信号更稳定
- [x] prepared-tier controller 已补齐 `prepared_cache_rebalance_pressure_step` 与 `prepared_cache_rebalance_pressure_ema`，prepared budget backoff 可同时参考累计、step 和 EMA 三类压力信号
- [x] `reset_offload_diagnostics()` 现已同步重置 prepared-tier 的 pressure/EMA 状态，单次 benchmark run 的 prepared controller 信号不再混入历史步数
- [x] migration queue 已接入 lifecycle 状态：`queued / prefetching / ready / deferred / applied`
- [x] decode 在 `decode_require_prefetch_ready` 模式下已改成 ready-only 消费，不再先 drain 全队列再回退
- [x] materialization manager 已支持后台 prefetch completion 轮询，ready 状态可在进入层前被主动刷新
- [x] migration manager 已支持 ready-only drain，decode 不再手写 ready subset 过滤逻辑
- [x] materialization 预取完成已接入 completion queue，ready 轮询不再扫描全部 future
- [x] ready 轮询已从 `HybridMoE.forward()` 入口上移到 engine/model 层，避免每层重复轮询
- [x] `SimpleEngine` 已抽出统一 offload refresh hook，prefill full/chunked 与 decode 共用同一入口
- [x] `activated -> applied` 已抽出显式 apply candidate queue；后台 worker 现在会先把 `ACTIVATED` expert 入队，前台再做 ready promotion commit
- [x] apply queue 现已具备独立预算、hotness-aware victim 选择和 queue 级诊断，后半段 resident commit 开始从“ opportunistic apply ”推进成“受控 staged commit”
- [x] background pipeline 现在只负责 `activated -> apply queue enqueue`；真正 resident commit 会留在后续前台/共享 staged commit 阶段，后台/前台边界更清晰
- [x] apply queue 现在也已接入 pressure / EMA / backoff 信号，并开始反向约束 activation / prebuild / prefetch aggressiveness，prepared-tier controller 已能感知 resident commit 拥塞
- [x] apply queue commit 现在已补上独立 batch metrics 和 adaptive commit limit，后台/前台 staged commit 的批次大小开始进入可调、可观测范围
- [x] apply queue 现已拆成 `apply_candidate_queue -> apply_commit_queue -> resident set` 两段 staged commit，后台路径只负责把候选推进到 commit queue，真正 resident commit 只消费 staged commit queue
- [x] apply commit queue 已补齐独立 `size / limit / utilization / enqueued / pruned / background_enqueued` 诊断，后半段 commit 拥塞开始能与 apply candidate queue 区分观测
- [x] apply commit queue 现已具备独立 `evictions` 与 hotness-aware victim policy，后半段 staged commit 现在也能单独衡量 budget 压力与回退行为
- [x] apply commit queue 现已新增独立 `pressure / step / ema / budget_backoff` 控制信号，并开始反向约束 activation / prebuild / prefetch aggressiveness
- [x] background runtime 现已单独累计 `offload_background_apply_commit_queue_enqueued_total`，可以区分后台 worker 往 staged commit queue 推进了多少 resident commit 候选
- [x] apply commit queue 现已进一步拆成 `apply_commit_queue -> apply_commit_batch_queue -> resident set` 三段，后台 worker 可先把 staged commit 候选推进到 batch queue，再由 resident commit 消费
- [x] apply commit batch queue 现已补齐独立 `size / limit / utilization / enqueued / pruned / evictions / background_enqueued` 诊断与 runtime totals，后半段 commit buffer 的推进和拥塞可单独观测
- [x] resident commit 现已支持从 `apply_commit_ready_cache` 做 batched module commit，再逐 expert 完成 residency/lifecycle finalize，后半段提交流水线已从“纯逐 expert 注入”推进成“两段式 batch commit”
- [x] apply commit batch queue 现已补齐独立 `pressure / step / ema / budget_backoff` 控制信号，并开始反向约束 activation / prebuild / prefetch aggressiveness，resident commit 的最终 batch buffer 也进入了闭环控制
- [x] apply commit batch queue 的 `pressure / step / ema / budget_backoff` 现已贯通到 scheduler summary / profile sweep，并开始参与不同 profile 的对比排序
- [x] resident commit 现已进一步拆成 `resident_commit_batch_queue -> resident_commit_finalize_queue -> resident set`，后台路径可先把 final resident batches 预推进到 finalize queue，前台/后台再消费 preexisting finalize batches
- [x] resident commit 现已进一步拆成 `resident_commit_batch_queue -> resident_commit_finalize_queue -> resident_commit_ready_cache -> resident set`，后台路径可先把 preexisting finalize batches 解析成 ready commit cache，前台/后台再消费 ready cache 中已有的 final batches
- [x] resident commit 现已进一步拆成 `resident_commit_batch_queue -> resident_commit_finalize_queue -> resident_commit_ready_cache -> resident_commit_apply_queue -> resident set`，后台路径可先把 ready resident batches 预推进到 apply queue，前台/后台再消费 tick 开始前已存在的 apply batches
- [x] resident commit 现已进一步拆成 `resident_commit_batch_queue -> resident_commit_finalize_queue -> resident_commit_ready_cache -> resident_commit_apply_queue -> resident_commit_finalize_ready_queue -> resident set`，后台路径可先把 preexisting apply batches 再预推进到 finalize-ready queue，前台/后台再消费 tick 开始前已存在的 finalize-ready batches
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
- [x] activated/warm cache 的 eviction 现在会同步回退 migration lifecycle：`ACTIVATED -> WARMED`、`WARMED -> READY`，cache 层次与状态机重新对齐
- [x] migration diagnostics 现在会显式统计 `activation_eviction_regressions` 和 `warm_eviction_regressions`，后续 benchmark 可直接看到缓存回退压力
- [x] profile sweep 现在会把 eviction regression 压力带进 comparison table / best-by-metric，能直接按“缓存回退更少”筛 profile
- [x] warm cache eviction 现在会按 lifecycle 优先级和 hotness 选 victim，不再只按 FIFO/LRU 式最旧对象回退
- [x] activated cache eviction 现在也会按 lifecycle 优先级和 hotness 选 victim，device-side 激活预算开始与 warm cache 保持一致的热点保留语义
- [x] warm/activated 两级准备缓存现在支持统一 prepared-cache 预算，warm cache 的有效容量会随 activated cache 占用动态收缩
- [x] prepared-cache 预算现在会做跨层级统一重平衡，activated 和 warm candidates 会在同一个保留策略下竞争 prepared slots
- [x] prepared-cache 预算现在已打通到 `LLM/example/benchmark` 入口，并在 scheduler summary 中暴露 utilization / effective warm limit 指标
- [x] profile sweep 现在也会显式汇总 prepared-cache 指标，prepared tier 开始进入自动比较与 best-by-metric 排序
- [x] prepared-cache 重平衡事件现在也会进入 scheduler summary/profile sweep，可直接区分 warm/activated 哪一层在挤压 prepared budget
- [x] prepared-cache retention policy 现在开始带最小自适应倾向，activated stage bonus 会随重平衡方向调整
- [x] prepared tier 现在开始用最小自适应限额驱动 activation/prebuild 候选数，prepared-cache 压力会反向收缩 activation/prebuild aggressiveness
- [x] cold promotion 现在会反馈回 prepared tier controller，冷路径比例偏高时会抬高 adaptive activation/prebuild limit
- [x] prepared tier controller 现在也开始直接约束 prefetch aggressiveness，新增 `adaptive_prefetch_pending_limit / adaptive_prefetch_candidate_budget`，prepared 压力与 cold penalty 会同时影响 pending promotion 预取和候选预取规模
- [x] prepared-cache budget 现在已从 profile heuristic 打通到诊断输出，`scheduler_profile_summary` 和 `LLM.get_offload_diagnostics()` 都会显式暴露 `prepared_cache_budget(_heuristic)`，便于 profile sweep 对照“控制器动作”和“静态预算基线”
- [x] prepared-cache budget heuristic 现在已开始随 scheduler profile 变化：`overlap_safe` 和 `eager` 会在 baseline 预算之上进一步抬高 prepared budget，为 strict ready-only 和更激进预热提供更稳定的 prepared tier
- [x] scheduler profile 现在也开始显式控制 prepared-tier aggressiveness：`baseline / overlap_safe / eager` 会分别对应不同的 `prepared_controller_aggressiveness`，直接影响 activation/prebuild/prefetch 三段的推进力度
- [x] materialization manager 已新增后台 resolve worker，`future.result() + cache store` 会在后台完成，前台 `poll_ready()` 基本只负责消费轻量 ready 通知
- [x] promotion batch 已收口成“先 batch resolve source/module，再 batch apply resident set”的骨架，后续继续做真正 batched apply 时边界更清晰
- [x] materialization manager 现已支持后台 ready callback，resolved expert 可直接把 migration lifecycle 推到 `READY`，前台 refresh hook 只保留兜底 drain 语义
- [x] migration manager 已补上内部锁，后台 ready callback 现在可以安全推进 lifecycle，不再默认假设所有迁移状态都只在前台单线程修改
- [x] engine/model 已补上 background offload tick hook，后台 ready callback 可以在每个 token-step 的主 refresh 前先推进一轮，不再只靠同一个前台函数串行承担全部工作
- [x] scheduler summary 现已汇总 `offload_background_ticks / offload_pipeline_background_ready_callback_total`，background tick 路径已进入 benchmark 可观测面
- [x] `LLM.reset_offload_diagnostics()` 现已同步清零 runtime 级 background tick 计数，单次 run 的 background offload 指标不再混入历史步数
- [x] background offload tick 现在不只推进 ready callback，也会提前推进一部分 `READY -> WARMED/ACTIVATED`，后台路径已开始覆盖 prepared tier 的前半段
- [x] 新增最小后台 offload worker 骨架，后台线程可周期性推进 `background_tick_offload_state()`，decode 主线程不再是唯一能推动 ready/warm/activation 前半段前进的入口
- [x] background worker 诊断现已并入 `offload_refresh_diagnostics()`，并支持 per-run reset，benchmark 可独立观察后台 worker tick 与 work tick 行为
- [x] `SimpleEngine/LLM` 现已接入 background worker 生命周期：生成前启动后台 worker，生成后停止并清理，后台推进不再只是模型内部的“可选对象”
- [x] background worker 现已改成显式启动，模型构造时默认不自动起线程，后台迁移执行器的生命周期边界更清晰
- [x] scheduler summary / profile sweep 现已纳入 background worker 指标，可直接比较后台 worker 的 tick、work ratio 与 decode 吞吐的关系
- [x] `LLM.get_offload_diagnostics()` 现已显式暴露 `prepared_cache_budget_heuristic`，profile 的静态 prepared 预算基线和 runtime controller 的实际行为已统一进入诊断面
- [x] 当前有 background worker 运行时，`SimpleEngine` 已不再手动重复调用 `background_tick_offload_state()`，前台 refresh 和后台 worker 的职责边界更清晰
- [x] background worker 的 runtime 指标现已细化到 `background_work_items_total` 与 `background_activation_applied_total`，后台线程推进了多少准备/提交工作已进入 summary 与 profile sweep
- [x] `HybridMoE` 现已引入内部 pipeline 锁，background worker 与前台 refresh/forward 对 prepared tier、migration lifecycle 和 resident set 的共享状态访问开始串行化
- [x] `benchmark_inference.py` 与 `example.py` 现已显式支持 `--enable-background-offload-worker` 与 `--background-offload-poll-interval-seconds`，真实 benchmark 路径可以直接接通后台 offload worker
- [x] benchmark 的 `run_single_generation()` 现已在单次生成前启动 background offload worker、结束后停止，真实 `cuda_pim` benchmark 开始能驱动后台 `prefetching -> ready -> warmed -> activated` 路径
- [x] apply commit 路径现已补上 `apply_commit_ready_cache`，background tick 可以先把 staged commit queue 中的 expert 解析成可直接 commit 的 ready entry，再由后续 resident commit 消费
- [x] background pipeline 现已允许“同一 tick 内新入队的 apply candidate 进入 apply commit queue 并完成 ready resolve”，但 resident set commit 仍只消费 tick 开始前已存在的 staged commit 候选，后台 enqueue / resolve / commit 边界更清晰
- [x] `_apply_promotion_batch()` 现已先批量将 ready-entry 中的 module 提交到 `gpu_experts` / `gpu_experts_mask`，再统一写回 residency 与 lifecycle，resident commit 的最后一段开始具备真正的 per-layer batch commit 语义
- [x] resident commit 现已拆分 staged resolve 与 final batch commit 的独立自适应预算，`apply_commit_batch_queue -> resident set` 拥有单独的 batch-limit 控制面
- [x] `apply_commit_batch_queue` 现已从逐 expert staged buffer 收敛成按 batch 组织的 commit buffer，resident commit 的最后一段开始以 batch 作为一等调度对象
- [x] resident commit 现已支持分离的 staged-queue budget 与 final batch-queue budget，后台/前台都能按 batch 粒度推进 commit buffer
- [x] resident commit 的 background 路径现已显式区分 `batch queue enqueue` 与 `batch prefinalize`，后台 worker 对最终 commit buffer 的推进开始可单独量化
- [x] resident commit 的 background 路径现在只会消费 tick 开始前已存在的 `apply_commit_batch_queue` batch，后台 enqueue 与 final commit prefinalize 已显式分离，避免同一 tick 内新入队 batch 被立即提交

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
- 当前虽然 benchmark 路径已接通 background worker，但还没重跑一轮开启 worker 的真实 `cuda_pim` benchmark，确认 staged queues / resident commit buffers 是否真正开始填充
- 当前 batch apply 的来源构成已经可见，但还没把这些来源构成真正用于 batch policy，例如按 `activated` 命中率自适应调整 activation/prebuild 预算
- 当前 cache eviction 已和 lifecycle 对齐，但还没有把这些回退事件显式汇总进 scheduler summary，后续 benchmark 仍难直接看出“冷热回退”压力
- 当前回退事件已经能进 scheduler summary，但 profile sweep 还没把这些 regression 指标纳入排序或 best-by-metric 比较
- 当前 regression 指标已经能参与 profile 对比，但 benchmark README 和宿主机真实 sweep 结果里还没有用这两个指标做解读
- 当前 activated cache 已按 hotness 裁剪、warm cache 也已按 hotness 选 victim，但两级 cache 之间还没有统一的全局 budget policy
- 当前 activated cache 虽已改成 hotness/lifecycle-aware victim 选择，但 warm/activated 两级 cache 仍是分开裁剪，还没有统一的 per-layer cache budget policy
- 当前 warm/activated 两级 cache 已开始共享 prepared-cache 总预算，但 budget 还没有和最近几步的回退压力、source mix 联动成自适应策略
- 当前 prepared-cache 已开始跨 warm/activated 统一重平衡，但 victim policy 还没有直接利用 recent source mix / eviction regression 做自适应预算调整
- 当前 prepared-cache 已可配置和观测，但 budget 仍是静态数值，还没有根据 source mix / regression pressure 自适应调整
- 当前 prepared-cache 指标已进入 profile sweep，但还没有真正反馈回 scheduler policy 去动态调整 prepared budget
- 当前 prepared-cache 重平衡压力已可观测，但 scheduler 还没有利用这些指标动态调节 prepared budget 或 prebuild/activation aggressiveness
- 当前 prepared-cache 已有最小自适应 stage bonus，但还没有真正把 recent source mix / regression pressure 接进完整的 budget controller
- 当前 adaptive activation/prebuild limit 已接上 prepared-cache 压力与 stage bonus，但还没有把它和 profile/scheduler 参数形成可学习或可调的 controller
- 当前 prepared tier 已开始同时感知 prepared-cache 压力和 cold-promotion 惩罚，但还没有把这些信号统一成稳定的多步 controller
- 当前 runtime batch totals 已是按 tick 增量统计，但还没有接入真实宿主机 benchmark 结果做长期趋势归档
- 当前 strict ready-only 语义已经覆盖 resident staging，但 `prefetching -> ready` 仍然依赖前台 refresh，而不是真 completion event 驱动
- 当前 materialization ready 的重活已下沉到后台 resolver，但 migration lifecycle 的 `READY` 标记仍通过前台 refresh hook 对齐，尚未变成真正后台 migration worker
- 当前 materialization ready 的 `READY` 标记已可由后台 callback 推进，但回调仍是单 expert 触发，尚未形成独立 migration worker / completion batching
- 当前后台 ready callback 已能安全推进 lifecycle，但 `READY -> WARMED/ACTIVATED/APPLIED` 仍然主要由前台 token-step runtime 驱动，后台 worker 还没有接管完整后半段流水线
- 当前 background tick 已经独立于主 refresh hook，但仍是 token-step 入口触发，不是真正常驻后台线程/事件循环
- 当前 background tick 已经进入 summary/benchmark 可观测面，但仍只覆盖 `prefetching -> ready`，尚未接管 `ready -> warmed/activated` 或 resident apply
- 当前 background tick 指标已支持 per-run reset，但 benchmark 还未把这组指标纳入 profile sweep 排序与 best-by-metric 对比
- 当前 background tick 已开始推进 warm/activation 准备，但真正的 `activated -> applied` 仍完全留在前台主流水线，后台路径还没触及 resident set commit
- 当前 background worker 已能推进一部分 `activated -> applied`，但目前仍是 opportunistic background apply，不是真正独立的 resident commit queue，也还不是底层 batched apply
- 当前后台路径已经从 opportunistic activated apply 推进到显式 apply queue，但 `apply queue -> resident set` 仍是前台/后台共享的 staged commit，不是底层 fully batched resident injection
- 当前 apply queue 已有自身预算和 victim policy，但 `apply queue -> resident set` 仍是共享 staged commit，不是独立 batched resident commit queue
- 当前 apply queue 压力已接回 prepared-tier controller，但 resident commit 仍是 staged shared commit；下一步仍需把 `apply queue -> resident set` 收成真正的 per-layer batch commit
- 当前 apply queue 现在虽然已经拆成 staged `apply_commit_queue`，但 resident 注入仍是 commit queue 内逐 expert 提交，还不是真正底层 batch resident commit
- 当前 apply commit queue 虽已具备独立 budget / eviction / utilization 信号，但 resident set 注入仍是 commit queue 内逐 expert commit，尚未变成真正的 per-layer batch resident commit
- 当前 apply commit queue 现在虽然也已接入独立 pressure controller，但 commit queue 与 resident set 之间仍是逐 expert staged commit，尚未形成真正底层 batched resident commit
- 当前后台 worker 虽然已经能稳定推进 `apply_commit_queue enqueue`，但 `apply_commit_queue -> resident set` 仍是共享 staged commit，而不是独立的 per-layer batch commit worker
- 当前 apply queue commit 已有独立 batch 指标和 adaptive commit limit，但 resident 注入本身仍是 batch 内逐 expert 提交；下一步仍要继续压成真正底层 batched resident commit
- 当前虽然已有 `HybridMoE` 级 pipeline 锁，但锁粒度仍偏粗；background apply 还只是“线程安全地 opportunistic apply”，不是独立 commit queue，也还没有真正 batched resident commit
- 当前 resident commit 已拆出独立 batch-limit，但后台 worker 仍主要负责 staged batch 准备，`apply_commit_batch_queue -> resident set` 还没有变成真正后台 batch commit worker
- 当前 `apply_commit_batch_queue` 已经是 batch 级 commit buffer，但后台 worker 还没有直接消费 batch buffer 做真正后台 batch commit，前台仍承担最后的 finalize
- 当前 resident commit 虽已进入 batch 粒度控制，但后台 worker 仍主要推进 batch buffer staging，真正的后台 batch commit worker 仍未形成
- 当前后台 worker 已能量化 batch buffer prefinalize，但真正的 resident set commit 仍未下沉成后台 batch commit worker，前台仍承担最终 finalize
- 当前后台 worker 虽已把 `batch queue enqueue` 与 `batch prefinalize` 拆开，但真正的 resident set final commit 仍未成为独立后台 batch commit worker，前台 decode 仍负责最后 finalize
- 当前 promotion batch 虽然已先统一 resolve source/module，再进入 apply，但 resident set 注入仍是 batch 内逐 expert 提交，不是真正底层 batched apply
- 当前 benchmark 已能稳定观察单次 run 的 pipeline 行为，但还缺少 profile sweep 结果表层面的自动对比汇总
- 当前 prebuild 已做候选裁剪，但 warm cache 还没有独立的“低优先级淘汰”策略，仍然主要依赖容量上限和 LRU
- 当前已经能量化 promotion source，但 benchmark 还缺少跨 profile 的自动排名/对比表
- 当前 prepared tier 已开始有“弱自适应 effective budget”语义，但还没有形成真正的 per-layer prepared budget controller；下一步应继续把 `cold_promotion_penalty`、rebalance pressure 与 prepared budget 收缩合成更完整的闭环
- 当前 prepared budget 已支持多级 backoff，但 activation/prebuild aggressiveness 仍与 budget controller 只做松耦合；下一步可继续把 backoff 直接反馈到 prebuild/activation batch aggressiveness，而不只是影响 effective prepared limit
- 当前 prepared budget backoff 已开始直接约束 activation/prebuild limits，但仍是 per-layer 局部 heuristic；下一步应继续把这组 controller 信号接到 profile 策略层，开始做更系统的 auto-tuning
- 当前 rebalance pressure 已按 step 归一，但仍是简单平均信号；下一步可以继续探索窗口化或 EMA 形式，让 controller 更贴近真实 decode 负载变化
- 当前 EMA 信号已补齐并接入 summary/profile sweep，但尚未做更系统的 profile 级 auto-tuning；下一步可开始围绕 `pressure_step / pressure_ema / cold_penalty` 联动 profile 策略
- 当前 prepared controller 的压力信号已具备 per-run 可复现性；下一步更适合开始做 sweep 结果驱动的 profile 调优，而不只是继续补本地状态变量
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
- resident commit 的最后一段现在新增了 `resident_commit_batch_queue`，后半段链路已明确成：
  `apply_candidate_queue -> apply_commit_queue -> apply_commit_batch_queue -> resident_commit_batch_queue -> resident set`
- `resident_commit_batch_queue` 现在已经接入 `HybridMoE.diagnostics()`、runtime background tick 汇总、`LLM.reset_offload_diagnostics()` 和 scheduler summary。
- 这意味着系统已经开始把最终 resident commit 从“apply batch buffer”再推进到“final resident commit buffer”，更接近真正后台 batch commit worker 的结构。
- background tick 现在还会区分：
  - 本轮开始前已经存在的 `resident_commit_batch_queue` batch：可进入本轮 background apply
  - 本轮新推进到 `resident_commit_batch_queue` 的 batch：只记为 `prefinalized`，留到下一轮再 commit
- 这让 resident commit 的最后一段也开始具备稳定的流水线边界，而不是在同一 tick 里边入队边消费。
- 新增 W4A32/GPTQ Int4 算子级实验路径：
- W4A32/GPTQ quantized runtime 现已补充分项 timing：host->DPU 输入下发、同步 launch/执行、DPU->host 结果回传与 runtime 总耗时都可在 `benchmark_quant_matvec.py` 中直接观察。
- quantized PIM operator benchmark 现已拆分权重加载阶段（qweight/scales）与稳态运行阶段（input transfer / launch / output transfer）；真实 Qwen3 GPTQ `gate/down` case 显示稳态时间约 89%~98% 落在 `launch_seconds_avg`，主瓶颈明确在 DPU kernel 执行。
- 新增 transfer-only kernel mode 后，真实 GPTQ `gate/down` case 的 breakdown 显示：去掉 DPU 计算后，纯输入/输出搬运仅约 `0.69ms~3.18ms`，而完整执行约 `20.3ms~35.2ms`；计算核本体占总时间约 90% 左右。
- 已完成真实 GPTQ `gate/down` 的 rank 与 batch breakdown sweep：无论 rank 还是 batch 怎么调，`estimated_compute_seconds` 都是主导项；batch 增长时主要是计算核近似线性变慢，纯传输只小幅增长。
- 已补充 quantized kernel mode 剖析：`transfer_only / unpack_only / dequant_only / full`。真实 GPTQ `gate/down` 表明，`unpack_only` 与 `dequant_only` 都显著增加 launch 时间，而从 `dequant_only` 到 `full` 的额外增量相对更小，当前瓶颈更偏向 nibble unpack + 反量化，而不是最终乘加本身。
- 已验证 block-level dequant LUT 优化：对真实 GPTQ `gate/down`，PIM operator-only 速度显著提升，`gate@rank24` 约从 `36ms` 降到 `23ms`，`down@rank2` 约从 `20ms` 降到 `11.8ms`；但仍未超过 CPU。
- 已验证一次 4-row tiled quantized DPU kernel 尝试在真实 `gate/down` case 上反而变慢，当前稳定实现仍保留原始 row-pair kernel；后续优化应优先避免增加 WRAM 占用和 inner-loop 分支。
- 已完成 quantized kernel 参数 sweep：当前真机上 `TASKLETS=8, BLOCK_FLOATS=64` 对真实 GPTQ `gate/down` 都略优于默认 `TASKLETS=16`，但提升仍很小，结论不变。
- 真实 Qwen3 GPTQ operator-only 剖析已表明：当前 PIM 路径的时间绝大部分落在 `launch_seconds_avg`，输入/输出传输只占很小比例；瓶颈已经明确偏向 DPU 侧 kernel 执行而不是 host 传输。
  - `WeightLoader` 现已支持读取 Qwen3 GPTQ expert linear 的 `qweight / scales / g_idx`
  - `GPTQLinearWeight` 统一承载最小可用的 4-bit 对称量化权重表示
  - `quantized_ops.py` 提供 CPU W4A32 operator-only matvec 与 synthetic quantizer
  - `pim_quantized_runtime.py` 与新的 quantized DPU kernel / host bridge 提供“量化权重常驻加载一次、重复执行 matvec”的 PIM runtime
- 新增 `benchmarks/benchmark_quant_matvec.py`：
  - 支持 synthetic W4A32 benchmark
  - 支持真实 GPTQ expert projection benchmark
  - 直接对比 CPU vs PIM 的单算子矩阵向量乘，不经过完整 decode 链路
- synthetic W4A32 operator-only benchmark 已在真实硬件上跑通：
  - `input_dim=2048`
  - `output_dim=768`
  - `group_size=128`
  - `rank_count=4`
  - CPU grouped avg 约 `8.42 ms`
  - CPU dense avg 约 `4.06 ms`
  - PIM avg 约 `52.27 ms`
  - `max_abs_error ≈ 1.68e-4`
  当前 synthetic case 下 PIM 仍明显慢于 CPU grouped / dense 两条基线，说明仅换成 W4A32 并不足以抵消当前 DPU 计算和 host orchestration 成本。
- `Qwen/Qwen3-30B-A3B-GPTQ-Int4` 已开始拉取：
  - tokenizer / config / quantize_config 已到本地
  - `model.safetensors` 仍在下载中
  - 下一步要在真实 GPTQ expert projection 上验证 tensor layout 与 operator-only CPU/PIM 对比
