---
updated: 2026-04-14
tags: [changelog]
---

# 📝 变更日志

## 2026-04-19

- 新增 GPTQ/W4A32 单算子实验路径：
  - `nano_ktrans/kernels/weight_loader.py` 新增 `GPTQLinearWeight`、Qwen3 GPTQ expert linear 读取和最小 dequant 支持
  - `nano_ktrans/kernels/quantized_ops.py` 新增 CPU W4A32 matvec 和 synthetic quantizer
  - `nano_ktrans/kernels/pim_quantized_runtime.py`、`pim_native/dpu_quantized_kernel.c`、`pim_native/host_quantized_bridge.c` 新增 PIM quantized runtime，支持量化权重常驻加载后重复执行 matvec
  - `benchmarks/benchmark_quant_matvec.py` 新增 operator-only benchmark，可直接比较 CPU/PIM 的量化矩阵向量乘
- 已在真实机器上跑通 synthetic W4A32 算子 benchmark：
  - shape=`2048 -> 768`
  - `group_size=128`
  - `rank_count=4`
  - CPU grouped avg ≈ `8.42 ms`
  - CPU dense avg ≈ `4.06 ms`
  - PIM avg ≈ `52.27 ms`
  - `max_abs_error ≈ 1.68e-4`
  当前 synthetic W4A32 operator-only 路径下，PIM 明显慢于 CPU grouped / dense 两条基线。
- 已开始拉取 `Qwen/Qwen3-30B-A3B-GPTQ-Int4`：
  - `config.json`、`quantize_config.json` 已就绪
  - `model.safetensors` 仍未完成下载
  - 后续需要在真实 GPTQ 权重上验证 tensor layout 和 operator-only benchmark

## 2026-04-17

### 2026-04-17 03:38 - Final staged resident commit queue

- 将后半段 resident commit 继续拆成 `apply_commit_queue -> apply_commit_batch_queue -> resident set` 三段式 staged commit。
- `HybridMoE` 新增 `apply_commit_batch_queue`，后台 pipeline 现在会先把 ready 的 staged commit 候选推进到 batch queue，再由 resident commit 消费。
- `apply_commit_batch_queue` 已补齐独立 `size / limit / utilization / enqueued / pruned / evictions / background_enqueued` 诊断，并接入 runtime totals 与 scheduler summary。
- `_apply_promotion_batch()` 现已支持 batched module commit：先按批量统一更新 `gpu_experts` 与 `gpu_experts_mask`，再逐 expert 完成 residency / lifecycle finalize。
- 重新跑过 `./.venv/bin/python -m pytest -q tests/test_core.py tests/test_pim_runtime.py`，结果 `121 passed, 1 warning`。

### 2026-04-17 03:50 - Close the loop around commit batch pressure

- `apply_commit_batch_queue` 新增独立 `pressure / step / ema / budget_backoff` 控制信号。
- prepared-tier controller 现在会同时感知 `apply_commit_batch_queue` 的拥塞，并反向约束 `adaptive_activation_limit / adaptive_prebuild_limit / adaptive_prefetch_*`。
- scheduler summary / profile sweep 已补齐 commit-batch queue 的压力指标，便于直接比较不同 profile 在最终 resident commit buffer 上的拥塞情况。
- 重新跑过 `./.venv/bin/python -m pytest -q tests/test_core.py tests/test_pim_runtime.py`，结果 `122 passed, 1 warning`。

### 2026-04-17 03:53 - Surface commit-batch pressure in sweep comparisons

- `apply_commit_batch_queue_pressure_avg / _ema_avg / _budget_backoff_avg` 已接入 profile sweep `comparison_table` 与 `best_by_metric`。
- summary 聚合测试和 prepared-tier backoff 测试已补齐，最终 commit buffer 的拥塞行为现在可直接参与 profile 排序。
- 重新跑过 `./.venv/bin/python -m pytest -q tests/test_core.py tests/test_pim_runtime.py`，结果 `122 passed, 1 warning`。

- <!-- updated: 2026-04-17 21:05 --> **[apply-commit-ready-cache]** `HybridMoE` 新增 `apply_commit_ready_cache`，`apply_commit_queue` 中的 staged commit 候选现在可以先在 background/foreground 路径上解析成可直接 resident commit 的 ready entry，`_apply_promotion_batch()` 也支持消费预解析 batch。
- <!-- updated: 2026-04-17 21:05 --> **[background-commit-staging]** background pipeline 现在允许“同一 tick 新入队的 apply candidate -> apply commit queue -> ready resolve”连续推进，但 resident commit 仍只消费 tick 开始前已存在的 staged commit 候选，避免 background tick 在同一轮里同时 enqueue 和 commit 同一 expert。
- <!-- updated: 2026-04-17 21:05 --> **[diagnostics]** 新增 `apply_commit_ready_cache_size / hits / stores / pruned / background_apply_commit_resolved`，用于区分 staged commit queue 的 ready resolve 命中与真正 resident commit 消费。
- <!-- updated: 2026-04-17 21:05 --> **[tests]** 新增 background staged commit resolve 路径覆盖，并更新 background apply queue 语义测试；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `119 passed, 1 warning`。
- <!-- updated: 2026-04-17 21:35 --> **[batched-resident-commit]** `_apply_promotion_batch()` 现已先批量把 ready-entry 中的 module 注入 `gpu_experts` 并统一更新 `gpu_experts_mask`，再逐 expert 写回 residency/history/lifecycle，resident commit 的最后一段已从纯逐 expert 注入推进到真正的 per-layer batched commit 语义。
- <!-- updated: 2026-04-17 21:35 --> **[tests]** 新增 batched resident commit 覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `120 passed, 1 warning`。

## 2026-04-16

- <!-- updated: 2026-04-17 01:53 --> **[apply-queue]** `HybridMoE` 新增显式 `apply_candidate_queue`，将 `ACTIVATED` expert 的 resident commit 从 opportunistic background apply 收敛为 staged commit 路径；background pipeline 现先执行 `ACTIVATED -> apply queue enqueue`，再由前台/后台从 apply queue 提交到 GPU resident set。
- <!-- updated: 2026-04-17 01:53 --> **[apply-queue-metrics]** 新增诊断：
  - `apply_queue_size`
  - `apply_queue_enqueued`
  - `apply_queue_committed`
  - `apply_queue_pruned`
  - `background_apply_queue_enqueued`
  以及 runtime 级 `offload_background_apply_queue_enqueued_total`，可以单独量化后台将激活 expert 推入 apply queue 的工作量。
- <!-- updated: 2026-04-17 02:01 --> **[apply-queue-policy]** apply queue 现在新增独立 budget 与 hotness-aware victim 选择；当 `ACTIVATED` candidate 超过 queue 容量时，会优先保留更热 expert，并显式统计 `apply_queue_evictions`。
- <!-- updated: 2026-04-17 02:01 --> **[tests]** 新增 apply queue enqueue / rebalance / summary 覆盖，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `113 passed, 1 warning`。
- <!-- updated: 2026-04-17 02:05 --> **[background-apply-boundary]** background pipeline 现只负责把 `ACTIVATED` expert 推入 apply queue；真正的 resident commit 留在后续 staged commit 路径，不再在同一 background tick 里立即消费“刚入队”的 apply candidate。
- <!-- updated: 2026-04-17 02:05 --> **[tests]** 调整 background apply queue 语义测试，并新增 apply queue 利用率/摘要覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `114 passed, 1 warning`。
- <!-- updated: 2026-04-17 16:20 --> **[apply-queue-controller]** apply queue 现新增 `apply_queue_pressure / step / ema / budget_backoff`，并把这组信号接回 prepared-tier controller；当 resident commit 阶段持续拥塞时，系统会主动收缩 activation/prebuild/prefetch aggressiveness，避免 prepared tier 继续向后半段无效堆积。
- <!-- updated: 2026-04-17 16:20 --> **[apply-queue-summaries]** scheduler summary / profile sweep 新增 `apply_queue_pressure_avg / apply_queue_pressure_ema_avg / apply_queue_budget_backoff_avg`，可以直接比较不同 profile 在 apply queue 拥塞下的 controller 反应。
- <!-- updated: 2026-04-17 16:20 --> **[tests]** 新增 apply queue pressure/backoff 行为测试与 summary/profile sweep 聚合覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `115 passed, 1 warning`。
- <!-- updated: 2026-04-17 16:45 --> **[apply-queue-commit-batches]** apply queue 现新增 `apply_queue_commit_batches / experts` 和 `background_apply_commit_batches / experts`，可以直接量化后台和前台 staged commit 的批次大小，不再只能看 enqueue/committed 总数。
- <!-- updated: 2026-04-17 16:45 --> **[apply-queue-commit-limit]** `HybridMoE` 新增 `adaptive_apply_commit_limit()`，apply queue staged commit 开始根据 apply queue 压力、EMA、cold penalty 和 profile aggressiveness 自适应调整每批 commit 的大小。
- <!-- updated: 2026-04-17 16:45 --> **[tests]** 新增 apply queue commit batch 指标和 adaptive commit path 的覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `116 passed, 1 warning`。
- <!-- updated: 2026-04-17 18:40 --> **[apply-commit-queue]** `HybridMoE` 现在把 resident commit 进一步拆成 `apply_candidate_queue -> apply_commit_queue -> resident set`；后台路径先将已激活 expert 分批推进到 staged commit queue，再由前台/后台消费 commit queue 做最终 resident 注入。
- <!-- updated: 2026-04-17 18:40 --> **[apply-commit-queue-metrics]** scheduler summary 新增 `apply_commit_queue_size / limit / utilization / enqueued / pruned / background_apply_commit_queue_enqueued`，可以单独量化后半段 staged commit queue 的拥塞与推进情况。
- <!-- updated: 2026-04-17 18:40 --> **[tests]** 扩展 apply commit queue 的后台 enqueue、前台 commit 和 summary 覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `116 passed, 1 warning`。
- <!-- updated: 2026-04-17 19:10 --> **[apply-commit-queue-policy]** apply commit queue 现在新增独立 `evictions` 计数，并按 hotness/lifecycle 选 victim；apply queue 压力信号已开始同时感知 candidate queue 与 commit queue 的预算回退。
- <!-- updated: 2026-04-17 19:10 --> **[tests]** 新增 apply commit queue rebalance 行为与 summary 聚合覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `117 passed, 1 warning`。
- <!-- updated: 2026-04-17 19:35 --> **[apply-commit-queue-controller]** apply commit queue 现在新增 `pressure / step / ema / budget_backoff`，并把这组信号反向接回 prepared-tier controller；当 staged commit queue 持续拥塞时，activation/prebuild/prefetch aggressiveness 也会一并收缩。
- <!-- updated: 2026-04-17 19:35 --> **[sweep]** scheduler summary / profile sweep 现已补充 `apply_commit_queue_pressure_avg / ema_avg / budget_backoff_avg`，可以直接比较不同 profile 在后半段 staged commit 压力下的反应。
- <!-- updated: 2026-04-17 19:35 --> **[tests]** 新增 apply commit queue pressure/backoff 行为与 summary/profile sweep 聚合覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `118 passed, 1 warning`。
- <!-- updated: 2026-04-17 20:00 --> **[background-commit-queue-runtime]** `MigrationPipelineRuntime` 现在会单独累计 `offload_background_apply_commit_queue_enqueued_total`，background worker 往 staged commit queue 推进的 resident commit 候选量已能与 `apply_queue enqueue` 分开观测。
- <!-- updated: 2026-04-17 20:00 --> **[tests]** 补充 background apply commit queue runtime totals 与 summary 覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `118 passed, 1 warning`。

- <!-- updated: 2026-04-17 15:18 --> **[pipeline-lock]** `HybridMoE` 新增内部 `RLock`，background worker 与前台 `refresh/advance/forward/diagnostics` 对 prepared-tier cache、migration lifecycle 和 resident set 的共享状态访问开始串行化，降低后台推进接入真实生成后出现竞态的风险。
- <!-- updated: 2026-04-17 15:18 --> **[tests]** 并发边界收口后重新回归 `tests/test_core.py + tests/test_pim_runtime.py`，当前为 `111 passed, 1 warning`。
- <!-- updated: 2026-04-17 15:05 --> **[background-apply-metrics]** background offload runtime 现已显式累计 `offload_background_work_items_total` 与 `offload_background_activation_applied_total`；`MixtralModel.background_tick_offload_state()` 返回值也已从“ready callback 数”扩展为“后台 tick 总 work items”。
- <!-- updated: 2026-04-17 15:05 --> **[sweep]** scheduler summary / profile sweep 新增 `offload_background_work_items_avg` 与 `offload_background_activation_applied_total`，可以直接比较后台 worker 是否在稳定推进 prepared/apply 工作，而不只看 tick/work ratio。
- <!-- updated: 2026-04-17 15:05 --> **[tests]** 扩展 background runtime reset、summary 和 sweep 覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `111 passed, 1 warning`。

- <!-- updated: 2026-04-16 09:08 --> **[prepared-controller-reset]** `LLM.reset_offload_diagnostics()` 现在会同步清零 prepared-tier controller 的 `prepared_cache_rebalance_pressure_ema`、`prepared_cache_rebalance_events_last_tick` 和 `prepared_cache_rebalance_events_prev_total`，避免单次 benchmark run 混入前序 step 的 controller 状态。
- <!-- updated: 2026-04-16 09:08 --> **[tests]** 扩展 `reset_offload_diagnostics()` 覆盖，验证 prepared controller 的 EMA / step counters 也会被清零；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `88 passed, 1 warning`。
- <!-- updated: 2026-04-16 09:02 --> **[prepared-pressure-signals]** prepared-tier 现在同时输出三类压力信号：累计 `prepared_cache_rebalance_pressure`、单步 `prepared_cache_rebalance_pressure_step` 和平滑后的 `prepared_cache_rebalance_pressure_ema`；prepared budget backoff 可以同时参考累计与 EMA，而不是只靠累计压力。
- <!-- updated: 2026-04-16 09:02 --> **[tests]** 新增 prepared pressure step/EMA 的控制器测试与 summary 聚合测试；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `88 passed, 1 warning`。
- <!-- updated: 2026-04-16 08:54 --> **[rebalance-pressure-normalization]** `prepared_cache_rebalance_pressure` 现按 `pipeline_ticks` 归一；prepared-tier controller 不再把长运行中的累计 eviction 直接当成瞬时高压，长期运行下的 backoff 信号更稳定。
- <!-- updated: 2026-04-16 08:54 --> **[tests]** 调整 prepared pressure/backoff 测试，验证 step 归一后的 effective prepared budget 行为；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `87 passed, 1 warning`。
- <!-- updated: 2026-04-16 08:47 --> **[prepared-controller-coupling]** `prepared_cache_budget_backoff` 不再只影响 `effective_prepared_cache_limit`，现在也会反馈到 `adaptive_activation_limit / adaptive_prebuild_limit`；prepared budget 收缩与候选准备 aggressiveness 已开始联动。
- <!-- updated: 2026-04-16 08:47 --> **[tests]** 新增 prepared controller engaged / backoff 影响 adaptive limit 的测试；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `87 passed, 1 warning`。
- <!-- updated: 2026-04-16 08:40 --> **[prepared-budget-backoff]** prepared tier controller 现在新增 `prepared_cache_budget_backoff`：会按 prepared-cache 重平衡压力分级收缩 `effective_prepared_cache_limit`，在高压时最多把 prepared tier 缩到仅保留最关键候选；若 `cold_promotion_penalty` 偏高，则会撤销 backoff，重新放宽 prepared budget。
- <!-- updated: 2026-04-16 08:40 --> **[prepared-budget-summary]** scheduler summary / profile sweep 现已输出 `prepared_cache_budget_backoff_avg`，可以直接比较不同 profile 的 prepared budget 收缩幅度。
- <!-- updated: 2026-04-16 08:40 --> **[tests]** 新增 prepared budget backoff 的行为与摘要覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `87 passed, 1 warning`。
- <!-- updated: 2026-04-16 08:29 --> **[effective-prepared-budget]** prepared tier 现在区分静态 `prepared_cache_limit` 与动态 `effective_prepared_cache_limit`：当重平衡压力持续偏高且 activation stage bonus 偏低时，会临时收缩 prepared tier 的有效预算，避免 warm/activated 两层在高回退压力下继续无效扩张。
- <!-- updated: 2026-04-16 08:29 --> **[prepared-budget-metrics]** scheduler summary / profile sweep 新增 `effective_prepared_cache_limit`、`effective_prepared_cache_utilization` 与 `prepared_cache_rebalance_pressure_avg`，prepared tier 的预算收缩行为与压力强度现在都可直接比较。
- <!-- updated: 2026-04-16 08:29 --> **[tests]** 新增 effective prepared budget 的诊断与摘要覆盖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `86 passed, 1 warning`。
- <!-- updated: 2026-04-16 08:12 --> **[cold-promotion-penalty]** prepared tier controller 现在会跟踪 `cold_promotion_penalty`：当 ready apply 中冷路径 promotion 占比偏高时，会提高后续 adaptive activation/prebuild limit，尝试增加 prepared overlap 以减少下一轮冷启动。
- <!-- updated: 2026-04-16 08:12 --> **[tests]** 新增 cold-promotion penalty 的摘要与行为测试；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `85 passed, 1 warning`。
- <!-- updated: 2026-04-16 07:54 --> **[adaptive-prepared-limits]** `HybridMoE` 现在会根据 prepared-cache 压力和 `prepared_cache_activation_stage_bonus` 动态调整 activation/prebuild 候选上限；在 prepared tier 吃紧且 activated 偏置较低时，会主动收缩 `adaptive_activation_limit` 和 `adaptive_prebuild_limit`。
- <!-- updated: 2026-04-16 07:54 --> **[tests]** 新增 adaptive activation/prebuild limit 的诊断与压力测试；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `84 passed, 1 warning`。
- <!-- updated: 2026-04-16 07:40 --> **[prepared-cache-stage-bonus]** prepared-cache retention policy 现在带最小自适应 stage bonus：当重平衡更频繁地打在 activated tier 或 warm tier 时，`prepared_cache_activation_stage_bonus` 会随之调整，开始为后续自适应 prepared-cache policy 预留动态信号。
- <!-- updated: 2026-04-16 07:40 --> **[tests]** 新增 prepared-cache stage-bonus 方向性测试；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `83 passed, 1 warning`。
- <!-- updated: 2026-04-16 07:29 --> **[prepared-cache-rebalance-metrics]** scheduler summary/profile sweep 现在会显式统计 prepared-cache 重平衡事件，包括 `prepared_cache_rebalance_evicted_warm / evicted_activated / demoted_to_warm / dropped_to_ready`，可以直接看 prepared budget 压力主要落在哪一层。
- <!-- updated: 2026-04-16 07:29 --> **[tests]** 新增 prepared-cache 重平衡摘要与 profile sweep 测试；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `82 passed, 1 warning`。
- <!-- updated: 2026-04-16 07:18 --> **[prepared-cache-sweep]** profile sweep 的 `profiles / comparison_table / best_by_metric` 现已包含 `prepared_cache_limit / prepared_cache_size / effective_warm_cache_limit / prepared_cache_utilization`，prepared tier 预算可以直接参与策略排序与对比。
- <!-- updated: 2026-04-16 07:18 --> **[tests]** 新增 prepared-cache profile sweep 汇总测试；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `82 passed, 1 warning`。
- <!-- updated: 2026-04-16 07:08 --> **[prepared-cache-plumbing]** `scheduler_prepared_cache_budget_per_layer` 已打通到 `LLM`、`example.py` 和 `benchmark_inference.py`，prepared-cache 预算不再只能在代码内硬编码测试。
- <!-- updated: 2026-04-16 07:08 --> **[prepared-cache-summary]** scheduler summary 现新增 `prepared_cache_limit / prepared_cache_size / effective_warm_cache_limit / prepared_cache_utilization`，便于在 benchmark/profile sweep 中直接观察 prepared tier 是否吃满、warm budget 是否被 activated 层挤压。
- <!-- updated: 2026-04-16 07:08 --> **[tests]** 新增 prepared-cache summary 聚合测试；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `81 passed, 1 warning`。
- <!-- updated: 2026-04-16 06:56 --> **[prepared-cache-rebalance]** prepared-cache 预算现在会在 `warm cache` 和 `activated cache` 两层之间统一重平衡；当总 prepared slots 超限时，系统会在两层候选中按 hotness 与 lifecycle 统一选 victim，而不再只先压 warm cache。
- <!-- updated: 2026-04-16 06:56 --> **[prepared-cache-rebalance-tests]** 新增 prepared-cache 重平衡测试，验证高 hotness 的 activated candidate 会优先保留，较冷的 warm candidate 会被回退到 `READY`；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `80 passed, 1 warning`。
- <!-- updated: 2026-04-16 06:44 --> **[prepared-cache-budget]** `HybridMoE` 新增统一的 `expert_prepared_cache_size`，用于约束 `warm cache + activated cache` 的总 prepared expert 数；activated cache 占用上升时，warm cache 的有效容量会动态收缩。
- <!-- updated: 2026-04-16 06:44 --> **[prepared-cache-tests]** 新增测试覆盖 unified prepared-cache budget，验证 activated cache 占满总预算时，较冷的 warm candidate 会被回退到 `READY`；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `79 passed, 1 warning`。
- <!-- updated: 2026-04-16 06:30 --> **[activated-cache-victims]** activated cache 的 victim 选择已从简单 FIFO/LRU 收敛为“lifecycle 优先级 + hotness”排序；更冷的 activated candidate 会优先回退到 warm cache，并把 lifecycle 从 `ACTIVATED` 降到 `WARMED`，与 warm cache 的热点保留策略保持一致。
- <!-- updated: 2026-04-16 06:30 --> **[tests]** 新增 activated cache eviction 测试，验证在容量不足时更冷的 activated expert 会被逐出到 warm cache，同时回归 `tests/test_core.py + tests/test_pim_runtime.py` 为 `78 passed, 1 warning`。
- <!-- updated: 2026-04-16 03:22 --> **[benchmark-sweep]** `profile_sweep_summary` 新增自动对比层，输出 `comparison_table`、`best_by_metric`、`metric_directions`，并补充 `pipeline_promotion_non_cold_total/ratio` 与 `runtime_apply_batch_size_avg`，便于直接比较 overlap 质量而不只看 decode TPS。
- <!-- updated: 2026-04-16 03:36 --> **[batch-apply-sources]** ready promotion 的批处理现在会统计 batch 内 `activated / warm / cold` 三类来源；对应指标已接到 `HybridMoE` 诊断、`MigrationPipelineRuntime` 汇总和 scheduler summary，便于判断批处理究竟是在消费热路径还是仍有大量冷启动。
- <!-- updated: 2026-04-16 03:48 --> **[lifecycle-alignment]** warm/activated cache 的 eviction 现在会同步回退 migration lifecycle：device-side activated candidate 被挤出时回退到 `WARMED`，CPU warm candidate 被挤出时回退到 `READY`，避免 cache 层次和状态机脱节。
- <!-- updated: 2026-04-16 03:58 --> **[eviction-regressions]** migration diagnostics 新增 `total_activation_eviction_regressions` 和 `total_warm_eviction_regressions`，可以直接统计缓存淘汰导致的 lifecycle 回退次数，为后续调整 warm/activated 预算提供依据。
- <!-- updated: 2026-04-16 04:05 --> **[profile-ranking]** profile sweep 的比较表和 metric ranking 现在会纳入 eviction regression 压力，后续可以直接按“更少 lifecycle 回退”筛选更稳的动态调度策略。
- <!-- updated: 2026-04-16 04:14 --> **[warm-cache-policy]** warm cache eviction 现在不再只按简单插入顺序，而会结合 lifecycle 优先级与 hotness 选择更冷的 victim，减少热点 expert 因短期缓存抖动被过早打回 `READY`。

## 2026-04-07

- **[init]** 初始化项目知识库
- <!-- updated: 2026-04-07 20:56 --> **[runtime]** 将 `flash-attn`、`triton`、`kt-kernel` 下放为可选依赖，新增 CPU-only fallback，`example.py` 改为显式传入模型路径，新增 `tests/test_smoke_cpu.py` 覆盖无 PIM 路径。
- <!-- updated: 2026-04-07 21:18 --> **[qwen3]** 修复 `SimpleEngine` 使用错误 `head_dim` 预分配 KV cache 的问题，确认 `example.py` 可在 `/home/yangfu/models/Qwen--Qwen3-30B-A3B-Base` 上以 `--device cpu --max-new-tokens 1` 跑通。

## 2026-04-08

- <!-- updated: 2026-04-08 10:12 --> **[benchmarks]** 新增 [benchmark_inference.py](../../benchmarks/benchmark_inference.py) 和 [benchmarks/README.md](../../benchmarks/README.md)，可统一测 `cpu`、`cuda`、`cuda_cpu_offload` 三类推理 backend。
- <!-- updated: 2026-04-08 10:12 --> **[pim]** 新增 [pim_microbench](../../benchmarks/pim_microbench/) 下的 host/DPU microbenchmark、build 脚本和 run 脚本；simulator 模式已跑通，硬件模式当前卡在 `dpu_alloc_ranks`。
- <!-- updated: 2026-04-08 11:02 --> **[host-validation]** 用户宿主机已确认存在 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 和 `/dev/dpu_rank*`，并成功跑出真实 PIM 硬件 benchmark；另已修复 inference benchmark，使 `cuda` backend OOM 时不阻断后续 `cuda_cpu_offload`。

## 2026-04-09

- <!-- updated: 2026-04-09 00:00 --> **[qwen3-layout]** 为 `Qwen3` 增加 checkpoint 自适应 expert 布局检测，支持从真实 safetensor 键名自动切换到 unpacked `gate_proj` / `up_proj` / `down_proj`。
- <!-- updated: 2026-04-09 00:00 --> **[offload-fixes]** 修复 `cuda_cpu_offload` 链路中的 CPU fallback 内存翻倍问题和 attention mask dtype 问题，确认 `/home/yangfu/models/Qwen--Qwen3-30B-A3B-Base` 可在 `--offload-device-experts 2` 下真机跑通。
- <!-- updated: 2026-04-09 00:00 --> **[pim-metrics]** 将 PIM microbenchmark 指标改为明确的整数 workload 度量，新增 `kernel_workload`、`kernel_element_gops`、`kernel_int32_gops_estimate`，避免误读为浮点算力。
- <!-- updated: 2026-04-09 00:00 --> **[backend-abstraction]** 新增 `ExpertOffloadBackend`、`PIMMoEBackend` 和 `offload_backend` 选择逻辑，`HybridMoE` / `LLM` / benchmark 入口已支持 `pim_shadow` 主链路。

## 2026-04-14

- <!-- updated: 2026-04-14 11:30 --> **[versioning]** 将仓库版本提升到 `v0.2.0`，用于标记“CPU baseline + Qwen3 修复 + cuda_cpu_offload + pim_shadow + UPMEM benchmarks”这一阶段性里程碑。
- <!-- updated: 2026-04-14 11:30 --> **[knowledge-sync]** 同步更新知识库中的架构说明、当前焦点、路线图和日志，避免继续沿用 2026-04-08 之前的过期状态描述。

## 2026-04-15

- <!-- updated: 2026-04-15 00:35 --> **[pim-runtime]** 新增 `pim_linear_runtime.py` 与 `pim_native/` 原生桥接代码，仓库现在具备最小真实 DPU 线性计算能力；已在真实硬件上用随机矩阵验证 DPU 结果与 CPU 对齐。
- <!-- updated: 2026-04-15 00:35 --> **[pim-backend]** `PIMMoEBackend` 新增实验性 `pim` 模式：expert 的线性投影可走真实 DPU，SiLU / gating 仍在 host 端执行，当前默认只对小 flattened batch 生效，其余情况自动回退 CPU。
- <!-- updated: 2026-04-15 15:10 --> **[pim-sharding]** 将原生 DPU linear host bridge 从单 DPU 扩展到多 rank / 多 DPU 行分片执行，新增 `runtime_dpu_count` 诊断，并让 `PIMLinearRuntime` 能按 `(profile, rank_count)` 复用不同运行时实例。
- <!-- updated: 2026-04-15 15:10 --> **[cuda-pim-validation]** 用真实 `/home/yangfu/models/Qwen--Qwen3-30B-A3B-Base` 跑通 `cuda_pim` 端到端链路；在 `--prompt Hi --offload-device-experts 2 --pim-rank-count 4 --pim-max-batch-tokens 1 --max-new-tokens 2` 下，确认所有层都发生了真实 DPU expert linear 调用并产出结果文件 `benchmarks/results/cuda_pim_2026-04-15.json`。
- <!-- updated: 2026-04-15 05:22 --> **[pim-fused]** 新增 `PIMExpertRuntime`、`dpu_expert_kernel.c` 和 `host_expert_bridge.c`，把 `gate/up/down + SiLU` 的完整 expert 子图接成实验性 fused DPU kernel，并让 `PIMMoEBackend` 支持 `pim_kernel_variant=fused`。
- <!-- updated: 2026-04-15 05:22 --> **[pim-fused-optimization]** 修正 fused expert kernel 的核心数据流，改为先在 WRAM 里计算完整 hidden 激活再复用到 down projection；真实 microbench 从原先二十到三十秒级降到约 `0.29s`，但仍显著慢于 `linear3` 的约 `0.03s`，因此当前默认策略保持 `linear`。
- <!-- updated: 2026-04-15 05:48 --> **[pim-fused-sharding]** 将 fused expert host bridge 从单纯 hidden 分片改成 `hidden_group x row_group` 二维分片；Qwen 级别单 expert microbench 进一步降到约 `0.26s`，并可看到 `expert_runtime_last_active_dpus=60`，但仍未优于 `linear3`。
- <!-- updated: 2026-04-15 06:04 --> **[dynamic-scheduler-skeleton]** 新增 `expert_runtime_state.py` 与 `scheduler/dynamic_expert_scheduler.py`，把系统目标从“静态 GPU expert mask”提升为“GPU/PIM 动态专家驻留”第一版骨架；`LLM`、`MixtralModel`、`HybridMoE` 已能携带驻留计划与调度器诊断，但真实迁移数据面尚未接入。
- <!-- updated: 2026-04-15 06:23 --> **[prefill-policy]** 为 `pim` backend 新增 `pim_prefill_policy` 和 `pim_prefill_token_threshold`，默认 prefill 走 CPU/GPU 路径，避免长 prompt 的大批量 token 直接压到 PIM。
- <!-- updated: 2026-04-15 06:23 --> **[dynamic-prefill-hook]** `HybridMoE` 现在会在 prefill 阶段基于路由结果更新调度器热度，并允许临时提升 GPU expert budget；当前只更新驻留状态与诊断，真实迁移数据面仍待实现。
- <!-- updated: 2026-04-15 06:31 --> **[migration-queue-semantics]** 调整动态调度语义：`HybridMoE` 现在只向 backend 排队 migration plan，不再在没有真实 GPU/PIM 数据面的前提下直接修改有效 `gpu_experts_mask`，避免控制面和执行面状态不一致。
- <!-- updated: 2026-04-15 06:40 --> **[migration-manager]** 新增 `expert_migration.py`，为 backend 提供每层 migration queue 与阶段历史记录；动态调度相关单测已补到 `tests/test_core.py`。
<!-- updated: 2026-04-15 06:58 -->

## 2026-04-15

- 新增 `nano_ktrans/layers/expert_mlp.py`，把 shared expert module 定义从 `mixtral.py` 抽离，供模型初始化和运行时 expert materialization 共用。
- `ExpertWeightLoader` 新增单 expert 加载接口，支持 decode 阶段按需从 safetensors 拉起单个 expert 权重。
- `HybridMoE` 新增最小 decode 迁移执行数据面：
  - drain 本层 migration queue
  - promotion 时动态构建 GPU expert 并注入 `gpu_experts`
  - demotion 时从 `gpu_experts` 移除并更新 mask
- 新增测试覆盖 decode 阶段 migration queue 被实际执行的路径。

<!-- updated: 2026-04-15 07:06 -->

- 新增 `nano_ktrans/kernels/expert_materialization.py`，提供单 expert 的 CPU staging cache、预取队列和基础诊断。
- `HybridMoE` 现在会在 `prefill` 阶段对候选 promotion expert 发起预取，并在 `decode` promotion 时优先命中 staging cache。
- 新增测试覆盖 prefill 阶段的 expert 预取路径。

<!-- updated: 2026-04-15 07:14 -->

- `HybridMoE` 的 decode migration 现在接入了 GPU budget 约束：若 promotion 时 GPU resident set 已满，会先按 hotness 驱逐冷 expert，再执行热点 expert promotion。
- 新增测试覆盖“为 promotion 驱逐冷 expert”的运行时路径。

<!-- updated: 2026-04-15 07:22 -->

- `HybridMoE` 现在会对 decode 阶段生成的 future promotions 也发起预取，不再仅限于 prefill 预热。
- decode migration 队列现按“当前活跃优先 + hotness 优先”排序，更接近真实热点 cache 的调度语义。
- 新增测试覆盖 decode 阶段的 future promotion 预取路径。

<!-- updated: 2026-04-15 07:30 -->

- decode promotion 队列进一步接入 `prefetch ready` 优先级，优先消费 staging cache 已就绪的专家。
- `HybridMoE` 诊断中新增 `decode_prefetch_hits` / `decode_prefetch_misses`，用于观察预热是否真正命中 decode promotion。

<!-- updated: 2026-04-15 07:37 -->

- `ExpertMaterializationManager.prefetch()` 现在会返回是否真的触发了新预取，避免把重复请求误记成有效预热。
- `HybridMoE` 诊断新增 `prefetch_enqueued`，用于区分“请求数”和“真正进入 staging cache 的次数”。

<!-- updated: 2026-04-15 07:44 -->

- `LayerExpertState` 新增 `last_access_step` 与 `last_residency_change_step`，scheduler 已开始维护这些 anti-thrashing 元数据。
- 当前 cooldown / idle-age 逻辑先以配置和诊断形式接入，默认值保持不改变现有行为。

<!-- updated: 2026-04-15 07:53 -->

- scheduler 新增 `prefill_collect_only`、`step_stride_prefill` 和 `step_stride_decode` 配置。
- `LLM`、`example.py`、`benchmark_inference.py` 已暴露这些入口，后续可以直接在真实 benchmark 中对比不同调度策略。

<!-- updated: 2026-04-15 08:01 -->

- decode migration 新增 `decode_require_prefetch_ready` 开关。
- 开启后，未完成 staging prefetch 的 promotion 会先 defer，而不是直接在 decode 关键路径上同步 materialize。
- 新增测试覆盖“defer until prefetch ready”的 decode migration 路径。

<!-- updated: 2026-04-15 08:08 -->

- `ExpertMigrationManager` 现在会按 expert 对 pending migration queue 去重。
- queue 诊断新增 `total_enqueued_ops`、`total_deduped_ops`、`total_drained_ops` 和 per-phase `deduped_plan_size`。

<!-- updated: 2026-04-15 08:16 -->

- scheduler 新增 `prefetch_candidate_budget_per_layer`，可按层从 offloaded experts 中挑选热点候选做预取。
- `HybridMoE` 新增 `prefetch_candidate_scans` 诊断，用于观察候选预取是否实际发生。

<!-- updated: 2026-04-15 10:58 -->

- scheduler 新增 profile 预设：`baseline`、`overlap_safe`、`eager`。
- `LLM`、`example.py`、`benchmark_inference.py` 现在都可直接选择 scheduler profile，而不必手工拼全部调度开关。
- benchmark 新增调度摘要输出，自动聚合：
  - `prefetch_requested / enqueued / materialized`
  - `decode_prefetch_hits / misses`
  - `runtime_evictions`
  - `runtime_deferred_for_prefetch`
  - migration queue 的 `enqueued / deduped / drained`
- 新增对应测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `45 passed`。

<!-- updated: 2026-04-15 11:08 -->

- migration manager 新增 lifecycle 跟踪：`queued / prefetching / ready / deferred / applied`。
- `HybridMoE` 现在会在预取、ready 命中、defer 和 applied 路径上写回 lifecycle 状态，benchmark 摘要也会同步聚合这些指标。
- 新增对应测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `46 passed`。

<!-- updated: 2026-04-15 11:20 -->

- `decode_require_prefetch_ready` 模式下，decode 入口现在只会消费“进入本层前已经 ready 的 promotion”。
- migration manager 新增 `take_layer()` / `peek_layer()`，让 decode 可以保留未 ready 的 pending op，而不是先 drain 再重排。
- 新增对应测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `47 passed`。

<!-- updated: 2026-04-15 11:27 -->

- `ExpertMaterializationManager` 新增 `poll_ready()`，可把后台完成的 prefetch future 主动转成 staging cache 命中。
- `HybridMoE` 现在会在进入本层前先轮询 ready prefetch，再把 migration lifecycle 更新为 `ready`。
- benchmark 摘要新增 `prefetch_polled_ready`，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `48 passed`。

<!-- updated: 2026-04-15 11:34 -->

- migration manager 新增 `take_ready_layer()` 和 `total_ready_drains` 统计。
- decode ready-only 路径已改成直接消费 migration manager 的 ready 子集，而不是由 `HybridMoE` 手写过滤逻辑。

<!-- updated: 2026-04-15 11:39 -->

- `ExpertMaterializationManager` 新增 completion queue；后台 prefetch 完成后会先进入 queue，再由 `poll_ready()` 消费。
- benchmark 摘要新增 `prefetch_completion_events`，用于区分“future 已完成”和“前台已轮询并入 cache”。

<!-- updated: 2026-04-15 11:45 -->

- `HybridMoE.forward()` 不再每层自行轮询 ready prefetch。
- `SimpleEngine` 现在会在每次 `prefill` / `decode_step` 进入模型前统一调用 `MixtralModel.refresh_offload_state()`，将 ready 刷新上移到 token-step 级别。
- 新增对应测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `49 passed`。

<!-- updated: 2026-04-15 11:50 -->

- `SimpleEngine` 新增统一 `_refresh_offload_state()` helper，full prefill、chunked prefill 和 decode 现在共用同一 refresh 入口。
- 新增对应测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `50 passed`。

<!-- updated: 2026-04-15 11:55 -->

- `MixtralModel` 新增 `offload_refresh_calls` 和 `offload_refresh_ready_total` 统计。
- `LLM.get_offload_diagnostics()` 与 benchmark 摘要现在会带出模型级 offload refresh 指标。

<!-- updated: 2026-04-15 11:59 -->

- `ExpertMaterializationManager` 新增 `has_pending_or_ready()`。
- `HybridMoE.refresh_offload_state()` 现在会在无 pending/ready prefetch 时直接短路返回，减少空轮询。

<!-- updated: 2026-04-15 12:05 -->

- benchmark 新增 `--scheduler-profile-sweep`，可在单次运行中依次比较多组 scheduler profile。
- `normalize_scheduler_profiles()` 会做 profile 归一化与去重，避免 sweep 配置重复。

<!-- updated: 2026-04-17 13:25 -->

- 新增 `nano_ktrans/kernels/offload_worker.py`，提供最小后台 offload worker 骨架，可在独立线程中周期性推进 `background_tick_offload_state()`。
- `MixtralModel` 现已支持：
  - `enable_background_offload_worker`
  - `background_offload_poll_interval_seconds`
  - `offload_refresh_diagnostics()` 暴露 background worker 诊断
  - `reset_offload_worker_diagnostics()` / `shutdown_offload_worker()`
- `LLM.reset_offload_diagnostics()` 已同步重置 background worker 计数，`LLM.shutdown()` 会在生成结束后关闭后台 worker。
- 新增对应测试覆盖 background worker 计数、reset 和 shutdown 路径。

<!-- updated: 2026-04-17 13:45 -->

- `SimpleEngine` 新增 `start_background_offload_worker()` / `stop_background_offload_worker()`，worker 生命周期现在可以由引擎统一管理。
- `LLM.generate()` 现在会在生成前启动 background offload worker，并在 `finally` 中停止 worker 与执行 shutdown，后台推进首次接入真实生成路径。
- `MixtralModel` 新增 `start_offload_worker()` / `offload_worker_running()`，后台 worker 从“模型里可选对象”进一步收敛成了正式 runtime 组件。

<!-- updated: 2026-04-17 14:05 -->

- `MixtralModel` 中的 background offload worker 现在默认 `auto_start=False`，模型构造时不再隐式起线程。
- worker 生命周期已明确改成“构造对象 -> 生成前显式启动 -> 生成后停止”，避免后台线程在未进入 decode 路径前就提前占用资源。

<!-- updated: 2026-04-17 14:20 -->

- `summarize_offload_diagnostics()` 现已汇总 background worker 的 `enabled / ticks / work_ticks / work_ratio`。
- `summarize_profile_sweep_results()` 现已将 `background_worker_work_ratio` 纳入 profile 对比和 `best_by_metric` 排名，后台 worker 的活跃度开始进入 benchmark 决策面。

<!-- updated: 2026-04-17 14:35 -->

- `LLM.get_offload_diagnostics()` 现已显式输出 `prepared_cache_budget_heuristic`，用户可以直接对照 profile 的静态 prepared 预算基线与 runtime controller 的实际 prepared-tier 行为。
- 新增对应测试，确保 prepared budget heuristic 不只存在于 profile summary，也会进入最终的 offload diagnostics。

<!-- updated: 2026-04-17 14:50 -->

- `SimpleEngine._refresh_offload_state()` 现在会在检测到 background worker 已运行时，跳过手动 `background_tick_offload_state()`，避免前台 hook 和后台线程对同一阶段做重复推进。
- 新增对应测试，验证 worker 运行时只保留主 refresh，不再重复调用 background tick。

## 2026-04-16

- <!-- updated: 2026-04-16 00:40 --> **[migration-pipeline-runtime]** 新增 `MigrationPipelineRuntime`，将 token-step 级 offload refresh 提升为最小流水线运行时；ready prefetch 轮询与 ready promotion 现在可在进入模型前统一推进，不再依赖层内 forward 临时收敛。
- <!-- updated: 2026-04-16 00:40 --> **[pipeline-diagnostics]** `MixtralModel.offload_refresh_diagnostics()` 与调度摘要新增 pipeline 指标，包括 `offload_pipeline_ticks`、`offload_pipeline_ready_applied_total`、`offload_pipeline_ready_deferred_total`，便于观察“ready 到 applied”是否开始形成流水线。
- <!-- updated: 2026-04-16 00:40 --> **[tests]** 新增 pipeline runtime 覆盖：模型级 refresh 已验证 phase-aware pipeline tick，`HybridMoE` 也新增 ready promotion 在 pipeline hook 中被提前应用的单测；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `52 passed`。
- <!-- updated: 2026-04-16 01:05 --> **[pipeline-priming]** `HybridMoE.advance_offload_pipeline()` 现在会在 decode 进入模型前主动检查 pending promotion，并统一推进 `queued -> prefetching/deferred`；层内 `decode_require_prefetch_ready` 路径不再重复承担这部分预取提交逻辑。
- <!-- updated: 2026-04-16 01:05 --> **[pipeline-counters]** pipeline runtime 新增 `offload_pipeline_prefetch_submitted_total` 统计，便于观察 token-step 级 runtime 是否真的在为后续 ready promotion 预热 pending experts；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `53 passed`。
- <!-- updated: 2026-04-16 01:30 --> **[resident-export]** `ExpertOffloadBackend` 新增 `export_expert_weights()` 接口，CPU/PIM backend 现在可直接导出 resident expert 权重；`HybridMoE` 的 prefetch 路径已优先尝试从 offload resident tier 直接 stage 到 materialization cache。
- <!-- updated: 2026-04-16 01:30 --> **[resident-staging]** `ExpertMaterializationManager` 新增 `stage_expert()` 和 `resident_stage_hits`，可以记录“从 resident tier 直接命中 staging cache”的次数，减少 decode promotion 对 checkpoint 扫描的依赖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `55 passed`。
- <!-- updated: 2026-04-16 01:50 --> **[warm-expert-cache]** `HybridMoE` 新增 demotion 后的 warm expert cache：GPU 驱逐下来的 expert module 可暂存到 CPU 侧 warm cache，后续短时间 re-promotion 时可直接复用 module，减少重复构建成本。
- <!-- updated: 2026-04-16 01:50 --> **[warm-cache-diagnostics]** 新增 `warm_cache_hits / stores / evictions / size` 诊断，并补充单测覆盖 demote 后缓存与 re-promotion 命中；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `57 passed`。
- <!-- updated: 2026-04-16 02:10 --> **[ready-prebuild]** token-step pipeline 现在会在 decode 进入模型前，对已经 `READY` 但尚未 materialize 的 promotion 预先构建 expert module，并放入 warm cache；后续 `READY -> APPLIED` promotion 可直接命中 warm cache。
- <!-- updated: 2026-04-16 02:10 --> **[ready-prebuild-tests]** 新增单测覆盖 `READY` expert 在 pipeline hook 中被 prebuild，再由 promotion 直接命中 warm cache 的路径；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `58 passed`。
- <!-- updated: 2026-04-16 02:25 --> **[cpu-prebuild]** ready expert 的 prebuild 现在固定在 CPU 上进行，promotion 再执行单次 device transfer；这让 token-step pipeline 的 prebuild 更接近“后台准备对象，前台只做激活”。
- <!-- updated: 2026-04-16 02:25 --> **[warm-transfer-diagnostics]** 新增 `warm_cache_device_transfers` 统计，用于观察 warm cache 命中后实际发生了多少次 CPU->device 激活拷贝。
- <!-- updated: 2026-04-16 02:40 --> **[warmed-lifecycle]** migration lifecycle 新增 `warmed` 状态，专门表示“expert 数据已 ready 且模块已在 warm cache 中预构建”；这样可以把 `ready -> warmed -> applied` 与简单 `ready -> applied` 区分开来。
- <!-- updated: 2026-04-16 03:05 --> **[activation-stage]** migration lifecycle 进一步新增 `activated` 状态，表示 warm cache 中的 expert module 已完成 device transfer、尚未正式进入 GPU resident set；`HybridMoE` 的 token-step pipeline 现在会在 decode 前先推进 `warmed -> activated`，再由最终 promotion 完成 `activated -> applied`。
- <!-- updated: 2026-04-16 03:05 --> **[activation-diagnostics]** pipeline/runtime/scheduler 摘要新增 activation 相关指标，包括 `offload_pipeline_activation_ready_total`、`activation_submitted / ready / applied` 和 `migration_activated_events`，便于区分“模块已预构建”和“设备激活已完成”。
- <!-- updated: 2026-04-16 03:20 --> **[activated-cache]** `HybridMoE` 现在为已完成 device transfer 的 expert 引入独立 activated cache；decode promotion 会优先命中 activated cache，再退到 CPU warm cache 或冷路径，进一步压缩 `activated -> applied` 关键路径。
- <!-- updated: 2026-04-16 03:32 --> **[activated-cache-priority]** activated cache 现在会按 lifecycle 优先级与 hotness 做预算保留；decode 前只把最值得保留的 warmed experts 提升到 device-side activated cache，避免较冷 expert 抢占有限激活预算。
- <!-- updated: 2026-04-16 03:43 --> **[deferred-state-preservation]** migration queue 重新排入 `*_deferred` op 时，若 expert 已处于 `prefetching/ready/warmed/activated`，现在会保留该中间态，不再把 pipeline 进度重置成 `deferred`，避免已完成一半的 promotion 在控制面上“掉回队尾”。
- <!-- updated: 2026-04-16 03:55 --> **[requeue-diagnostics]** migration queue 现新增 `total_requeue_preserved_states`，用于统计 deferred/queued 重排时保留了多少个中间 lifecycle；scheduler 摘要也同步输出这一指标，便于衡量流水线是否真的在“只前进不回退”。
- <!-- updated: 2026-04-16 04:07 --> **[ready-queue-drain]** decode 的 ready promotion 路径已改成“peek + selective consume”：只在真正 `applied` 时把 op 从 pending queue 里移除，预算不足导致的未消费 ready op 会保留在原队列中等待下一步，而不再重复 enqueue 成 `*_deferred`。
- <!-- updated: 2026-04-16 04:18 --> **[strict-ready-only]** `decode_require_prefetch_ready` 语义进一步收紧：即使 resident tier 直接 stage 成功，decode prime 阶段也不会在同一步立刻把它标成 `ready` 并消费，而是等待下一次 refresh/pipeline tick，使“ready-only”真正意味着“前一阶段已完成”。
- <!-- updated: 2026-04-16 04:30 --> **[per-run-scheduler-summary]** inference benchmark 现在会在每次 generation 前重置 offload/runtime 计数器，并把单次 run 的 `scheduler_summary` 直接挂到 run 结果上；这样 profile 对比时不再被 warmup 或前序 decode 步的累计诊断污染。
- <!-- updated: 2026-04-16 04:42 --> **[prebuild-target-budget]** `HybridMoE` 的 ready prebuild 现在不再对所有 ready 候选一视同仁，而是按 lifecycle 优先级、hotness 和 decode 预算只保留更有价值的一批 prebuild target，避免较冷 expert 过早占用 warm cache 和构建开销。
- <!-- updated: 2026-04-16 04:55 --> **[promotion-source-breakdown]** pipeline 现在会统计每次 promotion 究竟来自 `activated cache`、`warm cache` 还是冷路径 build，并聚合为 `pipeline_prefetch_overlap_hits` 与 source breakdown；这让 benchmark 可以直接回答“有多少次 promotion 已经不是冷启动”。 
- <!-- updated: 2026-04-16 05:20 --> **[decode-queue-retention]** `HybridMoE._apply_queued_migrations()` 已从旧的 `drain/take_ready -> deferred requeue` 路径收敛到 `peek + selective consume`：只有真正 `applied` 的 promotion / demotion 才会从 migration queue 中移除，预算不足或仍未 ready 的 op 会原地保留在 pending queue 中，避免 layer-forward 和 token-step pipeline 重复搬运同一批 decode 迁移。
- <!-- updated: 2026-04-16 05:20 --> **[decode-ready-strictness]** layer-forward 现在在 `decode_require_prefetch_ready=true` 时只消费 lifecycle 已推进到 `ready/warmed/activated` 的 promotion；即便 materialization cache 已同步命中，也不会在同一个 forward 中越级把 expert 直接视为 `ready`，进一步统一 strict ready-only 语义。
- <!-- updated: 2026-04-16 05:20 --> **[tests]** 新增两条 decode 队列语义测试：一条验证“预算不足的 ready promotion 会继续留在 pending queue 中”，另一条验证“只移除已执行的 demotion，active expert 对应的 demotion op 会继续保留”；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `70 passed, 1 warning`。
- <!-- updated: 2026-04-16 05:40 --> **[pipeline-apply-batch]** ready promotion 现在先经过 `_select_ready_promotion_batch()` 做同层小批量截断；虽然底层 apply 仍逐 expert 执行，但 pipeline 已开始按批次统计 `pipeline_apply_batches` / `pipeline_apply_batch_experts`，为后续真正的 layer-batched apply 打基础。
- <!-- updated: 2026-04-16 05:40 --> **[batch-metrics]** `HybridMoE.diagnostics()`、`LLM.reset_offload_diagnostics()` 和 scheduler summary 现在都会输出 pipeline apply 批次指标，并补了对应单测；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `71 passed, 1 warning`。
- <!-- updated: 2026-04-16 05:55 --> **[batch-eviction]** ready promotion 的同层批次现在会先统一计算需要腾出的 GPU slot，并通过 `_evict_for_promotion_batch()` 预先完成这批次的 eviction；这避免了 batch 内每个 expert 各自重复做一次 budget 检查。
- <!-- updated: 2026-04-16 05:55 --> **[batch-eviction-metrics]** 新增 `pipeline_apply_batch_evictions`，可直接观察一次 ready-apply 批次为了让位而预先执行了多少 GPU resident eviction；对应摘要和单测已补齐，当前 `tests/test_core.py + tests/test_pim_runtime.py` 仍为 `71 passed, 1 warning`。
- <!-- updated: 2026-04-16 06:10 --> **[runtime-batch-rollup]** `MigrationPipelineRuntime` 现在会汇总 token-step 级的 apply batch 指标，包括 `offload_pipeline_apply_batch_count_total`、`offload_pipeline_apply_batch_experts_total` 和 `offload_pipeline_apply_batch_evictions_total`，这样 benchmark 不只看层级局部状态，也能直接看每个 decode step 的批处理推进情况。
- <!-- updated: 2026-04-16 06:10 --> **[tests]** 新增 runtime 层的 batch 汇总测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `72 passed, 1 warning`。
- <!-- updated: 2026-04-16 06:25 --> **[profile-sweep-summary]** 新增 `summarize_profile_sweep_results()`，benchmark 现在会额外输出 `profile_sweep_summary`，自动汇总各 scheduler profile 的 `decode_tokens_per_second`、overlap 命中、promotion source breakdown、apply batch 指标和 deferred 数。
- <!-- updated: 2026-04-16 06:25 --> **[example-runtime-totals]** `example.py` 现在会额外打印 step 级 pipeline apply totals，方便快速肉眼查看本次生成是否真的出现批处理式 ready/apply 行为。
- <!-- updated: 2026-04-16 06:25 --> **[tests]** 新增 profile sweep 摘要测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `73 passed, 1 warning`。
- <!-- updated: 2026-04-16 06:35 --> **[runtime-batch-sweep]** profile sweep 摘要现在也纳入 step 级 runtime apply batch totals，包括 `runtime_offload_pipeline_apply_batch_count_total`、`runtime_offload_pipeline_apply_batch_experts_total` 和 `runtime_offload_pipeline_apply_batch_evictions_total`，benchmark/README 已同步说明这些指标的意义。
- <!-- updated: 2026-04-16 06:45 --> **[incremental-batch-metrics]** `HybridMoE.advance_offload_pipeline()` 现在返回的是本次 tick 新增的 apply batch 指标，而不是层上的累计值；新增单测验证连续两个 decode tick 会各自上报独立的批次数、专家数和 eviction 数，避免 runtime 汇总被累计计数放大。
- <!-- updated: 2026-04-16 09:35 --> **[adaptive-prefetch-controller]** prepared-tier controller 现在开始直接约束 prefetch aggressiveness：`HybridMoE` 新增 `adaptive_prefetch_pending_limit` 和 `adaptive_prefetch_candidate_budget`，会根据 `prepared_cache_budget_backoff`、rebalance step pressure 和 `cold_promotion_penalty` 同时调节 pending promotion 预取与候选预取预算。
- <!-- updated: 2026-04-16 09:35 --> **[prefetch-controller-diagnostics]** scheduler summary / profile sweep 现已纳入 `adaptive_prefetch_pending_limit_avg` 与 `adaptive_prefetch_candidate_budget_avg`，benchmarks README 也同步补充了 prepared-tier controller 的新指标；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `89 passed, 1 warning`。
- <!-- updated: 2026-04-16 09:48 --> **[prepared-budget-surface]** `scheduler_profile_summary()` 现在会显式给出 `prepared_cache_budget_heuristic`，`LLM.get_offload_diagnostics()` 也会输出实际采用的 `prepared_cache_budget`；这样 profile、diagnostics 和 benchmark 终于能把“静态 prepared 预算基线”与后续 controller 行为对应起来。
- <!-- updated: 2026-04-16 09:48 --> **[tests]** 新增 prepared-budget heuristic/diagnostics 覆盖，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `90 passed, 1 warning`。
- <!-- updated: 2026-04-16 09:56 --> **[profile-budget-heuristic]** prepared-cache budget heuristic 现在开始随 scheduler profile 变化：`baseline` 使用 `max(2 * decode_promote_k, prefetch_candidate_budget, 2)`，`overlap_safe` 和 `eager` 会在此基础上进一步上调 prepared budget，为 strict ready-only 和更激进的 prepared-tier 推进保留额外空间。
- <!-- updated: 2026-04-16 09:56 --> **[tests]** 新增 profile-aware prepared-budget 解析测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `91 passed, 1 warning`。
- <!-- updated: 2026-04-16 10:05 --> **[profile-aggressiveness]** prepared-tier controller 现在开始显式受 scheduler profile 影响：新增 `resolve_prepared_controller_aggressiveness()`，并通过 `LLM -> Mixtral -> HybridMoE` 传入 `prepared_controller_aggressiveness`，用于区分 `baseline / overlap_safe / eager` 在 activation / prebuild / prefetch 三段上的推进力度。
- <!-- updated: 2026-04-16 10:05 --> **[tests]** 新增 profile-aware controller aggressiveness 解析与 diagnostics 覆盖，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `92 passed, 1 warning`。
- <!-- updated: 2026-04-17 11:40 --> **[background-materialization-resolver]** `ExpertMaterializationManager` 新增后台 resolve worker：prefetch future 完成后会进入 resolve queue，由后台线程执行 `future.result() + cache store`，前台 `poll_ready()` 基本只负责消费轻量 ready 通知，不再在 decode refresh 路径里承担主要解析开销。
- <!-- updated: 2026-04-17 11:40 --> **[promotion-batch-resolve-apply]** `HybridMoE` 的 promotion batch 现在显式拆成两段：先统一 resolve `activated/warm/cold` source 和 module，再进入 batch apply resident set；这还不是真正底层 batched apply，但已经把后续批量 resident 注入的边界收清楚了。
- <!-- updated: 2026-04-17 11:40 --> **[diagnostics-tests]** 新增后台 materialization resolver 诊断：`background_resolver_enabled`、`prefetch_background_resolved`、`prefetch_background_failures`，并补充后台 resolve 和 batch resolve/apply 骨架的回归；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `94 passed, 1 warning`。
- <!-- updated: 2026-04-17 11:58 --> **[background-ready-callback]** `ExpertMaterializationManager` 现在支持后台 ready callback；resolved expert 会通过 callback 直接推动 migration lifecycle 进入 `READY`，前台 `refresh_offload_state()` 只保留 fallback drain 语义，不再承担主要的 `prefetching -> ready` 状态推进。
- <!-- updated: 2026-04-17 12:08 --> **[migration-manager-locking]** `ExpertMigrationManager` 已补上内部 `RLock`，后台 ready callback 可以安全调用 `mark_state()/state_for()/peek_layer()` 等接口，不再默认依赖前台单线程推进 migration lifecycle。
- <!-- updated: 2026-04-17 12:18 --> **[background-offload-tick]** `MixtralModel` 和 `MigrationPipelineRuntime` 已新增 background offload tick：每个 token-step 在主 refresh 前会先单独推进一轮后台 ready callback 统计，`offload_background_ticks` 与 `offload_pipeline_background_ready_callback_total` 现在可直接观察这条半独立推进路径。
- <!-- updated: 2026-04-17 12:32 --> **[background-tick-summary]** scheduler summary 现在会显式汇总 `offload_background_ticks` 和 `offload_pipeline_background_ready_callback_total`，background tick 不再只是 runtime 内部状态，已经进入 benchmark/profile 观察面。
- <!-- updated: 2026-04-17 12:46 --> **[background-tick-reset]** `LLM.reset_offload_diagnostics()` 现已同步清零 runtime 级 `background_ticks` 和 `background_ready_callback_total` 等计数，单次 benchmark run 的 background offload 指标不再混入历史 decode 步。
- <!-- updated: 2026-04-17 13:00 --> **[background-prepared-advance]** background offload tick 现在不只消费 ready callback，也会在 decode 期间提前推进一部分 `READY -> WARMED/ACTIVATED`，并新增 `offload_background_warm_prebuilt_total / offload_background_activation_ready_total` 两个 runtime 指标。
- <!-- updated: 2026-04-17 22:20 --> **[resident-commit-batch-limit]** `HybridMoE` 现已将 resident commit 的 staged resolve/apply queue 预算与 final batch commit 预算拆开：`_adaptive_apply_commit_limit()` 继续驱动 `apply_commit_queue` 的 resolve/staging，而新增的 `_adaptive_apply_commit_batch_limit()` 单独控制 `apply_commit_batch_queue -> resident set` 的 final batch commit，后半段 resident commit 现在拥有更清晰的双阶段 budget 控制面。
- <!-- updated: 2026-04-17 22:20 --> **[diagnostics-tests]** scheduler summary 已新增 `adaptive_apply_commit_limit_avg` 与 `adaptive_apply_commit_batch_limit_avg`，并补充 apply-commit-batch-limit 回归测试；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `123 passed, 1 warning`。
- <!-- updated: 2026-04-17 22:45 --> **[batch-commit-buffer]** `apply_commit_batch_queue` 现已从“逐 expert staged queue”收敛成真正的 batch 级 commit buffer：stage 阶段会把 ready commit 候选按 hotness/lifecycle 组装为 batch，后续 resident commit 直接消费 batch entries，而不是再逐 expert 构造 staged commit 单元。
- <!-- updated: 2026-04-17 22:45 --> **[batch-commit-buffer-diagnostics]** `HybridMoE.diagnostics()` 现已补充 `apply_commit_batch_queue_batches / committed_batches / background_apply_commit_batch_queue_committed_batches`，并新增 batch-queue grouping 回归；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `124 passed, 1 warning`。
- <!-- updated: 2026-04-17 23:05 --> **[batch-commit-budget-split]** resident commit 现在不仅有 batch buffer，还明确拆成 staged resolve 与 final batch commit 两级 budget：`max_commits / _adaptive_apply_commit_limit()` 负责 `apply_commit_queue` 的解析与 staging，`max_batch_commits / _adaptive_apply_commit_batch_limit()` 负责 `apply_commit_batch_queue -> resident set` 的 final batch commit。
- <!-- updated: 2026-04-17 23:22 --> **[resident-commit-batch-queue]** `HybridMoE` 新增 `resident_commit_batch_queue`，resident commit 的后半段链路已明确成 `apply_candidate_queue -> apply_commit_queue -> apply_commit_batch_queue -> resident_commit_batch_queue -> resident set`。background tick 现在会把 preexisting commit batch 再推进到 final resident-commit buffer，而 resident set commit 只消费该 buffer 中的 batch。
- <!-- updated: 2026-04-17 23:22 --> **[resident-commit-diagnostics]** `HybridMoE.diagnostics()`、`MigrationPipelineRuntime`、`LLM.reset_offload_diagnostics()` 和 scheduler summary 已补齐 `resident_commit_batch_queue_size / limit / utilization / enqueued / evictions / background_resident_commit_batch_queue_enqueued` 等指标，并新增 resident-commit staging 与 reset 回归；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `125 passed, 1 warning`。
- <!-- updated: 2026-04-17 23:34 --> **[resident-commit-prefinalize]** background tick 现在对 `resident_commit_batch_queue` 也区分“本轮之前已存在的 batch”和“本轮新推进的 batch”：只有前者会进入本轮 background apply，后者只计作 `resident_commit_batch_queue_prefinalized`，留到下一轮再进入最终 resident commit。
- <!-- updated: 2026-04-17 23:34 --> **[resident-commit-runtime-summary]** runtime、scheduler summary 和 reset 路径已补充 `offload_background_resident_commit_batch_queue_enqueued_total / prefinalized_total` 以及 layer 级 `resident_commit_batch_queue_committed_batches / background_resident_commit_batch_queue_committed_batches / prefinalized_batches`；resident commit 的 final buffer 现在真正进入可观测面。
- <!-- updated: 2026-04-17 05:18 --> **[resident-commit-finalize-queue]** `HybridMoE` 现已把 resident commit 再拆成 `resident_commit_batch_queue -> resident_commit_finalize_queue -> resident set`，background tick 可先把 final resident batches 预推进到 finalize queue，而真正 resident apply 只消费 finalize queue 中 preexisting 的 batch。
- <!-- updated: 2026-04-17 05:18 --> **[resident-commit-finalize-diagnostics]** `MigrationPipelineRuntime`、scheduler summary、`LLM.reset_offload_diagnostics()` 和单测已补齐 finalize queue 的 enqueue/prefinalize/committed 计数，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `126 passed, 1 warning`。
- <!-- updated: 2026-04-17 05:27 --> **[resident-commit-ready-cache]** resident commit 现已再拆成 `resident_commit_finalize_queue -> resident_commit_ready_cache -> resident set`：background tick 可先把 preexisting finalize batches 解析为 ready cache entries，而真正 resident apply 只消费 tick 开始前已存在的 ready commit batches。
- <!-- updated: 2026-04-17 05:27 --> **[resident-commit-ready-cache-diagnostics]** `MigrationPipelineRuntime`、scheduler summary、`LLM.reset_offload_diagnostics()` 和单测已补齐 ready cache 的 stores/utilization/runtime totals，当前 `tests/test_core.py + tests/test_pim_runtime.py` 仍为 `126 passed, 1 warning`。
- <!-- updated: 2026-04-17 05:42 --> **[resident-commit-apply-queue]** resident commit 后半段现已进一步拆成 `resident_commit_ready_cache -> resident_commit_apply_queue -> resident set`：background tick 可先把 preexisting ready resident batches 推进到 apply queue，真正 final resident apply 只消费 tick 开始前已存在的 apply batches，后台/前台边界进一步清晰。
- <!-- updated: 2026-04-17 05:42 --> **[resident-commit-apply-queue-diagnostics]** `HybridMoE.diagnostics()`、`MigrationPipelineRuntime`、scheduler summary 与 `LLM.reset_offload_diagnostics()` 已补齐 `resident_commit_apply_queue` 的 size/limit/utilization/enqueued/committed/background_enqueued 指标，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `126 passed, 1 warning`。
- <!-- updated: 2026-04-17 05:58 --> **[resident-commit-finalize-ready-queue]** resident commit 现已继续拆成 `resident_commit_apply_queue -> resident_commit_finalize_ready_queue -> resident set`：background tick 可先把 preexisting apply batches 预推进到 finalize-ready queue，真正 final resident apply 只消费 tick 开始前已存在的 finalize-ready batches。
- <!-- updated: 2026-04-17 05:58 --> **[resident-commit-finalize-ready-queue-diagnostics]** `HybridMoE.diagnostics()`、`MigrationPipelineRuntime`、scheduler summary 与 `LLM.reset_offload_diagnostics()` 已补齐 `resident_commit_finalize_ready_queue` 的 size/limit/utilization/enqueued/committed/background_enqueued 指标，当前 `tests/test_core.py + tests/test_pim_runtime.py` 仍为 `126 passed, 1 warning`。
- <!-- updated: 2026-04-17 06:20 --> **[benchmark-background-worker]** `benchmark_inference.py` 与 `example.py` 现已显式支持 `--enable-background-offload-worker` 和 `--background-offload-poll-interval-seconds`，真实 benchmark 路径不再只能走前台 refresh hook，可直接接通后台 offload worker。
- <!-- updated: 2026-04-17 06:20 --> **[benchmark-run-worker-lifecycle]** `run_single_generation()` 现在会像 `LLM.generate()` 一样，在单次生成前启动 background offload worker、结束后停止；补充 benchmark worker 生命周期回归后，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `127 passed, 1 warning`。

<!-- updated: 2026-04-19 02:20 -->

- 为 W4A32/GPTQ PIM quantized runtime 补充了分项 profiling，现可单独观察 host->DPU 输入传输、DPU launch/执行、DPU->host 回传与 runtime 总耗时。
- 在真实 `Qwen/Qwen3-30B-A3B-GPTQ-Int4` 上复测 `gate/down` operator-only case 后确认：当前 PIM 时间绝大部分集中在 `launch_seconds_avg`，说明瓶颈主要在 DPU 侧 kernel 执行而不是 host 传输。

<!-- updated: 2026-04-19 02:35 -->

- quantized PIM runtime 现已拆分权重加载与稳态运行 profiling：可分别观察 qweight/scales 常驻加载时间，以及运行期的 input transfer / DPU launch / output transfer。
- 在真实 Qwen3 GPTQ `gate/down` case 上，稳态 PIM 时间约 89%~98% 集中在 `launch_seconds_avg`，进一步确认当前瓶颈主要在 DPU kernel 执行而不是 host 传输。

<!-- updated: 2026-04-19 02:55 -->

- 试验性将 quantized DPU kernel 从 2-row 改为 4-row tile 以复用 input block，但在真实 Qwen3 GPTQ `gate/down` case 上反而变慢；该尝试未保留到代码。

<!-- updated: 2026-04-19 03:10 -->

- 对 quantized DPU kernel 做了参数 sweep：`TASKLETS=8` 在真实 GPTQ `gate/down` case 上略优于默认 `16`，而 `BLOCK_FLOATS=32/128` 基本无帮助；当前最佳稳定配置仍只带来很小改进。

<!-- updated: 2026-04-19 03:25 -->

- 为 quantized DPU kernel 增加了 transfer-only 模式，并在 benchmark 中输出 `pim_breakdown`。
- 真实 Qwen3 GPTQ `gate/down` case 表明：纯输入/输出搬运只占几毫秒，完整 PIM operator 时间的绝大部分仍是 DPU kernel 计算本体。

<!-- updated: 2026-04-19 03:45 -->

- 在真实 GPTQ `gate/down` 上做了 rank 与 batch 的 transfer-only breakdown sweep。结果表明：rank 调节主要改变少量传输与常数项，batch 增长时主导时间几乎全部落在 DPU 计算核，问题不是 host 传输扩展性。
