---
section: 领域知识
created: 2026-04-21
updated: 2026-04-21
tags: [related-work, pim, moe, offloading, literature]
---

# PIM + MoE Offloading 相关研究工作速查

> 本文件是 [ADR-001](../architecture/decisions/001-pim-moe-offloading-literature.md) 的配套速查表，用于在讨论设计方向时快速定位"已经有哪篇论文做过类似思路"。详细借鉴分析见 ADR。

## MoE Offloading（CPU/GPU 派）

| 方法 | 放专家的粒度 | Cache 策略 | 预取预测信号 | 是否可借鉴 |
|------|-------------|-----------|-------------|-----------|
| MoE-Infinity | Expert | activation-aware | Request-level EAM | ⚠️ 已被 fMoE 超越 |
| Fiddler | Expert，就地在 CPU 算 | — | Cost model（CPU 算 vs 拷贝到 GPU） | ✅ 可借鉴（P3） |
| HOBBIT | Expert + Mixed-precision | 三层分层 token/layer/sequence | Cache-miss + importance | ✅ 可借鉴（P4） |
| HybriMoE | Expert | **MRS score-aware** | 3 层 gating 预测 + impact-driven | ✅ **首选借鉴**（P1） |
| fMoE / FineMoE | Iteration-level | LFU + expert map | **Prompt 语义 + 轨迹相似度** | ✅ **首选借鉴**（P2） |
| FloE | Expert | — | On-the-fly 内存约束 | 📚 参考 |
| AdapMoE | Expert | LRU | 静态 | 📚 参考 |
| PowerInfer | Neuron | LFU | Decode only | 📚 参考（粒度更细） |

## PIM 派

| 方法 | 硬件 | 映射 | 是否可借鉴 |
|------|------|------|-----------|
| PIMoE | NPU + UPMEM | Throttle-aware expert 分配 | ✅ 可借鉴（P6，软件侧） |
| NeuPIMs | NPU + HBM-PIM | GEMM→NPU, GEMV→PIM + sub-batch 交织 | 🟡 部分可借鉴（P5） |
| AttAcc! | xPU + HBM-PIM | KV cache 常驻 PIM | ❌ 硬件侧 |
| PIM-AI | DDR5/LPDDR5 PIM | Decode 阶段整体卸载 | ❌ 硬件侧 |
| P3-LLM | NPU + PIM | 混合数值格式 + 算子融合 | ❌ 硬件侧 |
| UpDLRM | UPMEM | 推荐系统 embedding | 📚 同硬件的 lookalike baseline |

## 关键术语对照

| 论文术语 | nano-ktrans 对应概念 |
|---------|---------------------|
| EAM (Expert Activation Matrix) | `activation_freq[L, E]` + `LayerExpertState.hotness` |
| Prefetch cache (HybriMoE) | `ExpertMaterializationManager.staging_cache` |
| Warm / Hot expert (HOBBIT) | `warm_cache` / `activated_cache` |
| MRS score (HybriMoE) | 建议新增 `router_score_ema` |
| Expert Map (fMoE) | 建议新增 `ExpertMapStore` |
| Apply queue / Commit queue | nano-ktrans 独有的多级 staged commit |
| Throttle signal (PIMoE) | `runtime.last_active_dpus` + 新增 DPU 利用率采样 |

## 诊断指标对照

| 论文指标 | nano-ktrans 等价指标 |
|---------|---------------------|
| Expert hit rate | `decode_prefetch_hits / (hits + misses)` |
| Cold path ratio | `pipeline_promotion_source_cold / pipeline_promotion_source_*` |
| Batch apply size | `runtime_apply_batch_size_avg` |
| Prefetch redundancy | `prefetch_requested - prefetch_enqueued` |

## 快速引用

- 需要论据证明"score-aware cache 显著优于 LFU/LRU" → 引 HybriMoE Table / §IV
- 需要论据证明"prompt 语义能预测专家" → 引 fMoE §5.2（Pearson 相关性）
- 需要论据证明"小 batch 下 CPU 本地算 > PCIe 搬运" → 引 Fiddler beam search 11.57× 结果
- 需要论据证明"混合精度 offloading 不破坏精度" → 引 HOBBIT §3.2
