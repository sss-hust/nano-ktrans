---
project: nano-ktrans
created: 2026-04-07
updated: 2026-04-22
---

# nano-ktrans

> 一个面向学习和实验的 Hybrid MoE 推理框架。**ADR-002 M-1 ~ M-9 已在真机 Qwen3-30B-A3B-GPTQ-Int4 上全部关闭**（`dev_gate check` = 全 PASS，累计 77 条 acceptance）。**M-9** 把 routing-locality histogram 做成 `PIMMoEBackend.diagnostics()` 一等公民 + 跑 5 个 group_size 真机 sweep，决定性地证明 Qwen3 top_k=8 decode 的 Jaccard similarity 均值只有 **0.14**（45.7% 样本 <10% 重叠），**slot-based caching 理论上限就 14%，被 32-rank-pool 的协调开销吃掉 10×**；默认翻回 `group_size=48` + `speculative_preload_gptq=False`，decode_tps 从 M-8 的 0.242 恢复到 **0.2844 (+18%)**。M-5/M-6/M-7/M-8 的 caching 栈 infra 全部保留，默认关闭，给 high-locality 场景的用户留着 `--pim-layer-group-size 3` 旋钮。**M-4** fused gate+up 仍是历史最高 decode_tps = 0.317 (−33% DPU calls, bit-exact)。**M-3** BackendCostModel 让 prefill 比 cuda_cpu_offload 快 13.3×。**M-2** 真 T-MAC `kernel_mode=7` 作为 publishable 负结果 (ADR §10) 保留。Decode 仍差 CPU baseline ~10.8× (ratio 0.093×), **M-10 做 `dpu_launch(DPU_ASYNCHRONOUS)`** — 第一个不依赖 routing locality 的 perf 杠杆, 估算差距可缩到 5-6×。`pytest tests` = **238 passed** (+55 新单测覆盖 cost_model / dev_gate 扩展 / concat prep / dual runtime / slot LRU / layer group scoping / handle-based / locality diagnostic / CLI flag).

## 技术栈

- **语言**: Python
- **框架**: PyTorch + Transformers
- **运行环境**: Python 3.10，CPU-only 可运行；CUDA / `flash-attn` / `kt-kernel` 为可选加速项
- **关键依赖**: `torch`、`transformers`、`safetensors`、`huggingface_hub`

## 当前状态

- **阶段**: 🟡 最小真实 PIM 数值链路已跑通，正在从"静态专家放置"转向"GPU/PIM 动态专家调度"
- **当前焦点**: → [current-focus.md](development/current-focus.md)

## 知识库导航

| 目录 | 说明 | 入口 |
|---|---|---|
| `architecture/` | 系统架构 & 设计决策 | [_INDEX.md](architecture/_INDEX.md) |
| `development/` | 开发进度 & 路线图 | [_INDEX.md](development/_INDEX.md) |
| `conventions/` | 编码约定 & 规范 | [_INDEX.md](conventions/_INDEX.md) |
| `context/` | 领域知识 & 注意事项 | [_INDEX.md](context/_INDEX.md) |
| `journal/` | 开发日志 | [_INDEX.md](journal/_INDEX.md) |

## 关键入口

- 🔥 [当前工作焦点](development/current-focus.md)
- 📋 [开发路线图](development/roadmap.md)
- ⚠️ [已知陷阱](context/gotchas.md)
- 📚 [相关研究工作（PIM+MoE）](context/related-work.md)
- 🧭 [ADR-001：PIM+MoE 研究综述与可借鉴创新点](architecture/decisions/001-pim-moe-offloading-literature.md)
