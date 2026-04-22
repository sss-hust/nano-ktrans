---
project: nano-ktrans
created: 2026-04-07
updated: 2026-04-22
---

# nano-ktrans

> 一个面向学习和实验的 Hybrid MoE 推理框架。当前已打通 CPU、`cuda_cpu_offload` 和实验性真实 DPU `pim` backend；主目标是让非专家层常驻 GPU、专家在 GPU/PIM 间动态迁移。**ADR-002 M-1 ~ M-6 已在真机 Qwen3-30B-A3B-GPTQ-Int4 上全部关闭**（`dev_gate check` = 全 PASS，共 49 条 acceptance 规则）。**M-6** 落地 DPU binary multi-slot MRAM + host-side 8-slot LRU：micro-bench 证明 hit 正确且 bit-exact，但 e2e hit=0% 因 48 层共享 `get_shared()` 单例洗穿 8 slot（项目内第三次 null e2e 负结果，M-7 per-layer scoping 是单一解）。**M-4** 的 fused gate+up 仍是最大单点胜利（DPU 调用 −33.4%，decode_tps +39.2%，bit-exact）。**M-3** 的 `BackendCostModel` 让 prefill 比 cuda_cpu_offload 快 13.3×。**M-2** 的真 T-MAC `kernel_mode=7` 是 publishable 负结果（ADR-002 §10）。decode 仍差 CPU baseline ~10×（ratio 0.098×），M-7 攻破 48 层共享 + async launch 后预期追回一半。`pytest tests` 现为 **224 passed** (+38 新单测覆盖 cost_model / dev_gate 扩展 / concat prep / dual runtime / slot LRU)。

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
