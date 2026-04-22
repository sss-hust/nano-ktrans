---
project: nano-ktrans
created: 2026-04-07
updated: 2026-04-22
---

# nano-ktrans

> 一个面向学习和实验的 Hybrid MoE 推理框架。当前已打通 CPU、`cuda_cpu_offload` 和实验性真实 DPU `pim` backend；新的主目标是让非专家层常驻 GPU，而专家在 GPU/PIM 间动态迁移与调度。已在 `2026-04-21` 落地 P1 (MRS score-aware hotness) 与 P2 (Expert Map Store + prompt 语义预取) 的最小可用版本，两者默认关闭，需显式启用。`2026-04-22` 修复了 v0.3.0-rc1 测试套件的 3 个历史回归；**ADR-002 M-1 已在真机 Qwen3-30B-A3B-GPTQ-Int4 上验收通过**（`dev_gate check M-1` = PASS 6/6 rules），期间修了 GPTQ checkpoint layout 下游 4 处级联 bug + loader/index 进程级缓存（加速 ≈150×）；实测 mode=4 (int8 fixed) peak ratio = **3.36× CPU grouped**，确证 mode=6 "T-MAC" 伪实现 peak 仅 1.11×。`pytest tests` 现为 **186 passed**。

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
