---
project: nano-ktrans
created: 2026-04-07
updated: 2026-04-22
---

# nano-ktrans

> 一个面向学习和实验的 Hybrid MoE 推理框架。**ADR-002 M-1 ~ M-11 已在真机 Qwen3-30B-A3B-GPTQ-Int4 上全部关闭**（`dev_gate check` = 全 PASS，累计 97 条 acceptance）。**M-11** 系统扫 `offload_device_experts`，把 `benchmark_inference.py` 默认从 2 改为 **88**：M-11 final `decode_tps=0.6226`，相对 M-9 final 0.284 **+119%**，相对 M-4 peak 0.317 **+96%**。offload=94 是 short/medium peak 但 long prompt OOM，95/96 short OOM，因此 88 是当前 47GB GPU 的安全高性能默认。PIM 端到端 decode 仍低于 CPU baseline 3.07（ratio 0.203×，差 4.9×），但差距已从 M-9 的 10.8× 缩半。`pytest tests` = **246 passed**。M-12 将在 offload=88 新 baseline 上做 OOM envelope 扩展与 C-level batched/async DPU launch。

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
