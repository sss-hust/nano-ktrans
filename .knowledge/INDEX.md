---
project: nano-ktrans
created: 2026-04-07
updated: 2026-04-22
---

# nano-ktrans

> 一个面向学习和实验的 Hybrid MoE 推理框架。当前已打通 CPU、`cuda_cpu_offload` 和实验性真实 DPU `pim` backend；主目标是让非专家层常驻 GPU、专家在 GPU/PIM 间动态迁移。**ADR-002 M-1 / M-2 / M-3 / M-4 已在真机 Qwen3-30B-A3B-GPTQ-Int4 上全部关闭**（`dev_gate check` = 全 PASS，共 34 条 acceptance 规则）。**M-4** 落地 fused gate+up：host 端把一个 expert 的 gate 和 up W4A32 权重沿 row 轴 concat，一次 DPU launch 同时算出两者，DPU binary 零改动。真机 decode TPS **0.228 → 0.317 (+39.2%)**，DPU 调用 −33.4%，数值 bit-exact。M-2 的真 T-MAC `kernel_mode=7` 被锁定为 publishable 负结果（ADR-002 §10）。M-3 的 `BackendCostModel` 数据驱动路由让 prefill 比 CPU baseline 快 13.3×，decode 因仍是同步 DPU launch 还差 CPU ~9.7×，留给 **M-5**（async + overlap + batched preload）。M-3 还顺带修了 M-1 遗留的 `CPUMoEBackend` 在 GPTQ + 无-AMX 时写 zeros 的假 baseline 回归。`pytest tests` 现为 **213 passed** (+27 新增单测覆盖 cost_model、dev_gate 扩展、concat prep)。

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
