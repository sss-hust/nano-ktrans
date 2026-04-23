---
project: nano-ktrans
created: 2026-04-07
updated: 2026-04-22
---

# nano-ktrans

> 一个面向学习和实验的 Hybrid MoE 推理框架。**ADR-002 M-1 ~ M-10 已在真机 Qwen3-30B-A3B-GPTQ-Int4 上全部关闭**（`dev_gate check` = 全 PASS，累计 87 条 acceptance）。**M-10** 实装 Python `threading.Thread` async PIM submit 试图用 overlap 藏 PIM 时间，A/B 实测 @ offload=2 (-4.7%) / offload=32 (-3.1%) 都输 — Python GIL 争用 + 线程 spawn/join 开销 ~5 ms/call × 1488 call 完全吃掉收益，默认翻回 async OFF. **意外副产品**: A/B 里测出 `offload_device_experts=32 + async OFF` 跑出 **decode_tps = 0.3506, 超 M-4 peak 0.317 +10.6%** — 是 M-4 以来项目第一次 decode_tps 真正前进, 证明 weight residency 旋钮被严重低估. **M-9** routing Jaccard 均值 0.14 一次 1 行 diagnostic 证伪了 M-5~M-8 caching 栈假设. **M-4** 的 fused gate+up 仍是单点设计最大真实胜利. **M-3** BackendCostModel 让 prefill 比 cuda_cpu_offload 快 13.3×. **M-2** 真 T-MAC `kernel_mode=7` 是 publishable 负结果. Decode 仍差 CPU baseline ~8.7× (offload=32 ratio 0.114×), **M-11 双轨**: 选项 A 做 C-level `dpu_launch(DPU_ASYNCHRONOUS)` 消 Python roundtrip overhead; 选项 B 扫 offload_device_experts OOM 边界看能否推 32 做新默认. `pytest tests` = **242 passed** (+59 新单测覆盖 cost_model / dev_gate 扩展 / concat prep / dual runtime / slot LRU / layer group scoping / handle-based / locality diagnostic / CLI flag / async thread lifecycle).

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
