---
updated: 2026-04-14 11:30
---

# 🔥 当前工作焦点

## 正在进行

- [x] 建立 CPU-only、非 PIM 的最小可运行路径
- [x] 在真实权重上验证 CPU 路径并补齐 benchmark 入口
- [x] 在宿主机上打通 `cuda_cpu_offload` 并拿到真实 Qwen3 benchmark
- [x] 在真实 UPMEM 硬件上跑通多 rank PIM microbenchmark
- [x] 将 `pim_shadow` 接入主推理链路并记录 PIM 可见性与路由统计
- [ ] 为 `PIMMoEBackend` 接入真实 DPU 数值执行
- [ ] 为 CPU offload 接入 `kt-kernel` 或等价高性能实现

## 阻塞项

- `Qwen3-30B-A3B` 在本机 `47.41 GiB` GPU 上走全专家纯 `cuda` 路径仍然 OOM
- 当前 `pim_shadow` 仍由 CPU fallback 负责数值正确性，尚未把专家 MLP 真正放到 DPU 上执行
- Python 侧直接驱动 UPMEM 的尝试仍不稳定，当前更可靠的是独立 C host benchmark
- `HTTP_PROXY` / `HTTPS_PROXY` 指向 `127.0.0.1:7897` 时会阻塞 `pip install`

## 下一步

- 在 `PIMMoEBackend` 中先落一个最小可验证的单 token / 单 expert DPU kernel
- 建立 Python -> UPMEM host bridge，把 `pim_shadow` 升级成真正的 `pim` backend
- 对比 `cpu`、`cuda_cpu_offload`、`pim` 三条链路的 prefill/decode 延迟与 offload 命中分布
- 继续补充架构说明、依赖说明和版本化文档

## 本轮对话上下文

> 当前 repo 已完成阶段性里程碑：CPU 基线、Qwen3 真实权重适配、`cuda_cpu_offload` 真机验证、`pim_shadow` 主链路接入，以及真实 UPMEM 多 rank benchmark。未完成项是把专家数值计算真正迁移到 DPU。
