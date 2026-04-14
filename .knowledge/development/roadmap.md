---
updated: 2026-04-14
tags: [planning, roadmap]
---

# 📋 开发路线图

## 里程碑

### M1: 项目初始化 — 2026-04-07
<!-- status: ✅ 完成 -->

- [x] 初始化知识库
- [x] 建立 CPU-only、非 PIM 的最小可运行路径
- [x] 修复可选依赖缺失时的导入与运行问题
- [x] 用真实 Qwen3 checkpoint 验证 CPU 路径

### M2: Hybrid Offload Bring-up — 2026-04-09
<!-- status: ✅ 完成 -->

- [x] 接入 `cuda_cpu_offload`
- [x] 修复 Qwen3 expert layout 适配问题
- [x] 修复 CPU fallback 内存翻倍问题
- [x] 修复 attention fallback dtype 问题
- [x] 增加统一 inference benchmark
- [x] 增加 PIM microbenchmark 并完成真实硬件多 rank 实测
- [x] 将 `pim_shadow` 接入主推理链路

### M3: True PIM Expert Backend — 目标阶段
<!-- status: 🟡 进行中 -->

- [ ] 实现最小可运行的 DPU expert kernel
- [ ] 建立稳定的 Python / host / DPU 调用桥接
- [ ] 将 `PIMMoEBackend` 从 shadow 升级为真实数值 backend
- [ ] 形成 CPU / CUDA offload / PIM 的端到端对比基准
