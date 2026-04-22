# nano-ktrans Agent Context

## 项目定位

nano-ktrans 是 [KTransformers](https://github.com/kvcache-ai/ktransformers) 的学习简化版，聚焦在 **CPU/GPU/PIM 混合 MoE 推理** 的教学级实现。
当前支持的模型族：

- **Mixtral-8x7B**（原始目标）
- **Qwen2-MoE**（带共享专家的稀疏 MLP）
- **Qwen3-MoE**（packed 与 unpacked 两种专家布局，带 q/k norm）
- DeepSeek-V2/V3：未实现（阻塞在 MLA attention 适配）

相比上游 ktransformers，仍然裁剪了：YAML 注入系统、Marlin GPU 量化、CUDA Graph、GGUF、LoRA、多 GPU 张量并行、FlashInfer PagedAttention、DeepSeek V3 的 KV 压缩。

---

## 架构概览

```
nano_ktrans/
├── llm.py                      # 用户入口：加载模型 → 生成文本
├── engine/
│   └── simple_engine.py        # 推理引擎：prefill / chunked prefill / decode
├── models/
│   ├── config.py               # GenericMoeConfig + architecture specs
│   └── mixtral.py              # 通用 MoE 模型定义（mixtral/qwen2/qwen3 共用）
├── layers/
│   ├── attention.py            # FlashAttention + Triton KV Cache Store + torch fallback
│   ├── hybrid_moe.py           # CPU/GPU/PIM 混合 MoE 调度层（核心）
│   ├── expert_mlp.py           # GPU 常驻专家 MLP（含 packed gate_up 支持）
│   ├── linear.py               # QKV/Column/Row/Merged Parallel Linear
│   ├── norm.py                 # RMSNorm（含 fused residual）
│   └── rotary_embedding.py     # RoPE（支持 identity 类型 scaling）
├── kernels/
│   ├── cpu_infer.py            # CPU 异步推理线程池（封装 kt_kernel_ext）
│   ├── cpu_moe.py              # CPU MoE 后端：Pinned Buffer + AMX GEMM
│   ├── offload_backend.py      # ExpertOffloadBackend 抽象基类
│   ├── pim_moe.py              # PIM MoE 后端（UPMEM DPU 实专家路径）
│   ├── pim_expert_runtime.py   # FP32 DPU 专家桥接
│   ├── pim_linear_runtime.py   # INT8 DPU 线性桥接
│   ├── pim_quantized_runtime.py# GPTQ DPU 路径
│   ├── pim_native/             # DPU 侧 C 源码
│   ├── expert_migration.py     # 迁移生命周期 (Requested → Ready → Committed)
│   ├── expert_materialization.py # 后台专家物化线程池
│   ├── migration_runtime.py    # 流水线化 tick 管理
│   ├── quantized_ops.py        # 量化算子
│   └── weight_loader.py        # SafeTensor 专家权重 + GPTQ 加载器
├── scheduler/
│   ├── dynamic_expert_scheduler.py  # 运行时热度驱动的 GPU↔PIM 专家迁移
│   ├── profiles.py                  # 预设 profile (baseline / eager / overlap_safe)
│   └── diagnostics.py               # counter/profile 汇总
└── utils/
    ├── context.py              # 全局推理上下文（prefill/decode/chunked 状态）
    ├── expert_selection.py     # GPU 专家热度选择（激活频率驱动）
    ├── expert_runtime_state.py # ExpertResidencyPlan / Migration Op
    └── loader.py               # 模型权重加载（packed_modules_mapping）
```

---

## 核心机制详解

### 1. Hybrid MoE（核心创新）

**文件**: `layers/hybrid_moe.py` + `kernels/cpu_moe.py` + `kernels/pim_moe.py`

数据流：
```
hidden_states → gate → topk routing
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   OffloadBackend.submit_forward()   GPU experts forward()
   (异步，非阻塞)                     (同步，PyTorch)
          │                             │
          ▼                             ▼
   OffloadBackend.sync_forward()     final_gpu_states
   (等待+拷贝回GPU)                      │
          │                             │
          └──────────┬──────────────────┘
                     ▼
              output = offload_output + gpu_output
```

关键点：
- CPU/PIM 专家与 GPU 专家**并行计算**，不阻塞 GPU
- Pinned Memory 双缓冲避免每次重新分配
- `submit_with_cuda_stream` 保证 GPU→CPU 拷贝完成后再启动 CPU 计算
- PIM 路径额外支持 FP32 / INT8 线性 / GPTQ 三套 DPU runtime，并在失败时回落到 CPU

### 2. GPU 专家选择 & 动态调度

**静态选择** (`utils/expert_selection.py`)：
- `generate_gpu_experts_masks(activation_freq, num_gpu_experts)` — 基于激活频率的数据驱动选择
- `uniform_gpu_experts_masks()` — 均匀选前 N 个（无校准数据时的 fallback）
- `profile_expert_activation()` — 在校准数据上收集频率，通过 gate hook + bincount 统计

**动态调度** (`scheduler/dynamic_expert_scheduler.py`)：
- 推理过程中按 token 统计热度 EMA，周期性产出迁移计划
- GPU↔PIM/CPU 之间按预算进行 promotion/demotion
- `hybrid_moe.py` 在 demotion 时通过 `notify_expert_evicted()` 通知后端清理 DPU 缓存

### 3. Attention 三模式

**文件**: `layers/attention.py`

| 模式 | 函数 | 场景 |
|------|------|------|
| 标准 Prefill | `flash_attn_varlen_func` | 短序列，一次处理完 |
| Chunked Prefill | `flash_attn_with_kvcache` + k/v 追加 | 长序列分块处理 |
| Decode | `flash_attn_with_kvcache` | 单 token 逐步生成 |
| Torch fallback | `F.scaled_dot_product_attention` + 手写 KV 写入 | 无 flash-attn/CUDA 环境 |

Triton `store_kvcache_kernel` 用于标准 prefill 和 decode 时写入 KV Cache。
Chunked prefill 利用 flash_attn_with_kvcache 的 `k=/v=` 参数自动追加并包含历史 KV。

### 4. 推理引擎

**文件**: `engine/simple_engine.py`

- KV Cache 预分配: `[1, max_seq_len, num_kv_heads, head_dim]`，挂载到每层的 `Attention` 实例
- 长序列自动切换 chunked prefill（阈值 = `chunk_size`，默认 512）
- Batch size = 1，单序列推理
- 可选 background offload worker，与 dynamic scheduler 协同

### 5. 权重加载

**文件**: `utils/loader.py` + `kernels/weight_loader.py`

两套加载路径：
- **GPU 参数**: `loader.py` 的 `load_model()` 通过 `packed_modules_mapping` 映射 HF 权重名 → 模型参数名，自动处理 QKV 合并投影；加载结束打印 loaded/skipped 计数
- **CPU/PIM 专家权重**: `weight_loader.py` 的 `ExpertWeightLoader` 直接从 SafeTensor 加载 → stack → 传给 C++ AMX 在线量化 / DPU preload；支持 GPTQ INT4/INT8 自动检测

### 6. 张量形状约定

模型内部统一使用 **2D `[total_tokens, hidden_size]`**：
- `MixtralModel.forward()` 入口 flatten `[batch, seq, hidden] → [batch*seq, hidden]`
- 各层（attention, MoE, norm）全程 2D 传递
- 出口 restore 回 3D 给 `lm_head`

---

## 与 KTransformers 的功能覆盖对照

| ktransformers 功能 | nano-ktrans | 说明 |
|-------------------|-------------|------|
| Hybrid MoE (CPU/GPU 混合) | ✅ | 核心架构一致 |
| PIM / UPMEM DPU offload | ✅ | 实 DPU 专家路径 + shadow 路径 |
| CPU 异步线程池 + CUDA 流同步 | ✅ | CPUInferEngine 单例 |
| AMX/AVX CPU GEMM | ✅ | via kt-kernel C++ |
| Pinned Memory 双缓冲 | ✅ | PinnedBufferPool |
| FlashAttention (varlen + kvcache) | ✅ | 三模式 + torch fallback |
| Triton KV Cache Store | ✅ | store_kvcache_kernel |
| GPU 专家激活频率选择 | ✅ | generate_gpu_experts_masks |
| 动态 GPU↔PIM 专家调度 | ✅ | DynamicExpertScheduler |
| Chunked Prefill | ✅ | 长序列自动分块 |
| SafeTensor 权重加载 | ✅ | packed_modules_mapping |
| GPTQ INT4/INT8 权重加载 | ✅ | GPTQLinearWeight |
| RoPE + RMSNorm | ✅ | torch.compile RoPE，支持 identity scaling |
| Mixtral / Qwen2-MoE / Qwen3-MoE | ✅ | GenericMoeConfig + 自动布局检测 |
| YAML 注入优化系统 | ❌ | 裁剪：直接编码替代 |
| DeepSeek V2/V3 (MLA) | ❌ | 阻塞在 MLA 适配 |
| Marlin GPU 量化 | ❌ | 裁剪：硬件特定 |
| CUDA Graph | ❌ | 裁剪：高级优化 |
| GGUF 格式 | ❌ | 裁剪：只保留 SafeTensor |
| LoRA 微调 | ❌ | 裁剪：非推理核心 |
| 多 GPU 张量并行 | ❌ | 裁剪：专注单节点 |
| FlashInfer 分页 KV | ❌ | 裁剪：用连续 KV Cache |

---

## 依赖

- `torch >= 2.4.0` — 核心框架
- `triton >= 2.1.0` — KV Cache Store kernel (`[cuda]`/`[accel]`)
- `flash-attn == 2.6.3` — 高性能注意力 (`[cuda]`/`[accel]`)
- `transformers >= 4.36.0` — Tokenizer + Config
- `safetensors >= 0.4.0` — 权重文件读取
- `huggingface_hub >= 0.23.0` — 远程 checkpoint 拉取
- `kt-kernel` — CPU AMX/AVX GEMM C++ 扩展 (`[cpu-kernel]`/`[accel]`)

CPU-only 场景默认回落到纯 PyTorch：flash-attn / triton / kt-kernel 均可选。

---

## 测试

```bash
pytest tests -v
```

测试文件：
- `tests/test_core.py` — RMSNorm、Linear 系列、RoPE、MixtralConfig、MoE 路由逻辑、Context 生命周期、权重加载、专家前向、GPU 专家选择、迁移 eviction 通知等
- `tests/test_smoke_cpu.py` — CPU-only 端到端 smoke
- `tests/test_pim_runtime.py` — 真实 DPU 路径（需要 `/dev/dpu_rank*` + UPMEM toolchain）
- `tests/test_quantized_ops.py` — 量化算子

依赖 `flash_attn` 和 `triton` 的测试需要 CUDA 环境。纯 CPU / 调度 / 迁移逻辑测试无需 GPU。

---

## 已知的设计决策

1. **单序列推理** — 不支持 batch > 1，简化 KV Cache 管理
2. **连续 KV Cache** — 不做分页（FlashInfer PagedAttention），减少复杂度
3. **在线量化** — CPU 专家权重通过 C++ `moe.load_weights_task` 在线量化为 INT4/INT8
4. **全局 Context** — 用全局 dataclass 传递 prefill/decode 状态，避免层层传参
5. **gate 在 GPU** — Router gate 始终在 GPU 执行，只有专家 FFN 分配到 CPU/GPU/PIM
6. **PIM 最佳努力** — DPU 运行失败时静默回落到 CPU，并通过 `fallback_counts` 记录原因
