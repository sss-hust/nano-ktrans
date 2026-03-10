# nano-ktrans Agent Context

## 项目定位

nano-ktrans 是 [KTransformers](https://github.com/kvcache-ai/ktransformers) 的学习简化版，专注于 **Mixtral-8x7B** 的 CPU/GPU 混合 MoE 推理。裁剪了大量工程细节（YAML 注入系统、多模型适配、CUDA Graph、Marlin 量化、GGUF、LoRA、多 GPU 并行），只保留核心架构思想，代码可读性优先。

---

## 架构概览

```
nano_ktrans/
├── llm.py                      # 用户入口：加载模型 → 生成文本
├── engine/
│   └── simple_engine.py        # 推理引擎：prefill / chunked prefill / decode
├── models/
│   └── mixtral.py              # Mixtral-8x7B 完整模型定义
├── layers/
│   ├── attention.py            # FlashAttention + Triton KV Cache Store
│   ├── hybrid_moe.py           # CPU/GPU 混合 MoE 调度层（核心）
│   ├── linear.py               # QKV/Column/Row Parallel Linear
│   ├── norm.py                 # RMSNorm（含 fused residual）
│   └── rotary_embedding.py     # RoPE
├── kernels/
│   ├── cpu_infer.py            # CPU 异步推理线程池（封装 kt_kernel_ext）
│   ├── cpu_moe.py              # CPU MoE 后端：Pinned Buffer + AMX GEMM
│   └── weight_loader.py        # SafeTensor 专家权重加载器
└── utils/
    ├── context.py              # 全局推理上下文（prefill/decode/chunked 状态）
    ├── expert_selection.py     # GPU 专家热度选择（激活频率驱动）
    └── loader.py               # 模型权重加载（packed_modules_mapping）
```

---

## 核心机制详解

### 1. Hybrid MoE（核心创新）

**文件**: `layers/hybrid_moe.py` + `kernels/cpu_moe.py`

数据流：
```
hidden_states → gate → topk routing
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   CPUMoEBackend.submit_forward()   GPU experts forward()
   (异步，非阻塞)                    (同步，PyTorch)
          │                             │
          ▼                             ▼
   CPUMoEBackend.sync_forward()     final_gpu_states
   (等待+拷贝回GPU)                     │
          │                             │
          └──────────┬──────────────────┘
                     ▼
              output = cpu_output + gpu_output
```

关键点：
- CPU 和 GPU 专家**并行计算**，CPU 不阻塞 GPU
- Pinned Memory 双缓冲避免每次重新分配
- `submit_with_cuda_stream` 保证 GPU→CPU 拷贝完成后再启动 CPU 计算

### 2. GPU 专家选择策略

**文件**: `utils/expert_selection.py`

- `generate_gpu_experts_masks(activation_freq, num_gpu_experts)` — 基于激活频率的数据驱动选择，**每层独立选热门专家**（ktransformers 的核心策略）
- `uniform_gpu_experts_masks()` — 均匀选前 N 个（无校准数据时的 fallback）
- `profile_expert_activation()` — 在校准数据上收集频率，通过 gate hook 统计

### 3. Attention 三模式

**文件**: `layers/attention.py`

| 模式 | 函数 | 场景 |
|------|------|------|
| 标准 Prefill | `flash_attn_varlen_func` | 短序列，一次处理完 |
| Chunked Prefill | `flash_attn_with_kvcache` + k/v 追加 | 长序列分块处理 |
| Decode | `flash_attn_with_kvcache` | 单 token 逐步生成 |

Triton `store_kvcache_kernel` 用于标准 prefill 和 decode 时写入 KV Cache。
Chunked prefill 利用 flash_attn_with_kvcache 的 `k=/v=` 参数自动追加并包含历史 KV。

### 4. 推理引擎

**文件**: `engine/simple_engine.py`

- KV Cache 预分配: `[1, max_seq_len, num_kv_heads, head_dim]`，挂载到每层的 `Attention` 实例
- 长序列自动切换 chunked prefill（阈值 = `chunk_size`，默认 512）
- Batch size = 1，单序列推理

### 5. 权重加载

**文件**: `utils/loader.py` + `kernels/weight_loader.py`

两套加载路径：
- **GPU 参数**: `loader.py` 的 `load_model()` 通过 `packed_modules_mapping` 映射 HF 权重名 → 模型参数名，自动处理 QKV 合并投影
- **CPU 专家权重**: `weight_loader.py` 的 `ExpertWeightLoader` 直接从 SafeTensor 加载 → stack → 传给 C++ AMX 在线量化

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
| CPU 异步线程池 + CUDA 流同步 | ✅ | CPUInferEngine 单例 |
| AMX/AVX CPU GEMM | ✅ | via kt-kernel C++ |
| Pinned Memory 双缓冲 | ✅ | PinnedBufferPool |
| FlashAttention (varlen + kvcache) | ✅ | 三模式 |
| Triton KV Cache Store | ✅ | store_kvcache_kernel |
| GPU 专家激活频率选择 | ✅ | generate_gpu_experts_masks |
| Chunked Prefill | ✅ | 长序列自动分块 |
| SafeTensor 权重加载 | ✅ | packed_modules_mapping |
| RoPE + RMSNorm | ✅ | torch.compile RoPE |
| YAML 注入优化系统 | ❌ | 裁剪：直接编码替代 |
| 多模型 (DeepSeek V2/V3, Qwen) | ❌ | 裁剪：专注 Mixtral |
| Marlin GPU 量化 | ❌ | 裁剪：硬件特定 |
| CUDA Graph | ❌ | 裁剪：高级优化 |
| GGUF 格式 | ❌ | 裁剪：只保留 SafeTensor |
| LoRA 微调 | ❌ | 裁剪：非推理核心 |
| 多 GPU 张量并行 | ❌ | 裁剪：专注单节点 |
| FlashInfer 分页 KV | ❌ | 裁剪：用连续 KV Cache |
| KV 压缩 (DeepSeek V3) | ❌ | 裁剪：模型特定 |

---

## 依赖

- `torch >= 2.1.0` — 核心框架
- `triton >= 2.1.0` — KV Cache Store kernel
- `flash-attn >= 2.5.0` — 高性能注意力
- `transformers >= 4.36.0` — Tokenizer + Config
- `safetensors >= 0.4.0` — 权重文件读取
- `kt-kernel` — CPU AMX/AVX GEMM C++ 扩展

---

## 测试

```bash
pytest tests -v
```

22 个测试覆盖：RMSNorm、Linear 系列、RoPE、MixtralConfig、MoE 路由逻辑、Context 生命周期（含 chunked prefill）、权重加载、单专家前向、**GPU 专家选择**（基本/全 GPU/均匀回落/逐层差异化）。

依赖 `flash_attn` 和 `triton` 的测试需要 CUDA 环境。纯 CPU 逻辑测试无需 GPU。

---

## 已知的设计决策

1. **单序列推理** — 不支持 batch > 1，简化 KV Cache 管理
2. **连续 KV Cache** — 不做分页（FlashInfer PagedAttention），减少复杂度
3. **在线量化** — CPU 专家权重通过 C++ `moe.load_weights_task` 在线量化为 INT4/INT8
4. **全局 Context** — 用全局 dataclass 传递 prefill/decode 状态，避免层层传参
5. **gate 在 GPU** — Router gate 始终在 GPU 执行，只有专家 FFN 分配到 CPU/GPU
