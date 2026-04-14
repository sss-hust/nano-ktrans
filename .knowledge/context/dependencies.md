---
updated: 2026-04-07
tags: [dependencies]
---

# 📦 外部依赖 & 注意事项

<!-- updated: 2026-04-07 20:56 -->

## Core 依赖

- `torch>=2.4.0`
- `transformers>=4.36.0`
- `safetensors>=0.4.0`
- `huggingface_hub>=0.23.0`

这些依赖足够让项目在 CPU-only、无 PIM、无 `flash-attn`、无 `kt-kernel` 的模式下运行和测试。

<!-- updated: 2026-04-07 20:56 -->

## Optional extras

- `.[cuda]`：安装 `triton` + `flash-attn`
- `.[cpu-kernel]`：安装 `kt-kernel`
- `.[accel]`：一次性安装全部加速依赖

<!-- updated: 2026-04-07 20:56 -->

## 环境注意事项

- 如果环境变量 `HTTP_PROXY` / `HTTPS_PROXY` 指向不可用的本地代理，`pip install` 会失败。当前机器需要临时 `env -u HTTP_PROXY -u HTTPS_PROXY -u NO_PROXY ...` 才能正常安装。
