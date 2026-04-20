#define _POSIX_C_SOURCE 200809L

#include <dpu/dpu.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

#ifndef MAX_QWEIGHT_WORDS
#define MAX_QWEIGHT_WORDS 2097152
#endif

#ifndef MAX_SCALE_FLOATS
#define MAX_SCALE_FLOATS 65536
#endif

#ifndef MAX_INPUT_FLOATS
#define MAX_INPUT_FLOATS 65536
#endif

#ifndef MAX_INPUT_INT8
#define MAX_INPUT_INT8 MAX_INPUT_FLOATS
#endif

#ifndef MAX_OUTPUT_FLOATS
#define MAX_OUTPUT_FLOATS 65536
#endif

#ifndef MAX_OUTPUT_INT32
#define MAX_OUTPUT_INT32 MAX_OUTPUT_FLOATS
#endif

#ifndef MAX_LUT_INT16
#define MAX_LUT_INT16 (MAX_SCALE_FLOATS * (1 << BITS_PER_WEIGHT))
#endif

#ifndef BLOCK_FLOATS
#define BLOCK_FLOATS 64
#endif

#ifndef MAX_RUNTIME_LUT_INT16
#define MAX_RUNTIME_LUT_INT16 131072
#endif

#ifndef BITS_PER_WEIGHT
#define BITS_PER_WEIGHT 4
#endif

#ifndef WEIGHTS_PER_WORD
#define WEIGHTS_PER_WORD (32 / BITS_PER_WEIGHT)
#endif

static struct dpu_set_t g_set;
static bool g_initialized = false;
static bool g_weights_loaded = false;
static uint64_t g_last_cycles = 0;
static double g_last_load_qweight_transfer_seconds = 0.0;
static double g_last_load_scale_transfer_seconds = 0.0;
static double g_last_load_total_seconds = 0.0;
static double g_last_input_transfer_seconds = 0.0;
static double g_last_launch_seconds = 0.0;
static double g_last_output_transfer_seconds = 0.0;
static double g_last_total_seconds = 0.0;
static uint32_t g_nr_dpus = 0;
static uint32_t g_input_dim = 0;
static uint32_t g_output_dim = 0;
static uint32_t g_group_size = 0;
static uint32_t g_num_groups = 0;
static uint32_t g_kernel_mode = 0;
static float g_input_scale = 1.0f;
static size_t g_rows_per_dpu = 0;
static size_t g_shard_output_dim = 0;
static uint32_t *g_valid_rows = NULL;
static int16_t *g_lut_i16_shards = NULL;
static int8_t *g_input_i8_shards = NULL;
static int32_t *g_output_i32_shards = NULL;
static int16_t *g_runtime_lut_i16_shards = NULL;

static void
set_error(char *error_buffer, size_t error_buffer_len, const char *fmt, ...)
{
    if (error_buffer == NULL || error_buffer_len == 0) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    vsnprintf(error_buffer, error_buffer_len, fmt, args);
    va_end(args);
}

static double
timespec_diff_seconds(const struct timespec *start, const struct timespec *end)
{
    const time_t sec = end->tv_sec - start->tv_sec;
    const long nsec = end->tv_nsec - start->tv_nsec;
    return (double)sec + ((double)nsec / 1000000000.0);
}

static int
check_dpu_error(
    dpu_error_t error,
    char *error_buffer,
    size_t error_buffer_len,
    const char *context)
{
    if (error == DPU_OK) {
        return 0;
    }

    set_error(
        error_buffer,
        error_buffer_len,
        "%s failed: %s",
        context,
        dpu_error_to_string(error));
    return -1;
}

int
pim_quantized_init(
    const char *binary_path,
    const char *profile,
    uint32_t rank_count,
    char *error_buffer,
    size_t error_buffer_len)
{
    if (g_initialized) {
        return 0;
    }

    if (binary_path == NULL || binary_path[0] == '\0') {
        set_error(error_buffer, error_buffer_len, "binary_path is required");
        return -1;
    }
    if (rank_count == 0) {
        set_error(error_buffer, error_buffer_len, "rank_count must be positive");
        return -1;
    }

    const char *effective_profile = (profile != NULL && profile[0] != '\0') ? profile : NULL;
    if (check_dpu_error(
            dpu_alloc_ranks(rank_count, effective_profile, &g_set),
            error_buffer,
            error_buffer_len,
            "dpu_alloc_ranks") != 0) {
        return -1;
    }
    if (check_dpu_error(
            dpu_get_nr_dpus(g_set, &g_nr_dpus),
            error_buffer,
            error_buffer_len,
            "dpu_get_nr_dpus") != 0) {
        dpu_free(g_set);
        memset(&g_set, 0, sizeof(g_set));
        g_nr_dpus = 0;
        return -1;
    }
    if (check_dpu_error(dpu_load(g_set, binary_path, NULL), error_buffer, error_buffer_len, "dpu_load") != 0) {
        dpu_free(g_set);
        memset(&g_set, 0, sizeof(g_set));
        g_nr_dpus = 0;
        return -1;
    }

    g_initialized = true;
    g_weights_loaded = false;
    g_last_cycles = 0;
    g_last_load_qweight_transfer_seconds = 0.0;
    g_last_load_scale_transfer_seconds = 0.0;
    g_last_load_total_seconds = 0.0;
    g_last_input_transfer_seconds = 0.0;
    g_last_launch_seconds = 0.0;
    g_last_output_transfer_seconds = 0.0;
    g_last_total_seconds = 0.0;
    return 0;
}

int
pim_quantized_load_weights(
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t group_size,
    uint32_t kernel_mode,
    const void *packed_qweights,
    const void *scales,
    char *error_buffer,
    size_t error_buffer_len)
{
    struct dpu_set_t dpu;
    uint32_t dpu_index = 0;
    int rc = -1;
    uint32_t *qweight_shards = NULL;
    float *scale_shards = NULL;
    int16_t *lut_i16_shards = NULL;
    struct timespec total_start;
    struct timespec total_end;
    struct timespec qweight_start;
    struct timespec qweight_end;
    struct timespec scale_start;
    struct timespec scale_end;

    const uint32_t words_per_row = input_dim / WEIGHTS_PER_WORD;
    const uint32_t num_groups = input_dim / group_size;
    size_t shard_qweight_words = 0;
    size_t shard_scale_floats = 0;
    size_t shard_lut_i16 = 0;

    if (!g_initialized) {
        set_error(error_buffer, error_buffer_len, "pim_quantized_init must be called first");
        return -1;
    }
    if (packed_qweights == NULL || scales == NULL) {
        set_error(error_buffer, error_buffer_len, "packed_qweights and scales must be non-null");
        return -1;
    }
    if ((input_dim % WEIGHTS_PER_WORD) != 0 || (input_dim % group_size) != 0) {
        set_error(error_buffer, error_buffer_len, "input_dim must be divisible by pack factor and group_size");
        return -1;
    }

    g_input_dim = input_dim;
    g_output_dim = output_dim;
    g_group_size = group_size;
    g_num_groups = num_groups;
    g_kernel_mode = kernel_mode;
    g_rows_per_dpu = ((size_t)output_dim + (size_t)g_nr_dpus - 1u) / (size_t)g_nr_dpus;
    g_shard_output_dim = g_rows_per_dpu + (g_rows_per_dpu % 2u);
    shard_qweight_words = g_shard_output_dim * (size_t)words_per_row;
    shard_scale_floats = g_shard_output_dim * (size_t)num_groups;
    shard_lut_i16 = shard_scale_floats * (size_t)(1u << BITS_PER_WEIGHT);

    if (shard_qweight_words > MAX_QWEIGHT_WORDS) {
        set_error(error_buffer, error_buffer_len, "qweight shard too large");
        return -1;
    }
    if (shard_scale_floats > MAX_SCALE_FLOATS) {
        set_error(error_buffer, error_buffer_len, "scale shard too large");
        return -1;
    }
    if (shard_lut_i16 > MAX_LUT_INT16) {
        set_error(error_buffer, error_buffer_len, "lut shard too large");
        return -1;
    }

    free(g_valid_rows);
    free(g_lut_i16_shards);
    g_lut_i16_shards = NULL;
    free(g_runtime_lut_i16_shards);
    g_runtime_lut_i16_shards = NULL;
    g_valid_rows = calloc(g_nr_dpus, sizeof(*g_valid_rows));
    qweight_shards = calloc((size_t)g_nr_dpus * shard_qweight_words, sizeof(*qweight_shards));
    scale_shards = calloc((size_t)g_nr_dpus * shard_scale_floats, sizeof(*scale_shards));
    lut_i16_shards = calloc((size_t)g_nr_dpus * shard_lut_i16, sizeof(*lut_i16_shards));
    if (g_valid_rows == NULL || qweight_shards == NULL || scale_shards == NULL || lut_i16_shards == NULL) {
        set_error(error_buffer, error_buffer_len, "failed to allocate weight shard buffers");
        goto cleanup;
    }

    clock_gettime(CLOCK_MONOTONIC, &total_start);

    const uint32_t *qweight_src = (const uint32_t *)packed_qweights;
    const float *scale_src = (const float *)scales;
    for (dpu_index = 0; dpu_index < g_nr_dpus; ++dpu_index) {
        const size_t row_start = (size_t)dpu_index * g_rows_per_dpu;
        const size_t rows_remaining = row_start < (size_t)output_dim ? (size_t)output_dim - row_start : 0;
        const size_t local_rows = rows_remaining < g_rows_per_dpu ? rows_remaining : g_rows_per_dpu;
        uint32_t *qweight_ptr = qweight_shards + ((size_t)dpu_index * shard_qweight_words);
        float *scale_ptr = scale_shards + ((size_t)dpu_index * shard_scale_floats);
        int16_t *lut_ptr = lut_i16_shards + ((size_t)dpu_index * shard_lut_i16);

        g_valid_rows[dpu_index] = (uint32_t)local_rows;
        for (size_t local_row = 0; local_row < local_rows; ++local_row) {
            memcpy(
                qweight_ptr + (local_row * (size_t)words_per_row),
                qweight_src + (((row_start + local_row) * (size_t)words_per_row)),
                (size_t)words_per_row * sizeof(uint32_t));
            memcpy(
                scale_ptr + (local_row * (size_t)num_groups),
                scale_src + (((row_start + local_row) * (size_t)num_groups)),
                (size_t)num_groups * sizeof(float));
            for (size_t group_idx = 0; group_idx < (size_t)num_groups; ++group_idx) {
                const float scale =
                    scale_src[((row_start + local_row) * (size_t)num_groups) + group_idx];
                int16_t *group_lut = lut_ptr
                    + (((local_row * (size_t)num_groups) + group_idx) * (1u << BITS_PER_WEIGHT));
                for (uint32_t q = 0; q < (1u << BITS_PER_WEIGHT); ++q) {
                    const int32_t centered = (int32_t)q - (int32_t)(1u << (BITS_PER_WEIGHT - 1));
                    int32_t value = (int32_t)(centered * scale * 256.0f);
                    if (value > INT16_MAX) {
                        value = INT16_MAX;
                    } else if (value < INT16_MIN) {
                        value = INT16_MIN;
                    }
                    group_lut[q] = (int16_t)value;
                }
            }
        }
    }

    if (check_dpu_error(
            dpu_broadcast_to(g_set, "input_dim", 0, &g_input_dim, sizeof(g_input_dim), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(input_dim)") != 0) {
        goto cleanup;
    }
    const uint32_t shard_output_dim_u32 = (uint32_t)g_shard_output_dim;
    if (check_dpu_error(
            dpu_broadcast_to(g_set, "output_dim", 0, &shard_output_dim_u32, sizeof(shard_output_dim_u32), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(output_dim)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(g_set, "group_size", 0, &g_group_size, sizeof(g_group_size), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(group_size)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(g_set, "num_groups", 0, &g_num_groups, sizeof(g_num_groups), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(num_groups)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(g_set, "kernel_mode", 0, &g_kernel_mode, sizeof(g_kernel_mode), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(kernel_mode)") != 0) {
        goto cleanup;
    }

    if (g_kernel_mode == 4 || g_kernel_mode == 5) {
        dpu_index = 0;
        DPU_FOREACH(g_set, dpu, dpu_index)
        {
            if (check_dpu_error(
                    dpu_prepare_xfer(dpu, lut_i16_shards + ((size_t)dpu_index * shard_lut_i16)),
                    error_buffer, error_buffer_len, "dpu_prepare_xfer(lut_mram)") != 0) {
                goto cleanup;
            }
        }
        if (check_dpu_error(
                dpu_push_xfer(g_set, DPU_XFER_TO_DPU, "lut_mram", 0, shard_lut_i16 * sizeof(int16_t), DPU_XFER_DEFAULT),
                error_buffer, error_buffer_len, "dpu_push_xfer(lut_mram)") != 0) {
            goto cleanup;
        }
    }

    dpu_index = 0;
    clock_gettime(CLOCK_MONOTONIC, &qweight_start);
    DPU_FOREACH(g_set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(dpu, qweight_shards + ((size_t)dpu_index * shard_qweight_words)),
                error_buffer, error_buffer_len, "dpu_prepare_xfer(qweight_mram)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(g_set, DPU_XFER_TO_DPU, "qweight_mram", 0, shard_qweight_words * sizeof(uint32_t), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_push_xfer(qweight_mram)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &qweight_end);

    dpu_index = 0;
    clock_gettime(CLOCK_MONOTONIC, &scale_start);
    DPU_FOREACH(g_set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(dpu, scale_shards + ((size_t)dpu_index * shard_scale_floats)),
                error_buffer, error_buffer_len, "dpu_prepare_xfer(scales_mram)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(g_set, DPU_XFER_TO_DPU, "scales_mram", 0, shard_scale_floats * sizeof(float), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_push_xfer(scales_mram)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &scale_end);

    g_weights_loaded = true;
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    g_last_load_qweight_transfer_seconds = timespec_diff_seconds(&qweight_start, &qweight_end);
    g_last_load_scale_transfer_seconds = timespec_diff_seconds(&scale_start, &scale_end);
    g_last_load_total_seconds = timespec_diff_seconds(&total_start, &total_end);
    rc = 0;
    g_lut_i16_shards = lut_i16_shards;
    lut_i16_shards = NULL;

cleanup:
    free(lut_i16_shards);
    free(scale_shards);
    free(qweight_shards);
    return rc;
}

int
pim_quantized_run(
    uint32_t batch_size,
    const void *inputs,
    void *outputs,
    char *error_buffer,
    size_t error_buffer_len)
{
    struct dpu_set_t dpu;
    uint32_t dpu_index = 0;
    int rc = -1;
    uint64_t max_cycles = 0;
    uint64_t *kernel_cycles = NULL;
    float *output_shards = NULL;
    int8_t *input_i8_shards = NULL;
    int32_t *output_i32_shards = NULL;
    int16_t *runtime_lut_i16_shards = NULL;
    float *input_scales = NULL;
    struct timespec total_start;
    struct timespec total_end;
    struct timespec input_start;
    struct timespec input_end;
    struct timespec launch_start;
    struct timespec launch_end;
    struct timespec output_start;
    struct timespec output_end;
    const size_t input_floats = (size_t)batch_size * (size_t)g_input_dim;
    const size_t shard_output_floats = (size_t)batch_size * g_shard_output_dim;
    const size_t input_i8_count = input_floats;
    const size_t shard_output_i32 = shard_output_floats;
    const size_t blocks_per_batch = (size_t)g_input_dim / BLOCK_FLOATS;
    const size_t runtime_lut_i16_count =
        (size_t)batch_size * (size_t)g_output_dim * blocks_per_batch * (1u << BITS_PER_WEIGHT);

    if (!g_initialized || !g_weights_loaded) {
        set_error(error_buffer, error_buffer_len, "weights must be loaded before pim_quantized_run");
        return -1;
    }
    if (inputs == NULL || outputs == NULL) {
        set_error(error_buffer, error_buffer_len, "inputs and outputs must be non-null");
        return -1;
    }
    if (input_floats > MAX_INPUT_FLOATS || shard_output_floats > MAX_OUTPUT_FLOATS) {
        set_error(error_buffer, error_buffer_len, "input/output shape too large");
        return -1;
    }
    if ((g_kernel_mode == 4 || g_kernel_mode == 5)
        && (input_i8_count > MAX_INPUT_INT8 || shard_output_i32 > MAX_OUTPUT_INT32)) {
        set_error(error_buffer, error_buffer_len, "int8/int32 input/output shape too large");
        return -1;
    }
    if (g_kernel_mode == 5 && runtime_lut_i16_count > MAX_RUNTIME_LUT_INT16) {
        set_error(error_buffer, error_buffer_len, "runtime int16 lut too large");
        return -1;
    }

    kernel_cycles = calloc(g_nr_dpus, sizeof(*kernel_cycles));
    output_shards = calloc((size_t)g_nr_dpus * shard_output_floats, sizeof(*output_shards));
    if (g_kernel_mode == 4 || g_kernel_mode == 5) {
        input_i8_shards = calloc(input_i8_count, sizeof(*input_i8_shards));
        output_i32_shards = calloc((size_t)g_nr_dpus * shard_output_i32, sizeof(*output_i32_shards));
        if (g_kernel_mode == 5) {
            runtime_lut_i16_shards = calloc(runtime_lut_i16_count, sizeof(*runtime_lut_i16_shards));
        }
        if (g_kernel_mode == 4) {
            input_scales = calloc(batch_size, sizeof(*input_scales));
        }
    }
    if (
        kernel_cycles == NULL || output_shards == NULL
        || ((g_kernel_mode == 4 || g_kernel_mode == 5)
            && (input_i8_shards == NULL || output_i32_shards == NULL))
        || (g_kernel_mode == 4 && input_scales == NULL)
        || (g_kernel_mode == 5 && runtime_lut_i16_shards == NULL)
    ) {
        set_error(error_buffer, error_buffer_len, "failed to allocate run buffers");
        goto cleanup;
    }

    clock_gettime(CLOCK_MONOTONIC, &total_start);

    if (check_dpu_error(
            dpu_broadcast_to(g_set, "batch_size", 0, &batch_size, sizeof(batch_size), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(batch_size)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &input_start);
    if (g_kernel_mode == 4) {
        const float *inputs_f32 = (const float *)inputs;
        for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const size_t batch_offset = (size_t)batch_idx * (size_t)g_input_dim;
            float max_abs = 0.0f;
            for (uint32_t col = 0; col < g_input_dim; ++col) {
                const float value = inputs_f32[batch_offset + col];
                const float abs_value = value >= 0.0f ? value : -value;
                if (abs_value > max_abs) {
                    max_abs = abs_value;
                }
            }
            input_scales[batch_idx] = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
            for (uint32_t col = 0; col < g_input_dim; ++col) {
                float scaled = inputs_f32[batch_offset + col] / input_scales[batch_idx];
                if (scaled > 127.0f) {
                    scaled = 127.0f;
                } else if (scaled < -127.0f) {
                    scaled = -127.0f;
                }
                input_i8_shards[batch_offset + col] =
                    (int8_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
            }
        }
        g_input_scale = batch_size > 0 ? input_scales[0] : 1.0f;
        if (check_dpu_error(
                dpu_broadcast_to(g_set, "inputs_i8_mram", 0, input_i8_shards, input_i8_count * sizeof(int8_t), DPU_XFER_DEFAULT),
                error_buffer, error_buffer_len, "dpu_broadcast_to(inputs_i8_mram)") != 0) {
            goto cleanup;
        }
    } else if (g_kernel_mode == 5) {
        const float *inputs_f32 = (const float *)inputs;
        for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            for (size_t block_idx = 0; block_idx < blocks_per_batch; ++block_idx) {
                const size_t block_offset = ((size_t)batch_idx * (size_t)g_input_dim) + (block_idx * BLOCK_FLOATS);
                float max_abs = 0.0f;
                for (size_t lane = 0; lane < BLOCK_FLOATS; ++lane) {
                    const float value = inputs_f32[block_offset + lane];
                    const float abs_value = value >= 0.0f ? value : -value;
                    if (abs_value > max_abs) {
                        max_abs = abs_value;
                    }
                }
                g_input_scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
                for (size_t lane = 0; lane < BLOCK_FLOATS; ++lane) {
                    float scaled = inputs_f32[block_offset + lane] / g_input_scale;
                    if (scaled > 127.0f) {
                        scaled = 127.0f;
                    } else if (scaled < -127.0f) {
                        scaled = -127.0f;
                    }
                    input_i8_shards[block_offset + lane] =
                        (int8_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
                }
                if (g_kernel_mode == 5) {
                    for (uint32_t row = 0; row < g_output_dim; ++row) {
                        const uint32_t group_idx = (uint32_t)((block_idx * BLOCK_FLOATS) / g_group_size);
                        const int16_t *base_lut =
                            g_lut_i16_shards + (((size_t)row * (size_t)g_num_groups) + group_idx) * (1u << BITS_PER_WEIGHT);
                        int16_t *runtime_lut =
                            runtime_lut_i16_shards
                            + ((((size_t)batch_idx * (size_t)g_output_dim) + row) * blocks_per_batch + block_idx)
                                * (1u << BITS_PER_WEIGHT);
                        for (uint32_t q = 0; q < (1u << BITS_PER_WEIGHT); ++q) {
                            int32_t value = (int32_t)((float)base_lut[q] * g_input_scale);
                            if (value > INT16_MAX) {
                                value = INT16_MAX;
                            } else if (value < INT16_MIN) {
                                value = INT16_MIN;
                            }
                            runtime_lut[q] = (int16_t)value;
                        }
                    }
                }
            }
        }
        if (check_dpu_error(
                dpu_broadcast_to(g_set, "inputs_i8_mram", 0, input_i8_shards, input_i8_count * sizeof(int8_t), DPU_XFER_DEFAULT),
                error_buffer, error_buffer_len, "dpu_broadcast_to(inputs_i8_mram)") != 0) {
            goto cleanup;
        }
        if (check_dpu_error(
                dpu_broadcast_to(
                    g_set,
                    "runtime_lut_mram",
                    0,
                    runtime_lut_i16_shards,
                    runtime_lut_i16_count * sizeof(int16_t),
                    DPU_XFER_DEFAULT),
                error_buffer,
                error_buffer_len,
                "dpu_broadcast_to(runtime_lut_mram)") != 0) {
            goto cleanup;
        }
    } else if (check_dpu_error(
                   dpu_broadcast_to(g_set, "inputs_mram", 0, inputs, input_floats * sizeof(float), DPU_XFER_DEFAULT),
                   error_buffer, error_buffer_len, "dpu_broadcast_to(inputs_mram)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &input_end);
    clock_gettime(CLOCK_MONOTONIC, &launch_start);
    if (check_dpu_error(dpu_launch(g_set, DPU_SYNCHRONOUS), error_buffer, error_buffer_len, "dpu_launch") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &launch_end);

    dpu_index = 0;
    clock_gettime(CLOCK_MONOTONIC, &output_start);
    if (g_kernel_mode == 4) {
        DPU_FOREACH(g_set, dpu, dpu_index)
        {
            if (check_dpu_error(
                    dpu_prepare_xfer(dpu, output_i32_shards + ((size_t)dpu_index * shard_output_i32)),
                    error_buffer, error_buffer_len, "dpu_prepare_xfer(outputs_i32_mram)") != 0) {
                goto cleanup;
            }
        }
        if (check_dpu_error(
                dpu_push_xfer(g_set, DPU_XFER_FROM_DPU, "outputs_i32_mram", 0, shard_output_i32 * sizeof(int32_t), DPU_XFER_DEFAULT),
                error_buffer, error_buffer_len, "dpu_push_xfer(outputs_i32_mram)") != 0) {
            goto cleanup;
        }
    } else {
        DPU_FOREACH(g_set, dpu, dpu_index)
        {
            if (check_dpu_error(
                    dpu_prepare_xfer(dpu, output_shards + ((size_t)dpu_index * shard_output_floats)),
                    error_buffer, error_buffer_len, "dpu_prepare_xfer(outputs_mram)") != 0) {
                goto cleanup;
            }
        }
        if (check_dpu_error(
                dpu_push_xfer(g_set, DPU_XFER_FROM_DPU, "outputs_mram", 0, shard_output_floats * sizeof(float), DPU_XFER_DEFAULT),
                error_buffer, error_buffer_len, "dpu_push_xfer(outputs_mram)") != 0) {
            goto cleanup;
        }
    }

    dpu_index = 0;
    DPU_FOREACH(g_set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(dpu, kernel_cycles + dpu_index),
                error_buffer, error_buffer_len, "dpu_prepare_xfer(kernel_cycles)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(g_set, DPU_XFER_FROM_DPU, "kernel_cycles", 0, sizeof(*kernel_cycles), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_push_xfer(kernel_cycles)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &output_end);

    float *output_dst = (float *)outputs;
    for (dpu_index = 0; dpu_index < g_nr_dpus; ++dpu_index) {
        const uint32_t local_rows = g_valid_rows[dpu_index];
        const size_t row_start = (size_t)dpu_index * g_rows_per_dpu;
        if (kernel_cycles[dpu_index] > max_cycles) {
            max_cycles = kernel_cycles[dpu_index];
        }
        for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            if (g_kernel_mode == 4 || g_kernel_mode == 5) {
                const int32_t *shard_ptr_i32 =
                    output_i32_shards + ((size_t)dpu_index * shard_output_i32) + ((size_t)batch_idx * g_shard_output_dim);
                for (uint32_t local_row = 0; local_row < local_rows; ++local_row) {
                    if (g_kernel_mode == 5) {
                        output_dst[((size_t)batch_idx * (size_t)g_output_dim) + row_start + local_row] =
                            ((float)shard_ptr_i32[local_row]) / 256.0f;
                    } else if (g_kernel_mode == 4) {
                        output_dst[((size_t)batch_idx * (size_t)g_output_dim) + row_start + local_row] =
                            ((float)shard_ptr_i32[local_row]) * (input_scales[batch_idx] / 256.0f);
                    }
                }
            } else {
                const float *shard_ptr =
                    output_shards + ((size_t)dpu_index * shard_output_floats) + ((size_t)batch_idx * g_shard_output_dim);
                memcpy(
                    output_dst + (((size_t)batch_idx * (size_t)g_output_dim) + row_start),
                    shard_ptr,
                    (size_t)local_rows * sizeof(float));
            }
        }
    }

    g_last_cycles = max_cycles;
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    g_last_input_transfer_seconds = timespec_diff_seconds(&input_start, &input_end);
    g_last_launch_seconds = timespec_diff_seconds(&launch_start, &launch_end);
    g_last_output_transfer_seconds = timespec_diff_seconds(&output_start, &output_end);
    g_last_total_seconds = timespec_diff_seconds(&total_start, &total_end);
    rc = 0;

cleanup:
    free(runtime_lut_i16_shards);
    free(input_scales);
    free(output_i32_shards);
    free(input_i8_shards);
    free(output_shards);
    free(kernel_cycles);
    return rc;
}

void
pim_quantized_shutdown(void)
{
    if (!g_initialized) {
        return;
    }

    free(g_valid_rows);
    free(g_lut_i16_shards);
    g_lut_i16_shards = NULL;
    free(g_runtime_lut_i16_shards);
    g_runtime_lut_i16_shards = NULL;
    g_valid_rows = NULL;
    dpu_free(g_set);
    memset(&g_set, 0, sizeof(g_set));
    g_initialized = false;
    g_weights_loaded = false;
    g_last_cycles = 0;
    g_last_load_qweight_transfer_seconds = 0.0;
    g_last_load_scale_transfer_seconds = 0.0;
    g_last_load_total_seconds = 0.0;
    g_last_input_transfer_seconds = 0.0;
    g_last_launch_seconds = 0.0;
    g_last_output_transfer_seconds = 0.0;
    g_last_total_seconds = 0.0;
    g_nr_dpus = 0;
    g_input_dim = 0;
    g_output_dim = 0;
    g_group_size = 0;
    g_num_groups = 0;
    g_kernel_mode = 0;
    g_rows_per_dpu = 0;
    g_shard_output_dim = 0;
}

uint64_t
pim_quantized_last_cycles(void)
{
    return g_last_cycles;
}

double
pim_quantized_last_input_transfer_seconds(void)
{
    return g_last_input_transfer_seconds;
}

double
pim_quantized_last_load_qweight_transfer_seconds(void)
{
    return g_last_load_qweight_transfer_seconds;
}

double
pim_quantized_last_load_scale_transfer_seconds(void)
{
    return g_last_load_scale_transfer_seconds;
}

double
pim_quantized_last_load_total_seconds(void)
{
    return g_last_load_total_seconds;
}

double
pim_quantized_last_launch_seconds(void)
{
    return g_last_launch_seconds;
}

double
pim_quantized_last_output_transfer_seconds(void)
{
    return g_last_output_transfer_seconds;
}

double
pim_quantized_last_total_seconds(void)
{
    return g_last_total_seconds;
}

uint32_t
pim_quantized_num_dpus(void)
{
    return g_nr_dpus;
}
