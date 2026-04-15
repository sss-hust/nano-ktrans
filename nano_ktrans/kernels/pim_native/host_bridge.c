#define _POSIX_C_SOURCE 200809L

#include <dpu/dpu.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#ifndef MAX_WEIGHT_FLOATS
#define MAX_WEIGHT_FLOATS 2097152
#endif

#ifndef MAX_INPUT_FLOATS
#define MAX_INPUT_FLOATS 65536
#endif

#ifndef MAX_OUTPUT_FLOATS
#define MAX_OUTPUT_FLOATS 65536
#endif

static struct dpu_set_t g_set;
static bool g_initialized = false;
static uint64_t g_last_cycles = 0;
static uint32_t g_nr_dpus = 0;

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
pim_linear_init(
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
    g_last_cycles = 0;
    return 0;
}

void
pim_linear_shutdown(void)
{
    if (!g_initialized) {
        return;
    }

    dpu_free(g_set);
    memset(&g_set, 0, sizeof(g_set));
    g_initialized = false;
    g_last_cycles = 0;
    g_nr_dpus = 0;
}

uint64_t
pim_linear_last_cycles(void)
{
    return g_last_cycles;
}

uint32_t
pim_linear_num_dpus(void)
{
    return g_nr_dpus;
}

int
pim_linear_run(
    uint32_t batch_size,
    uint32_t input_dim,
    uint32_t output_dim,
    const float *inputs,
    const float *weights,
    float *outputs,
    char *error_buffer,
    size_t error_buffer_len)
{
    struct dpu_set_t dpu;
    uint32_t dpu_index = 0;
    int rc = -1;
    uint64_t max_cycles = 0;
    uint64_t *kernel_cycles = NULL;
    uint32_t *valid_rows = NULL;
    float *weight_shards = NULL;
    float *output_shards = NULL;

    const size_t input_floats = (size_t)batch_size * (size_t)input_dim;
    size_t rows_per_dpu = 0;
    size_t shard_output_dim = 0;
    size_t shard_weight_floats = 0;
    size_t shard_output_floats = 0;

    if (!g_initialized) {
        set_error(error_buffer, error_buffer_len, "pim_linear_init must be called first");
        return -1;
    }

    if (inputs == NULL || weights == NULL || outputs == NULL) {
        set_error(error_buffer, error_buffer_len, "inputs, weights, and outputs must be non-null");
        return -1;
    }

    if (batch_size == 0 || input_dim == 0 || output_dim == 0) {
        set_error(error_buffer, error_buffer_len, "batch_size, input_dim, and output_dim must be positive");
        return -1;
    }

    if (g_nr_dpus == 0) {
        set_error(error_buffer, error_buffer_len, "no DPUs are available");
        return -1;
    }

    rows_per_dpu = ((size_t)output_dim + (size_t)g_nr_dpus - 1u) / (size_t)g_nr_dpus;
    shard_output_dim = rows_per_dpu + (rows_per_dpu % 2u);
    shard_weight_floats = shard_output_dim * (size_t)input_dim;
    shard_output_floats = (size_t)batch_size * shard_output_dim;

    if (shard_weight_floats > MAX_WEIGHT_FLOATS) {
        set_error(
            error_buffer,
            error_buffer_len,
            "weight shard too large for DPU bridge: %zu floats > %u",
            shard_weight_floats,
            MAX_WEIGHT_FLOATS);
        return -1;
    }

    if (input_floats > MAX_INPUT_FLOATS) {
        set_error(
            error_buffer,
            error_buffer_len,
            "input matrix too large for DPU bridge: %zu floats > %u",
            input_floats,
            MAX_INPUT_FLOATS);
        return -1;
    }

    if (shard_output_floats > MAX_OUTPUT_FLOATS) {
        set_error(
            error_buffer,
            error_buffer_len,
            "output shard too large for DPU bridge: %zu floats > %u",
            shard_output_floats,
            MAX_OUTPUT_FLOATS);
        return -1;
    }

    valid_rows = calloc(g_nr_dpus, sizeof(*valid_rows));
    kernel_cycles = calloc(g_nr_dpus, sizeof(*kernel_cycles));
    weight_shards = calloc((size_t)g_nr_dpus * shard_weight_floats, sizeof(*weight_shards));
    output_shards = calloc((size_t)g_nr_dpus * shard_output_floats, sizeof(*output_shards));
    if (valid_rows == NULL || kernel_cycles == NULL || weight_shards == NULL || output_shards == NULL) {
        set_error(error_buffer, error_buffer_len, "failed to allocate host-side shard buffers");
        goto cleanup;
    }

    for (dpu_index = 0; dpu_index < g_nr_dpus; ++dpu_index) {
        const size_t row_start = (size_t)dpu_index * rows_per_dpu;
        const size_t rows_remaining = row_start < (size_t)output_dim ? (size_t)output_dim - row_start : 0;
        const size_t local_rows = rows_remaining < rows_per_dpu ? rows_remaining : rows_per_dpu;
        float *shard_ptr = weight_shards + ((size_t)dpu_index * shard_weight_floats);

        valid_rows[dpu_index] = (uint32_t)local_rows;
        for (size_t local_row = 0; local_row < local_rows; ++local_row) {
            memcpy(
                shard_ptr + (local_row * (size_t)input_dim),
                weights + (((row_start + local_row) * (size_t)input_dim)),
                (size_t)input_dim * sizeof(float));
        }
    }

    if (check_dpu_error(
            dpu_broadcast_to(g_set, "batch_size", 0, &batch_size, sizeof(batch_size), DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_broadcast_to(batch_size)") != 0) {
        goto cleanup;
    }

    if (check_dpu_error(
            dpu_broadcast_to(g_set, "input_dim", 0, &input_dim, sizeof(input_dim), DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_broadcast_to(input_dim)") != 0) {
        goto cleanup;
    }

    const uint32_t shard_output_dim_u32 = (uint32_t)shard_output_dim;
    if (check_dpu_error(
            dpu_broadcast_to(
                g_set,
                "output_dim",
                0,
                &shard_output_dim_u32,
                sizeof(shard_output_dim_u32),
                DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_broadcast_to(output_dim)") != 0) {
        goto cleanup;
    }

    dpu_index = 0;
    DPU_FOREACH(g_set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(
                    dpu,
                    weight_shards + ((size_t)dpu_index * shard_weight_floats)),
                error_buffer,
                error_buffer_len,
                "dpu_prepare_xfer(weights_mram)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(
                g_set,
                DPU_XFER_TO_DPU,
                "weights_mram",
                0,
                shard_weight_floats * sizeof(float),
                DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_push_xfer(weights_mram)") != 0) {
        goto cleanup;
    }

    if (check_dpu_error(
            dpu_broadcast_to(
                g_set,
                "inputs_mram",
                0,
                inputs,
                input_floats * sizeof(float),
                DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_broadcast_to(inputs_mram)") != 0) {
        goto cleanup;
    }

    if (check_dpu_error(dpu_launch(g_set, DPU_SYNCHRONOUS), error_buffer, error_buffer_len, "dpu_launch") != 0) {
        goto cleanup;
    }

    dpu_index = 0;
    DPU_FOREACH(g_set, dpu)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(
                    dpu,
                    output_shards + ((size_t)dpu_index * shard_output_floats)),
                error_buffer,
                error_buffer_len,
                "dpu_prepare_xfer(outputs)") != 0) {
            goto cleanup;
        }
        ++dpu_index;
    }

    if (check_dpu_error(
            dpu_push_xfer(
                g_set,
                DPU_XFER_FROM_DPU,
                "outputs_mram",
                0,
                shard_output_floats * sizeof(float),
                DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_push_xfer(outputs_mram)") != 0) {
        goto cleanup;
    }

    dpu_index = 0;
    DPU_FOREACH(g_set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(dpu, kernel_cycles + dpu_index),
                error_buffer,
                error_buffer_len,
                "dpu_prepare_xfer(kernel_cycles)") != 0) {
            goto cleanup;
        }
    }

    if (check_dpu_error(
            dpu_push_xfer(
                g_set,
                DPU_XFER_FROM_DPU,
                "kernel_cycles",
                0,
                sizeof(kernel_cycles),
                DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_push_xfer(kernel_cycles)") != 0) {
        goto cleanup;
    }

    for (dpu_index = 0; dpu_index < g_nr_dpus; ++dpu_index) {
        const uint32_t local_rows = valid_rows[dpu_index];
        const size_t row_start = (size_t)dpu_index * rows_per_dpu;
        const float *shard_ptr = output_shards + ((size_t)dpu_index * shard_output_floats);

        if (kernel_cycles[dpu_index] > max_cycles) {
            max_cycles = kernel_cycles[dpu_index];
        }

        for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            memcpy(
                outputs + (((size_t)batch_idx * (size_t)output_dim) + row_start),
                shard_ptr + ((size_t)batch_idx * shard_output_dim),
                (size_t)local_rows * sizeof(float));
        }
    }

    g_last_cycles = max_cycles;
    rc = 0;

cleanup:
    free(output_shards);
    free(weight_shards);
    free(kernel_cycles);
    free(valid_rows);
    return rc;
}
