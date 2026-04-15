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

#ifndef MAX_INTERMEDIATE_DIM
#define MAX_INTERMEDIATE_DIM 2048
#endif

static struct dpu_set_t g_set;
static bool g_initialized = false;
static uint64_t g_last_cycles = 0;
static uint32_t g_nr_dpus = 0;
static uint32_t g_last_active_dpus = 0;

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
pim_expert_init(
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
        g_last_active_dpus = 0;
        return -1;
    }

    if (check_dpu_error(dpu_load(g_set, binary_path, NULL), error_buffer, error_buffer_len, "dpu_load") != 0) {
        dpu_free(g_set);
        memset(&g_set, 0, sizeof(g_set));
        g_nr_dpus = 0;
        g_last_active_dpus = 0;
        return -1;
    }

    g_initialized = true;
    g_last_cycles = 0;
    g_last_active_dpus = 0;
    return 0;
}

void
pim_expert_shutdown(void)
{
    if (!g_initialized) {
        return;
    }

    dpu_free(g_set);
    memset(&g_set, 0, sizeof(g_set));
    g_initialized = false;
    g_last_cycles = 0;
    g_nr_dpus = 0;
    g_last_active_dpus = 0;
}

uint64_t
pim_expert_last_cycles(void)
{
    return g_last_cycles;
}

uint32_t
pim_expert_num_dpus(void)
{
    return g_nr_dpus;
}

uint32_t
pim_expert_last_active_dpus(void)
{
    return g_last_active_dpus;
}

int
pim_expert_run(
    uint32_t batch_size,
    uint32_t input_dim,
    uint32_t intermediate_dim,
    uint32_t output_dim,
    const float *inputs,
    const float *gate_proj,
    const float *up_proj,
    const float *down_proj,
    float *outputs,
    char *error_buffer,
    size_t error_buffer_len)
{
    struct dpu_set_t dpu;
    uint32_t dpu_index = 0;
    int rc = -1;
    uint64_t max_cycles = 0;
    uint64_t *kernel_cycles = NULL;
    uint32_t *valid_hidden = NULL;
    uint32_t *valid_rows = NULL;
    float *gate_shards = NULL;
    float *up_shards = NULL;
    float *down_shards = NULL;
    float *output_shards = NULL;

    const size_t input_floats = (size_t)batch_size * (size_t)input_dim;
    size_t hidden_groups = 0;
    size_t row_groups = 0;
    size_t active_dpus = 0;
    size_t hidden_per_group = 0;
    size_t rows_per_group = 0;
    size_t shard_intermediate_dim = 0;
    size_t shard_output_dim = 0;
    size_t gate_shard_floats = 0;
    size_t up_shard_floats = 0;
    size_t down_shard_floats = 0;
    size_t output_shard_floats = 0;

    if (!g_initialized) {
        set_error(error_buffer, error_buffer_len, "pim_expert_init must be called first");
        return -1;
    }

    if (inputs == NULL || gate_proj == NULL || up_proj == NULL || down_proj == NULL || outputs == NULL) {
        set_error(error_buffer, error_buffer_len, "inputs, projections, and outputs must be non-null");
        return -1;
    }

    if (batch_size == 0 || input_dim == 0 || intermediate_dim == 0 || output_dim == 0) {
        set_error(error_buffer, error_buffer_len, "all dimensions must be positive");
        return -1;
    }
    if ((input_dim % BLOCK_FLOATS) != 0) {
        set_error(
            error_buffer,
            error_buffer_len,
            "input_dim must be padded to a multiple of %u, got %u",
            (unsigned)BLOCK_FLOATS,
            input_dim);
        return -1;
    }
    if ((intermediate_dim % BLOCK_FLOATS) != 0) {
        set_error(
            error_buffer,
            error_buffer_len,
            "intermediate_dim must be padded to a multiple of %u, got %u",
            (unsigned)BLOCK_FLOATS,
            intermediate_dim);
        return -1;
    }
    if ((output_dim % 2u) != 0) {
        set_error(error_buffer, error_buffer_len, "output_dim must be padded to an even value, got %u", output_dim);
        return -1;
    }
    if (intermediate_dim > MAX_INTERMEDIATE_DIM) {
        set_error(
            error_buffer,
            error_buffer_len,
            "intermediate_dim=%u exceeds MAX_INTERMEDIATE_DIM=%u",
            intermediate_dim,
            (unsigned)MAX_INTERMEDIATE_DIM);
        return -1;
    }

    if (g_nr_dpus == 0) {
        set_error(error_buffer, error_buffer_len, "no DPUs are available");
        return -1;
    }

    hidden_groups = ((size_t)intermediate_dim + (size_t)BLOCK_FLOATS - 1u) / (size_t)BLOCK_FLOATS;
    if (hidden_groups == 0) {
        hidden_groups = 1;
    }
    if (hidden_groups > (size_t)g_nr_dpus) {
        hidden_groups = (size_t)g_nr_dpus;
    }
    row_groups = (size_t)g_nr_dpus / hidden_groups;
    if (row_groups == 0) {
        row_groups = 1;
    }
    {
        const size_t max_useful_row_groups = ((size_t)output_dim + 1u) / 2u;
        if (row_groups > max_useful_row_groups) {
            row_groups = max_useful_row_groups;
        }
    }
    active_dpus = hidden_groups * row_groups;
    hidden_per_group = ((size_t)intermediate_dim + hidden_groups - 1u) / hidden_groups;
    rows_per_group = ((size_t)output_dim + row_groups - 1u) / row_groups;
    shard_intermediate_dim =
        ((hidden_per_group + (size_t)BLOCK_FLOATS - 1u) / (size_t)BLOCK_FLOATS) * (size_t)BLOCK_FLOATS;
    shard_output_dim = rows_per_group + (rows_per_group % 2u);
    gate_shard_floats = shard_intermediate_dim * (size_t)input_dim;
    up_shard_floats = shard_intermediate_dim * (size_t)input_dim;
    down_shard_floats = shard_output_dim * shard_intermediate_dim;
    output_shard_floats = (size_t)batch_size * shard_output_dim;

    if (gate_shard_floats > MAX_WEIGHT_FLOATS || up_shard_floats > MAX_WEIGHT_FLOATS) {
        set_error(
            error_buffer,
            error_buffer_len,
            "gate/up projection shard too large for DPU bridge: gate=%zu up=%zu limit=%u",
            gate_shard_floats,
            up_shard_floats,
            MAX_WEIGHT_FLOATS);
        return -1;
    }

    if (down_shard_floats > MAX_WEIGHT_FLOATS) {
        set_error(
            error_buffer,
            error_buffer_len,
            "down projection shard too large for DPU bridge: %zu floats > %u",
            down_shard_floats,
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

    if (output_shard_floats > MAX_OUTPUT_FLOATS) {
        set_error(
            error_buffer,
            error_buffer_len,
            "output shard too large for DPU bridge: %zu floats > %u",
            output_shard_floats,
            MAX_OUTPUT_FLOATS);
        return -1;
    }

    valid_rows = calloc(g_nr_dpus, sizeof(*valid_rows));
    valid_hidden = calloc(g_nr_dpus, sizeof(*valid_hidden));
    kernel_cycles = calloc(g_nr_dpus, sizeof(*kernel_cycles));
    gate_shards = calloc((size_t)g_nr_dpus * gate_shard_floats, sizeof(*gate_shards));
    up_shards = calloc((size_t)g_nr_dpus * up_shard_floats, sizeof(*up_shards));
    down_shards = calloc((size_t)g_nr_dpus * down_shard_floats, sizeof(*down_shards));
    output_shards = calloc((size_t)g_nr_dpus * output_shard_floats, sizeof(*output_shards));
    if (valid_rows == NULL || valid_hidden == NULL || kernel_cycles == NULL || gate_shards == NULL || up_shards == NULL
        || down_shards == NULL || output_shards == NULL) {
        set_error(error_buffer, error_buffer_len, "failed to allocate expert shard buffers");
        goto cleanup;
    }

    g_last_active_dpus = (uint32_t)active_dpus;

    for (dpu_index = 0; dpu_index < g_nr_dpus; ++dpu_index) {
        const size_t hidden_group = (size_t)dpu_index % hidden_groups;
        const size_t row_group = (size_t)dpu_index / hidden_groups;
        const size_t hidden_start = hidden_group * hidden_per_group;
        const size_t row_start = row_group * rows_per_group;
        const size_t hidden_remaining =
            hidden_start < (size_t)intermediate_dim ? (size_t)intermediate_dim - hidden_start : 0;
        const size_t rows_remaining =
            row_start < (size_t)output_dim ? (size_t)output_dim - row_start : 0;
        const size_t local_hidden =
            (dpu_index < active_dpus && hidden_remaining < hidden_per_group) ? hidden_remaining : hidden_per_group;
        const size_t local_rows =
            (dpu_index < active_dpus && rows_remaining < rows_per_group) ? rows_remaining : rows_per_group;
        float *gate_ptr = gate_shards + ((size_t)dpu_index * gate_shard_floats);
        float *up_ptr = up_shards + ((size_t)dpu_index * up_shard_floats);
        float *shard_ptr = down_shards + ((size_t)dpu_index * down_shard_floats);

        valid_hidden[dpu_index] = (uint32_t)((dpu_index < active_dpus) ? local_hidden : 0u);
        valid_rows[dpu_index] = (uint32_t)((dpu_index < active_dpus) ? local_rows : 0u);
        if (dpu_index >= active_dpus || local_hidden == 0 || local_rows == 0) {
            continue;
        }

        for (size_t local_hidden_idx = 0; local_hidden_idx < local_hidden; ++local_hidden_idx) {
            memcpy(
                gate_ptr + (local_hidden_idx * (size_t)input_dim),
                gate_proj + (((hidden_start + local_hidden_idx) * (size_t)input_dim)),
                (size_t)input_dim * sizeof(float));
            memcpy(
                up_ptr + (local_hidden_idx * (size_t)input_dim),
                up_proj + (((hidden_start + local_hidden_idx) * (size_t)input_dim)),
                (size_t)input_dim * sizeof(float));
        }

        for (size_t row = 0; row < local_rows; ++row) {
            memcpy(
                shard_ptr + (row * shard_intermediate_dim),
                down_proj + (((row_start + row) * (size_t)intermediate_dim) + hidden_start),
                local_hidden * sizeof(float));
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

    if (check_dpu_error(
            dpu_broadcast_to(
                g_set,
                "intermediate_dim",
                0,
                &(uint32_t){(uint32_t)shard_intermediate_dim},
                sizeof(uint32_t),
                DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_broadcast_to(intermediate_dim)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(
                g_set,
                "output_dim",
                0,
                &(uint32_t){(uint32_t)shard_output_dim},
                sizeof(uint32_t),
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
                    gate_shards + ((size_t)dpu_index * gate_shard_floats)),
                error_buffer,
                error_buffer_len,
                "dpu_prepare_xfer(gate_proj_mram)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(
                g_set,
                DPU_XFER_TO_DPU,
                "gate_proj_mram",
                0,
                gate_shard_floats * sizeof(float),
                DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_push_xfer(gate_proj_mram)") != 0) {
        goto cleanup;
    }

    dpu_index = 0;
    DPU_FOREACH(g_set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(
                    dpu,
                    up_shards + ((size_t)dpu_index * up_shard_floats)),
                error_buffer,
                error_buffer_len,
                "dpu_prepare_xfer(up_proj_mram)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(
                g_set,
                DPU_XFER_TO_DPU,
                "up_proj_mram",
                0,
                up_shard_floats * sizeof(float),
                DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_push_xfer(up_proj_mram)") != 0) {
        goto cleanup;
    }

    dpu_index = 0;
    DPU_FOREACH(g_set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(
                    dpu,
                    down_shards + ((size_t)dpu_index * down_shard_floats)),
                error_buffer,
                error_buffer_len,
                "dpu_prepare_xfer(down_proj_mram)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(
                g_set,
                DPU_XFER_TO_DPU,
                "down_proj_mram",
                0,
                down_shard_floats * sizeof(float),
                DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_push_xfer(down_proj_mram)") != 0) {
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
    DPU_FOREACH(g_set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(
                    dpu,
                    output_shards + ((size_t)dpu_index * output_shard_floats)),
                error_buffer,
                error_buffer_len,
                "dpu_prepare_xfer(outputs)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(
                g_set,
                DPU_XFER_FROM_DPU,
                "outputs_mram",
                0,
                output_shard_floats * sizeof(float),
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
                sizeof(*kernel_cycles),
                DPU_XFER_DEFAULT),
            error_buffer,
            error_buffer_len,
            "dpu_push_xfer(kernel_cycles)") != 0) {
        goto cleanup;
    }

    memset(outputs, 0, (size_t)batch_size * (size_t)output_dim * sizeof(*outputs));
    for (dpu_index = 0; dpu_index < g_nr_dpus; ++dpu_index) {
        const size_t row_group = (size_t)dpu_index / hidden_groups;
        const size_t row_start = row_group * rows_per_group;
        const float *shard_ptr = output_shards + ((size_t)dpu_index * output_shard_floats);
        const uint32_t local_rows = valid_rows[dpu_index];
        if (valid_hidden[dpu_index] == 0 || local_rows == 0) {
            continue;
        }
        if (kernel_cycles[dpu_index] > max_cycles) {
            max_cycles = kernel_cycles[dpu_index];
        }
        for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            float *dst = outputs + (((size_t)batch_idx * (size_t)output_dim) + row_start);
            const float *src = shard_ptr + ((size_t)batch_idx * shard_output_dim);
            for (uint32_t row = 0; row < local_rows; ++row) {
                dst[row] += src[row];
            }
        }
    }

    g_last_cycles = max_cycles;
    rc = 0;

cleanup:
    free(up_shards);
    free(gate_shards);
    free(output_shards);
    free(down_shards);
    free(kernel_cycles);
    free(valid_rows);
    free(valid_hidden);
    return rc;
}
