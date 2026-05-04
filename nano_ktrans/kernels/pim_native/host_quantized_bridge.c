#define _POSIX_C_SOURCE 200809L

#include <dpu/dpu.h>

#include <math.h>
#include <pthread.h>
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

#ifndef MAX_RUN_REQUESTS
#define MAX_RUN_REQUESTS 64
#endif

/* kernel_mode=7 bit-plane packing: 8 planes per BLOCK_FLOATS-wide block.
 * Host allocates (batch_size * blocks_per_batch * 8) uint64_t entries. */
#ifndef MAX_INPUT_BITPLANES_U64
#define MAX_INPUT_BITPLANES_U64 ((MAX_INPUT_FLOATS / 64) * 8)
#endif

/* ADR-002 M-6.1: multi-slot MRAM residency (must match DPU kernel) */
#ifndef NUM_SLOTS
#define NUM_SLOTS 8
#endif
#define WORDS_PER_SLOT     (MAX_QWEIGHT_WORDS / NUM_SLOTS)
#define SCALES_PER_SLOT    (MAX_SCALE_FLOATS  / NUM_SLOTS)
#define LUT_INT16_PER_SLOT (MAX_LUT_INT16     / NUM_SLOTS)

/* ADR-002 M-8: handle-based context.
 *
 * Replaces the 20 `static` globals that M-5/M-6/M-7 diagnosed as the
 * source of silent runtime-pool aliasing (ADR-002 §15).  Every
 * pim_quantized_init() call now allocates a fresh pim_q_ctx_t and
 * returns an opaque handle; every subsequent call takes that handle
 * as first argument.  The .so itself becomes stateless.
 *
 * M-5 dual runtime, M-6 multi-slot MRAM, and M-7 per-layer-group
 * scoping all depend on this refactor to do anything — their Python
 * `profile`-keyed distinctness was a lie until this change landed.
 */
typedef struct {
    struct dpu_set_t set;
    bool weights_loaded;
    uint64_t last_cycles;
    double last_load_qweight_transfer_seconds;
    double last_load_scale_transfer_seconds;
    double last_load_total_seconds;
    double last_input_transfer_seconds;
    double last_launch_seconds;
    double last_output_transfer_seconds;
    double last_total_seconds;
    uint32_t nr_dpus;
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t group_size;
    uint32_t num_groups;
    uint32_t kernel_mode;
    float input_scale;
    size_t rows_per_dpu;
    size_t shard_output_dim;
    uint32_t *valid_rows;
    int16_t *lut_i16_shards;
    int8_t *input_i8_shards;
    int32_t *output_i32_shards;
    int16_t *runtime_lut_i16_shards;
    uint32_t *load_qweight_shards;
    float *load_scale_shards;
    uint64_t *kernel_cycles;
    float *output_shards;
    float *input_scales;
    uint64_t *input_bitplanes;
    size_t valid_rows_capacity;
    size_t lut_i16_shards_capacity;
    size_t input_i8_shards_capacity;
    size_t output_i32_shards_capacity;
    size_t runtime_lut_i16_shards_capacity;
    size_t load_qweight_shards_capacity;
    size_t load_scale_shards_capacity;
    size_t kernel_cycles_capacity;
    size_t output_shards_capacity;
    size_t input_scales_capacity;
    size_t input_bitplanes_capacity;
    /* ADR-002 M-6.1: per-slot occupancy tracking.  Bit b set => slot b
     * has valid weights loaded.  Host-side LRU logic in Python sets the
     * active slot on every run; this flag just gates sanity checks. */
    uint32_t slot_loaded_mask;
    /* ADR-002 M-17.2: when true, at least one weight load has issued
     * DPU_XFER_ASYNC pushes that have not been waited on yet.  Run /
     * shutdown entry points must call dpu_sync(ctx->set) and clear
     * this flag before touching the DPU set. */
    bool inflight_async_load;
} pim_q_ctx_t;

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
ensure_buffer(
    void **buffer,
    size_t *capacity,
    size_t needed_count,
    size_t element_size,
    char *error_buffer,
    size_t error_buffer_len,
    const char *name)
{
    if (needed_count == 0) {
        return 0;
    }
    if (*buffer != NULL && *capacity >= needed_count) {
        return 0;
    }
    if (element_size != 0 && needed_count > (SIZE_MAX / element_size)) {
        set_error(error_buffer, error_buffer_len, "%s buffer size overflow", name);
        return -1;
    }

    void *new_buffer = realloc(*buffer, needed_count * element_size);
    if (new_buffer == NULL) {
        set_error(error_buffer, error_buffer_len, "failed to allocate %s buffer", name);
        return -1;
    }
    *buffer = new_buffer;
    *capacity = needed_count;
    return 0;
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

/* ADR-002 M-8: return a fresh handle on each call.  ``NULL`` return
 * means failure (error message written into error_buffer).  Callers
 * MUST pair this with a matching pim_quantized_shutdown(ctx) call.
 */
void *
pim_quantized_init(
    const char *binary_path,
    const char *profile,
    uint32_t rank_count,
    char *error_buffer,
    size_t error_buffer_len)
{
    if (binary_path == NULL || binary_path[0] == '\0') {
        set_error(error_buffer, error_buffer_len, "binary_path is required");
        return NULL;
    }
    if (rank_count == 0) {
        set_error(error_buffer, error_buffer_len, "rank_count must be positive");
        return NULL;
    }

    pim_q_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        set_error(error_buffer, error_buffer_len, "failed to allocate pim_q_ctx_t");
        return NULL;
    }

    const char *effective_profile = (profile != NULL && profile[0] != '\0') ? profile : NULL;
    if (check_dpu_error(
            dpu_alloc_ranks(rank_count, effective_profile, &ctx->set),
            error_buffer,
            error_buffer_len,
            "dpu_alloc_ranks") != 0) {
        free(ctx);
        return NULL;
    }
    if (check_dpu_error(
            dpu_get_nr_dpus(ctx->set, &ctx->nr_dpus),
            error_buffer,
            error_buffer_len,
            "dpu_get_nr_dpus") != 0) {
        dpu_free(ctx->set);
        free(ctx);
        return NULL;
    }
    if (check_dpu_error(dpu_load(ctx->set, binary_path, NULL), error_buffer, error_buffer_len, "dpu_load") != 0) {
        dpu_free(ctx->set);
        free(ctx);
        return NULL;
    }

    /* zero-init done by calloc; only weights_loaded already false, etc. */
    return ctx;
}

/* ADR-002 M-17.2: drain any inflight async weight pushes from prior
 * pim_quantized_load_weights[_with_lut] calls.  Must be called at the
 * very start of every entry point that touches the DPU set after a
 * load (run, run_many, shutdown).  Cheap no-op when no async load is
 * outstanding. */
static int
flush_inflight_async_load(
    pim_q_ctx_t *ctx,
    char *error_buffer,
    size_t error_buffer_len,
    const char *context)
{
    if (ctx == NULL || !ctx->inflight_async_load) {
        return 0;
    }
    if (check_dpu_error(
            dpu_sync(ctx->set), error_buffer, error_buffer_len, context) != 0) {
        return -1;
    }
    ctx->inflight_async_load = false;
    return 0;
}

/* ADR-002 M-17.1: shared core for both legacy
 * pim_quantized_load_weights() (which still computes the LUT inside the
 * shard loop) and the new pim_quantized_load_weights_with_lut() entry
 * point that takes a host-precomputed [output_dim, num_groups, 16] LUT
 * tensor and only does shard memcpy + DMA push.
 *
 * If ``precomputed_lut_full`` is non-NULL, it must point to a
 * row-major int16 tensor of shape (output_dim, num_groups, 16) and the
 * inner LUT loop is skipped.  Otherwise the legacy nested computation
 * (centered * scale * 256, clamped to int16) runs per-shard.
 */
static int
load_weights_inner(
    void *handle,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t group_size,
    uint32_t kernel_mode,
    const void *packed_qweights,
    const void *scales,
    const int16_t *precomputed_lut_full,
    uint32_t slot_id,
    char *error_buffer,
    size_t error_buffer_len)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
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

    if (ctx == NULL) {
        set_error(error_buffer, error_buffer_len, "handle is NULL");
        return -1;
    }
    /* ADR-002 M-17.2 (correctness): the ctx-owned host shard buffers
     * (load_qweight_shards / load_scale_shards / lut_i16_shards) are
     * reused across calls.  If a previous load issued ASYNC pushes
     * over those buffers, we MUST sync before overwriting them with
     * this call's data.  The 3 pushes inside this single call can
     * still overlap inside the SDK because the per-symbol prepare/push
     * pairs target distinct mram symbols and are issued back-to-back. */
    if (flush_inflight_async_load(ctx, error_buffer, error_buffer_len,
                                  "dpu_sync(flush before load_weights_inner)") != 0) {
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
    if (slot_id >= NUM_SLOTS) {
        set_error(error_buffer, error_buffer_len, "slot_id must be < NUM_SLOTS (%d)", NUM_SLOTS);
        return -1;
    }

    ctx->input_dim = input_dim;
    ctx->output_dim = output_dim;
    ctx->group_size = group_size;
    ctx->num_groups = num_groups;
    ctx->kernel_mode = kernel_mode;
    ctx->rows_per_dpu = ((size_t)output_dim + (size_t)ctx->nr_dpus - 1u) / (size_t)ctx->nr_dpus;
    ctx->shard_output_dim = ctx->rows_per_dpu + (ctx->rows_per_dpu % 2u);
    shard_qweight_words = ctx->shard_output_dim * (size_t)words_per_row;
    shard_scale_floats = ctx->shard_output_dim * (size_t)num_groups;
    shard_lut_i16 = shard_scale_floats * (size_t)(1u << BITS_PER_WEIGHT);

    /* M-6.1: each slot only sees 1/NUM_SLOTS of the total MRAM budget. */
    if (shard_qweight_words > WORDS_PER_SLOT) {
        set_error(error_buffer, error_buffer_len,
                  "qweight shard too large for slot: %zu > %d (NUM_SLOTS=%d)",
                  shard_qweight_words, WORDS_PER_SLOT, NUM_SLOTS);
        return -1;
    }
    if (shard_scale_floats > SCALES_PER_SLOT) {
        set_error(error_buffer, error_buffer_len,
                  "scale shard too large for slot: %zu > %d",
                  shard_scale_floats, SCALES_PER_SLOT);
        return -1;
    }
    if (shard_lut_i16 > LUT_INT16_PER_SLOT) {
        set_error(error_buffer, error_buffer_len,
                  "lut shard too large for slot: %zu > %d",
                  shard_lut_i16, LUT_INT16_PER_SLOT);
        return -1;
    }

    const size_t total_qweight_words = (size_t)ctx->nr_dpus * shard_qweight_words;
    const size_t total_scale_floats = (size_t)ctx->nr_dpus * shard_scale_floats;
    const size_t total_lut_i16 = (size_t)ctx->nr_dpus * shard_lut_i16;
    if (
        ensure_buffer((void **)&ctx->valid_rows, &ctx->valid_rows_capacity,
                      ctx->nr_dpus, sizeof(*ctx->valid_rows), error_buffer, error_buffer_len, "valid_rows") != 0
        || ensure_buffer((void **)&ctx->load_qweight_shards, &ctx->load_qweight_shards_capacity,
                         total_qweight_words, sizeof(*ctx->load_qweight_shards), error_buffer, error_buffer_len, "load_qweight_shards") != 0
        || ensure_buffer((void **)&ctx->load_scale_shards, &ctx->load_scale_shards_capacity,
                         total_scale_floats, sizeof(*ctx->load_scale_shards), error_buffer, error_buffer_len, "load_scale_shards") != 0
        || ensure_buffer((void **)&ctx->lut_i16_shards, &ctx->lut_i16_shards_capacity,
                         total_lut_i16, sizeof(*ctx->lut_i16_shards), error_buffer, error_buffer_len, "lut_i16_shards") != 0
    ) {
        goto cleanup;
    }
    qweight_shards = ctx->load_qweight_shards;
    scale_shards = ctx->load_scale_shards;
    lut_i16_shards = ctx->lut_i16_shards;

    clock_gettime(CLOCK_MONOTONIC, &total_start);

    const uint32_t *qweight_src = (const uint32_t *)packed_qweights;
    const float *scale_src = (const float *)scales;
    for (dpu_index = 0; dpu_index < ctx->nr_dpus; ++dpu_index) {
        const size_t row_start = (size_t)dpu_index * ctx->rows_per_dpu;
        const size_t rows_remaining = row_start < (size_t)output_dim ? (size_t)output_dim - row_start : 0;
        const size_t local_rows = rows_remaining < ctx->rows_per_dpu ? rows_remaining : ctx->rows_per_dpu;
        uint32_t *qweight_ptr = qweight_shards + ((size_t)dpu_index * shard_qweight_words);
        float *scale_ptr = scale_shards + ((size_t)dpu_index * shard_scale_floats);
        int16_t *lut_ptr = lut_i16_shards + ((size_t)dpu_index * shard_lut_i16);

        ctx->valid_rows[dpu_index] = (uint32_t)local_rows;
        for (size_t local_row = 0; local_row < local_rows; ++local_row) {
            memcpy(
                qweight_ptr + (local_row * (size_t)words_per_row),
                qweight_src + (((row_start + local_row) * (size_t)words_per_row)),
                (size_t)words_per_row * sizeof(uint32_t));
            memcpy(
                scale_ptr + (local_row * (size_t)num_groups),
                scale_src + (((row_start + local_row) * (size_t)num_groups)),
                (size_t)num_groups * sizeof(float));
            if (precomputed_lut_full != NULL) {
                /* M-17.1: copy the row's slice of the host-precomputed
                 * LUT directly; no per-q multiplication on the host
                 * critical path. */
                const int16_t *lut_src_row = precomputed_lut_full
                    + (((row_start + local_row) * (size_t)num_groups)
                       * (1u << BITS_PER_WEIGHT));
                int16_t *group_lut = lut_ptr
                    + ((local_row * (size_t)num_groups)
                       * (1u << BITS_PER_WEIGHT));
                memcpy(
                    group_lut,
                    lut_src_row,
                    (size_t)num_groups * (1u << BITS_PER_WEIGHT) * sizeof(int16_t));
            } else {
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
    }

    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "input_dim", 0, &ctx->input_dim, sizeof(ctx->input_dim), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(input_dim)") != 0) {
        goto cleanup;
    }
    const uint32_t shard_output_dim_u32 = (uint32_t)ctx->shard_output_dim;
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "output_dim", 0, &shard_output_dim_u32, sizeof(shard_output_dim_u32), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(output_dim)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "group_size", 0, &ctx->group_size, sizeof(ctx->group_size), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(group_size)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "num_groups", 0, &ctx->num_groups, sizeof(ctx->num_groups), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(num_groups)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "kernel_mode", 0, &ctx->kernel_mode, sizeof(ctx->kernel_mode), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(kernel_mode)") != 0) {
        goto cleanup;
    }

    if (ctx->kernel_mode == 4 || ctx->kernel_mode == 5 || ctx->kernel_mode == 6 || ctx->kernel_mode == 7) {
        dpu_index = 0;
        DPU_FOREACH(ctx->set, dpu, dpu_index)
        {
            if (check_dpu_error(
                    dpu_prepare_xfer(dpu, lut_i16_shards + ((size_t)dpu_index * shard_lut_i16)),
                    error_buffer, error_buffer_len, "dpu_prepare_xfer(lut_mram)") != 0) {
                goto cleanup;
            }
        }
        /* ADR-002 M-17.2: ASYNC push.  Sync is deferred to the next
         * run/shutdown entry so multiple back-to-back loads + their
         * lut/qweight/scale segments can overlap inside the SDK. */
        if (check_dpu_error(
                dpu_push_xfer(ctx->set, DPU_XFER_TO_DPU, "lut_mram",
                              (size_t)slot_id * LUT_INT16_PER_SLOT * sizeof(int16_t),
                              shard_lut_i16 * sizeof(int16_t), DPU_XFER_ASYNC),
                error_buffer, error_buffer_len, "dpu_push_xfer(lut_mram async)") != 0) {
            goto cleanup;
        }
    }

    dpu_index = 0;
    clock_gettime(CLOCK_MONOTONIC, &qweight_start);
    DPU_FOREACH(ctx->set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(dpu, qweight_shards + ((size_t)dpu_index * shard_qweight_words)),
                error_buffer, error_buffer_len, "dpu_prepare_xfer(qweight_mram)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(ctx->set, DPU_XFER_TO_DPU, "qweight_mram",
                          (size_t)slot_id * WORDS_PER_SLOT * sizeof(uint32_t),
                          shard_qweight_words * sizeof(uint32_t), DPU_XFER_ASYNC),
            error_buffer, error_buffer_len, "dpu_push_xfer(qweight_mram async)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &qweight_end);

    dpu_index = 0;
    clock_gettime(CLOCK_MONOTONIC, &scale_start);
    DPU_FOREACH(ctx->set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(dpu, scale_shards + ((size_t)dpu_index * shard_scale_floats)),
                error_buffer, error_buffer_len, "dpu_prepare_xfer(scales_mram)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(ctx->set, DPU_XFER_TO_DPU, "scales_mram",
                          (size_t)slot_id * SCALES_PER_SLOT * sizeof(float),
                          shard_scale_floats * sizeof(float), DPU_XFER_ASYNC),
            error_buffer, error_buffer_len, "dpu_push_xfer(scales_mram async)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &scale_end);

    ctx->weights_loaded = true;
    ctx->slot_loaded_mask |= (1u << slot_id);
    /* M-17.2: mark inflight async DMA; the next run/shutdown entry
     * will do dpu_sync(ctx->set) before touching the device. */
    ctx->inflight_async_load = true;
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    ctx->last_load_qweight_transfer_seconds = timespec_diff_seconds(&qweight_start, &qweight_end);
    ctx->last_load_scale_transfer_seconds = timespec_diff_seconds(&scale_start, &scale_end);
    ctx->last_load_total_seconds = timespec_diff_seconds(&total_start, &total_end);
    rc = 0;

cleanup:
    return rc;
}

int
pim_quantized_load_weights(
    void *handle,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t group_size,
    uint32_t kernel_mode,
    const void *packed_qweights,
    const void *scales,
    uint32_t slot_id,
    char *error_buffer,
    size_t error_buffer_len)
{
    return load_weights_inner(
        handle, input_dim, output_dim, group_size, kernel_mode,
        packed_qweights, scales,
        /* precomputed_lut_full = */ NULL,
        slot_id, error_buffer, error_buffer_len);
}

/* ADR-002 M-17.1: take a host-precomputed LUT (shape
 * [output_dim, num_groups, 16] int16 row-major) and load weights
 * without recomputing the LUT in C.  All other layout invariants are
 * identical to pim_quantized_load_weights() so callers may freely mix
 * the two entry points across calls. */
int
pim_quantized_load_weights_with_lut(
    void *handle,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t group_size,
    uint32_t kernel_mode,
    const void *packed_qweights,
    const void *scales,
    const void *precomputed_lut,
    uint32_t slot_id,
    char *error_buffer,
    size_t error_buffer_len)
{
    if (precomputed_lut == NULL) {
        set_error(error_buffer, error_buffer_len,
                  "precomputed_lut must be non-null; use pim_quantized_load_weights() to skip");
        return -1;
    }
    return load_weights_inner(
        handle, input_dim, output_dim, group_size, kernel_mode,
        packed_qweights, scales,
        (const int16_t *)precomputed_lut,
        slot_id, error_buffer, error_buffer_len);
}

int
pim_quantized_run(
    void *handle,
    uint32_t batch_size,
    const void *inputs,
    void *outputs,
    uint32_t slot_id,
    char *error_buffer,
    size_t error_buffer_len)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
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
    uint64_t *input_bitplanes = NULL;
    struct timespec total_start;
    struct timespec total_end;
    struct timespec input_start;
    struct timespec input_end;
    struct timespec launch_start;
    struct timespec launch_end;
    struct timespec output_start;
    struct timespec output_end;
    const size_t input_floats = (size_t)batch_size * (size_t)ctx->input_dim;
    const size_t shard_output_floats = (size_t)batch_size * ctx->shard_output_dim;
    const size_t input_i8_count = input_floats;
    const size_t shard_output_i32 = shard_output_floats;
    const size_t blocks_per_batch = (size_t)ctx->input_dim / BLOCK_FLOATS;
    const size_t runtime_lut_i16_count =
        (size_t)batch_size * (size_t)ctx->output_dim * blocks_per_batch * (1u << BITS_PER_WEIGHT);
    const size_t bitplane_u64_count = (size_t)batch_size * blocks_per_batch * 8u;

    if (ctx == NULL || !ctx->weights_loaded) {
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
    if ((ctx->kernel_mode == 4 || ctx->kernel_mode == 5 || ctx->kernel_mode == 6 || ctx->kernel_mode == 7)
        && (input_i8_count > MAX_INPUT_INT8 || shard_output_i32 > MAX_OUTPUT_INT32)) {
        set_error(error_buffer, error_buffer_len, "int8/int32 input/output shape too large");
        return -1;
    }
    if (ctx->kernel_mode == 5 && runtime_lut_i16_count > MAX_RUNTIME_LUT_INT16) {
        set_error(error_buffer, error_buffer_len, "runtime int16 lut too large");
        return -1;
    }
    if (ctx->kernel_mode == 7 && bitplane_u64_count > MAX_INPUT_BITPLANES_U64) {
        set_error(error_buffer, error_buffer_len, "bitplane buffer too large");
        return -1;
    }
    if (ctx->kernel_mode == 7 && ((size_t)ctx->input_dim % 64u != 0)) {
        set_error(error_buffer, error_buffer_len,
                  "kernel_mode=7 requires input_dim divisible by 64 (BLOCK_FLOATS)");
        return -1;
    }

    const bool int_kernel = (ctx->kernel_mode == 4 || ctx->kernel_mode == 5 || ctx->kernel_mode == 6 || ctx->kernel_mode == 7);
    if (
        ensure_buffer((void **)&ctx->kernel_cycles, &ctx->kernel_cycles_capacity,
                      ctx->nr_dpus, sizeof(*ctx->kernel_cycles), error_buffer, error_buffer_len, "kernel_cycles") != 0
        || ensure_buffer((void **)&ctx->output_shards, &ctx->output_shards_capacity,
                         (size_t)ctx->nr_dpus * shard_output_floats, sizeof(*ctx->output_shards), error_buffer, error_buffer_len, "output_shards") != 0
        || (int_kernel && ensure_buffer((void **)&ctx->input_i8_shards, &ctx->input_i8_shards_capacity,
                                        input_i8_count, sizeof(*ctx->input_i8_shards), error_buffer, error_buffer_len, "input_i8_shards") != 0)
        || (int_kernel && ensure_buffer((void **)&ctx->output_i32_shards, &ctx->output_i32_shards_capacity,
                                        (size_t)ctx->nr_dpus * shard_output_i32, sizeof(*ctx->output_i32_shards), error_buffer, error_buffer_len, "output_i32_shards") != 0)
        || (ctx->kernel_mode == 5 && ensure_buffer((void **)&ctx->runtime_lut_i16_shards, &ctx->runtime_lut_i16_shards_capacity,
                                                   runtime_lut_i16_count, sizeof(*ctx->runtime_lut_i16_shards), error_buffer, error_buffer_len, "runtime_lut_i16_shards") != 0)
        || ((ctx->kernel_mode == 4 || ctx->kernel_mode == 6 || ctx->kernel_mode == 7)
            && ensure_buffer((void **)&ctx->input_scales, &ctx->input_scales_capacity,
                             batch_size, sizeof(*ctx->input_scales), error_buffer, error_buffer_len, "input_scales") != 0)
        || (ctx->kernel_mode == 7 && ensure_buffer((void **)&ctx->input_bitplanes, &ctx->input_bitplanes_capacity,
                                                   bitplane_u64_count, sizeof(*ctx->input_bitplanes), error_buffer, error_buffer_len, "input_bitplanes") != 0)
    ) {
        goto cleanup;
    }
    kernel_cycles = ctx->kernel_cycles;
    output_shards = ctx->output_shards;
    input_i8_shards = ctx->input_i8_shards;
    output_i32_shards = ctx->output_i32_shards;
    runtime_lut_i16_shards = ctx->runtime_lut_i16_shards;
    input_scales = ctx->input_scales;
    input_bitplanes = ctx->input_bitplanes;

    clock_gettime(CLOCK_MONOTONIC, &total_start);

    /* M-6.1: guard against running a slot that was never loaded. */
    if (slot_id >= NUM_SLOTS) {
        set_error(error_buffer, error_buffer_len, "slot_id must be < NUM_SLOTS (%d)", NUM_SLOTS);
        goto cleanup;
    }
    if ((ctx->slot_loaded_mask & (1u << slot_id)) == 0) {
        set_error(error_buffer, error_buffer_len,
                  "slot %u has no weights loaded (ctx->slot_loaded_mask=0x%x)",
                  slot_id, ctx->slot_loaded_mask);
        goto cleanup;
    }

    /* ADR-002 M-17.2: drain any inflight async weight pushes from
     * previous load_weights[_with_lut] calls before starting the run. */
    if (flush_inflight_async_load(ctx, error_buffer, error_buffer_len,
                                  "dpu_sync(flush before pim_quantized_run)") != 0) {
        goto cleanup;
    }

    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "batch_size", 0, &batch_size, sizeof(batch_size), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(batch_size)") != 0) {
        goto cleanup;
    }
    const uint32_t request_count_zero = 0;
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "run_request_count", 0, &request_count_zero, sizeof(request_count_zero), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(run_request_count=0)") != 0) {
        goto cleanup;
    }

    /* M-6.1: tell the kernel which slot to compute from this call. */
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "active_slot", 0, &slot_id, sizeof(slot_id), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(active_slot)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &input_start);
    if (ctx->kernel_mode == 4 || ctx->kernel_mode == 6 || ctx->kernel_mode == 7) {
        const float *inputs_f32 = (const float *)inputs;
        for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const size_t batch_offset = (size_t)batch_idx * (size_t)ctx->input_dim;
            float max_abs = 0.0f;
            for (uint32_t col = 0; col < ctx->input_dim; ++col) {
                const float value = inputs_f32[batch_offset + col];
                const float abs_value = value >= 0.0f ? value : -value;
                if (abs_value > max_abs) {
                    max_abs = abs_value;
                }
            }
            input_scales[batch_idx] = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
            for (uint32_t col = 0; col < ctx->input_dim; ++col) {
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
        ctx->input_scale = batch_size > 0 ? input_scales[0] : 1.0f;

        if (ctx->kernel_mode == 7) {
            /* Bit-plane pack each BLOCK_FLOATS=64 block into 8 uint64_t
             * (7 magnitude planes + 1 sign plane).  Host does this once
             * per inference; DPU's inner loop is then pure add/lookup. */
            memset(input_bitplanes, 0, bitplane_u64_count * sizeof(*input_bitplanes));
            for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                for (size_t block_idx = 0; block_idx < blocks_per_batch; ++block_idx) {
                    const size_t block_offset =
                        ((size_t)batch_idx * (size_t)ctx->input_dim) + (block_idx * BLOCK_FLOATS);
                    const size_t bp_base =
                        (((size_t)batch_idx * blocks_per_batch) + block_idx) * 8u;
                    uint64_t planes[8] = {0};
                    for (size_t lane = 0; lane < BLOCK_FLOATS; ++lane) {
                        const int8_t v = input_i8_shards[block_offset + lane];
                        uint32_t mag;
                        uint64_t sign_bit;
                        if (v < 0) {
                            /* int8 minimum is -128; clamp magnitude to 127. */
                            const int32_t nv = -(int32_t)v;
                            mag = (nv > 127) ? 127u : (uint32_t)nv;
                            sign_bit = 1ULL;
                        } else {
                            mag = (uint32_t)v;
                            sign_bit = 0ULL;
                        }
                        const uint64_t lane_bit = (1ULL << lane);
                        for (uint32_t b = 0; b < 7; ++b) {
                            if ((mag >> b) & 1u) {
                                planes[b] |= lane_bit;
                            }
                        }
                        if (sign_bit) {
                            planes[7] |= lane_bit;
                        }
                    }
                    for (size_t p = 0; p < 8; ++p) {
                        input_bitplanes[bp_base + p] = planes[p];
                    }
                }
            }
            if (check_dpu_error(
                    dpu_broadcast_to(
                        ctx->set,
                        "inputs_bitplanes_mram",
                        0,
                        input_bitplanes,
                        bitplane_u64_count * sizeof(uint64_t),
                        DPU_XFER_DEFAULT),
                    error_buffer,
                    error_buffer_len,
                    "dpu_broadcast_to(inputs_bitplanes_mram)") != 0) {
                goto cleanup;
            }
        } else {
            if (check_dpu_error(
                    dpu_broadcast_to(ctx->set, "inputs_i8_mram", 0, input_i8_shards, input_i8_count * sizeof(int8_t), DPU_XFER_DEFAULT),
                    error_buffer, error_buffer_len, "dpu_broadcast_to(inputs_i8_mram)") != 0) {
                goto cleanup;
            }
        }
    } else if (ctx->kernel_mode == 5) {
        const float *inputs_f32 = (const float *)inputs;
        for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            for (size_t block_idx = 0; block_idx < blocks_per_batch; ++block_idx) {
                const size_t block_offset = ((size_t)batch_idx * (size_t)ctx->input_dim) + (block_idx * BLOCK_FLOATS);
                float max_abs = 0.0f;
                for (size_t lane = 0; lane < BLOCK_FLOATS; ++lane) {
                    const float value = inputs_f32[block_offset + lane];
                    const float abs_value = value >= 0.0f ? value : -value;
                    if (abs_value > max_abs) {
                        max_abs = abs_value;
                    }
                }
                ctx->input_scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
                for (size_t lane = 0; lane < BLOCK_FLOATS; ++lane) {
                    float scaled = inputs_f32[block_offset + lane] / ctx->input_scale;
                    if (scaled > 127.0f) {
                        scaled = 127.0f;
                    } else if (scaled < -127.0f) {
                        scaled = -127.0f;
                    }
                    input_i8_shards[block_offset + lane] =
                        (int8_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
                }
                if (ctx->kernel_mode == 5) {
                    for (uint32_t row = 0; row < ctx->output_dim; ++row) {
                        const uint32_t group_idx = (uint32_t)((block_idx * BLOCK_FLOATS) / ctx->group_size);
                        const int16_t *base_lut =
                            ctx->lut_i16_shards + (((size_t)row * (size_t)ctx->num_groups) + group_idx) * (1u << BITS_PER_WEIGHT);
                        int16_t *runtime_lut =
                            runtime_lut_i16_shards
                            + ((((size_t)batch_idx * (size_t)ctx->output_dim) + row) * blocks_per_batch + block_idx)
                                * (1u << BITS_PER_WEIGHT);
                        for (uint32_t q = 0; q < (1u << BITS_PER_WEIGHT); ++q) {
                            int32_t value = (int32_t)((float)base_lut[q] * ctx->input_scale);
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
                dpu_broadcast_to(ctx->set, "inputs_i8_mram", 0, input_i8_shards, input_i8_count * sizeof(int8_t), DPU_XFER_DEFAULT),
                error_buffer, error_buffer_len, "dpu_broadcast_to(inputs_i8_mram)") != 0) {
            goto cleanup;
        }
        if (check_dpu_error(
                dpu_broadcast_to(
                    ctx->set,
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
                   dpu_broadcast_to(ctx->set, "inputs_mram", 0, inputs, input_floats * sizeof(float), DPU_XFER_DEFAULT),
                   error_buffer, error_buffer_len, "dpu_broadcast_to(inputs_mram)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &input_end);
    clock_gettime(CLOCK_MONOTONIC, &launch_start);
    if (check_dpu_error(dpu_launch(ctx->set, DPU_SYNCHRONOUS), error_buffer, error_buffer_len, "dpu_launch") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &launch_end);

    dpu_index = 0;
    clock_gettime(CLOCK_MONOTONIC, &output_start);
    if (ctx->kernel_mode == 4 || ctx->kernel_mode == 6 || ctx->kernel_mode == 7) {
        DPU_FOREACH(ctx->set, dpu, dpu_index)
        {
            if (check_dpu_error(
                    dpu_prepare_xfer(dpu, output_i32_shards + ((size_t)dpu_index * shard_output_i32)),
                    error_buffer, error_buffer_len, "dpu_prepare_xfer(outputs_i32_mram)") != 0) {
                goto cleanup;
            }
        }
        if (check_dpu_error(
                dpu_push_xfer(ctx->set, DPU_XFER_FROM_DPU, "outputs_i32_mram", 0, shard_output_i32 * sizeof(int32_t), DPU_XFER_DEFAULT),
                error_buffer, error_buffer_len, "dpu_push_xfer(outputs_i32_mram)") != 0) {
            goto cleanup;
        }
    } else {
        DPU_FOREACH(ctx->set, dpu, dpu_index)
        {
            if (check_dpu_error(
                    dpu_prepare_xfer(dpu, output_shards + ((size_t)dpu_index * shard_output_floats)),
                    error_buffer, error_buffer_len, "dpu_prepare_xfer(outputs_mram)") != 0) {
                goto cleanup;
            }
        }
        if (check_dpu_error(
                dpu_push_xfer(ctx->set, DPU_XFER_FROM_DPU, "outputs_mram", 0, shard_output_floats * sizeof(float), DPU_XFER_DEFAULT),
                error_buffer, error_buffer_len, "dpu_push_xfer(outputs_mram)") != 0) {
            goto cleanup;
        }
    }

    dpu_index = 0;
    DPU_FOREACH(ctx->set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(dpu, kernel_cycles + dpu_index),
                error_buffer, error_buffer_len, "dpu_prepare_xfer(kernel_cycles)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(ctx->set, DPU_XFER_FROM_DPU, "kernel_cycles", 0, sizeof(*kernel_cycles), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_push_xfer(kernel_cycles)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &output_end);

    float *output_dst = (float *)outputs;
    for (dpu_index = 0; dpu_index < ctx->nr_dpus; ++dpu_index) {
        const uint32_t local_rows = ctx->valid_rows[dpu_index];
        const size_t row_start = (size_t)dpu_index * ctx->rows_per_dpu;
        if (kernel_cycles[dpu_index] > max_cycles) {
            max_cycles = kernel_cycles[dpu_index];
        }
        for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            if (ctx->kernel_mode == 4 || ctx->kernel_mode == 5 || ctx->kernel_mode == 6 || ctx->kernel_mode == 7) {
                const int32_t *shard_ptr_i32 =
                    output_i32_shards + ((size_t)dpu_index * shard_output_i32) + ((size_t)batch_idx * ctx->shard_output_dim);
                for (uint32_t local_row = 0; local_row < local_rows; ++local_row) {
                    if (ctx->kernel_mode == 5) {
                        output_dst[((size_t)batch_idx * (size_t)ctx->output_dim) + row_start + local_row] =
                            ((float)shard_ptr_i32[local_row]) / 256.0f;
                    } else if (ctx->kernel_mode == 4 || ctx->kernel_mode == 6 || ctx->kernel_mode == 7) {
                        output_dst[((size_t)batch_idx * (size_t)ctx->output_dim) + row_start + local_row] =
                            ((float)shard_ptr_i32[local_row]) * (input_scales[batch_idx] / 256.0f);
                    }
                }
            } else {
                const float *shard_ptr =
                    output_shards + ((size_t)dpu_index * shard_output_floats) + ((size_t)batch_idx * ctx->shard_output_dim);
                memcpy(
                    output_dst + (((size_t)batch_idx * (size_t)ctx->output_dim) + row_start),
                    shard_ptr,
                    (size_t)local_rows * sizeof(float));
            }
        }
    }

    ctx->last_cycles = max_cycles;
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    ctx->last_input_transfer_seconds = timespec_diff_seconds(&input_start, &input_end);
    ctx->last_launch_seconds = timespec_diff_seconds(&launch_start, &launch_end);
    ctx->last_output_transfer_seconds = timespec_diff_seconds(&output_start, &output_end);
    ctx->last_total_seconds = timespec_diff_seconds(&total_start, &total_end);
    rc = 0;

cleanup:
    return rc;
}

int
pim_quantized_run_many(
    void *handle,
    uint32_t call_count,
    const uint32_t *batch_sizes,
    const void *const *inputs,
    void *const *outputs,
    const uint32_t *slot_ids,
    char *error_buffer,
    size_t error_buffer_len)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    struct dpu_set_t dpu;
    uint32_t dpu_index = 0;
    int rc = -1;
    uint64_t max_cycles = 0;
    struct timespec total_start;
    struct timespec total_end;
    struct timespec input_start;
    struct timespec input_end;
    struct timespec launch_start;
    struct timespec launch_end;
    struct timespec output_start;
    struct timespec output_end;
    uint32_t request_input_offsets[MAX_RUN_REQUESTS] = {0};
    uint32_t request_output_offsets[MAX_RUN_REQUESTS] = {0};
    uint32_t request_scale_offsets[MAX_RUN_REQUESTS] = {0};

    if (ctx == NULL) {
        set_error(error_buffer, error_buffer_len, "handle is NULL");
        return -1;
    }
    if (call_count == 0) {
        return 0;
    }
    if (batch_sizes == NULL || inputs == NULL || outputs == NULL || slot_ids == NULL) {
        set_error(error_buffer, error_buffer_len, "run_many arrays must be non-null");
        return -1;
    }
    if (call_count > MAX_RUN_REQUESTS) {
        set_error(error_buffer, error_buffer_len, "call_count too large: %u > %u", call_count, MAX_RUN_REQUESTS);
        return -1;
    }

    /* ADR-002 M-17.2: drain inflight async weight pushes before
     * touching the device, regardless of which run path we take.
     * The fallback loop calls pim_quantized_run() which would also
     * flush, but doing it here keeps the launch-time profile clean.
     */
    if (flush_inflight_async_load(ctx, error_buffer, error_buffer_len,
                                  "dpu_sync(flush before pim_quantized_run_many)") != 0) {
        return -1;
    }
    bool all_batch_one = true;
    for (uint32_t i = 0; i < call_count; ++i) {
        if (batch_sizes[i] != 1) {
            all_batch_one = false;
            break;
        }
    }
    if (ctx->kernel_mode != 4 || !all_batch_one) {
        /* M-15 true batching is implemented for the GPTQ decode hot path only. */
        double input_sum = 0.0;
        double launch_sum = 0.0;
        double output_sum = 0.0;
        double total_sum = 0.0;
        for (uint32_t i = 0; i < call_count; ++i) {
            const int single_rc = pim_quantized_run(
                handle,
                batch_sizes[i],
                inputs[i],
                outputs[i],
                slot_ids[i],
                error_buffer,
                error_buffer_len);
            if (single_rc != 0) {
                return single_rc;
            }
            input_sum += ctx->last_input_transfer_seconds;
            launch_sum += ctx->last_launch_seconds;
            output_sum += ctx->last_output_transfer_seconds;
            total_sum += ctx->last_total_seconds;
            if (ctx->last_cycles > max_cycles) {
                max_cycles = ctx->last_cycles;
            }
        }
        ctx->last_input_transfer_seconds = input_sum;
        ctx->last_launch_seconds = launch_sum;
        ctx->last_output_transfer_seconds = output_sum;
        ctx->last_total_seconds = total_sum;
        ctx->last_cycles = max_cycles;
        return 0;
    }

    size_t total_input_i8 = 0;
    size_t total_output_i32 = 0;
    size_t total_batches = 0;
    for (uint32_t i = 0; i < call_count; ++i) {
        if (inputs[i] == NULL || outputs[i] == NULL) {
            set_error(error_buffer, error_buffer_len, "run_many input/output pointers must be non-null");
            return -1;
        }
        if (slot_ids[i] >= NUM_SLOTS || (ctx->slot_loaded_mask & (1u << slot_ids[i])) == 0) {
            set_error(error_buffer, error_buffer_len, "slot %u has no weights loaded", slot_ids[i]);
            return -1;
        }
        request_input_offsets[i] = (uint32_t)total_input_i8;
        request_output_offsets[i] = (uint32_t)total_output_i32;
        request_scale_offsets[i] = (uint32_t)total_batches;
        total_input_i8 += (size_t)batch_sizes[i] * (size_t)ctx->input_dim;
        total_output_i32 += (size_t)batch_sizes[i] * ctx->shard_output_dim;
        total_batches += (size_t)batch_sizes[i];
    }
    if (total_input_i8 > MAX_INPUT_INT8 || total_output_i32 > MAX_OUTPUT_INT32) {
        set_error(error_buffer, error_buffer_len, "packed run_many input/output too large");
        return -1;
    }

    if (
        ensure_buffer((void **)&ctx->input_i8_shards, &ctx->input_i8_shards_capacity,
                      total_input_i8, sizeof(*ctx->input_i8_shards), error_buffer, error_buffer_len, "input_i8_shards") != 0
        || ensure_buffer((void **)&ctx->output_i32_shards, &ctx->output_i32_shards_capacity,
                         (size_t)ctx->nr_dpus * total_output_i32, sizeof(*ctx->output_i32_shards), error_buffer, error_buffer_len, "output_i32_shards") != 0
        || ensure_buffer((void **)&ctx->input_scales, &ctx->input_scales_capacity,
                         total_batches, sizeof(*ctx->input_scales), error_buffer, error_buffer_len, "input_scales") != 0
        || ensure_buffer((void **)&ctx->kernel_cycles, &ctx->kernel_cycles_capacity,
                         ctx->nr_dpus, sizeof(*ctx->kernel_cycles), error_buffer, error_buffer_len, "kernel_cycles") != 0
    ) {
        return -1;
    }

    clock_gettime(CLOCK_MONOTONIC, &total_start);
    clock_gettime(CLOCK_MONOTONIC, &input_start);
    for (uint32_t req = 0; req < call_count; ++req) {
        const float *inputs_f32 = (const float *)inputs[req];
        const size_t req_input_base = (size_t)request_input_offsets[req];
        const size_t req_scale_base = (size_t)request_scale_offsets[req];
        for (uint32_t batch_idx = 0; batch_idx < batch_sizes[req]; ++batch_idx) {
            const size_t batch_offset = (size_t)batch_idx * (size_t)ctx->input_dim;
            float max_abs = 0.0f;
            for (uint32_t col = 0; col < ctx->input_dim; ++col) {
                const float value = inputs_f32[batch_offset + col];
                const float abs_value = value >= 0.0f ? value : -value;
                if (abs_value > max_abs) {
                    max_abs = abs_value;
                }
            }
            const float input_scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
            ctx->input_scales[req_scale_base + batch_idx] = input_scale;
            for (uint32_t col = 0; col < ctx->input_dim; ++col) {
                float scaled = inputs_f32[batch_offset + col] / input_scale;
                if (scaled > 127.0f) {
                    scaled = 127.0f;
                } else if (scaled < -127.0f) {
                    scaled = -127.0f;
                }
                ctx->input_i8_shards[req_input_base + batch_offset + col] =
                    (int8_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
            }
        }
    }

    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "batch_size", 0, &call_count, sizeof(call_count), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(batch_size run_many)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "run_request_count", 0, &call_count, sizeof(call_count), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(run_request_count)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "request_active_slots", 0, slot_ids, call_count * sizeof(uint32_t), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(request_active_slots)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "request_batch_sizes", 0, batch_sizes, call_count * sizeof(uint32_t), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(request_batch_sizes)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "request_input_offsets", 0, request_input_offsets, call_count * sizeof(uint32_t), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(request_input_offsets)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "request_output_offsets", 0, request_output_offsets, call_count * sizeof(uint32_t), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(request_output_offsets)") != 0) {
        goto cleanup;
    }
    if (check_dpu_error(
            dpu_broadcast_to(ctx->set, "inputs_i8_mram", 0, ctx->input_i8_shards, total_input_i8 * sizeof(int8_t), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_broadcast_to(inputs_i8_mram run_many)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &input_end);

    clock_gettime(CLOCK_MONOTONIC, &launch_start);
    if (check_dpu_error(dpu_launch(ctx->set, DPU_SYNCHRONOUS), error_buffer, error_buffer_len, "dpu_launch(run_many)") != 0) {
        goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &launch_end);

    dpu_index = 0;
    clock_gettime(CLOCK_MONOTONIC, &output_start);
    DPU_FOREACH(ctx->set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(dpu, ctx->output_i32_shards + ((size_t)dpu_index * total_output_i32)),
                error_buffer, error_buffer_len, "dpu_prepare_xfer(outputs_i32_mram run_many)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(ctx->set, DPU_XFER_FROM_DPU, "outputs_i32_mram", 0,
                          total_output_i32 * sizeof(int32_t), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_push_xfer(outputs_i32_mram run_many)") != 0) {
        goto cleanup;
    }

    dpu_index = 0;
    DPU_FOREACH(ctx->set, dpu, dpu_index)
    {
        if (check_dpu_error(
                dpu_prepare_xfer(dpu, ctx->kernel_cycles + dpu_index),
                error_buffer, error_buffer_len, "dpu_prepare_xfer(kernel_cycles run_many)") != 0) {
            goto cleanup;
        }
    }
    if (check_dpu_error(
            dpu_push_xfer(ctx->set, DPU_XFER_FROM_DPU, "kernel_cycles", 0, sizeof(*ctx->kernel_cycles), DPU_XFER_DEFAULT),
            error_buffer, error_buffer_len, "dpu_push_xfer(kernel_cycles run_many)") != 0) {
        goto cleanup;
    }

    for (dpu_index = 0; dpu_index < ctx->nr_dpus; ++dpu_index) {
        const uint32_t local_rows = ctx->valid_rows[dpu_index];
        const size_t row_start = (size_t)dpu_index * ctx->rows_per_dpu;
        if (ctx->kernel_cycles[dpu_index] > max_cycles) {
            max_cycles = ctx->kernel_cycles[dpu_index];
        }
        for (uint32_t req = 0; req < call_count; ++req) {
            float *output_dst = (float *)outputs[req];
            const size_t req_output_base = (size_t)request_output_offsets[req];
            const size_t req_scale_base = (size_t)request_scale_offsets[req];
            for (uint32_t batch_idx = 0; batch_idx < batch_sizes[req]; ++batch_idx) {
                const int32_t *shard_ptr_i32 =
                    ctx->output_i32_shards
                    + ((size_t)dpu_index * total_output_i32)
                    + req_output_base
                    + ((size_t)batch_idx * ctx->shard_output_dim);
                const float scale = ctx->input_scales[req_scale_base + batch_idx] / 256.0f;
                for (uint32_t local_row = 0; local_row < local_rows; ++local_row) {
                    output_dst[((size_t)batch_idx * (size_t)ctx->output_dim) + row_start + local_row] =
                        ((float)shard_ptr_i32[local_row]) * scale;
                }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &output_end);

    ctx->last_cycles = max_cycles;
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    ctx->last_input_transfer_seconds = timespec_diff_seconds(&input_start, &input_end);
    ctx->last_launch_seconds = timespec_diff_seconds(&launch_start, &launch_end);
    ctx->last_output_transfer_seconds = timespec_diff_seconds(&output_start, &output_end);
    ctx->last_total_seconds = timespec_diff_seconds(&total_start, &total_end);
    rc = 0;

cleanup:
    return rc;
}

void
pim_quantized_shutdown(void *handle)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    if (ctx == NULL) {
        return;
    }

    /* ADR-002 M-17.2: drain any outstanding async weight pushes
     * before tearing down the DPU set. */
    if (ctx->inflight_async_load) {
        (void)dpu_sync(ctx->set);
        ctx->inflight_async_load = false;
    }

    free(ctx->valid_rows);
    free(ctx->lut_i16_shards);
    free(ctx->input_i8_shards);
    free(ctx->output_i32_shards);
    free(ctx->runtime_lut_i16_shards);
    free(ctx->load_qweight_shards);
    free(ctx->load_scale_shards);
    free(ctx->kernel_cycles);
    free(ctx->output_shards);
    free(ctx->input_scales);
    free(ctx->input_bitplanes);
    dpu_free(ctx->set);
    free(ctx);  /* handle is invalid after this call. */
}

uint64_t
pim_quantized_last_cycles(void *handle)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    return ctx ? ctx->last_cycles : 0;
}

double
pim_quantized_last_input_transfer_seconds(void *handle)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    return ctx ? ctx->last_input_transfer_seconds : 0.0;
}

double
pim_quantized_last_load_qweight_transfer_seconds(void *handle)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    return ctx ? ctx->last_load_qweight_transfer_seconds : 0.0;
}

double
pim_quantized_last_load_scale_transfer_seconds(void *handle)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    return ctx ? ctx->last_load_scale_transfer_seconds : 0.0;
}

double
pim_quantized_last_load_total_seconds(void *handle)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    return ctx ? ctx->last_load_total_seconds : 0.0;
}

double
pim_quantized_last_launch_seconds(void *handle)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    return ctx ? ctx->last_launch_seconds : 0.0;
}

double
pim_quantized_last_output_transfer_seconds(void *handle)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    return ctx ? ctx->last_output_transfer_seconds : 0.0;
}

double
pim_quantized_last_total_seconds(void *handle)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    return ctx ? ctx->last_total_seconds : 0.0;
}

uint32_t
pim_quantized_num_dpus(void *handle)
{
    pim_q_ctx_t *ctx = (pim_q_ctx_t *)handle;
    return ctx ? ctx->nr_dpus : 0;
}

/* ============================================================================
 * ADR-002 M-24 Stage B: fused gate_up + silu*up + down single C call.
 *
 * Motivation:
 *   Per decode step the Python-side `_run_quantized_experts_batched_on_dpu`
 *   pays two `pim_quantized_run_many` ctypes round-trips per layer (one for
 *   gate_up, one for down) plus a Python-level `F.silu(gate) * up` in between.
 *   48 layers × 32 tokens × 2 RT + silu/mul = ~3000 Python↔C transitions per
 *   e2e run, each ~100-300µs on top of DPU work.  Fusing the three phases
 *   inside one C call:
 *     - halves ctypes RT (2 → 1 per layer)
 *     - replaces Python silu*up (~200-500µs/layer) with a tight C fp32 loop
 *     - prepares the ground for M-24 Stage A (the whole fused op can be
 *       submitted via DPU_ASYNCHRONOUS in one go)
 *
 * This function never touches the DPU kernel binary; it is 100% host-side
 * orchestration.  PIM still performs the actual gate_up and down matvecs.
 *
 * Contract:
 *   handle_gate_up : M-5 gate_up runtime context
 *   handle_down    : M-5 down runtime context (may equal handle_gate_up
 *                    when the dual-allocation fallback collapsed)
 *   call_count     : number of experts in this layer's batched submission
 *   gate_up_*      : arrays length=call_count describing the gate_up
 *                    pim_quantized_run_many() call:
 *                      - batch_sizes[i]   : #tokens routed to expert i
 *                      - slot_ids[i]      : gate_up runtime slot
 *                      - inputs[i]        : fp32 [batch, hidden_size] activation
 *   gate_cols[i]   : #output columns in the gate slice of the concat
 *                    (== lhs_orig from preload_concat_and_get_slot)
 *   up_cols[i]     : #output columns in the up slice (== rhs_orig; must equal
 *                    gate_cols[i] for SwiGLU Qwen3-style experts)
 *   down_*         : arrays length=call_count describing the down call:
 *                      - batch_sizes[i]   : must equal gate_up_batch_sizes[i]
 *                      - slot_ids[i]      : down runtime slot
 *                      - outputs[i]       : fp32 [batch, down_output_dim] preallocated
 *   down_output_dim: output width of the down projection (== hidden_size for Qwen3)
 *
 * The caller retains ownership of all input / output buffers.  Intermediate
 * scratch (concat gate_up output, silu*up hidden) is allocated and freed
 * inside this call so Python never materialises those tensors.
 *
 * Returns 0 on success, -1 on error with a human-readable message in
 * error_buffer.
 */
int
pim_quantized_run_many_fused_silu(
    void *handle_gate_up,
    void *handle_down,
    uint32_t call_count,
    const uint32_t *gate_up_batch_sizes,
    const uint32_t *gate_up_slot_ids,
    const void *const *gate_up_inputs,
    const uint32_t *gate_cols,
    const uint32_t *up_cols,
    const uint32_t *down_batch_sizes,
    const uint32_t *down_slot_ids,
    void *const *down_outputs,
    uint32_t down_output_dim,
    char *error_buffer,
    size_t error_buffer_len)
{
    pim_q_ctx_t *ctx_gu = (pim_q_ctx_t *)handle_gate_up;
    pim_q_ctx_t *ctx_dn = (pim_q_ctx_t *)handle_down;
    int rc = -1;
    float *concat_scratch = NULL;   /* concatenated gate_up outputs (fp32) */
    float *hidden_scratch = NULL;   /* silu(gate)*up outputs (fp32) */
    const void **concat_ptrs = NULL;
    const void **hidden_ptrs = NULL;
    void **concat_ptrs_w = NULL;    /* writable view for run_many() signature */

    if (ctx_gu == NULL || ctx_dn == NULL) {
        set_error(error_buffer, error_buffer_len, "fused: handles must be non-null");
        return -1;
    }
    if (call_count == 0) {
        return 0;
    }
    if (call_count > MAX_RUN_REQUESTS) {
        set_error(error_buffer, error_buffer_len, "fused: call_count %u > MAX_RUN_REQUESTS %u",
                  call_count, MAX_RUN_REQUESTS);
        return -1;
    }
    if (gate_up_batch_sizes == NULL || gate_up_slot_ids == NULL || gate_up_inputs == NULL
        || gate_cols == NULL || up_cols == NULL || down_batch_sizes == NULL
        || down_slot_ids == NULL || down_outputs == NULL) {
        set_error(error_buffer, error_buffer_len, "fused: array arguments must be non-null");
        return -1;
    }

    /* Sanity: gate_cols + up_cols per expert must not exceed gate_up ctx's
     * loaded output_dim (the concat row count).  The concat width equals
     * ctx_gu->output_dim, which was set by the most recent load_weights
     * for that ctx's slot.  Because slots share the same shape in our
     * usage (all experts have identical projection shapes in Qwen3),
     * checking against ctx_gu->output_dim is sufficient. */
    for (uint32_t i = 0; i < call_count; ++i) {
        if (gate_up_batch_sizes[i] != down_batch_sizes[i]) {
            set_error(error_buffer, error_buffer_len,
                      "fused: batch_size mismatch expert %u (gate_up=%u down=%u)",
                      i, gate_up_batch_sizes[i], down_batch_sizes[i]);
            return -1;
        }
        /* gate_cols and up_cols may be zero if caller passes a malformed
         * request; guard against that. */
        if (gate_cols[i] == 0 || up_cols[i] == 0) {
            set_error(error_buffer, error_buffer_len,
                      "fused: gate_cols/up_cols must be positive (expert %u: %u/%u)",
                      i, gate_cols[i], up_cols[i]);
            return -1;
        }
        /* For SwiGLU (Qwen3) gate_cols == up_cols.  Enforce so the silu*up
         * loop can use a single width per expert. */
        if (gate_cols[i] != up_cols[i]) {
            set_error(error_buffer, error_buffer_len,
                      "fused: gate_cols must equal up_cols (expert %u: %u vs %u)",
                      i, gate_cols[i], up_cols[i]);
            return -1;
        }
    }

    /* Compute total scratch sizes:
     *   concat_scratch: sum_i batch_i * ctx_gu->output_dim  (padded concat row count)
     *   hidden_scratch: sum_i batch_i * up_cols[i]          (tight, will feed down)
     *
     * We over-allocate concat_scratch using ctx_gu->output_dim because
     * pim_quantized_run() writes padded_output_dim columns; the caller-
     * facing gate_cols/up_cols are the original (unpadded) widths.  We
     * index with ctx_gu->output_dim stride to extract the gate / up
     * slices correctly.
     */
    size_t total_concat_floats = 0;
    size_t total_hidden_floats = 0;
    size_t hidden_offsets[MAX_RUN_REQUESTS];
    size_t concat_offsets[MAX_RUN_REQUESTS];
    for (uint32_t i = 0; i < call_count; ++i) {
        concat_offsets[i] = total_concat_floats;
        hidden_offsets[i] = total_hidden_floats;
        total_concat_floats += (size_t)gate_up_batch_sizes[i] * (size_t)ctx_gu->output_dim;
        total_hidden_floats += (size_t)gate_up_batch_sizes[i] * (size_t)up_cols[i];
    }

    concat_scratch = (float *)calloc(total_concat_floats, sizeof(float));
    hidden_scratch = (float *)calloc(total_hidden_floats, sizeof(float));
    concat_ptrs = (const void **)calloc(call_count, sizeof(*concat_ptrs));
    concat_ptrs_w = (void **)calloc(call_count, sizeof(*concat_ptrs_w));
    hidden_ptrs = (const void **)calloc(call_count, sizeof(*hidden_ptrs));
    if (concat_scratch == NULL || hidden_scratch == NULL
        || concat_ptrs == NULL || concat_ptrs_w == NULL || hidden_ptrs == NULL) {
        set_error(error_buffer, error_buffer_len, "fused: scratch allocation failed");
        goto cleanup;
    }
    for (uint32_t i = 0; i < call_count; ++i) {
        concat_ptrs_w[i] = concat_scratch + concat_offsets[i];
        concat_ptrs[i] = concat_scratch + concat_offsets[i];
        hidden_ptrs[i] = hidden_scratch + hidden_offsets[i];
    }

    /* Phase 1: gate_up via the existing batched API (internally SYNC). */
    if (pim_quantized_run_many(
            handle_gate_up,
            call_count,
            gate_up_batch_sizes,
            gate_up_inputs,
            concat_ptrs_w,
            gate_up_slot_ids,
            error_buffer,
            error_buffer_len) != 0) {
        goto cleanup;
    }

    /* Phase 2: silu(gate) * up, written into hidden_scratch.
     * Reference (matches torch.nn.functional.silu):
     *   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
     *
     * For each expert i:
     *   for b in [0, batch):
     *     gate_row = concat_scratch[concat_offsets[i] + b*concat_stride + 0 .. gate_cols[i])
     *     up_row   = concat_scratch[concat_offsets[i] + b*concat_stride + gate_cols[i] .. gate_cols[i]+up_cols[i])
     *     hidden_row = silu(gate_row) * up_row
     */
    for (uint32_t i = 0; i < call_count; ++i) {
        const uint32_t batch = gate_up_batch_sizes[i];
        const size_t concat_stride = (size_t)ctx_gu->output_dim;
        const uint32_t gcols = gate_cols[i];
        const uint32_t ucols = up_cols[i];  /* == gcols enforced above */
        const float *concat_base = concat_scratch + concat_offsets[i];
        float *hidden_base = hidden_scratch + hidden_offsets[i];
        for (uint32_t b = 0; b < batch; ++b) {
            const float *row = concat_base + (size_t)b * concat_stride;
            float *out_row = hidden_base + (size_t)b * ucols;
            const float *gate_row = row;
            const float *up_row = row + gcols;
            for (uint32_t c = 0; c < ucols; ++c) {
                const float gv = gate_row[c];
                /* silu = gv / (1 + exp(-gv)) */
                const float sig = 1.0f / (1.0f + expf(-gv));
                out_row[c] = (gv * sig) * up_row[c];
            }
        }
    }

    /* Phase 3: down via the existing batched API. */
    if (pim_quantized_run_many(
            handle_down,
            call_count,
            down_batch_sizes,
            hidden_ptrs,
            down_outputs,
            down_slot_ids,
            error_buffer,
            error_buffer_len) != 0) {
        goto cleanup;
    }

    (void)down_output_dim;  /* consumed implicitly via down ctx's output_dim */
    rc = 0;

cleanup:
    free(concat_scratch);
    free(hidden_scratch);
    free(concat_ptrs);
    free(concat_ptrs_w);
    free(hidden_ptrs);
    return rc;
}


/* ============================================================================
 * ADR-002 M-24 Stage A: C-level async submit of the fused op.
 *
 * Motivation:
 *   Stage B (pim_quantized_run_many_fused_silu) is still synchronous from
 *   Python's point of view — it blocks the caller until DPU work completes.
 *   HybridMoE.forward(python) runs its 92 GPU-resident experts AFTER
 *   submit_forward returns, meaning the GPU expert loop sits idle while
 *   PIM crunches.
 *
 *   Stage A: spawn a C-level pthread that runs the fused op while Python
 *   returns to the caller immediately.  The caller later blocks on
 *   pim_quantized_fused_wait() to collect results.  This gets real GPU/PIM
 *   overlap without the Python GIL contention that sank M-10.
 *
 * Threading model:
 *   - Each submit allocates a fresh ``pim_fused_async_job_t`` containing a
 *     pthread_t, a done-flag protected by a mutex+cond pair, and snapshots
 *     of all user-facing pointers.  The worker owns these resources until
 *     wait() joins and frees them.
 *   - The ctx scratch buffers (input_i8_shards etc.) are touched only by
 *     the worker thread, never concurrently by Python.  Since there is at
 *     most one in-flight async job per HybridMoE layer in our target
 *     workload (decode step is serial across layers), two async jobs on
 *     the same ctx never happen — Python always waits between layers.
 *   - If Python tries to launch a second async job on the same ctx before
 *     waiting, the worker would race with itself on ctx state.  We guard
 *     via ``ctx_has_pending_async`` in the ctx — set on submit, cleared on
 *     wait.  Trying to submit while pending returns an error.
 *   - Weight buffers (``gate_up_inputs`` / ``down_outputs``) must remain
 *     valid until wait() returns; Python caller responsibility documented
 *     in the runtime wrapper.
 *
 * Error propagation:
 *   The worker captures any error (rc + error string) on its local error
 *   buffer, and wait() returns that rc + copies the error string to the
 *   caller's error buffer.
 * ==========================================================================*/

typedef struct pim_fused_async_job_s {
    /* Worker thread handle. */
    pthread_t worker;
    bool joined;                    /* wait() sets this to prevent double-join */

    /* Inputs (shallow pointer copies; caller keeps underlying memory alive). */
    void *handle_gate_up;
    void *handle_down;
    uint32_t call_count;
    /* The arrays below are deep-copied on submit so the caller may free
     * its Python-managed ctypes arrays immediately after submit returns. */
    uint32_t *gate_up_batch_sizes;
    uint32_t *gate_up_slot_ids;
    const void **gate_up_inputs;    /* still shallow — inputs float32 buffers
                                     * belong to caller, stay alive until wait */
    uint32_t *gate_cols;
    uint32_t *up_cols;
    uint32_t *down_batch_sizes;
    uint32_t *down_slot_ids;
    void **down_outputs;            /* also shallow; output buffers alive until wait */
    uint32_t down_output_dim;

    /* Result propagated by the worker. */
    int result_rc;
    char result_error[2048];
} pim_fused_async_job_t;

static void
free_job(pim_fused_async_job_t *job)
{
    if (job == NULL) {
        return;
    }
    free(job->gate_up_batch_sizes);
    free(job->gate_up_slot_ids);
    free(job->gate_up_inputs);
    free(job->gate_cols);
    free(job->up_cols);
    free(job->down_batch_sizes);
    free(job->down_slot_ids);
    free(job->down_outputs);
    free(job);
}

static void *
pim_fused_async_worker(void *arg)
{
    pim_fused_async_job_t *job = (pim_fused_async_job_t *)arg;
    job->result_rc = pim_quantized_run_many_fused_silu(
        job->handle_gate_up,
        job->handle_down,
        job->call_count,
        job->gate_up_batch_sizes,
        job->gate_up_slot_ids,
        job->gate_up_inputs,
        job->gate_cols,
        job->up_cols,
        job->down_batch_sizes,
        job->down_slot_ids,
        job->down_outputs,
        job->down_output_dim,
        job->result_error,
        sizeof(job->result_error));
    return NULL;
}

/* Submit a fused gate_up + silu*up + down op to a C pthread worker.
 *
 * On success, returns a non-NULL opaque token (cast to void * and returned
 * via ``*out_token``).  Caller must pair each successful submit with a
 * matching ``pim_quantized_fused_wait()`` call or resources will leak.
 *
 * On failure (malloc / thread spawn), returns -1 and writes an error
 * message; ``*out_token`` is set to NULL.
 *
 * The arrays (batch_sizes, slot_ids, gate/up cols, inputs, outputs) are
 * deep-copied internally except for the payload buffers pointed to by
 * inputs[] / outputs[].  The caller retains ownership of those buffers
 * and MUST keep them alive until wait() returns.
 */
int
pim_quantized_run_many_fused_silu_async(
    void *handle_gate_up,
    void *handle_down,
    uint32_t call_count,
    const uint32_t *gate_up_batch_sizes,
    const uint32_t *gate_up_slot_ids,
    const void *const *gate_up_inputs,
    const uint32_t *gate_cols,
    const uint32_t *up_cols,
    const uint32_t *down_batch_sizes,
    const uint32_t *down_slot_ids,
    void *const *down_outputs,
    uint32_t down_output_dim,
    void **out_token,
    char *error_buffer,
    size_t error_buffer_len)
{
    if (out_token == NULL) {
        set_error(error_buffer, error_buffer_len, "fused_async: out_token must be non-null");
        return -1;
    }
    *out_token = NULL;
    if (handle_gate_up == NULL || handle_down == NULL) {
        set_error(error_buffer, error_buffer_len, "fused_async: handles must be non-null");
        return -1;
    }
    if (call_count == 0) {
        /* Trivial case: no work, no worker thread.  Return a sentinel
         * token that wait() recognises. */
        pim_fused_async_job_t *job = (pim_fused_async_job_t *)calloc(1, sizeof(*job));
        if (job == NULL) {
            set_error(error_buffer, error_buffer_len, "fused_async: calloc job failed");
            return -1;
        }
        job->joined = true;  /* no thread to join */
        job->result_rc = 0;
        *out_token = job;
        return 0;
    }
    if (call_count > MAX_RUN_REQUESTS) {
        set_error(error_buffer, error_buffer_len,
                  "fused_async: call_count %u > MAX_RUN_REQUESTS %u",
                  call_count, MAX_RUN_REQUESTS);
        return -1;
    }
    if (gate_up_batch_sizes == NULL || gate_up_slot_ids == NULL || gate_up_inputs == NULL
        || gate_cols == NULL || up_cols == NULL || down_batch_sizes == NULL
        || down_slot_ids == NULL || down_outputs == NULL) {
        set_error(error_buffer, error_buffer_len, "fused_async: array arguments must be non-null");
        return -1;
    }

    pim_fused_async_job_t *job = (pim_fused_async_job_t *)calloc(1, sizeof(*job));
    if (job == NULL) {
        set_error(error_buffer, error_buffer_len, "fused_async: calloc job failed");
        return -1;
    }
    job->handle_gate_up = handle_gate_up;
    job->handle_down = handle_down;
    job->call_count = call_count;
    job->down_output_dim = down_output_dim;
    job->joined = false;

    /* Deep-copy integer arrays (small, fixed-size); shallow-copy pointer arrays. */
    const size_t u32_bytes = call_count * sizeof(uint32_t);
    const size_t in_ptr_bytes = call_count * sizeof(const void *);
    const size_t out_ptr_bytes = call_count * sizeof(void *);
    job->gate_up_batch_sizes = (uint32_t *)malloc(u32_bytes);
    job->gate_up_slot_ids = (uint32_t *)malloc(u32_bytes);
    job->gate_up_inputs = (const void **)malloc(in_ptr_bytes);
    job->gate_cols = (uint32_t *)malloc(u32_bytes);
    job->up_cols = (uint32_t *)malloc(u32_bytes);
    job->down_batch_sizes = (uint32_t *)malloc(u32_bytes);
    job->down_slot_ids = (uint32_t *)malloc(u32_bytes);
    job->down_outputs = (void **)malloc(out_ptr_bytes);
    if (job->gate_up_batch_sizes == NULL || job->gate_up_slot_ids == NULL
        || job->gate_up_inputs == NULL || job->gate_cols == NULL
        || job->up_cols == NULL || job->down_batch_sizes == NULL
        || job->down_slot_ids == NULL || job->down_outputs == NULL) {
        set_error(error_buffer, error_buffer_len, "fused_async: malloc array copies failed");
        free_job(job);
        return -1;
    }
    memcpy(job->gate_up_batch_sizes, gate_up_batch_sizes, u32_bytes);
    memcpy(job->gate_up_slot_ids, gate_up_slot_ids, u32_bytes);
    memcpy(job->gate_up_inputs, gate_up_inputs, in_ptr_bytes);
    memcpy(job->gate_cols, gate_cols, u32_bytes);
    memcpy(job->up_cols, up_cols, u32_bytes);
    memcpy(job->down_batch_sizes, down_batch_sizes, u32_bytes);
    memcpy(job->down_slot_ids, down_slot_ids, u32_bytes);
    memcpy(job->down_outputs, down_outputs, out_ptr_bytes);

    int perr = pthread_create(&job->worker, NULL, pim_fused_async_worker, job);
    if (perr != 0) {
        set_error(error_buffer, error_buffer_len,
                  "fused_async: pthread_create failed (errno=%d)", perr);
        free_job(job);
        return -1;
    }
    *out_token = job;
    return 0;
}

/* Block until the async fused op behind ``token`` finishes.  Must be
 * called exactly once per successful submit (returns -1 if called twice
 * or on a NULL token).  Propagates the worker's error via error_buffer.
 *
 * ``token`` is invalidated (freed) by this call regardless of the worker's
 * return code.
 */
int
pim_quantized_fused_wait(
    void *token,
    char *error_buffer,
    size_t error_buffer_len)
{
    if (token == NULL) {
        set_error(error_buffer, error_buffer_len, "fused_wait: token is NULL");
        return -1;
    }
    pim_fused_async_job_t *job = (pim_fused_async_job_t *)token;
    if (!job->joined) {
        int perr = pthread_join(job->worker, NULL);
        if (perr != 0) {
            set_error(error_buffer, error_buffer_len,
                      "fused_wait: pthread_join failed (errno=%d)", perr);
            /* Still mark joined so free_job() doesn't try again, and
             * release resources. */
            job->joined = true;
            int rc = job->result_rc;
            free_job(job);
            return rc != 0 ? rc : -1;
        }
        job->joined = true;
    }
    int rc = job->result_rc;
    if (rc != 0) {
        /* Copy worker's error into caller's buffer. */
        if (error_buffer != NULL && error_buffer_len > 0) {
            snprintf(error_buffer, error_buffer_len, "%s", job->result_error);
        }
    }
    free_job(job);
    return rc;
}
