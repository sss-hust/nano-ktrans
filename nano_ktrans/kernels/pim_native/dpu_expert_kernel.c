#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>

#include "silu_lut_4096.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

#ifndef BLOCK_FLOATS
#define BLOCK_FLOATS 64
#endif

#ifndef MAX_WEIGHT_FLOATS
#define MAX_WEIGHT_FLOATS 2097152
#endif

#ifndef MAX_INPUT_FLOATS
#define MAX_INPUT_FLOATS 65536
#endif

#ifndef MAX_OUTPUT_FLOATS
#define MAX_OUTPUT_FLOATS 65536
#endif

BARRIER_INIT(work_barrier, NR_TASKLETS);

__host uint32_t batch_size;
__host uint32_t input_dim;
__host uint32_t intermediate_dim;
__host uint32_t output_dim;
__host uint64_t kernel_cycles;

__mram_noinit float gate_proj_mram[MAX_WEIGHT_FLOATS];
__mram_noinit float up_proj_mram[MAX_WEIGHT_FLOATS];
__mram_noinit float down_proj_mram[MAX_WEIGHT_FLOATS];
__mram_noinit float inputs_mram[MAX_INPUT_FLOATS];
__mram_noinit float outputs_mram[MAX_OUTPUT_FLOATS];
__dma_aligned float hidden_wram[MAX_INTERMEDIATE_DIM];

static inline float
silu_lookup(float x)
{
    if (x <= SILU_LUT_MIN) {
        return 0.0f;
    }
    if (x >= SILU_LUT_MAX) {
        return x;
    }
    const float scaled = (x - SILU_LUT_MIN) * ((float)SILU_LUT_SIZE / (SILU_LUT_MAX - SILU_LUT_MIN));
    const uint32_t idx = (uint32_t)scaled;
    const float frac = scaled - (float)idx;
    const float v0 = kSiluLut[idx];
    const float v1 = kSiluLut[idx + 1];
    return v0 + (v1 - v0) * frac;
}

int
main(void)
{
    const uint32_t tasklet_id = me();

    if (tasklet_id == 0) {
        mem_reset();
        perfcounter_config(COUNT_CYCLES, true);
    }

    barrier_wait(&work_barrier);
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&work_barrier);

    __dma_aligned float input_cache[BLOCK_FLOATS];
    __dma_aligned float proj0_cache[BLOCK_FLOATS];
    __dma_aligned float proj1_cache[BLOCK_FLOATS];
    __dma_aligned float hidden_cache[BLOCK_FLOATS];
    __dma_aligned float output_pair[2];

    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (uint32_t hidden_idx = tasklet_id; hidden_idx < intermediate_dim; hidden_idx += NR_TASKLETS) {
            float gate_acc = 0.0f;
            float up_acc = 0.0f;

            for (uint32_t col = 0; col < input_dim; col += BLOCK_FLOATS) {
                mram_read(
                    (__mram_ptr void const *)(inputs_mram + ((batch_idx * input_dim) + col)),
                    input_cache,
                    BLOCK_FLOATS * sizeof(float));
                mram_read(
                    (__mram_ptr void const *)(gate_proj_mram + ((hidden_idx * input_dim) + col)),
                    proj0_cache,
                    BLOCK_FLOATS * sizeof(float));
                mram_read(
                    (__mram_ptr void const *)(up_proj_mram + ((hidden_idx * input_dim) + col)),
                    proj1_cache,
                    BLOCK_FLOATS * sizeof(float));
                for (uint32_t i = 0; i < BLOCK_FLOATS; ++i) {
                    gate_acc += input_cache[i] * proj0_cache[i];
                    up_acc += input_cache[i] * proj1_cache[i];
                }
            }
            hidden_wram[hidden_idx] = silu_lookup(gate_acc) * up_acc;
        }

        barrier_wait(&work_barrier);

        for (uint32_t row = tasklet_id * 2; row < output_dim; row += NR_TASKLETS * 2) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;

            for (uint32_t hidden_base = 0; hidden_base < intermediate_dim; hidden_base += BLOCK_FLOATS) {
                for (uint32_t i = 0; i < BLOCK_FLOATS; ++i) {
                    hidden_cache[i] = hidden_wram[hidden_base + i];
                }
                mram_read(
                    (__mram_ptr void const *)(down_proj_mram + ((row * intermediate_dim) + hidden_base)),
                    proj0_cache,
                    BLOCK_FLOATS * sizeof(float));
                mram_read(
                    (__mram_ptr void const *)(down_proj_mram + (((row + 1) * intermediate_dim) + hidden_base)),
                    proj1_cache,
                    BLOCK_FLOATS * sizeof(float));
                for (uint32_t i = 0; i < BLOCK_FLOATS; ++i) {
                    acc0 += hidden_cache[i] * proj0_cache[i];
                    acc1 += hidden_cache[i] * proj1_cache[i];
                }
            }

            output_pair[0] = acc0;
            output_pair[1] = acc1;
            mram_write(
                output_pair,
                (__mram_ptr void *)(outputs_mram + ((batch_idx * output_dim) + row)),
                2 * sizeof(float));
        }

        barrier_wait(&work_barrier);
    }

    barrier_wait(&work_barrier);
    if (tasklet_id == 0) {
        kernel_cycles = perfcounter_get();
    }

    return 0;
}
