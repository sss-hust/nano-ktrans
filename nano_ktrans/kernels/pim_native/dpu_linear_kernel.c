#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>

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
__host uint32_t output_dim;
__host uint64_t kernel_cycles;

__mram_noinit float weights_mram[MAX_WEIGHT_FLOATS];
__mram_noinit float inputs_mram[MAX_INPUT_FLOATS];
__mram_noinit float outputs_mram[MAX_OUTPUT_FLOATS];

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
    __dma_aligned float weight0_cache[BLOCK_FLOATS];
    __dma_aligned float weight1_cache[BLOCK_FLOATS];
    __dma_aligned float output_cache[2];

    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (uint32_t row = tasklet_id * 2; row < output_dim; row += NR_TASKLETS * 2) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint32_t col = 0; col < input_dim; col += BLOCK_FLOATS) {
                mram_read(
                    (__mram_ptr void const *)(inputs_mram + ((batch_idx * input_dim) + col)),
                    input_cache,
                    BLOCK_FLOATS * sizeof(float));
                mram_read(
                    (__mram_ptr void const *)(weights_mram + ((row * input_dim) + col)),
                    weight0_cache,
                    BLOCK_FLOATS * sizeof(float));
                mram_read(
                    (__mram_ptr void const *)(weights_mram + (((row + 1) * input_dim) + col)),
                    weight1_cache,
                    BLOCK_FLOATS * sizeof(float));

                for (uint32_t i = 0; i < BLOCK_FLOATS; ++i) {
                    acc0 += input_cache[i] * weight0_cache[i];
                    acc1 += input_cache[i] * weight1_cache[i];
                }
            }

            output_cache[0] = acc0;
            output_cache[1] = acc1;
            mram_write(
                output_cache,
                (__mram_ptr void *)(outputs_mram + ((batch_idx * output_dim) + row)),
                2 * sizeof(float));
        }
    }

    barrier_wait(&work_barrier);
    if (tasklet_id == 0) {
        kernel_cycles = perfcounter_get();
    }

    return 0;
}
