#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

#ifndef BLOCK_BYTES
#define BLOCK_BYTES 1024
#endif

#ifndef MAX_BYTES_PER_DPU
#define MAX_BYTES_PER_DPU (8 * 1024 * 1024)
#endif

#define BLOCK_WORDS (BLOCK_BYTES / sizeof(uint32_t))

BARRIER_INIT(work_barrier, NR_TASKLETS);

__host uint32_t bytes_per_dpu;
__host uint32_t repetitions;
__host uint32_t scalar;
__host uint64_t kernel_cycles;

__mram_noinit uint8_t input_mram[MAX_BYTES_PER_DPU];
__mram_noinit uint8_t output_mram[MAX_BYTES_PER_DPU];

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

    __dma_aligned uint32_t cache[BLOCK_WORDS];

    for (uint32_t rep = 0; rep < repetitions; ++rep) {
        for (uint32_t offset = tasklet_id * BLOCK_BYTES; offset < bytes_per_dpu; offset += NR_TASKLETS * BLOCK_BYTES) {
            mram_read((__mram_ptr void const *)(input_mram + offset), cache, BLOCK_BYTES);
            for (uint32_t i = 0; i < BLOCK_WORDS; ++i) {
                cache[i] = cache[i] * scalar + 1u;
            }
            mram_write(cache, (__mram_ptr void *)(output_mram + offset), BLOCK_BYTES);
        }
    }

    barrier_wait(&work_barrier);
    if (tasklet_id == 0) {
        kernel_cycles = perfcounter_get();
    }

    return 0;
}
