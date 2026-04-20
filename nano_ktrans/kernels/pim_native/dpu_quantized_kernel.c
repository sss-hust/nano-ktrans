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

#ifndef BITS_PER_WEIGHT
#define BITS_PER_WEIGHT 4
#endif

#ifndef WEIGHTS_PER_WORD
#define WEIGHTS_PER_WORD (32 / BITS_PER_WEIGHT)
#endif

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

#ifndef MAX_GROUPS
#define MAX_GROUPS 64
#endif

#ifndef MAX_LUT_INT16
#define MAX_LUT_INT16 (MAX_SCALE_FLOATS * (1 << BITS_PER_WEIGHT))
#endif

#ifndef MAX_RUNTIME_LUT_INT16
#define MAX_RUNTIME_LUT_INT16 131072
#endif

#ifndef FIXED_BATCH_TILE
#define FIXED_BATCH_TILE 4
#endif

BARRIER_INIT(work_barrier, NR_TASKLETS);

__host uint32_t batch_size;
__host uint32_t input_dim;
__host uint32_t output_dim;
__host uint32_t group_size;
__host uint32_t num_groups;
__host uint32_t kernel_mode;
__host uint64_t kernel_cycles;

__mram_noinit uint32_t qweight_mram[MAX_QWEIGHT_WORDS];
__mram_noinit float scales_mram[MAX_SCALE_FLOATS];
__mram_noinit float inputs_mram[MAX_INPUT_FLOATS];
__mram_noinit float outputs_mram[MAX_OUTPUT_FLOATS];
__mram_noinit int16_t lut_mram[MAX_LUT_INT16];
__mram_noinit int8_t inputs_i8_mram[MAX_INPUT_INT8];
__mram_noinit int32_t outputs_i32_mram[MAX_OUTPUT_INT32];
__mram_noinit int16_t runtime_lut_mram[MAX_RUNTIME_LUT_INT16];

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
    __dma_aligned uint32_t qweight0_cache[BLOCK_FLOATS / WEIGHTS_PER_WORD];
    __dma_aligned uint32_t qweight1_cache[BLOCK_FLOATS / WEIGHTS_PER_WORD];
    __dma_aligned float scales0_cache[MAX_GROUPS];
    __dma_aligned float scales1_cache[MAX_GROUPS];
    __dma_aligned float output_cache[2];
    __dma_aligned int8_t input_i8_cache[BLOCK_FLOATS];
    __dma_aligned int8_t input_i8_tile_cache[FIXED_BATCH_TILE][BLOCK_FLOATS];
    __dma_aligned int16_t lut0_i16[1 << BITS_PER_WEIGHT];
    __dma_aligned int16_t lut1_i16[1 << BITS_PER_WEIGHT];
    __dma_aligned int32_t output_i32_cache[2];
    float lut0[1 << BITS_PER_WEIGHT];
    float lut1[1 << BITS_PER_WEIGHT];

    const uint32_t words_per_row = input_dim / WEIGHTS_PER_WORD;
    const float zero_point = (float)(1 << (BITS_PER_WEIGHT - 1));

    for (uint32_t row = tasklet_id * 2; row < output_dim; row += NR_TASKLETS * 2) {
        for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            if (kernel_mode == 5) {
                int32_t acc0 = 0;
                int32_t acc1 = 0;

                for (uint32_t col = 0; col < input_dim; col += BLOCK_FLOATS) {
                    const uint32_t word_offset = col / WEIGHTS_PER_WORD;
                    const uint32_t words_this_block = BLOCK_FLOATS / WEIGHTS_PER_WORD;
                    const uint32_t group_idx = col / group_size;
                    const uint32_t block_idx = col / BLOCK_FLOATS;
                    const uint32_t blocks_per_batch = input_dim / BLOCK_FLOATS;
                    const uint32_t runtime_lut_offset0 =
                        ((((batch_idx * output_dim) + row) * blocks_per_batch) + block_idx)
                        * (1u << BITS_PER_WEIGHT);
                    const uint32_t runtime_lut_offset1 =
                        (((((batch_idx * output_dim) + (row + 1)) * blocks_per_batch) + block_idx)
                        * (1u << BITS_PER_WEIGHT));
                    const uint32_t lut_row_offset0 =
                        ((row * num_groups) + group_idx) * (1u << BITS_PER_WEIGHT);
                    const uint32_t lut_row_offset1 =
                        (((row + 1) * num_groups) + group_idx) * (1u << BITS_PER_WEIGHT);
                    int32_t block_acc0 = 0;
                    int32_t block_acc1 = 0;

                    mram_read(
                        (__mram_ptr void const *)(inputs_i8_mram + ((batch_idx * input_dim) + col)),
                        input_i8_cache,
                        BLOCK_FLOATS * sizeof(int8_t));
                    mram_read(
                        (__mram_ptr void const *)(runtime_lut_mram + runtime_lut_offset0),
                        lut0_i16,
                        sizeof(lut0_i16));
                    mram_read(
                        (__mram_ptr void const *)(runtime_lut_mram + runtime_lut_offset1),
                        lut1_i16,
                        sizeof(lut1_i16));
                    mram_read(
                        (__mram_ptr void const *)(qweight_mram + ((row * words_per_row) + word_offset)),
                        qweight0_cache,
                        words_this_block * sizeof(uint32_t));
                    mram_read(
                        (__mram_ptr void const *)(qweight_mram + (((row + 1) * words_per_row) + word_offset)),
                        qweight1_cache,
                        words_this_block * sizeof(uint32_t));

                    for (uint32_t word_idx = 0; word_idx < words_this_block; ++word_idx) {
                        const uint32_t packed0 = qweight0_cache[word_idx];
                        const uint32_t packed1 = qweight1_cache[word_idx];
                        for (uint32_t nibble = 0; nibble < WEIGHTS_PER_WORD; ++nibble) {
                            const uint32_t shift = nibble * BITS_PER_WEIGHT;
                            const uint32_t q0 = (packed0 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                            const uint32_t q1 = (packed1 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                            const uint32_t input_idx = (word_idx * WEIGHTS_PER_WORD) + nibble;
                            const int32_t x = (int32_t)input_i8_cache[input_idx];
                            block_acc0 += x * (int32_t)lut0_i16[q0];
                            block_acc1 += x * (int32_t)lut1_i16[q1];
                        }
                    }

                    acc0 += block_acc0;
                    acc1 += block_acc1;
                }

                output_i32_cache[0] = acc0;
                output_i32_cache[1] = acc1;
                mram_write(
                    output_i32_cache,
                    (__mram_ptr void *)(outputs_i32_mram + ((batch_idx * output_dim) + row)),
                    2 * sizeof(int32_t));
                continue;
            }

            if (kernel_mode == 4) {
                const bool use_batch_tile = (FIXED_BATCH_TILE > 1) && (batch_size > 1) && (input_dim >= 1024);
                if (use_batch_tile) {
                    const uint32_t tile_count =
                        ((batch_size - batch_idx) < FIXED_BATCH_TILE) ? (batch_size - batch_idx) : FIXED_BATCH_TILE;
                    int32_t acc0[FIXED_BATCH_TILE] = {0};
                    int32_t acc1[FIXED_BATCH_TILE] = {0};

                    for (uint32_t col = 0; col < input_dim; col += BLOCK_FLOATS) {
                        const uint32_t word_offset = col / WEIGHTS_PER_WORD;
                        const uint32_t words_this_block = BLOCK_FLOATS / WEIGHTS_PER_WORD;
                        const uint32_t group_idx = col / group_size;
                        const uint32_t lut_row_offset0 =
                            ((row * num_groups) + group_idx) * (1u << BITS_PER_WEIGHT);
                        const uint32_t lut_row_offset1 =
                            (((row + 1) * num_groups) + group_idx) * (1u << BITS_PER_WEIGHT);

                        mram_read(
                            (__mram_ptr void const *)(lut_mram + lut_row_offset0),
                            lut0_i16,
                            sizeof(lut0_i16));
                        mram_read(
                            (__mram_ptr void const *)(lut_mram + lut_row_offset1),
                            lut1_i16,
                            sizeof(lut1_i16));
                        mram_read(
                            (__mram_ptr void const *)(qweight_mram + ((row * words_per_row) + word_offset)),
                            qweight0_cache,
                            words_this_block * sizeof(uint32_t));
                        mram_read(
                            (__mram_ptr void const *)(qweight_mram + (((row + 1) * words_per_row) + word_offset)),
                            qweight1_cache,
                            words_this_block * sizeof(uint32_t));
                        for (uint32_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                            mram_read(
                                (__mram_ptr void const *)(inputs_i8_mram + (((batch_idx + tile_idx) * input_dim) + col)),
                                input_i8_tile_cache[tile_idx],
                                BLOCK_FLOATS * sizeof(int8_t));
                        }

                        for (uint32_t word_idx = 0; word_idx < words_this_block; ++word_idx) {
                            const uint32_t packed0 = qweight0_cache[word_idx];
                            const uint32_t packed1 = qweight1_cache[word_idx];
                            for (uint32_t nibble = 0; nibble < WEIGHTS_PER_WORD; ++nibble) {
                                const uint32_t shift = nibble * BITS_PER_WEIGHT;
                                const uint32_t q0 = (packed0 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                                const uint32_t q1 = (packed1 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                                const uint32_t input_idx = (word_idx * WEIGHTS_PER_WORD) + nibble;
                                const int32_t w0 = (int32_t)lut0_i16[q0];
                                const int32_t w1 = (int32_t)lut1_i16[q1];
                                for (uint32_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                                    const int32_t x = (int32_t)input_i8_tile_cache[tile_idx][input_idx];
                                    acc0[tile_idx] += x * w0;
                                    acc1[tile_idx] += x * w1;
                                }
                            }
                        }
                    }

                    for (uint32_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                        output_i32_cache[0] = acc0[tile_idx];
                        output_i32_cache[1] = acc1[tile_idx];
                        mram_write(
                            output_i32_cache,
                            (__mram_ptr void *)(outputs_i32_mram + (((batch_idx + tile_idx) * output_dim) + row)),
                            2 * sizeof(int32_t));
                    }
                    batch_idx += tile_count - 1;
                } else {
                    int32_t acc0 = 0;
                    int32_t acc1 = 0;

                    for (uint32_t col = 0; col < input_dim; col += BLOCK_FLOATS) {
                        const uint32_t word_offset = col / WEIGHTS_PER_WORD;
                        const uint32_t words_this_block = BLOCK_FLOATS / WEIGHTS_PER_WORD;
                        const uint32_t group_idx = col / group_size;
                        const uint32_t lut_row_offset0 =
                            ((row * num_groups) + group_idx) * (1u << BITS_PER_WEIGHT);
                        const uint32_t lut_row_offset1 =
                            (((row + 1) * num_groups) + group_idx) * (1u << BITS_PER_WEIGHT);

                        mram_read(
                            (__mram_ptr void const *)(inputs_i8_mram + ((batch_idx * input_dim) + col)),
                            input_i8_cache,
                            BLOCK_FLOATS * sizeof(int8_t));
                        mram_read(
                            (__mram_ptr void const *)(lut_mram + lut_row_offset0),
                            lut0_i16,
                            sizeof(lut0_i16));
                        mram_read(
                            (__mram_ptr void const *)(lut_mram + lut_row_offset1),
                            lut1_i16,
                            sizeof(lut1_i16));
                        mram_read(
                            (__mram_ptr void const *)(qweight_mram + ((row * words_per_row) + word_offset)),
                            qweight0_cache,
                            words_this_block * sizeof(uint32_t));
                        mram_read(
                            (__mram_ptr void const *)(qweight_mram + (((row + 1) * words_per_row) + word_offset)),
                            qweight1_cache,
                            words_this_block * sizeof(uint32_t));

                        for (uint32_t word_idx = 0; word_idx < words_this_block; ++word_idx) {
                            const uint32_t packed0 = qweight0_cache[word_idx];
                            const uint32_t packed1 = qweight1_cache[word_idx];
                            for (uint32_t nibble = 0; nibble < WEIGHTS_PER_WORD; ++nibble) {
                                const uint32_t shift = nibble * BITS_PER_WEIGHT;
                                const uint32_t q0 = (packed0 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                                const uint32_t q1 = (packed1 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                                const uint32_t input_idx = (word_idx * WEIGHTS_PER_WORD) + nibble;
                                const int32_t x = (int32_t)input_i8_cache[input_idx];
                                acc0 += x * (int32_t)lut0_i16[q0];
                                acc1 += x * (int32_t)lut1_i16[q1];
                            }
                        }
                    }

                    output_i32_cache[0] = acc0;
                    output_i32_cache[1] = acc1;
                    mram_write(
                        output_i32_cache,
                        (__mram_ptr void *)(outputs_i32_mram + ((batch_idx * output_dim) + row)),
                        2 * sizeof(int32_t));
                }
                continue;
            }

            if (kernel_mode != 0) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;

                if (kernel_mode == 1) {
                    output_cache[0] = 0.0f;
                    output_cache[1] = 0.0f;
                    mram_write(
                        output_cache,
                        (__mram_ptr void *)(outputs_mram + ((batch_idx * output_dim) + row)),
                        2 * sizeof(float));
                    continue;
                }

                mram_read(
                    (__mram_ptr void const *)(scales_mram + (row * num_groups)),
                    scales0_cache,
                    num_groups * sizeof(float));
                mram_read(
                    (__mram_ptr void const *)(scales_mram + ((row + 1) * num_groups)),
                    scales1_cache,
                    num_groups * sizeof(float));

                for (uint32_t col = 0; col < input_dim; col += BLOCK_FLOATS) {
                    const uint32_t word_offset = col / WEIGHTS_PER_WORD;
                    const uint32_t words_this_block = BLOCK_FLOATS / WEIGHTS_PER_WORD;
                    const uint32_t group_idx = col / group_size;
                    const float scale0 = scales0_cache[group_idx];
                    const float scale1 = scales1_cache[group_idx];

                    for (uint32_t q = 0; q < (1u << BITS_PER_WEIGHT); ++q) {
                        lut0[q] = ((float)q - zero_point) * scale0;
                        lut1[q] = ((float)q - zero_point) * scale1;
                    }

                    mram_read(
                        (__mram_ptr void const *)(inputs_mram + ((batch_idx * input_dim) + col)),
                        input_cache,
                        BLOCK_FLOATS * sizeof(float));
                    mram_read(
                        (__mram_ptr void const *)(qweight_mram + ((row * words_per_row) + word_offset)),
                        qweight0_cache,
                        words_this_block * sizeof(uint32_t));
                    mram_read(
                        (__mram_ptr void const *)(qweight_mram + (((row + 1) * words_per_row) + word_offset)),
                        qweight1_cache,
                        words_this_block * sizeof(uint32_t));

                    for (uint32_t word_idx = 0; word_idx < words_this_block; ++word_idx) {
                        const uint32_t packed0 = qweight0_cache[word_idx];
                        const uint32_t packed1 = qweight1_cache[word_idx];
                        for (uint32_t nibble = 0; nibble < WEIGHTS_PER_WORD; ++nibble) {
                            const uint32_t shift = nibble * BITS_PER_WEIGHT;
                            const uint32_t q0 = (packed0 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                            const uint32_t q1 = (packed1 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                            const uint32_t input_idx = (word_idx * WEIGHTS_PER_WORD) + nibble;
                            const float x = input_cache[input_idx];

                            if (kernel_mode == 2) {
                                acc0 += (float)q0;
                                acc1 += (float)q1;
                            } else {
                                const float dq0 = lut0[q0];
                                const float dq1 = lut1[q1];
                                if (kernel_mode == 3) {
                                    acc0 += dq0;
                                    acc1 += dq1;
                                } else {
                                    acc0 += x * dq0;
                                    acc1 += x * dq1;
                                }
                            }
                        }
                    }
                }

                output_cache[0] = acc0;
                output_cache[1] = acc1;
                mram_write(
                    output_cache,
                    (__mram_ptr void *)(outputs_mram + ((batch_idx * output_dim) + row)),
                    2 * sizeof(float));
                continue;
            }

            float acc0 = 0.0f;
            float acc1 = 0.0f;

            mram_read(
                (__mram_ptr void const *)(scales_mram + (row * num_groups)),
                scales0_cache,
                num_groups * sizeof(float));
            mram_read(
                (__mram_ptr void const *)(scales_mram + ((row + 1) * num_groups)),
                scales1_cache,
                num_groups * sizeof(float));

            for (uint32_t col = 0; col < input_dim; col += BLOCK_FLOATS) {
                const uint32_t word_offset = col / WEIGHTS_PER_WORD;
                const uint32_t words_this_block = BLOCK_FLOATS / WEIGHTS_PER_WORD;
                const uint32_t group_idx = col / group_size;
                const float scale0 = scales0_cache[group_idx];
                const float scale1 = scales1_cache[group_idx];

                for (uint32_t q = 0; q < (1u << BITS_PER_WEIGHT); ++q) {
                    lut0[q] = ((float)q - zero_point) * scale0;
                    lut1[q] = ((float)q - zero_point) * scale1;
                }

                mram_read(
                    (__mram_ptr void const *)(inputs_mram + ((batch_idx * input_dim) + col)),
                    input_cache,
                    BLOCK_FLOATS * sizeof(float));
                mram_read(
                    (__mram_ptr void const *)(qweight_mram + ((row * words_per_row) + word_offset)),
                    qweight0_cache,
                    words_this_block * sizeof(uint32_t));
                mram_read(
                    (__mram_ptr void const *)(qweight_mram + (((row + 1) * words_per_row) + word_offset)),
                    qweight1_cache,
                    words_this_block * sizeof(uint32_t));

                for (uint32_t word_idx = 0; word_idx < words_this_block; ++word_idx) {
                    const uint32_t packed0 = qweight0_cache[word_idx];
                    const uint32_t packed1 = qweight1_cache[word_idx];
                    for (uint32_t nibble = 0; nibble < WEIGHTS_PER_WORD; ++nibble) {
                        const uint32_t shift = nibble * BITS_PER_WEIGHT;
                        const uint32_t q0 = (packed0 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                        const uint32_t q1 = (packed1 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                        const uint32_t input_idx = (word_idx * WEIGHTS_PER_WORD) + nibble;
                        const float x = input_cache[input_idx];
                        acc0 += x * lut0[q0];
                        acc1 += x * lut1[q1];
                    }
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
