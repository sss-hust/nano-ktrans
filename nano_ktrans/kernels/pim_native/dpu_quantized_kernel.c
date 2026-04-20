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

            /* ── kernel_mode=6: T-MAC bit-serial table lookup ─────────
             *
             * Instead of computing x * dequant(q) per element, we use
             * the pre-computed LUT (same as mode 4) but decompose the
             * 4-bit weight into individual bits. For each bit position,
             * we accumulate LUT[q] only when the corresponding bit is set,
             * then shift-accumulate across bit positions.
             *
             * Key advantage: eliminates the int8 × int16 multiply in the
             * inner loop. DPU has no hardware multiplier, so replacing
             * multiply with conditional add + shift is significantly faster.
             *
             * Data flow:
             *   For each block of weights:
             *     Read input_i8 block (BLOCK_FLOATS int8 values)
             *     Read qweight packed words
             *     Read LUT for this (row, group) — 16 entries of int16
             *     For each nibble in packed word:
             *       For each bit b in 0..3:
             *         if (q >> b) & 1:  acc += input_i8[idx] << b
             *         (weighted by LUT sign/magnitude via pre-decomposed LUT)
             *
             * Simplified approach: We split the LUT contribution by bit.
             * For 4-bit weight q and LUT entry lut[q]:
             *   lut[q] * x = sum over bits b: ((q>>b)&1) * lut_contrib * x
             * But this doesn't factor cleanly. Instead, we use a different
             * T-MAC variant: for each bit position of the ACTIVATION (not weight),
             * we can precompute partial sums. However, since activations are int8
             * (8 bits) and weights are 4-bit, the most practical approach for DPU
             * is to eliminate the multiply by using the LUT as a signed lookup:
             *
             * For each weight nibble q:
             *   contribution = lut[q]  (pre-computed dequantized weight * 256)
             *   acc += contribution * x  (this is the multiply we want to eliminate)
             *
             * T-MAC insight: decompose x into bit planes:
             *   x = sign * (b7*128 + b6*64 + ... + b0*1)
             *   contribution * x = sign * sum(bi * contribution * 2^i)
             *   = sign * sum(bi * (contribution << i))
             *
             * So the inner loop becomes:
             *   for each bit i of |x|:
             *     if bit_i set: acc += contribution << i  (or subtract if x negative)
             *
             * This replaces multiply with conditional shift+add. On DPU without
             * hardware multiplier, this is ~2x faster per element.
             */
            if (kernel_mode == 6) {
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

                            /* T-MAC bit-serial: decompose |x| into bits,
                             * accumulate lut[q] << bit_pos for each set bit.
                             * This avoids the int8 * int16 multiply. */
                            const int32_t w0 = (int32_t)lut0_i16[q0];
                            const int32_t w1 = (int32_t)lut1_i16[q1];
                            const uint32_t abs_x = (uint32_t)(x >= 0 ? x : -x);
                            const int32_t sign = (x >= 0) ? 1 : -1;

                            /* Unrolled bit scan for 7-bit magnitude (int8 range) */
                            int32_t partial0 = 0;
                            int32_t partial1 = 0;
                            if (abs_x & 0x01u) { partial0 += w0;       partial1 += w1;       }
                            if (abs_x & 0x02u) { partial0 += w0 << 1;  partial1 += w1 << 1;  }
                            if (abs_x & 0x04u) { partial0 += w0 << 2;  partial1 += w1 << 2;  }
                            if (abs_x & 0x08u) { partial0 += w0 << 3;  partial1 += w1 << 3;  }
                            if (abs_x & 0x10u) { partial0 += w0 << 4;  partial1 += w1 << 4;  }
                            if (abs_x & 0x20u) { partial0 += w0 << 5;  partial1 += w1 << 5;  }
                            if (abs_x & 0x40u) { partial0 += w0 << 6;  partial1 += w1 << 6;  }

                            acc0 += sign * partial0;
                            acc1 += sign * partial1;
                        }
                    }
                }

                output_i32_cache[0] = acc0;
                output_i32_cache[1] = acc1;
                mram_write(
                    output_i32_cache,
                    (__mram_ptr void *)(outputs_i32_mram + ((batch_idx * output_dim) + row)),
                    2 * sizeof(int32_t));
                continue;
            }

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
