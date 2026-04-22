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

/* ADR-002 M-6.1: Multi-slot MRAM residency.
 *
 * Rather than a single qweight/scales/lut buffer per DPU, we carve the
 * same total MRAM budget into NUM_SLOTS equally-sized slots.  The host
 * tells us which slot holds the weights for the current call via
 * ``active_slot`` and can keep up to NUM_SLOTS expert bundles resident
 * simultaneously.  M-4 and M-5 showed preload miss cost ~= 0.96 ms/call
 * pure DPU DMA; multi-slot residency lets a decode step hit MRAM when
 * its active expert was already loaded by an earlier step.
 *
 * All slots share the same (input_dim, output_dim, group_size, kernel_mode)
 * — a single-layer MoE backend always works with one bundle shape, so
 * the host-side pair (PIMMoEBackend.quantized_runtime for gate_up,
 * quantized_runtime_down for down) keeps that invariant trivially.
 *
 * The per-slot capacity is MAX_QWEIGHT_WORDS / NUM_SLOTS words etc, so
 * total MRAM footprint is unchanged relative to M-5. */
#ifndef NUM_SLOTS
#define NUM_SLOTS 8
#endif

#define WORDS_PER_SLOT     (MAX_QWEIGHT_WORDS / NUM_SLOTS)
#define SCALES_PER_SLOT    (MAX_SCALE_FLOATS  / NUM_SLOTS)
#define LUT_INT16_PER_SLOT (MAX_LUT_INT16     / NUM_SLOTS)

/* kernel_mode=7 (genuine T-MAC bit-serial): we pre-decompose activations
 * into bit-plane bitmasks on the host.  For each batch/block we store
 * a sign bitmask plus 7 magnitude bit-plane bitmasks, each packing 64
 * int8 activations into a single uint64_t.  The DPU inner loop becomes
 * a pure "lookup + conditional add" (no multiply, no per-element shift
 * loop).  Max capacity = MAX_INPUT_INT8 bits per plane / 64, times 8
 * planes (7 magnitude + 1 sign). */
#ifndef MAX_INPUT_BITPLANES_U64
#define MAX_INPUT_BITPLANES_U64 ((MAX_INPUT_INT8 / 64) * 8)
#endif

BARRIER_INIT(work_barrier, NR_TASKLETS);

__host uint32_t batch_size;
__host uint32_t input_dim;
__host uint32_t output_dim;
__host uint32_t group_size;
__host uint32_t num_groups;
__host uint32_t kernel_mode;
__host uint32_t active_slot;   /* ADR-002 M-6.1: which slot to compute from */
__host uint64_t kernel_cycles;

__mram_noinit uint32_t qweight_mram[MAX_QWEIGHT_WORDS];
__mram_noinit float scales_mram[MAX_SCALE_FLOATS];
__mram_noinit float inputs_mram[MAX_INPUT_FLOATS];
__mram_noinit float outputs_mram[MAX_OUTPUT_FLOATS];
__mram_noinit int16_t lut_mram[MAX_LUT_INT16];
__mram_noinit int8_t inputs_i8_mram[MAX_INPUT_INT8];
__mram_noinit int32_t outputs_i32_mram[MAX_OUTPUT_INT32];
__mram_noinit int16_t runtime_lut_mram[MAX_RUNTIME_LUT_INT16];
/* Bit-plane layout for kernel_mode=7:
 *   [batch][block][plane] where plane in [0..7]:
 *     plane 0..6 = magnitude bit 0..6 bitmasks (64 bits per block)
 *     plane 7   = sign bitmask (bit set = activation negative)
 * Stored contiguously per batch, block-major, plane-minor.
 */
__mram_noinit uint64_t inputs_bitplanes_mram[MAX_INPUT_BITPLANES_U64];

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

    /* kernel_mode=7 scratchpads live in WRAM heap (not stack) to avoid
     * blowing STACK_SIZE_DEFAULT=2048.  We only need them for mode=7 but
     * allocate lazily per tasklet to keep startup cost off other paths. */
    int32_t *w0_block_heap = NULL;
    int32_t *w1_block_heap = NULL;
    uint64_t *bp_cache_heap = NULL;
    if (kernel_mode == 7) {
        w0_block_heap = (int32_t *)mem_alloc(BLOCK_FLOATS * sizeof(int32_t));
        w1_block_heap = (int32_t *)mem_alloc(BLOCK_FLOATS * sizeof(int32_t));
        /* bp_cache needs 8-byte alignment for the uint64_t DMA transfer. */
        bp_cache_heap = (uint64_t *)mem_alloc(8 * sizeof(uint64_t));
    }

    const uint32_t words_per_row = input_dim / WEIGHTS_PER_WORD;
    const float zero_point = (float)(1 << (BITS_PER_WEIGHT - 1));

    /* M-6.1: compute slot-scoped base offsets once.  The kernel
     * addresses qweight/scales/lut as if they were per-slot arrays;
     * host bridge is responsible for writing into the matching slot. */
    const uint32_t qw_slot_base    = active_slot * WORDS_PER_SLOT;
    const uint32_t scale_slot_base = active_slot * SCALES_PER_SLOT;
    const uint32_t lut_slot_base   = active_slot * LUT_INT16_PER_SLOT;

    for (uint32_t row = tasklet_id * 2; row < output_dim; row += NR_TASKLETS * 2) {
        for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {

            /* ── kernel_mode=7: GENUINE T-MAC bit-serial (no software multiply,
             *    no per-element shift loop).
             *
             *  Host has pre-decomposed activations into 8 bit-plane bitmasks
             *  per BLOCK_FLOATS=64-wide block:
             *    plane b (b=0..6) holds the b-th magnitude bit of each |x_i|
             *    plane 7 holds the sign bit of each x_i
             *  Each plane is 64 bits = exactly one uint64_t per block.
             *
             *  Inner loop per block:
             *    For each of the 64 activations we compute:
             *      x_i * lut[q_i]
             *    = sign_i * (sum over b of bit_b(|x_i|) * 2^b) * lut[q_i]
             *    = sign_i * sum_b 2^b * (bit_b(|x_i|) * lut[q_i])
             *  So if we maintain per-bit-plane accumulators
             *      S_b = sum_{i : bit_b(|x_i|)=1} sign_i * lut[q_i]
             *  the final answer is simply
             *      acc = sum_b S_b << b.
             *  No software multiply anywhere in the hot path. */
            if (kernel_mode == 7) {
                int32_t S[7][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0},
                                   {0, 0}, {0, 0}, {0, 0}};
                const uint32_t blocks_per_batch = input_dim / BLOCK_FLOATS;

                for (uint32_t col = 0; col < input_dim; col += BLOCK_FLOATS) {
                    const uint32_t word_offset = col / WEIGHTS_PER_WORD;
                    const uint32_t words_this_block = BLOCK_FLOATS / WEIGHTS_PER_WORD;
                    const uint32_t group_idx = col / group_size;
                    const uint32_t block_idx = col / BLOCK_FLOATS;
                    const uint32_t lut_row_offset0 =
                        ((row * num_groups) + group_idx) * (1u << BITS_PER_WEIGHT);
                    const uint32_t lut_row_offset1 =
                        (((row + 1) * num_groups) + group_idx) * (1u << BITS_PER_WEIGHT);
                    /* 8 planes per block, stored plane-minor.  Load all
                     * 8 * 8 = 64 bytes for this batch/block in one DMA. */
                    const uint32_t bp_offset =
                        (((batch_idx * blocks_per_batch) + block_idx) * 8u);

                    mram_read(
                        (__mram_ptr void const *)(inputs_bitplanes_mram + bp_offset),
                        bp_cache_heap,
                        8 * sizeof(uint64_t));
                    mram_read(
                        (__mram_ptr void const *)(lut_mram + (lut_slot_base + lut_row_offset0)),
                        lut0_i16,
                        sizeof(lut0_i16));
                    mram_read(
                        (__mram_ptr void const *)(lut_mram + (lut_slot_base + lut_row_offset1)),
                        lut1_i16,
                        sizeof(lut1_i16));
                    mram_read(
                        (__mram_ptr void const *)(qweight_mram + (qw_slot_base + (row * words_per_row) + word_offset)),
                        qweight0_cache,
                        words_this_block * sizeof(uint32_t));
                    mram_read(
                        (__mram_ptr void const *)(qweight_mram + (qw_slot_base + ((row + 1) * words_per_row) + word_offset)),
                        qweight1_cache,
                        words_this_block * sizeof(uint32_t));

                    const uint64_t sign_mask = bp_cache_heap[7];

                    /* Unpack nibbles into a linear per-block weight table.
                     * Doing this once per block (rather than per bit-plane)
                     * amortises the shift/mask work across 7 plane passes. */
                    for (uint32_t word_idx = 0; word_idx < words_this_block; ++word_idx) {
                        const uint32_t packed0 = qweight0_cache[word_idx];
                        const uint32_t packed1 = qweight1_cache[word_idx];
                        for (uint32_t nibble = 0; nibble < WEIGHTS_PER_WORD; ++nibble) {
                            const uint32_t shift = nibble * BITS_PER_WEIGHT;
                            const uint32_t q0 = (packed0 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                            const uint32_t q1 = (packed1 >> shift) & ((1u << BITS_PER_WEIGHT) - 1u);
                            const uint32_t input_idx = (word_idx * WEIGHTS_PER_WORD) + nibble;
                            /* Flip the weight's sign now if the activation
                             * is negative; this lets every bit plane use
                             * a single add (no per-element branch on sign). */
                            const int32_t s = ((sign_mask >> input_idx) & 1u) ? -1 : 1;
                            w0_block_heap[input_idx] = s * (int32_t)lut0_i16[q0];
                            w1_block_heap[input_idx] = s * (int32_t)lut1_i16[q1];
                        }
                    }

                    /* T-MAC on DPU: sparse bit-scan accumulation.
                     *
                     * Mode 4 (int8 * int16 software multiply) uses a
                     * single highly-tuned mult per element; on UPMEM DPU
                     * that is ~8-10 cycles.  A bit-serial decomposition
                     * replaces the multiply with conditional adds.
                     *
                     * Empirically (see sweeps in benchmarks/results/),
                     * mode=4 beats mode=7 across every shape/batch/rank
                     * we have measured.  This kernel is retained because
                     *   a) It is a genuine T-MAC (no multiplies in the
                     *      inner loop, unlike mode=6 which pays a 7x
                     *      conditional branch ladder without LUT reuse);
                     *   b) ADR-002 requires we ship the negative result
                     *      as honestly as any positive one.
                     *
                     * We walk only the set bits of each magnitude plane
                     * (via __builtin_ctz on 32-bit halves), which on
                     * gaussian activations visits ~25-50%% of bits.
                     * Dense scan was benchmarked (in mode=7 sweep v2,
                     * 2026-04-22) and found 30-50%% slower than this
                     * sparse variant on Qwen3 GPTQ activations. */
                    for (uint32_t bp = 0; bp < 7; ++bp) {
                        const uint64_t mask64 = bp_cache_heap[bp];
                        const uint32_t mask_lo = (uint32_t)(mask64);
                        const uint32_t mask_hi = (uint32_t)(mask64 >> 32);
                        int32_t acc0_b = 0;
                        int32_t acc1_b = 0;

                        uint32_t m = mask_lo;
                        uint32_t base = 0;
                        while (m != 0) {
                            const uint32_t bit = m & (-m);              /* isolate LSB */
                            const uint32_t i = base + (uint32_t)__builtin_ctz(bit);
                            m &= m - 1u;
                            acc0_b += w0_block_heap[i];
                            acc1_b += w1_block_heap[i];
                        }
                        m = mask_hi;
                        base = 32;
                        while (m != 0) {
                            const uint32_t bit = m & (-m);
                            const uint32_t i = base + (uint32_t)__builtin_ctz(bit);
                            m &= m - 1u;
                            acc0_b += w0_block_heap[i];
                            acc1_b += w1_block_heap[i];
                        }
                        S[bp][0] += acc0_b;
                        S[bp][1] += acc1_b;
                    }
                }

                int32_t acc0 = 0;
                int32_t acc1 = 0;
                for (uint32_t bp = 0; bp < 7; ++bp) {
                    acc0 += S[bp][0] << bp;
                    acc1 += S[bp][1] << bp;
                }

                output_i32_cache[0] = acc0;
                output_i32_cache[1] = acc1;
                mram_write(
                    output_i32_cache,
                    (__mram_ptr void *)(outputs_i32_mram + ((batch_idx * output_dim) + row)),
                    2 * sizeof(int32_t));
                continue;
            }

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
                        (__mram_ptr void const *)(lut_mram + (lut_slot_base + lut_row_offset0)),
                        lut0_i16,
                        sizeof(lut0_i16));
                    mram_read(
                        (__mram_ptr void const *)(lut_mram + (lut_slot_base + lut_row_offset1)),
                        lut1_i16,
                        sizeof(lut1_i16));
                    mram_read(
                        (__mram_ptr void const *)(qweight_mram + (qw_slot_base + (row * words_per_row) + word_offset)),
                        qweight0_cache,
                        words_this_block * sizeof(uint32_t));
                    mram_read(
                        (__mram_ptr void const *)(qweight_mram + (qw_slot_base + ((row + 1) * words_per_row) + word_offset)),
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
                        (__mram_ptr void const *)(qweight_mram + (qw_slot_base + (row * words_per_row) + word_offset)),
                        qweight0_cache,
                        words_this_block * sizeof(uint32_t));
                    mram_read(
                        (__mram_ptr void const *)(qweight_mram + (qw_slot_base + ((row + 1) * words_per_row) + word_offset)),
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
                            (__mram_ptr void const *)(lut_mram + (lut_slot_base + lut_row_offset0)),
                            lut0_i16,
                            sizeof(lut0_i16));
                        mram_read(
                            (__mram_ptr void const *)(lut_mram + (lut_slot_base + lut_row_offset1)),
                            lut1_i16,
                            sizeof(lut1_i16));
                        mram_read(
                            (__mram_ptr void const *)(qweight_mram + (qw_slot_base + (row * words_per_row) + word_offset)),
                            qweight0_cache,
                            words_this_block * sizeof(uint32_t));
                        mram_read(
                            (__mram_ptr void const *)(qweight_mram + (qw_slot_base + ((row + 1) * words_per_row) + word_offset)),
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
                            (__mram_ptr void const *)(lut_mram + (lut_slot_base + lut_row_offset0)),
                            lut0_i16,
                            sizeof(lut0_i16));
                        mram_read(
                            (__mram_ptr void const *)(lut_mram + (lut_slot_base + lut_row_offset1)),
                            lut1_i16,
                            sizeof(lut1_i16));
                        mram_read(
                            (__mram_ptr void const *)(qweight_mram + (qw_slot_base + (row * words_per_row) + word_offset)),
                            qweight0_cache,
                            words_this_block * sizeof(uint32_t));
                        mram_read(
                            (__mram_ptr void const *)(qweight_mram + (qw_slot_base + ((row + 1) * words_per_row) + word_offset)),
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
                    (__mram_ptr void const *)(scales_mram + (scale_slot_base + row * num_groups)),
                    scales0_cache,
                    num_groups * sizeof(float));
                mram_read(
                    (__mram_ptr void const *)(scales_mram + (scale_slot_base + (row + 1) * num_groups)),
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
                        (__mram_ptr void const *)(qweight_mram + (qw_slot_base + (row * words_per_row) + word_offset)),
                        qweight0_cache,
                        words_this_block * sizeof(uint32_t));
                    mram_read(
                        (__mram_ptr void const *)(qweight_mram + (qw_slot_base + ((row + 1) * words_per_row) + word_offset)),
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
                (__mram_ptr void const *)(scales_mram + (scale_slot_base + row * num_groups)),
                scales0_cache,
                num_groups * sizeof(float));
            mram_read(
                (__mram_ptr void const *)(scales_mram + (scale_slot_base + (row + 1) * num_groups)),
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
                    (__mram_ptr void const *)(qweight_mram + (qw_slot_base + (row * words_per_row) + word_offset)),
                    qweight0_cache,
                    words_this_block * sizeof(uint32_t));
                mram_read(
                    (__mram_ptr void const *)(qweight_mram + (qw_slot_base + ((row + 1) * words_per_row) + word_offset)),
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
