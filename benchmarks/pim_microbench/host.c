#define _POSIX_C_SOURCE 200809L

#include <dpu/dpu.h>

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DPU_BINARY_PATH
#define DPU_BINARY_PATH "benchmarks/pim_microbench/build/pim_kernel"
#endif

struct options {
    uint32_t ranks;
    uint32_t bytes_per_dpu;
    uint32_t repetitions;
    uint32_t scalar;
    bool verify;
    const char *profile;
};

static double
now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint32_t
parse_u32(const char *value, const char *name)
{
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX) {
        fprintf(stderr, "Invalid %s: %s\n", name, value);
        exit(EXIT_FAILURE);
    }
    return (uint32_t)parsed;
}

static void
parse_args(int argc, char **argv, struct options *opts)
{
    opts->ranks = 1;
    opts->bytes_per_dpu = 8u * 1024u * 1024u;
    opts->repetitions = 4;
    opts->scalar = 7;
    opts->verify = true;
    opts->profile = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--ranks") == 0 && i + 1 < argc) {
            opts->ranks = parse_u32(argv[++i], "ranks");
        } else if (strcmp(argv[i], "--bytes-per-dpu") == 0 && i + 1 < argc) {
            opts->bytes_per_dpu = parse_u32(argv[++i], "bytes-per-dpu");
        } else if (strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc) {
            opts->repetitions = parse_u32(argv[++i], "repetitions");
        } else if (strcmp(argv[i], "--scalar") == 0 && i + 1 < argc) {
            opts->scalar = parse_u32(argv[++i], "scalar");
        } else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            opts->profile = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opts->verify = false;
        } else {
            fprintf(stderr,
                "Usage: %s [--ranks N] [--bytes-per-dpu BYTES] [--repetitions N] [--scalar N] [--profile STR] [--no-verify]\n",
                argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (opts->bytes_per_dpu == 0 || (opts->bytes_per_dpu % 1024u) != 0) {
        fprintf(stderr, "--bytes-per-dpu must be a non-zero multiple of 1024.\n");
        exit(EXIT_FAILURE);
    }
}

static void
fill_input(uint32_t *buffer, size_t words_per_dpu, size_t dpu_index)
{
    for (size_t i = 0; i < words_per_dpu; ++i) {
        buffer[i] = (uint32_t)(i + 17u * (uint32_t)dpu_index);
    }
}

static bool
verify_output(const uint32_t *input, const uint32_t *output, size_t words_per_dpu, uint32_t scalar)
{
    size_t sample = words_per_dpu < 4096 ? words_per_dpu : 4096;
    for (size_t i = 0; i < sample; ++i) {
        uint32_t expected = input[i] * scalar + 1u;
        if (output[i] != expected) {
            fprintf(stderr, "Verification failed at word %zu: got %" PRIu32 ", expected %" PRIu32 "\n", i, output[i], expected);
            return false;
        }
    }
    return true;
}

int
main(int argc, char **argv)
{
    struct options opts;
    parse_args(argc, argv, &opts);

    struct dpu_set_t set, dpu;
    DPU_ASSERT(dpu_alloc_ranks(opts.ranks, opts.profile, &set));

    uint32_t nr_dpus = 0;
    uint32_t nr_ranks = 0;
    DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
    DPU_ASSERT(dpu_get_nr_ranks(set, &nr_ranks));
    DPU_ASSERT(dpu_load(set, DPU_BINARY_PATH, NULL));

    const size_t words_per_dpu = opts.bytes_per_dpu / sizeof(uint32_t);
    const size_t total_words = words_per_dpu * (size_t)nr_dpus;
    const size_t total_bytes = (size_t)nr_dpus * (size_t)opts.bytes_per_dpu;

    uint32_t *host_input = aligned_alloc(8, total_bytes);
    uint32_t *host_output = aligned_alloc(8, total_bytes);
    uint64_t *kernel_cycles = calloc(nr_dpus, sizeof(uint64_t));
    if (host_input == NULL || host_output == NULL || kernel_cycles == NULL) {
        fprintf(stderr, "Failed to allocate host buffers.\n");
        return EXIT_FAILURE;
    }

    for (uint32_t dpu_index = 0; dpu_index < nr_dpus; ++dpu_index) {
        fill_input(host_input + ((size_t)dpu_index * words_per_dpu), words_per_dpu, dpu_index);
    }

    DPU_ASSERT(dpu_broadcast_to(set, "bytes_per_dpu", 0, &opts.bytes_per_dpu, sizeof(opts.bytes_per_dpu), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, "repetitions", 0, &opts.repetitions, sizeof(opts.repetitions), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, "scalar", 0, &opts.scalar, sizeof(opts.scalar), DPU_XFER_DEFAULT));

    uint32_t dpu_index = 0;
    DPU_FOREACH(set, dpu, dpu_index)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, host_input + ((size_t)dpu_index * words_per_dpu)));
    }

    double upload_start = now_seconds();
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "input_mram", 0, opts.bytes_per_dpu, DPU_XFER_DEFAULT));
    double upload_seconds = now_seconds() - upload_start;

    double kernel_start = now_seconds();
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    double kernel_seconds = now_seconds() - kernel_start;

    dpu_index = 0;
    DPU_FOREACH(set, dpu, dpu_index)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, kernel_cycles + dpu_index));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "kernel_cycles", 0, sizeof(uint64_t), DPU_XFER_DEFAULT));

    dpu_index = 0;
    DPU_FOREACH(set, dpu, dpu_index)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, host_output + ((size_t)dpu_index * words_per_dpu)));
    }

    double download_start = now_seconds();
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "output_mram", 0, opts.bytes_per_dpu, DPU_XFER_DEFAULT));
    double download_seconds = now_seconds() - download_start;

    if (opts.verify && !verify_output(host_input, host_output, words_per_dpu, opts.scalar)) {
        DPU_ASSERT(dpu_free(set));
        free(kernel_cycles);
        free(host_output);
        free(host_input);
        return EXIT_FAILURE;
    }

    uint64_t min_cycles = UINT64_MAX;
    uint64_t max_cycles = 0;
    long double sum_cycles = 0.0;
    for (uint32_t i = 0; i < nr_dpus; ++i) {
        if (kernel_cycles[i] < min_cycles) {
            min_cycles = kernel_cycles[i];
        }
        if (kernel_cycles[i] > max_cycles) {
            max_cycles = kernel_cycles[i];
        }
        sum_cycles += kernel_cycles[i];
    }
    long double avg_cycles = nr_dpus > 0 ? sum_cycles / nr_dpus : 0.0;

    double end_to_end_seconds = upload_seconds + kernel_seconds + download_seconds;
    double upload_gbps = (double)total_bytes / upload_seconds / 1e9;
    double download_gbps = (double)total_bytes / download_seconds / 1e9;
    double kernel_effective_gbps = (double)(total_bytes * 2ull * opts.repetitions) / kernel_seconds / 1e9;
    double element_updates = (double)total_words * (double)opts.repetitions;
    double int32_ops = element_updates * 2.0; /* one multiply + one add per element */
    double kernel_element_gops = element_updates / kernel_seconds / 1e9;
    double kernel_int32_gops_estimate = int32_ops / kernel_seconds / 1e9;

    printf("{\n");
    printf("  \"status\": \"ok\",\n");
    printf("  \"ranks\": %" PRIu32 ",\n", nr_ranks);
    printf("  \"dpus\": %" PRIu32 ",\n", nr_dpus);
    printf("  \"bytes_per_dpu\": %" PRIu32 ",\n", opts.bytes_per_dpu);
    printf("  \"repetitions\": %" PRIu32 ",\n", opts.repetitions);
    printf("  \"scalar\": %" PRIu32 ",\n", opts.scalar);
    printf("  \"profile\": \"%s\",\n", opts.profile == NULL ? "" : opts.profile);
    printf("  \"host_to_pim_seconds\": %.9f,\n", upload_seconds);
    printf("  \"host_to_pim_gbps\": %.6f,\n", upload_gbps);
    printf("  \"kernel_seconds\": %.9f,\n", kernel_seconds);
    printf("  \"kernel_workload\": \"int32 affine transform: y = x * scalar + 1\",\n");
    printf("  \"kernel_effective_gbps\": %.6f,\n", kernel_effective_gbps);
    printf("  \"kernel_element_gops\": %.6f,\n", kernel_element_gops);
    printf("  \"kernel_int32_gops_estimate\": %.6f,\n", kernel_int32_gops_estimate);
    printf("  \"pim_to_host_seconds\": %.9f,\n", download_seconds);
    printf("  \"pim_to_host_gbps\": %.6f,\n", download_gbps);
    printf("  \"end_to_end_seconds\": %.9f,\n", end_to_end_seconds);
    printf("  \"kernel_cycles_min\": %" PRIu64 ",\n", min_cycles);
    printf("  \"kernel_cycles_avg\": %.3Lf,\n", avg_cycles);
    printf("  \"kernel_cycles_max\": %" PRIu64 "\n", max_cycles);
    printf("}\n");

    DPU_ASSERT(dpu_free(set));
    free(kernel_cycles);
    free(host_output);
    free(host_input);
    return EXIT_SUCCESS;
}
