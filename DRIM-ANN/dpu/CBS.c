/*
Author: Chen Mingkai
Date: 2024/2/27
Function: Entry to the cluster searching phase on DPUs.
*/

#include "cluster.h"
#include <stdio.h>

#define POINT_MEM_SIZE (30 << 20)
#define QUERY_MEM_SIZE (5 << 20)
#define NEIGHBOR_MEM_SIZE (24 << 20)  // >= K * Q * 16B
#define SQUARE_RES_MEM_SIZE (1 << 9)  // For uint16: 64K * 4B = 256KB; for uint8: 256 * 2B = 512B
#define CODEBOOK_MEM_SIZE (1 << 19)
#define LUT_MEM_SIZE (16 << 10)  // |CB| * M * sizeof(VECSUMTYPE)
#define POINT_ADDR_MEM_SIZE (8 << 10)
#define POINT_SIZE_MEM_SIZE (8 << 10)
#define CLUSTER_ID_INDEX_MEM_SIZE (80 << 10)
#define CLUSTER_ID_MEM_SIZE (80 << 10)
#define QUERY_ID_MEM_SIZE (12 << 10)
#define CENTROID_MEM_SIZE (52 << 10)
#define RADIUS_MEM_SIZE POINT_SIZE_MEM_SIZE
#define LARGE_SQUARE_RES_MEM_SIZE (LARGE_SQUARE_RES_SIZE << 2)  // LARGE_SQUARE_RES_SIZE * sizeof(VECSUMTYPE)

// Kept
__host uint16_t squareRes[SQUARE_RES_MEM_SIZE / sizeof(uint16_t)];
__mram_noinit CB_TYPE codebook[CODEBOOK_MEM_SIZE / sizeof(CB_TYPE)];
__host uint32_t dimAmt;
__host uint32_t sliceAmt;
__host uint32_t neighborAmt;
__host uint32_t svDimAmt;  // svDimAmt = ceil(dimAmt / sliceAmt)
// Inputs
__mram_noinit POINTTYPE points[POINT_MEM_SIZE / sizeof(POINTTYPE)];
__mram_noinit ELEMTYPE queries[QUERY_MEM_SIZE / sizeof(ELEMTYPE)];
__host uint32_t queryAmt;
__host ADDRTYPE codebookEntryAmt;
__mram_noinit CENTROIDS_TYPE centroids[CENTROID_MEM_SIZE / sizeof(CENTROIDS_TYPE)];
__host ADDRTYPE localClusterAddrs[POINT_ADDR_MEM_SIZE / sizeof(ADDRTYPE)];
__host CLUSTER_SIZES_TYPE localClusterSizes[POINT_SIZE_MEM_SIZE / sizeof(CLUSTER_SIZES_TYPE)];
__mram_noinit ADDRTYPE localClusterIDIdxs[CLUSTER_ID_INDEX_MEM_SIZE / sizeof(ADDRTYPE) + 1];
__mram_noinit ADDRTYPE localClusterIDs[CLUSTER_ID_MEM_SIZE / sizeof(ADDRTYPE)];
__mram_noinit ADDRTYPE localQueryIDs[QUERY_ID_MEM_SIZE / sizeof(ADDRTYPE)];
__host SQUAREROOTDISTTYPE radii[RADIUS_MEM_SIZE / sizeof(SQUAREROOTDISTTYPE)];
__host VECSUMTYPE largeSquareRes[LARGE_SQUARE_RES_MEM_SIZE / sizeof(VECSUMTYPE)];
// Built at runtime
__host VECSUMTYPE lookUpTable[LUT_MEM_SIZE / sizeof(VECSUMTYPE)];  // Note: store the sum of the corresponding subvectors; may shrink to uint32_t when sizeof(ELEMTYPE) == 1
// Outputs
__mram_noinit pqueue_elem_t_mram neighbors[NEIGHBOR_MEM_SIZE / sizeof(pqueue_elem_t_mram)];
#ifdef PERF_EVAL_SIM
__host perfcounter_t exec_time;
MUTEX_INIT(mutex_exec_time);
#endif
#if (defined MODULE_PERF_EVAL || defined CYCLE_PERF_EVAL)
__mram perfcounter_t exec_times[MODULE_TYPES_END * NR_TASKLETS];
#endif

int main() {
#ifdef PERF_EVAL_SIM
    if (me() == 0) {
        exec_time = 0;
        perfcounter_config(COUNT_CYCLES, true);  // `The main difference between counting cycles and instructions is that cycles include the execution time of instructions AND the memory transfers.`
        // perfcounter_config(COUNT_INSTRUCTIONS, true);
    }
#endif
#if (defined MODULE_PERF_EVAL || defined CYCLE_PERF_EVAL)
    if (me() == 0) {
        perfcounter_config(COUNT_CYCLES, true);  // `The main difference between counting cycles and instructions is that cycles include the execution time of instructions AND the memory transfers.`
        // perfcounter_config(COUNT_INSTRUCTIONS, true);
    }
    for (int32_t exec_time_idx = MODULE_TYPES_END * NR_TASKLETS - 1 - me(); exec_time_idx >= 0; exec_time_idx -= NR_TASKLETS)
        exec_times[exec_time_idx] = 0;
#endif
#if (defined MODULE_PERF_EVAL || defined CYCLE_PERF_EVAL)
    clusterSearching(points, dimAmt, sliceAmt, codebookEntryAmt, queries, queryAmt, neighbors, neighborAmt, 0, lookUpTable, centroids, localClusterAddrs, localClusterSizes, localClusterIDIdxs, localClusterIDs, localQueryIDs, squareRes, codebook, svDimAmt, largeSquareRes, radii, exec_times);
#else
    clusterSearching(points, dimAmt, sliceAmt, codebookEntryAmt, queries, queryAmt, neighbors, neighborAmt, 0, lookUpTable, centroids, localClusterAddrs, localClusterSizes, localClusterIDIdxs, localClusterIDs, localQueryIDs, squareRes, codebook, svDimAmt, largeSquareRes, radii);
#endif
#ifdef PERF_EVAL_SIM
    perfcounter_t exec_time_me = perfcounter_get();
    mutex_lock(mutex_exec_time);
    if (exec_time_me > exec_time) {
        exec_time = exec_time_me;
    }
    mutex_unlock(mutex_exec_time);
#endif
    return 0;
}
