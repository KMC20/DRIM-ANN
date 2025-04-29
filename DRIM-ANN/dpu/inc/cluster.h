/*
Author: KMC20
Date: 2024/2/27
Function: Operations for the cluster searching phase on DPUs.
*/

#ifndef CLUSTER_SEARCHING_H
#define CLUSTER_SEARCHING_H

#include "pqueue.h"  // libqueue: max heap
#include <string.h>  // memcpy, memmove
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <perfcounter.h>
#include <stdint.h>
#include <mram_unaligned.h>
#include <seqread.h>
#include "request.h"

#define increAddr(addr) addr + sizeof(pqueue_elem_t_mram)  // Increase the address by sizeof(ELEMTYPE *)

#define littleEndian true
#define localClusterIDsStride 2  // 8 / sizeof(ADDRTYPE)
#define localQueryIDsStride 2  // 8 / sizeof(ADDRTYPE)

#if (defined MODULE_PERF_EVAL || defined CYCLE_PERF_EVAL)
void clusterSearching(const __mram_ptr POINTTYPE *const points, const unsigned short dimAmt, const unsigned short sliceAmt, const ADDRTYPE codebookEntryAmt, const __mram_ptr ELEMTYPE *const queries, const uint32_t queryAmt, __mram_ptr pqueue_elem_t_mram *neighbors, const unsigned short neighborAmt, const ADDRTYPE queryNeighborStartAddr, VECSUMTYPE *lookUpTable, const __mram_ptr CENTROIDS_TYPE *const centroids, const ADDRTYPE *const localClusterAddrs, const CLUSTER_SIZES_TYPE *const localClusterSizes, const __mram_ptr ADDRTYPE *const localClusterIDIdxs, const __mram_ptr ADDRTYPE *const localClusterIDs, const __mram_ptr ADDRTYPE *const localQueryIDs, const uint16_t *const squareRes, const __mram_ptr CB_TYPE *const codebook, const uint32_t svDimAmt, const VECSUMTYPE *largeSquareRes, const SQUAREROOTDISTTYPE *radii, __mram_ptr perfcounter_t *exec_times);
#else
void clusterSearching(const __mram_ptr POINTTYPE *const points, const unsigned short dimAmt, const unsigned short sliceAmt, const ADDRTYPE codebookEntryAmt, const __mram_ptr ELEMTYPE *const queries, const uint32_t queryAmt, __mram_ptr pqueue_elem_t_mram *neighbors, const unsigned short neighborAmt, const ADDRTYPE queryNeighborStartAddr, VECSUMTYPE *lookUpTable, const __mram_ptr CENTROIDS_TYPE *const centroids, const ADDRTYPE *const localClusterAddrs, const CLUSTER_SIZES_TYPE *const localClusterSizes, const __mram_ptr ADDRTYPE *const localClusterIDIdxs, const __mram_ptr ADDRTYPE *const localClusterIDs, const __mram_ptr ADDRTYPE *const localQueryIDs, const uint16_t *const squareRes, const __mram_ptr CB_TYPE *const codebook, const uint32_t svDimAmt, const VECSUMTYPE *largeSquareRes, const SQUAREROOTDISTTYPE *radii);
#endif

#endif  // CLUSTER_SEARCHING_H
