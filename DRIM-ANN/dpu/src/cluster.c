/*
Author: KMC20
Date: 2024/2/27
Function: Operations for the cluster searching phase on DPUs.
*/

#include "cluster.h"
#include <built_ins.h>

#define min(a, b) (a < b ? a : b)
#define INFINITY 0xFFFFFFFFFFFFFFFF  // Note: Only for the cases that the type of distance results is integer. If the type changes into float, this macro should be changed correspondingly.
#define U32INFINITY 0xFFFFFFFF  // Note: Only for the cases that the type of distance results is integer. If the type changes into float, this macro should be changed correspondingly.

MUTEX_INIT(mutex_pqueue_update);
BARRIER_INIT(barrier_tasklets, NR_TASKLETS);

/**************************************************************************************************************************************************************************************************************/
/*                                                              Functions for the usage of priority queue (Copied from `sample.c` of `libpqueue`)                                                             */
/**************************************************************************************************************************************************************************************************************/
static int cmp_pri(pqueue_pri_t next, pqueue_pri_t curr) {
	return (next < curr);
}
static pqueue_pri_t get_pri(void *a) {
	return ((pqueue_elem_t *) a)->pri;
}
static void set_pri(void *a, pqueue_pri_t pri) {
	((pqueue_elem_t *) a)->pri = pri;
}
static size_t get_pos(void *a) {
	return ((pqueue_elem_t *) a)->pos;
}
static void set_pos(void *a, size_t pos) {
	((pqueue_elem_t *) a)->pos = pos;
}	
/***************************************************************************************************************************************************************************************************************/

VECSUMTYPE distCalVec_prunePartDis;
VECSUMTYPE distCalVec_remainPartDis;
VECSUMTYPE distCalVec(const POINTTYPE *const vec1, const VECSUMTYPE *const LUT, const unsigned short sliceAmt, const ADDRTYPE codebookEntryAmt, const uint32_t savedPruneSliceAmt, bool *insertDist) {
    register VECSUMTYPE res = 0;
    POINTTYPE *vec1SavedEnd = (POINTTYPE *)vec1 + savedPruneSliceAmt;
    POINTTYPE *vec1End = (POINTTYPE *)vec1 + sliceAmt;
    VECSUMTYPE *LUTBasePt = (VECSUMTYPE *)LUT;
    for (POINTTYPE *vec1pt = (POINTTYPE *)vec1; vec1pt < vec1SavedEnd; ++vec1pt, LUTBasePt += codebookEntryAmt) {
        res += *(LUTBasePt + *vec1pt);
    }
    if (res >= distCalVec_remainPartDis) {  // Prune redundant vector slices
        *insertDist = false;
        return res;
    }
    for (POINTTYPE *vec1pt = vec1SavedEnd; vec1pt < vec1End; ++vec1pt, LUTBasePt += codebookEntryAmt) {
        res += *(LUTBasePt + *vec1pt);
    }
    return res;
}

MUTEX_INIT(mutex_unaligned_neighbors);
void save_pq_into_mram(pqueue_t *pq, const ADDRTYPE pointNeighborStartAddr, __mram_ptr pqueue_elem_t_mram *pointsMram) {
    uint32_t priUnalignedBytes = pointNeighborStartAddr & MRAM_ALIGN_BYTES - 1;
    if (priUnalignedBytes != 0) {  // Unaligned address. This case should be avoided and hardly happens since the auto alignment of struct data type
        pqueue_elem_t *pqTop = NULL;
        fsb_allocator_t wramWriteBufAllocator = fsb_alloc(sizeof(pqueue_elem_t_mram) * pq->size, 1);
        __dma_aligned uint8_t *wramWriteBuf = fsb_get(wramWriteBufAllocator);
        uint32_t wramWriteBufSize = 0;
        for (; wramWriteBufSize < priUnalignedBytes && (pqTop = pqueue_pop(pq)) != NULL; wramWriteBufSize += sizeof(pqueue_elem_t_mram))
            memcpy(wramWriteBuf, pqTop, sizeof(pqueue_elem_t_mram));
        __dma_aligned uint64_t priBuf;
        __mram_ptr pqueue_elem_t_mram *pointsMramAligned = pointsMram + pointNeighborStartAddr - priUnalignedBytes;
        uint32_t priCopiedBytes = MRAM_ALIGN_BYTES - priUnalignedBytes;
        mutex_lock(mutex_unaligned_neighbors);
        mram_read(pointsMramAligned, &priBuf, sizeof(uint64_t));
        memcpy((uint8_t *)&priBuf + priUnalignedBytes, wramWriteBuf, priCopiedBytes);
        mram_write(&priBuf, pointsMramAligned, sizeof(uint64_t));
        mutex_unlock(mutex_unaligned_neighbors);
        ++pointsMramAligned;
        wramWriteBufSize -= priCopiedBytes;
        memmove(wramWriteBuf, wramWriteBuf + priCopiedBytes, wramWriteBufSize);
        __dma_aligned uint8_t *wramWriteBufPt = wramWriteBuf;
        while ((pqTop = pqueue_pop(pq)) != NULL) {
            memcpy(wramWriteBufPt, pqTop, sizeof(pqueue_elem_t_mram));
            increAddr(wramWriteBufPt);
        }
        mutex_lock(mutex_unaligned_neighbors);
        mram_write_unaligned(wramWriteBuf, pointsMramAligned, wramWriteBufPt - wramWriteBuf);
        mutex_unlock(mutex_unaligned_neighbors);
        fsb_free(wramWriteBufAllocator, wramWriteBuf);
    } else {
        __mram_ptr pqueue_elem_t_mram *pointsMramAddr = pointsMram + pointNeighborStartAddr + pqueue_size(pq) - 1;
        __dma_aligned pqueue_elem_t *pqTop = NULL;
        while ((pqTop = pqueue_pop(pq)) != NULL) {
            mram_write(pqTop, pointsMramAddr, sizeof(pqueue_elem_t_mram));
            --pointsMramAddr;
        }
    }
}

inline VECSUMTYPE distCalSubVec(const RESIDUAL_TYPE *const vec1, const CB_TYPE *const vec2, const unsigned short dimAmt, const uint16_t *const squareRes) {
    register VECSUMTYPE res = 0, diff;
    RESIDUAL_TYPE *vec1pt = (RESIDUAL_TYPE *)vec1, *vec1End = (RESIDUAL_TYPE *)vec1 + dimAmt;
    for (CB_TYPE *vec2pt = (CB_TYPE *)vec2; vec1pt < vec1End; ++vec1pt, ++vec2pt) {
        int32_t vec1op = *vec1pt;
        int32_t vec2op = *vec2pt;
        diff = vec1op > vec2op ? vec1op - vec2op : vec2op - vec1op;
        res += squareRes[diff];
    }
    return res;
}

VECSUMTYPE *constrLUT_partDis;
fsb_allocator_t constrLUT_partDisAllocator;
MUTEX_INIT(mutex_prune_constrLUT);
void constrLUT(const RESIDUAL_TYPE *const vec1, const __mram_ptr CB_TYPE *const codebook, VECSUMTYPE *LUT, const uint16_t maxSvDimAmt, const uint16_t paddedSvDimAmt, const uint16_t dimAmt, const uint32_t savedPruneSliceAmt, const ADDRTYPE codebookEntryAmt, CB_TYPE *CBSliceBuf, const uint16_t CB_SLICE_BUF_SIZE, const uint16_t *const squareRes, const uint16_t PARA_CB_SLICE_INC) {
    uint16_t sliceEnd;
    RESIDUAL_TYPE *vec1Slice = (RESIDUAL_TYPE *)vec1;
    __mram_ptr CB_TYPE *CBIter = (__mram_ptr CB_TYPE *)codebook + me() * paddedSvDimAmt;
    VECSUMTYPE *LUTIter = LUT + me();
    uint16_t svDimAmt = maxSvDimAmt;
    uint16_t sliceSize = CB_SLICE_BUF_SIZE;
    uint16_t sliceBeg = 0;
    ADDRTYPE codebookEntryCntEnd = codebookEntryAmt + me();
    for (uint16_t sliceCnt = 0; sliceCnt < savedPruneSliceAmt; sliceBeg = sliceEnd, ++sliceCnt) {
        sliceEnd = sliceBeg + svDimAmt;
        if (sliceEnd > dimAmt) {
            svDimAmt = dimAmt - sliceBeg;
            sliceSize = CB_SLICE_BUF_SIZE;
        }
        ADDRTYPE codebookEntryCnt;
        for (codebookEntryCnt = me(); codebookEntryCnt < codebookEntryAmt; codebookEntryCnt += NR_TASKLETS) {
            mram_read(CBIter, CBSliceBuf, sliceSize);
            __dma_aligned VECSUMTYPE partDis = distCalSubVec(vec1Slice, CBSliceBuf, svDimAmt, squareRes);
            *LUTIter = partDis;
            LUTIter += NR_TASKLETS;
            CBIter += PARA_CB_SLICE_INC;
        }
        if (codebookEntryCnt > codebookEntryCntEnd) {  // The last loop is a bubble
            ADDRTYPE bubbleSize = codebookEntryCnt - codebookEntryCntEnd;
            LUTIter -= bubbleSize;
            CBIter -= paddedSvDimAmt * bubbleSize;
        } else {  // The last loop is a real execution
            ADDRTYPE bubbleSize = codebookEntryCntEnd - codebookEntryCnt;
            LUTIter += bubbleSize;
            CBIter += paddedSvDimAmt * bubbleSize;
        }
        vec1Slice += svDimAmt;
    }
    for (uint16_t pruneSliceCnt = 0; sliceBeg < dimAmt; sliceBeg = sliceEnd, ++pruneSliceCnt) {
        sliceEnd = sliceBeg + svDimAmt;
        if (sliceEnd > dimAmt) {
            svDimAmt = dimAmt - sliceBeg;
            sliceSize = CB_SLICE_BUF_SIZE;
        }
        ADDRTYPE codebookEntryCnt;
        for (codebookEntryCnt = me(); codebookEntryCnt < codebookEntryAmt; codebookEntryCnt += NR_TASKLETS) {
            mram_read(CBIter, CBSliceBuf, sliceSize);
            __dma_aligned VECSUMTYPE partDis = distCalSubVec(vec1Slice, CBSliceBuf, svDimAmt, squareRes);
            *LUTIter = partDis;
            mutex_lock(mutex_prune_constrLUT);
            if (partDis < constrLUT_partDis[pruneSliceCnt])
                constrLUT_partDis[pruneSliceCnt] = partDis;
            mutex_unlock(mutex_prune_constrLUT);
            LUTIter += NR_TASKLETS;
            CBIter += PARA_CB_SLICE_INC;
        }
        if (codebookEntryCnt > codebookEntryCntEnd) {  // The last loop is a bubble
            ADDRTYPE bubbleSize = codebookEntryCnt - codebookEntryCntEnd;
            LUTIter -= bubbleSize;
            CBIter -= paddedSvDimAmt * bubbleSize;
        } else {  // The last loop is a real execution
            ADDRTYPE bubbleSize = codebookEntryCntEnd - codebookEntryCnt;
            LUTIter += bubbleSize;
            CBIter += paddedSvDimAmt * bubbleSize;
        }
        vec1Slice += svDimAmt;
    }
}

void calResidualInPlace(const __dma_aligned ELEMTYPE *const queryBuf, const __dma_aligned CENTROIDS_TYPE *const centroidBuf, __dma_aligned RESIDUAL_TYPE* residualBuf, const unsigned short dimAmt) {  // Calculate the residual of query to centroid and store the result in centroid buffer
    for (ADDRTYPE dim = 0; dim < dimAmt; ++dim) {
        residualBuf[dim] = queryBuf[dim] - (RESIDUAL_TYPE)(centroidBuf[dim]);
    }
}

// return: the square root, so it falls in [0, 2 ^ sizeof(ELEMTYPE) * squareRoot(sliceAmt)]
inline SQUAREROOTDISTTYPE getSquareRoot(const VECSUMTYPE *largeSquareRes, const VECSUMTYPE squareVal) {
    if (!squareVal)
        return 0;
    SQUAREROOTDISTTYPE beg = 0, end = LARGE_SQUARE_RES_SIZE;
    SQUAREROOTDISTTYPE mid;
    VECSUMTYPE midVal;
    while (beg < end) {
        mid = (beg + end) >> 1;
        midVal = largeSquareRes[mid];
        if (midVal < squareVal) {
            beg = mid + 1;
        } else if (midVal > squareVal) {
            end = mid;
        } else {
            return mid;
        }
    }
    return midVal < squareVal ? mid : mid - 1;  // Expect that mid - 1 >= 0 since the case squareVal == 0 has been judged at the beginning
}

inline VECSUMTYPE distCalSelfVec(const ELEMTYPE *const vec1, const unsigned short dimAmt, const uint16_t *const squareRes) {
    register VECSUMTYPE res = 0, diff;
    ELEMTYPE *vec1End = (ELEMTYPE *)vec1 + dimAmt;
    for (ELEMTYPE *vec1pt = (ELEMTYPE *)vec1; vec1pt < vec1End; ++vec1pt) {
        diff = *vec1pt;
        res += squareRes[diff];
    }
    return res;
}

pqueue_t *clusterSearching_pq;
pqueue_elem_t *clusterSearching_pqElems;
uint32_t clusterSearching_pqElemSize;
fsb_allocator_t clusterSearching_pqElemsAllocator, clusterSearching_pq_allocator;
#if (defined MODULE_PERF_EVAL || defined CYCLE_PERF_EVAL)
void clusterSearching(const __mram_ptr POINTTYPE *const points, const unsigned short dimAmt, const unsigned short sliceAmt, const ADDRTYPE codebookEntryAmt, const __mram_ptr ELEMTYPE *const queries, const uint32_t queryAmt, __mram_ptr pqueue_elem_t_mram *neighbors, const unsigned short neighborAmt, const ADDRTYPE queryNeighborStartAddr, VECSUMTYPE *lookUpTable, const __mram_ptr CENTROIDS_TYPE *const centroids, const ADDRTYPE *const localClusterAddrs, const CLUSTER_SIZES_TYPE *const localClusterSizes, const __mram_ptr ADDRTYPE *const localClusterIDIdxs, const __mram_ptr ADDRTYPE *const localClusterIDs, const __mram_ptr ADDRTYPE *const localQueryIDs, const uint16_t *const squareRes, const __mram_ptr CB_TYPE *const codebook, const uint32_t svDimAmt, const VECSUMTYPE *largeSquareRes, const SQUAREROOTDISTTYPE *radii, __mram_ptr perfcounter_t *exec_times) {
    __mram_ptr perfcounter_t *exec_times_me_base = exec_times + me() * MODULE_TYPES_END;
    perfcounter_t exec_time_me = perfcounter_get();
#else
void clusterSearching(const __mram_ptr POINTTYPE *const points, const unsigned short dimAmt, const unsigned short sliceAmt, const ADDRTYPE codebookEntryAmt, const __mram_ptr ELEMTYPE *const queries, const uint32_t queryAmt, __mram_ptr pqueue_elem_t_mram *neighbors, const unsigned short neighborAmt, const ADDRTYPE queryNeighborStartAddr, VECSUMTYPE *lookUpTable, const __mram_ptr CENTROIDS_TYPE *const centroids, const ADDRTYPE *const localClusterAddrs, const CLUSTER_SIZES_TYPE *const localClusterSizes, const __mram_ptr ADDRTYPE *const localClusterIDIdxs, const __mram_ptr ADDRTYPE *const localClusterIDs, const __mram_ptr ADDRTYPE *const localQueryIDs, const uint16_t *const squareRes, const __mram_ptr CB_TYPE *const codebook, const uint32_t svDimAmt, const VECSUMTYPE *largeSquareRes, const SQUAREROOTDISTTYPE *radii) {
#endif
    mem_reset();
    const uint32_t PRUNE_SLICE_SIZE = PRUNE_SLICE_AMT * sizeof(VECSUMTYPE);
    barrier_wait(&barrier_tasklets);
    if (me() == 0) {
        clusterSearching_pqElemsAllocator = fsb_alloc(neighborAmt * sizeof(pqueue_elem_t), 1);
        clusterSearching_pqElems = (pqueue_elem_t *)fsb_get(clusterSearching_pqElemsAllocator);
        clusterSearching_pq = pqueue_init(neighborAmt, cmp_pri, get_pri, set_pri, get_pos, set_pos, &clusterSearching_pq_allocator);
        clusterSearching_pqElemSize = 0;
        constrLUT_partDisAllocator = fsb_alloc(PRUNE_SLICE_SIZE, 1);
        constrLUT_partDis = (VECSUMTYPE *)fsb_get(constrLUT_partDisAllocator);
    }
    barrier_wait(&barrier_tasklets);
    const uint16_t savedPruneSliceAmt = sliceAmt - PRUNE_SLICE_AMT;
    const uint32_t pointSize = sizeof(POINTTYPE) * sliceAmt;
    const uint32_t pointStride = pointSize * NR_TASKLETS;
    const uint16_t querySize = sizeof(ELEMTYPE) * dimAmt;
    const uint16_t residualSize = sizeof(RESIDUAL_TYPE) * dimAmt;
    uint16_t CB_SLICE_BUF_SIZE = sizeof(CB_TYPE) * svDimAmt;
    uint16_t PARA_CB_SLICE_INC = NR_TASKLETS;
    uint16_t paddedSvDimAmt = svDimAmt;
    {  // Padding
        uint16_t CB_SLICE_BUF_SIZE_REMAIN = CB_SLICE_BUF_SIZE & 7;
        if (CB_SLICE_BUF_SIZE_REMAIN != 0) {
            CB_SLICE_BUF_SIZE += 8 - CB_SLICE_BUF_SIZE_REMAIN;
            paddedSvDimAmt = CB_SLICE_BUF_SIZE / sizeof(CB_TYPE);
            PARA_CB_SLICE_INC *= paddedSvDimAmt;
        } else {
            PARA_CB_SLICE_INC *= svDimAmt;
        }
    }
    fsb_allocator_t CBBufAllocator = fsb_alloc(CB_SLICE_BUF_SIZE, 1);
    __dma_aligned CB_TYPE *CBSliceBuf = fsb_get(CBBufAllocator);
    fsb_allocator_t curPointBufAllocator = fsb_alloc(pointSize, 1);
    __dma_aligned POINTTYPE *curPointBuf = fsb_get(curPointBufAllocator);
    fsb_allocator_t queryBufAllocator = fsb_alloc(querySize, 1);
    __dma_aligned ELEMTYPE *queryBuf = fsb_get(queryBufAllocator);
    fsb_allocator_t centroidBufAllocator = fsb_alloc(querySize, 1);
    __dma_aligned CENTROIDS_TYPE *centroidBuf = fsb_get(centroidBufAllocator);
    fsb_allocator_t residualBufAllocator = fsb_alloc(residualSize, 1);
    __dma_aligned RESIDUAL_TYPE *residualBuf = fsb_get(residualBufAllocator);
    const uint32_t neighborSize = sizeof(pqueue_elem_t) * neighborAmt;
    const uint32_t MRAM_WRITE_MAX_SIZE = 2048;
    const uint32_t neighborCopyStart = MRAM_WRITE_MAX_SIZE * me();
    const uint32_t neighborCopyStride = MRAM_WRITE_MAX_SIZE * NR_TASKLETS;
    __mram_ptr pqueue_elem_t_mram *neighborsWrite = neighbors;
    uint32_t neighborsWriteStride = neighborAmt;
    const __dma_aligned pqueue_elem_t_mram neighborsDefaultValue = { U32INFINITY, {U32INFINITY} };
    ADDRTYPE localClusterIDIdxStart = 0;
    ADDRTYPE *localClusterAddrsRead = localClusterAddrs;
    CLUSTER_SIZES_TYPE *localClusterSizesRead = localClusterSizes;
#ifdef MODULE_PERF_EVAL
    exec_times_me_base[PRE_DEFINING] += perfcounter_get() - exec_time_me;
    exec_time_me = perfcounter_get();
#endif
    // // ****************************************************************************************** Set for loop tilling ******************************************************************************************
    // distCalSubVec
    CB_TYPE *vec2pt0 = CBSliceBuf;
    CB_TYPE *vec2pt1 = vec2pt0 + 1;
    CB_TYPE *vec2pt2 = vec2pt1 + 1;
    CB_TYPE *vec2pt3 = vec2pt2 + 1;
    CB_TYPE *vec2pt4 = vec2pt3 + 1;
    CB_TYPE *vec2pt5 = vec2pt4 + 1;
    CB_TYPE *vec2pt6 = vec2pt5 + 1;
    CB_TYPE *vec2pt7 = vec2pt6 + 1;
    // distCalVec
    POINTTYPE *distCalVec_vec1pt0 = curPointBuf;
    POINTTYPE *distCalVec_vec1pt1 = distCalVec_vec1pt0 + 1;
    POINTTYPE *distCalVec_vec1pt2 = distCalVec_vec1pt1 + 1;
    POINTTYPE *distCalVec_vec1pt3 = distCalVec_vec1pt2 + 1;
    POINTTYPE *distCalVec_vec1pt4 = distCalVec_vec1pt3 + 1;
    POINTTYPE *distCalVec_vec1pt5 = distCalVec_vec1pt4 + 1;
    POINTTYPE *distCalVec_vec1pt6 = distCalVec_vec1pt5 + 1;
    POINTTYPE *distCalVec_vec1pt7 = distCalVec_vec1pt6 + 1;
    POINTTYPE *distCalVec_vec1pt8 = distCalVec_vec1pt7 + 1;
    POINTTYPE *distCalVec_vec1pt9 = distCalVec_vec1pt8 + 1;
    POINTTYPE *distCalVec_vec1pt10 = distCalVec_vec1pt9 + 1;
    POINTTYPE *distCalVec_vec1pt11 = distCalVec_vec1pt10 + 1;
    POINTTYPE *distCalVec_vec1pt12 = distCalVec_vec1pt11 + 1;
    POINTTYPE *distCalVec_vec1pt13 = distCalVec_vec1pt12 + 1;
    POINTTYPE *distCalVec_vec1pt14 = distCalVec_vec1pt13 + 1;
    POINTTYPE *distCalVec_vec1pt15 = distCalVec_vec1pt14 + 1;
    VECSUMTYPE *LUTBasePt0 = lookUpTable;
    VECSUMTYPE *LUTBasePt1 = LUTBasePt0 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt2 = LUTBasePt1 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt3 = LUTBasePt2 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt4 = LUTBasePt3 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt5 = LUTBasePt4 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt6 = LUTBasePt5 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt7 = LUTBasePt6 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt8 = LUTBasePt7 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt9 = LUTBasePt8 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt10 = LUTBasePt9 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt11 = LUTBasePt10 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt12 = LUTBasePt11 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt13 = LUTBasePt12 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt14 = LUTBasePt13 + codebookEntryAmt;
    VECSUMTYPE *LUTBasePt15 = LUTBasePt14 + codebookEntryAmt;
    // // ************************************************************************************ End of setting for loop tilling *************************************************************************************
    for (ADDRTYPE qBufId = 0, qBufEnd; qBufId < queryAmt; qBufId = qBufEnd) {
        qBufEnd = min(queryAmt, qBufId + localQueryIDsStride);
        __dma_aligned uint64_t localQueryIDsBuf;
        __dma_aligned uint64_t localClusterIDIdxsBuf;
        mram_read(localQueryIDs + qBufId, &localQueryIDsBuf, sizeof(uint64_t));
        mram_read(localClusterIDIdxs + qBufId, &localClusterIDIdxsBuf, sizeof(uint64_t));
        for (ADDRTYPE qId = qBufId; qId < qBufEnd; ++qId) {
#if (littleEndian == true)
// #if sizeof(ADDRTYPE) == 4
            ADDRTYPE localQueryIDsVal = localQueryIDsBuf & 0xFFFFFFFF;
            localQueryIDsBuf >>= 32;
            ADDRTYPE localClusterIDIdxsVal = localClusterIDIdxsBuf & 0xFFFFFFFF;
            localClusterIDIdxsBuf >>= 32;
// #endif
#else
// #if sizeof(ADDRTYPE) == 4
            ADDRTYPE localQueryIDsVal = localQueryIDsBuf >> 32;
            localQueryIDsBuf <<= 32;
            ADDRTYPE localClusterIDIdxsVal = localClusterIDIdxsBuf >> 32;
            localClusterIDIdxsBuf <<= 32;
// #endif
#endif
            mram_read((__mram_ptr uint8_t *)queries + localQueryIDsVal * querySize, queryBuf, querySize);
            ADDRTYPE localClusterIDIdxEnd = localClusterIDIdxStart + localClusterIDIdxsVal;  // local -> DPU global
            __dma_aligned uint64_t localClusterIDsBuf;
// #if sizeof(ADDRTYPE) == 4
            mram_read(localClusterIDs + (localClusterIDIdxStart & 0xFFFFFFFE), &localClusterIDsBuf, sizeof(uint64_t));
// #endif
#if (littleEndian == true)
// #if sizeof(ADDRTYPE) == 4
            {
                ADDRTYPE localClusterIDsBufShift = localClusterIDIdxStart & 1;  // & (8 / sizeof(ADDRTYPE) - 1)
                while (localClusterIDsBufShift-- > 0) {
                    localClusterIDsBuf >>= 32;
                }
            }
// #endif
#else
// #if sizeof(ADDRTYPE) == 4
            {
                ADDRTYPE localClusterIDsBufShift = localClusterIDIdxStart & 1;  // & (8 / sizeof(ADDRTYPE) - 1)
                while (localClusterIDsBufShift-- > 0) {
                    localClusterIDsBuf <<= 32;
                }
            }
// #endif
#endif
            for (ADDRTYPE localClusterIDIdxBuf = localClusterIDIdxStart, localClusterIDIdxBufEnd = min(localClusterIDIdxEnd, localClusterIDIdxStart + localClusterIDsStride - (localClusterIDIdxStart & 1)); localClusterIDIdxBuf < localClusterIDIdxEnd; ) {
                for (ADDRTYPE localClusterIDIdx = localClusterIDIdxBuf; localClusterIDIdx < localClusterIDIdxBufEnd; ++localClusterIDIdx) {
#if (littleEndian == true)
// #if sizeof(ADDRTYPE) == 4
                    ADDRTYPE localClusterIDsVal = localClusterIDsBuf & 0xFFFFFFFF;
                    localClusterIDsBuf >>= 32;
// #endif
#else
// #if sizeof(ADDRTYPE) == 4
                    ADDRTYPE localClusterIDsVal = localClusterIDsBuf >> 32;
                    localClusterIDsBuf <<= 32;
// #endif
#endif
                    ADDRTYPE localClusterAddrBeg = *localClusterAddrsRead;
                    ADDRTYPE localClusterAddrEnd = *localClusterSizesRead + localClusterAddrBeg;
                    ++localClusterAddrsRead, ++localClusterSizesRead;
                    mram_read((__mram_ptr uint8_t *)centroids + localClusterIDsVal * querySize, centroidBuf, querySize);
#ifdef MODULE_PERF_EVAL
                    exec_times_me_base[AFFILIATE_OPS] += perfcounter_get() - exec_time_me;
                    exec_time_me = perfcounter_get();
#endif
                    calResidualInPlace(queryBuf, centroidBuf, residualBuf, dimAmt);
#ifdef MODULE_PERF_EVAL
                    exec_times_me_base[CAL_RESIDUAL] += perfcounter_get() - exec_time_me;
                    exec_time_me = perfcounter_get();
#endif
                    if (clusterSearching_pqElemSize >= neighborAmt) {  // Prune redundant clusters
                        pqueue_elem_t *pqTop = pqueue_peek(clusterSearching_pq);
                        SQUAREROOTDISTTYPE distQueryCentroid = getSquareRoot(largeSquareRes, distCalSelfVec(centroidBuf, dimAmt, squareRes));
                        SQUAREROOTDISTTYPE distQueryCurPoints = getSquareRoot(largeSquareRes, pqTop->pri);
                        SQUAREROOTDISTTYPE radiusCluster = radii[localClusterIDsVal];  // Note: `radii` stores the Euclidean distance instead of the square Euclidean distance
                        if (distQueryCentroid > radiusCluster && distQueryCentroid > distQueryCurPoints && distQueryCentroid >= distQueryCurPoints + radiusCluster)
                            continue;
                    }
                    ADDRTYPE localClusterAddrBegMe = localClusterAddrBeg + me();
                    uint32_t pointAddr = localClusterAddrBegMe * pointSize;
#ifdef MODULE_PERF_EVAL
                    exec_times_me_base[AFFILIATE_OPS] += perfcounter_get() - exec_time_me;
                    exec_time_me = perfcounter_get();
#endif
                    // barrier_wait(&barrier_tasklets);
                    // if (me() == 0) {
                    //     memset(constrLUT_partDis, 0xFF, PRUNE_SLICE_SIZE);
                    // }
                    barrier_wait(&barrier_tasklets);
                    // constrLUT(residualBuf, codebook, lookUpTable, svDimAmt, paddedSvDimAmt, dimAmt, savedPruneSliceAmt, codebookEntryAmt, CBSliceBuf, CB_SLICE_BUF_SIZE, squareRes, PARA_CB_SLICE_INC);
                    {  // constrLUT
                        uint16_t sliceEnd;
                        RESIDUAL_TYPE *vec1Slice = (RESIDUAL_TYPE *)residualBuf;
                        __mram_ptr CB_TYPE *CBIter = (__mram_ptr CB_TYPE *)codebook + me() * paddedSvDimAmt;
                        VECSUMTYPE *LUTIter = lookUpTable + me();
                        uint16_t localSvDimAmt = svDimAmt;
                        uint16_t sliceSize = CB_SLICE_BUF_SIZE;
                        uint16_t sliceBeg = 0;
                        ADDRTYPE codebookEntryCntEnd = codebookEntryAmt + me();
                        for (uint16_t sliceCnt = 0; sliceCnt < savedPruneSliceAmt; sliceBeg = sliceEnd, ++sliceCnt) {
                            sliceEnd = sliceBeg + localSvDimAmt;
                            if (sliceEnd > dimAmt) {
                                localSvDimAmt = dimAmt - sliceBeg;
                                sliceSize = CB_SLICE_BUF_SIZE;
                            }
                            // // ****************************************************************************************** Set for loop tilling ******************************************************************************************
                            // distCalSubVec
                            RESIDUAL_TYPE *vec1pt0 = vec1Slice;
                            RESIDUAL_TYPE *vec1pt1 = vec1pt0 + 1;
                            RESIDUAL_TYPE *vec1pt2 = vec1pt1 + 1;
                            RESIDUAL_TYPE *vec1pt3 = vec1pt2 + 1;
                            RESIDUAL_TYPE *vec1pt4 = vec1pt3 + 1;
                            RESIDUAL_TYPE *vec1pt5 = vec1pt4 + 1;
                            RESIDUAL_TYPE *vec1pt6 = vec1pt5 + 1;
                            RESIDUAL_TYPE *vec1pt7 = vec1pt6 + 1;
                            // // ************************************************************************************ End of setting for loop tilling *************************************************************************************
                            ADDRTYPE codebookEntryCnt;
                            for (codebookEntryCnt = me(); codebookEntryCnt < codebookEntryAmt; codebookEntryCnt += NR_TASKLETS) {
                                mram_read(CBIter, CBSliceBuf, sliceSize);
                                __dma_aligned VECSUMTYPE partDis;
                                {  // distCalSubVec
                                    register VECSUMTYPE res = 0, diff;
                                    register int32_t vec1op, vec2op;
                                    vec1op = *vec1pt0;
                                    vec2op = *vec2pt0;
                                    diff = vec1op > vec2op ? vec1op - vec2op : vec2op - vec1op;
                                    res += squareRes[diff];
                                    vec1op = *vec1pt1;
                                    vec2op = *vec2pt1;
                                    diff = vec1op > vec2op ? vec1op - vec2op : vec2op - vec1op;
                                    res += squareRes[diff];
                                    vec1op = *vec1pt2;
                                    vec2op = *vec2pt2;
                                    diff = vec1op > vec2op ? vec1op - vec2op : vec2op - vec1op;
                                    res += squareRes[diff];
                                    vec1op = *vec1pt3;
                                    vec2op = *vec2pt3;
                                    diff = vec1op > vec2op ? vec1op - vec2op : vec2op - vec1op;
                                    res += squareRes[diff];
                                    vec1op = *vec1pt4;
                                    vec2op = *vec2pt4;
                                    diff = vec1op > vec2op ? vec1op - vec2op : vec2op - vec1op;
                                    res += squareRes[diff];
                                    vec1op = *vec1pt5;
                                    vec2op = *vec2pt5;
                                    diff = vec1op > vec2op ? vec1op - vec2op : vec2op - vec1op;
                                    res += squareRes[diff];
                                    if (localSvDimAmt > 6) {  // SIFT
                                        vec1op = *vec1pt6;
                                        vec2op = *vec2pt6;
                                        diff = vec1op > vec2op ? vec1op - vec2op : vec2op - vec1op;
                                        res += squareRes[diff];
                                        vec1op = *vec1pt7;
                                        vec2op = *vec2pt7;
                                        diff = vec1op > vec2op ? vec1op - vec2op : vec2op - vec1op;
                                        res += squareRes[diff];
                                    }  // else: DEEP
                                    partDis = res;                                
                                }
                                *LUTIter = partDis;
                                LUTIter += NR_TASKLETS;
                                CBIter += PARA_CB_SLICE_INC;
                            }
                            if (codebookEntryCnt > codebookEntryCntEnd) {  // The last loop is a bubble
                                ADDRTYPE bubbleSize = codebookEntryCnt - codebookEntryCntEnd;
                                LUTIter -= bubbleSize;
                                CBIter -= paddedSvDimAmt * bubbleSize;
                            } else {  // The last loop is really an execution
                                ADDRTYPE bubbleSize = codebookEntryCntEnd - codebookEntryCnt;
                                LUTIter += bubbleSize;
                                CBIter += paddedSvDimAmt * bubbleSize;
                            }
                            vec1Slice += localSvDimAmt;
                        }
                        for (uint16_t pruneSliceCnt = 0; sliceBeg < dimAmt; sliceBeg = sliceEnd, ++pruneSliceCnt) {
                            sliceEnd = sliceBeg + localSvDimAmt;
                            if (sliceEnd > dimAmt) {
                                localSvDimAmt = dimAmt - sliceBeg;
                                sliceSize = CB_SLICE_BUF_SIZE;
                            }
                            ADDRTYPE codebookEntryCnt;
                            for (codebookEntryCnt = me(); codebookEntryCnt < codebookEntryAmt; codebookEntryCnt += NR_TASKLETS) {
                                mram_read(CBIter, CBSliceBuf, sliceSize);
                                __dma_aligned VECSUMTYPE partDis = distCalSubVec(vec1Slice, CBSliceBuf, localSvDimAmt, squareRes);
                                *LUTIter = partDis;
                                mutex_lock(mutex_prune_constrLUT);
                                if (partDis < constrLUT_partDis[pruneSliceCnt])
                                    constrLUT_partDis[pruneSliceCnt] = partDis;
                                mutex_unlock(mutex_prune_constrLUT);
                                LUTIter += NR_TASKLETS;
                                CBIter += PARA_CB_SLICE_INC;
                            }
                            if (codebookEntryCnt > codebookEntryCntEnd) {  // The last loop is a bubble
                                ADDRTYPE bubbleSize = codebookEntryCnt - codebookEntryCntEnd;
                                LUTIter -= bubbleSize;
                                CBIter -= paddedSvDimAmt * bubbleSize;
                            } else {  // The last loop is a real execution
                                ADDRTYPE bubbleSize = codebookEntryCntEnd - codebookEntryCnt;
                                LUTIter += bubbleSize;
                                CBIter += paddedSvDimAmt * bubbleSize;
                            }
                            vec1Slice += localSvDimAmt;
                        }                    
                    }
                    distCalVec_prunePartDis = 0;
                    distCalVec_remainPartDis = U32INFINITY;
                    barrier_wait(&barrier_tasklets);
                    // if (me() == 0) {
                    //     distCalVec_prunePartDis = 0;
                    //     for (uint16_t pruneSliceCnt = 0; pruneSliceCnt < PRUNE_SLICE_AMT; ++pruneSliceCnt)
                    //         distCalVec_prunePartDis += constrLUT_partDis[pruneSliceCnt];
                    //     distCalVec_remainPartDis = U32INFINITY;
                    // }
                    // barrier_wait(&barrier_tasklets);
#ifdef MODULE_PERF_EVAL
                    exec_times_me_base[CONSTR_LUT] += perfcounter_get() - exec_time_me;
                    exec_time_me = perfcounter_get();
#endif

                    for (ADDRTYPE pId = localClusterAddrBegMe, localPId = 0; pId < localClusterAddrEnd; pId += NR_TASKLETS, ++localPId) {
#ifdef MODULE_PERF_EVAL
                        exec_times_me_base[AFFILIATE_OPS] += perfcounter_get() - exec_time_me;
                        exec_time_me = perfcounter_get();
#endif
                        mram_read((__mram_ptr uint8_t *)points + pointAddr, curPointBuf, pointSize);
#ifdef MODULE_PERF_EVAL
                        exec_times_me_base[CLUSTER_LOADING] += perfcounter_get() - exec_time_me;
                        exec_time_me = perfcounter_get();
#endif
                        bool insertDist = true;
                        // VECSUMTYPE dist = distCalVec(curPointBuf, lookUpTable, sliceAmt, codebookEntryAmt, savedPruneSliceAmt, &insertDist);
                        VECSUMTYPE dist;
                        {  // distCalVec
                            register VECSUMTYPE res = 0;
                            res += *(LUTBasePt0 + *distCalVec_vec1pt0);
                            res += *(LUTBasePt1 + *distCalVec_vec1pt1);
                            res += *(LUTBasePt2 + *distCalVec_vec1pt2);
                            res += *(LUTBasePt3 + *distCalVec_vec1pt3);
                            res += *(LUTBasePt4 + *distCalVec_vec1pt4);
                            res += *(LUTBasePt5 + *distCalVec_vec1pt5);
                            res += *(LUTBasePt6 + *distCalVec_vec1pt6);
                            res += *(LUTBasePt7 + *distCalVec_vec1pt7);
                            res += *(LUTBasePt8 + *distCalVec_vec1pt8);
                            res += *(LUTBasePt9 + *distCalVec_vec1pt9);
                            res += *(LUTBasePt10 + *distCalVec_vec1pt10);
                            res += *(LUTBasePt11 + *distCalVec_vec1pt11);
                            res += *(LUTBasePt12 + *distCalVec_vec1pt12);
                            res += *(LUTBasePt13 + *distCalVec_vec1pt13);
                            res += *(LUTBasePt14 + *distCalVec_vec1pt14);
                            res += *(LUTBasePt15 + *distCalVec_vec1pt15);
                            dist = res;
                            if (res >= distCalVec_remainPartDis) {  // Prune redundant vector slices
                                insertDist = false;
                            }
                        }
#ifdef MODULE_PERF_EVAL
                        exec_times_me_base[CAL_DISTANCE] += perfcounter_get() - exec_time_me;
                        exec_time_me = perfcounter_get();
#endif
                        if (insertDist) {  // Prune redundant distance insertion
                            mutex_lock(mutex_pqueue_update);
                            if (clusterSearching_pqElemSize < neighborAmt) {
                                clusterSearching_pqElems[clusterSearching_pqElemSize].pri = dist, clusterSearching_pqElems[clusterSearching_pqElemSize].val = pId << 8;  // clusterSearching_pqElems[clusterSearching_pqElemSize].val = pId << 8 * sizeof(clusterSearching_pqElems[clusterSearching_pqElemSize].pos);
                                pqueue_insert(clusterSearching_pq, clusterSearching_pqElems + clusterSearching_pqElemSize);
                                ++clusterSearching_pqElemSize;
                            } else {
                                pqueue_elem_t *pqTop = pqueue_peek(clusterSearching_pq);
                                if (pqTop->pri > dist) {
                                    pqTop->val = pId << 8;  // pqTop->val = pId << 8 * sizeof(pqTop->pos);
                                    pqTop->pri = dist;
                                    pqueue_pop(clusterSearching_pq);
                                    pqueue_insert(clusterSearching_pq, pqTop);
                                }
                            }
                            mutex_unlock(mutex_pqueue_update);
                        }
#ifdef MODULE_PERF_EVAL
                        exec_times_me_base[TOPK_SORT] += perfcounter_get() - exec_time_me;
                        exec_time_me = perfcounter_get();
#endif
                        if ((localPId & 0x3F) == neighborAmt) {  // Updating period: 64. Update the pruning threshold of `distCalVec`. Not mind the racing risk since it does not affect the correctness
                                                                 // Note: the updating period should be no smaller than neighborAmt! Thus, the right value of the judgment should be no smaller than `neighborAmt`
                            pqueue_elem_t *pqTop = pqueue_peek(clusterSearching_pq);
                            distCalVec_remainPartDis = pqTop->pri - distCalVec_prunePartDis;
                        }
                        pointAddr += pointStride;
                    }
                }
                localClusterIDIdxBuf = localClusterIDIdxBufEnd;
                localClusterIDIdxBufEnd = min(localClusterIDIdxEnd, localClusterIDIdxBufEnd + localClusterIDsStride);
                mram_read(localClusterIDs + localClusterIDIdxBuf, &localClusterIDsBuf, sizeof(uint64_t));
            }
            barrier_wait(&barrier_tasklets);
#ifdef MODULE_PERF_EVAL
            exec_times_me_base[AFFILIATE_OPS] += perfcounter_get() - exec_time_me;
            exec_time_me = perfcounter_get();
#endif
            // The third implementation of saving results out of order. Most parts of this version are implemented with multiple tasklets
            __mram_ptr pqueue_elem_t_mram *neighborsWriteBase = neighborsWrite + queryNeighborStartAddr;
            __dma_aligned uint8_t *clusterSearching_pqElemsFrom = (uint8_t *)(clusterSearching_pqElems);
            __mram_ptr uint8_t *neighborsWriteTo = (__mram_ptr uint8_t *)(neighborsWriteBase);
            for (uint32_t neighborsWritePos = neighborCopyStart; neighborsWritePos < neighborSize; neighborsWritePos += neighborCopyStride) {
                mram_write(clusterSearching_pqElemsFrom + neighborsWritePos, neighborsWriteTo + neighborsWritePos, min(MRAM_WRITE_MAX_SIZE, neighborSize - neighborsWritePos));
            }
            for (uint32_t neighborsWritePos = clusterSearching_pqElemSize + me(); neighborsWritePos < neighborAmt; neighborsWritePos += NR_TASKLETS) {
                mram_write(&neighborsDefaultValue, neighborsWriteBase + neighborsWritePos, sizeof(pqueue_elem_t_mram));
            }
            barrier_wait(&barrier_tasklets);
            if (me() == 0) {
                clusterSearching_pq->size = 1;
                clusterSearching_pqElemSize = 0;
            }  // End of the third implementation
#ifdef MODULE_PERF_EVAL
            exec_times_me_base[TOPK_SAVING] += perfcounter_get() - exec_time_me;
            exec_time_me = perfcounter_get();
#endif
            barrier_wait(&barrier_tasklets);
            neighborsWrite += neighborsWriteStride;
            localClusterIDIdxStart = localClusterIDIdxEnd;
        }
    }
    fsb_free(residualBufAllocator, residualBuf);
    fsb_free(centroidBufAllocator, centroidBuf);
    fsb_free(queryBufAllocator, queryBuf);
    fsb_free(curPointBufAllocator, curPointBuf);
    fsb_free(CBBufAllocator, CBSliceBuf);
    if (me() == 0) {
        pqueue_free(clusterSearching_pq, &clusterSearching_pq_allocator);
        fsb_free(clusterSearching_pqElemsAllocator, clusterSearching_pqElems);
        fsb_free(constrLUT_partDisAllocator, constrLUT_partDis);
    }
#if (defined MODULE_PERF_EVAL || defined CYCLE_PERF_EVAL)
    exec_times_me_base[AFFILIATE_OPS] += perfcounter_get() - exec_time_me;
    exec_time_me = perfcounter_get();
#endif
}
