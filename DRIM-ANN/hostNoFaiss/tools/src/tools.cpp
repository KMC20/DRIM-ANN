/*
Author: KMC20
Date: 2024/2/29
Function: Auxiliary functions for distance calculation and top-k sorting of the host side.
*/

#include "tools.h"

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

uint64_t Log(const uint64_t k) {
    uint64_t shiftBits = 32;
    uint64_t shiftVaryBits = 16;
    while (shiftVaryBits) {
        if (k >> shiftBits) {
            shiftBits += shiftVaryBits;
        } else {
            shiftBits -= shiftVaryBits;
        }
        shiftVaryBits >>= 1;
    }
    return shiftBits - (uint64_t)(!(k >> shiftBits));
}

// Complexity: ideal: O((k + nq) * log(k)); worst: O(k * nq * log(k))!
void mergeTopkQueueSort(const pqueue_elem_t_mram *topKQueues, const ADDRTYPE k, const ADDRTYPE nq, pqueue_elem_t_mram *topK) {  // if (k * (nq -1) > (k + nq) * Log(k))
    const pqueue_elem_t_mram *topKQueuesEnd = topKQueues + k * nq;
    pqueue_elem_t *pqElems = (pqueue_elem_t *)malloc(k * sizeof(pqueue_elem_t));
    pqueue_t *pq = pqueue_init(k, cmp_pri, get_pri, set_pri, get_pos, set_pos);  // Note: expect this to be a max heap
    ADDRTYPE pqElemSize = 0;
    ADDRTYPE *queuePos = (ADDRTYPE *)malloc(nq * sizeof(ADDRTYPE));
    memset(queuePos, 0, nq * sizeof(ADDRTYPE));
    while (pqElemSize < k) {
        for (pqueue_elem_t_mram *qAddr = (pqueue_elem_t_mram *)(topKQueues) + queuePos[0]; qAddr < topKQueuesEnd; qAddr += k)
            if (pqElemSize < k) {
                pqElems[pqElemSize].pri = qAddr->pri, pqElems[pqElemSize].val = qAddr->val;
                pqueue_insert(pq, pqElems + pqElemSize);
                ++pqElemSize;
            } else {
                pqueue_elem_t *pqTop = (pqueue_elem_t *)pqueue_peek(pq);
                if (pqTop->pri > qAddr->pri) {
                    memcpy(pqTop, qAddr, sizeof(pqueue_elem_t));
                    pqueue_pop(pq);
                    pqueue_insert(pq, pqTop);
                }
            }
        ++(queuePos[0]);
    }
    for (ADDRTYPE qId = 1; qId < nq; ++qId)
        queuePos[qId] = queuePos[0];
    ADDRTYPE startQueueId = 0, endQueueId = 0;
    pqueue_elem_t_mram *startQueueAddr = (pqueue_elem_t_mram *)topKQueues;
    do {
        pqueue_elem_t *pqTop = (pqueue_elem_t *)pqueue_peek(pq);
        pqueue_elem_t_mram *queueAddr = startQueueAddr + queuePos[startQueueId];
        while (queuePos[startQueueId] < k && pqTop->pri > queueAddr->pri) {
            memcpy(pqTop, queueAddr, sizeof(pqueue_elem_t));
            pqueue_pop(pq);
            pqueue_insert(pq, pqTop);
            pqTop = (pqueue_elem_t *)pqueue_peek(pq);
            ++(queuePos[startQueueId]);
            ++queueAddr;
            endQueueId = startQueueId;
        }
        if (++startQueueId >= nq)
            startQueueId = 0, startQueueAddr = (pqueue_elem_t_mram *)topKQueues;
        else
            startQueueAddr += k;
    } while (startQueueId != endQueueId);
    free(queuePos);
    pqueue_elem_t_mram *topKPos = topK + (k - 1);
    for (ADDRTYPE topKCnt = 0; topKCnt < k; ++topKCnt, --topKPos) {
        pqueue_elem_t *pqTop = (pqueue_elem_t *)pqueue_pop(pq);
        memcpy(topKPos, pqTop, sizeof(pqueue_elem_t_mram));
    }
    pqueue_free(pq);
    free(pqElems);
}

// Complexity: O(k * nq)
void mergeTopkPairByPair(const pqueue_elem_t_mram *topKQueues, const ADDRTYPE k, const ADDRTYPE nq, pqueue_elem_t_mram *topK) {  // if (k * (nq -1) < (k + nq) * Log(k))
    pqueue_elem_t_mram *nextQueue = (pqueue_elem_t_mram *)topKQueues + k, *topKEnd = topK + k;
    for (pqueue_elem_t_mram *q1 = (pqueue_elem_t_mram *)topKQueues, *q2 = nextQueue, *q3 = topK; q3 < topKEnd; ++q3)
        memcpy(q3, q1->pri < q2->pri ? q1++ : q2++, sizeof(pqueue_elem_t_mram));
    for (ADDRTYPE qId = 2; qId < nq; ++qId) {
        nextQueue += k;
        pqueue_elem_t_mram *q1 = nextQueue + k - 1, *q2 = topKEnd - 1;
        for (ADDRTYPE discardAmt = 0; discardAmt < k; ++discardAmt)
            if (q1->pri < q2->pri)
                --q2;
            else
                --q1;
        for (pqueue_elem_t_mram *q3 = topKEnd - 1; q1 >= nextQueue; )
            if (q1->pri < q2->pri)
                memcpy(q3--, q2--, sizeof(pqueue_elem_t_mram));
            else
                memcpy(q3--, q1--, sizeof(pqueue_elem_t_mram));
    }
}

// Complexity: O(k * nq)
void mergePairInPlace(const pqueue_elem_t_mram *topKQueue, const ADDRTYPE k, pqueue_elem_t_mram *topK) {  // if (k * (nq -1) < (k + nq) * Log(k))
    pqueue_elem_t_mram *topKEnd = topK + k;
    pqueue_elem_t_mram *q1 = (pqueue_elem_t_mram *)topKQueue + k - 1, *q2 = topKEnd - 1;
    for (ADDRTYPE discardAmt = 0; discardAmt < k; ++discardAmt)
        if (q1->pri < q2->pri)
            --q2;
        else
            --q1;
    if (q2 < topK) {
        memcpy(topK, topKQueue, sizeof(pqueue_elem_t_mram) * k);
    } else {
        for (pqueue_elem_t_mram *q3 = topKEnd - 1; q1 >= topKQueue && q2 >= topK; )
            if (q1->pri < q2->pri)
                memcpy(q3--, q2--, sizeof(pqueue_elem_t_mram));
            else
                memcpy(q3--, q1--, sizeof(pqueue_elem_t_mram));
        if (q1 >= topKQueue)
            memcpy(topK, topKQueue, sizeof(pqueue_elem_t_mram) * (q1 - topKQueue + 1));
    }
}

// Complexity: O(k * nq)
void mergePairOutOfPlace(const pqueue_elem_t_mram *const topKQueue1, const pqueue_elem_t_mram *const topKQueue2, const ADDRTYPE k, pqueue_elem_t_mram *topK) {  // if (k * (nq -1) < (k + nq) * Log(k))
    pqueue_elem_t_mram *topKEnd = topK + k;
    for (pqueue_elem_t_mram *q1 = (pqueue_elem_t_mram *)topKQueue1, *q2 = (pqueue_elem_t_mram *)topKQueue2, *q3 = topK; q3 < topKEnd; ++q3)
        memcpy(q3, q1->pri < q2->pri ? q1++ : q2++, sizeof(pqueue_elem_t_mram));
}

inline ADDRTYPE getDPUId(const ADDRTYPE &clusterId, const uint32_t &nr_all_dpus) {
    return clusterId % nr_all_dpus;
}
inline ADDRTYPE getDPUId(const ADDRTYPE &clusterId, const uint32_t &nr_all_dpus, const std::vector<ADDRTYPE> &clusterIDsStart) {
    ADDRTYPE nr_dpu = clusterId % nr_all_dpus;
    ADDRTYPE next_dpu;
    while ((next_dpu = nr_dpu + 1) < nr_all_dpus && clusterIDsStart[next_dpu] <= clusterId)
        ++nr_dpu;
    while (clusterId < clusterIDsStart[nr_dpu])  // clusterIDsStart[0] should always be 0!
        --nr_dpu;
    return nr_dpu;
}
inline ADDRTYPE getDPUGId(const ADDRTYPE &clusterId, const uint32_t &nr_groups, const std::vector<ADDRTYPE> &clusterIDsStart) {
    if (clusterIDsStart.size() < 2)
        return 0;
    ADDRTYPE nr_group = clusterId / clusterIDsStart[1];
    ADDRTYPE next_group;
    while ((next_group = nr_group + 1) < nr_groups && clusterIDsStart[next_group] <= clusterId)
        ++nr_group;
    while (clusterId < clusterIDsStart[nr_group])  // clusterIDsStart[0] should always be 0!
        --nr_group;
    return nr_group;
}

class coarseScheduler_pqType {
    public:
    UINT64 dpuHeat;
    ADDRTYPE dpuIdx;
    coarseScheduler_pqType(): dpuHeat(0) {}
    friend bool operator< (const coarseScheduler_pqType &a, const coarseScheduler_pqType &b) {
        return a.dpuHeat > b.dpuHeat;
    }
    ~coarseScheduler_pqType() {}
};
#define NR_JOB_PER_RANK 64
// Note: the order of elements of the input `clusterDirectory[clusterId]` may be changed in this function!
void coarseScheduler(const ADDRTYPE &clusterId, std::vector<std::vector<ClusterIndex>> &clusterDirectory, const std::vector<CLUSTER_SIZES_TYPE> &clusterSizesFlat, const float &pointAmtPerSlice, const std::vector<std::vector<ClusterLayout>>&clusterLayout, std::vector<ClusterLayout> &selDPUs, const std::vector<UINT64> &heatDPUs) {
    // Greedy assignment
    sort(clusterDirectory[clusterId].begin(), clusterDirectory[clusterId].end(), [&heatDPUs] (const ClusterIndex &a, const ClusterIndex&b) { return heatDPUs[a.dpuId] < heatDPUs[b.dpuId]; });
    std::vector<std::pair<ADDRTYPE, ADDRTYPE>> clusterSliceDPUIdxs(std::ceil(clusterSizesFlat[clusterId] / pointAmtPerSlice), std::make_pair(-1, -1));  // <dpuId, localClusterId>
    ADDRTYPE remainSliceAmt = clusterSliceDPUIdxs.size();
    for (auto &clusterSliceDPUIdx: clusterDirectory[clusterId]) {
        for (ADDRTYPE sliceStartId = clusterLayout[clusterSliceDPUIdx.dpuId][clusterSliceDPUIdx.localClusterId].startSliceId, sliceEndId = clusterLayout[clusterSliceDPUIdx.dpuId][clusterSliceDPUIdx.localClusterId].endSliceId; sliceStartId < sliceEndId; ++sliceStartId) {
            if (clusterSliceDPUIdxs[sliceStartId].first + 1 < 1) {  // Avoid `clusterSliceDPUIdxs[sliceStartId].first < 0` since the left value is non-negative, which would overflow with 1 added if it is in the initial state
                clusterSliceDPUIdxs[sliceStartId] = std::make_pair(clusterSliceDPUIdx.dpuId, clusterSliceDPUIdx.localClusterId);
                --remainSliceAmt;
                if (!remainSliceAmt)
                    break;
            }
        }
        if (!remainSliceAmt)
            break;
    }
    assert(remainSliceAmt == 0 && "Incomplete cluster in `coarseScheduler` in tools.cpp! Exit now!\n");
    {
        ADDRTYPE DPUId = clusterSliceDPUIdxs[0].first;
        ADDRTYPE startSliceId = 0;
        size_t clusterSliceDPUIdxAmt = clusterSliceDPUIdxs.size();
        for (size_t clusterSliceDPUIdx = 1; clusterSliceDPUIdx < clusterSliceDPUIdxAmt; ++clusterSliceDPUIdx) {
            if (clusterSliceDPUIdxs[clusterSliceDPUIdx].first != DPUId) {
                selDPUs.push_back(ClusterLayout(DPUId, clusterSliceDPUIdxs[clusterSliceDPUIdx - 1].second, startSliceId, clusterSliceDPUIdx));  // Note: the second cluster id is a local one instead of a global one
                DPUId = clusterSliceDPUIdxs[clusterSliceDPUIdx].first;
                startSliceId = clusterSliceDPUIdx;
            }
        }
        selDPUs.push_back(ClusterLayout(DPUId, clusterSliceDPUIdxs[clusterSliceDPUIdxAmt - 1].second, startSliceId, clusterSliceDPUIdxAmt));
    }
}

template<typename Tq, typename Tc, typename Td, typename Ti>
void knn_L2sqrAVX512(const Tq* queries, const Tc* centroids, size_t dim, size_t queryAmt, size_t centroidAmt, size_t k, Td* selClusterDists, Ti* selClusterIds) {
    const size_t avxSize = 512;
    const uint64_t dimU64 = dim;
    const size_t distStride = (avxSize >> 3) / sizeof(Tq);
    const size_t distHalfStride = distStride >> 1;
    const size_t paddingPointSize = std::ceil(dimU64 / static_cast<float>(distStride)) * static_cast<float>(distStride);
    const Td initDist = 0;
    #pragma omp parallel for num_threads(NR_JOB_PER_RANK)
    for (size_t queryId = 0; queryId < queryAmt; ++queryId) {
        Tq *queryPt = const_cast<Tq *>(queries) + queryId * dimU64;
        std::vector<std::pair<Td, Ti>> dists(centroidAmt);
        std::vector<uint16_t> dist(paddingPointSize);
        __m512i partQuery;
        __m512i partQuery_lo;
        __m512i partQuery_hi;
        __m512i partCentroid;
        __m512i partCentroid_lo;
        __m512i partCentroid_hi;
        __m512i partDiff_lo;
        __m512i partDiff_hi;
        __m512i partDist_lo;
        __m512i partDist_hi;
        for (size_t clusterId = 0; clusterId < centroidAmt; ++clusterId) {
            dists[clusterId].first = 0;
            dists[clusterId].second = clusterId;
            Tc *centroidPt = const_cast<Tc *>(centroids) + clusterId * dimU64;
            for (size_t dimCnt = 0; dimCnt < dim; dimCnt += distHalfStride) {
                // Load  
                partQuery = _mm512_loadu_si512(reinterpret_cast<__m512i *>(queryPt + dimCnt));  
                partCentroid = _mm512_loadu_si512(reinterpret_cast<__m512i *>(centroidPt + dimCnt));
                // Extend
                partQuery_lo = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(partQuery, 0));
                partQuery_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(partQuery, 1));
                partCentroid_lo = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(partCentroid, 0));
                partCentroid_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(partCentroid, 1));
                // Residual
                partDiff_lo = _mm512_sub_epi16(partQuery_lo, partCentroid_lo);
                partDiff_hi = _mm512_sub_epi16(partQuery_hi, partCentroid_hi);
                // Multiplication  
                partDist_lo = _mm512_mullo_epi16(partDiff_lo, partDiff_lo);
                partDist_hi = _mm512_mullo_epi16(partDiff_hi, partDiff_hi);
                // Store  
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(dist.data() + dimCnt), partDist_lo);
                dimCnt += distHalfStride;
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(dist.data() + dimCnt), partDist_hi);
            }
            dists[clusterId].first = std::accumulate(dist.begin(), dist.begin() + dim, initDist);
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [] (const std::pair<Td, Ti> &a, const std::pair<Td, Ti> &b) { return a.first < b.first; });
        Td *selClusterDist = selClusterDists + queryId * k;
        Ti *selClusterId = selClusterIds + queryId * k;
        for (size_t probe = 0; probe < k; ++probe) {
            selClusterDist[probe] = dists[probe].first, selClusterId[probe] = dists[probe].second;
        }
    }
}
template<typename Tq, typename Tc, typename Td, typename Ti>
void knn_L2sqr(const Tq* queries, const Tc* centroids, size_t dim, size_t queryAmt, size_t centroidAmt, size_t k, Td* selClusterDists, Ti* selClusterIds) {
    const size_t avxSize = 256;
    const uint64_t dimU64 = dim;
    const size_t distStride = (avxSize >> 3) / sizeof(Tq);
    const size_t distHalfStride = distStride >> 1;
    const size_t paddingPointSize = std::ceil(dimU64 / static_cast<float>(distStride)) * static_cast<float>(distStride);
    const Td initDist = 0;
    #pragma omp parallel for num_threads(NR_JOB_PER_RANK)
    for (size_t queryId = 0; queryId < queryAmt; ++queryId) {
        Tq *queryPt = const_cast<Tq *>(queries) + queryId * dimU64;
        std::vector<std::pair<Td, Ti>> dists(centroidAmt);
        std::vector<uint16_t> dist(paddingPointSize);
        __m256i partQuery;
        __m256i partQuery_lo;
        __m256i partQuery_hi;
        __m256i partCentroid;
        __m256i partCentroid_lo;
        __m256i partCentroid_hi;
        __m256i partDiff_lo;
        __m256i partDiff_hi;
        __m256i partDist_lo;
        __m256i partDist_hi;
        for (size_t clusterId = 0; clusterId < centroidAmt; ++clusterId) {
            dists[clusterId].first = 0;
            dists[clusterId].second = clusterId;
            Tc *centroidPt = const_cast<Tc *>(centroids) + clusterId * dimU64;
            for (size_t dimCnt = 0; dimCnt < dim; dimCnt += distHalfStride) {
                // Load
                partQuery = _mm256_loadu_si256(reinterpret_cast<__m256i *>(queryPt + dimCnt));
                partCentroid = _mm256_loadu_si256(reinterpret_cast<__m256i *>(centroidPt + dimCnt));
                // Extend
                partQuery_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(partQuery, 0));
                partQuery_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(partQuery, 1));
                partCentroid_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(partCentroid, 0));
                partCentroid_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(partCentroid, 1));
                // Residual
                partDiff_lo = _mm256_sub_epi16(partQuery_lo, partCentroid_lo);
                partDiff_hi = _mm256_sub_epi16(partQuery_hi, partCentroid_hi);
                // Multiplication
                partDist_lo = _mm256_mullo_epi16(partDiff_lo, partDiff_lo);
                partDist_hi = _mm256_mullo_epi16(partDiff_hi, partDiff_hi);
                // Store  
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dist.data() + dimCnt), partDist_lo);
                dimCnt += distHalfStride;
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dist.data() + dimCnt), partDist_hi);
            }
            dists[clusterId].first = std::accumulate(dist.begin(), dist.begin() + dim, initDist);
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [] (const std::pair<Td, Ti> &a, const std::pair<Td, Ti> &b) { return a.first < b.first; });
        Td *selClusterDist = selClusterDists + queryId * k;
        Ti *selClusterId = selClusterIds + queryId * k;
        for (size_t probe = 0; probe < k; ++probe) {
            selClusterDist[probe] = dists[probe].first, selClusterId[probe] = dists[probe].second;
        }
    }
}

void IVFsearch(const ELEMTYPE *const queries, const ADDRTYPE &queryAmt, size_t dimAmt, const CENTROIDS_TYPE *const centroids, const ADDRTYPE &centroidAmt, const ADDRTYPE &nprobe, const ADDRTYPE &pointAmtPerSlice, std::vector<std::vector<ADDRTYPE>> &selClusterIds, std::vector<std::vector<ADDRTYPE>> &selClusterSizes, std::vector<std::vector<ADDRTYPE>> &distribQueryIds, std::vector<std::vector<std::pair<ADDRTYPE, ADDRTYPE>>> &queryDPUIds, const std::vector<std::vector<ADDRTYPE>> &clusterAddrs, const std::vector<std::vector<CLUSTER_SIZES_TYPE>> &clusterSizes, const UINT64 *const latency, const std::vector<CLUSTER_SIZES_TYPE> &clusterSizesFlat, const std::vector<std::vector<ClusterLayout>>&clusterLayout, std::vector<std::vector<ClusterIndex>> &clusterDirectory, std::vector<std::vector<ADDRTYPE>> &DPUClusterAddrs, std::vector<std::vector<CLUSTER_SIZES_TYPE>> &DPUClusterSizes) {
    // metric_type == METRIC_L2
    // we see the distances and labels as heaps
    std::unique_ptr<ADDRTYPE[]> idx(new ADDRTYPE[queryAmt * nprobe]);
    std::unique_ptr<pqueue_pri_t[]> selClusterDists(new pqueue_pri_t[queryAmt * nprobe]);
    knn_L2sqr(queries, centroids, dimAmt, queryAmt, centroidAmt, nprobe, selClusterDists.get(), idx.get());
    // knn_L2sqrAVX512(queries, centroids, dimAmt, queryAmt, centroidAmt, nprobe, selClusterDists.get(), idx.get());
    uint32_t nr_all_dpus = selClusterIds.size();
    float pointAmtPerSliceF = pointAmtPerSlice;
    std::vector<UINT64> heatDPUs(nr_all_dpus, 0);
    for (ADDRTYPE qId = 0, idxId = 0; qId < queryAmt; ++qId) {
        std::map<ADDRTYPE, ADDRTYPE> queryCIds;
        for (ADDRTYPE cId = 0; cId < nprobe; ++cId, ++idxId) {
            ADDRTYPE selCId = idx[idxId];
            std::vector<ClusterLayout> selDPUs;
            coarseScheduler(selCId, clusterDirectory, clusterSizesFlat, pointAmtPerSliceF, clusterLayout, selDPUs, heatDPUs);
            for (auto &selDPUInfo: selDPUs) {
                ADDRTYPE nr_dpu = selDPUInfo.dpuId;
                selClusterIds[nr_dpu].push_back(selDPUInfo.clusterId);
                DPUClusterAddrs[nr_dpu].push_back(clusterAddrs[nr_dpu][selDPUInfo.clusterId] + (selDPUInfo.startSliceId - clusterLayout[nr_dpu][selDPUInfo.clusterId].startSliceId) * pointAmtPerSlice);
                DPUClusterSizes[nr_dpu].push_back(selDPUInfo.endSliceId != clusterLayout[nr_dpu][selDPUInfo.clusterId].endSliceId ?
                                                        (selDPUInfo.endSliceId - selDPUInfo.startSliceId) * pointAmtPerSlice :
                                                        clusterSizes[nr_dpu][selDPUInfo.clusterId] - (selDPUInfo.startSliceId - clusterLayout[nr_dpu][selDPUInfo.clusterId].startSliceId) * pointAmtPerSlice);
                heatDPUs[nr_dpu] += latency[resiCal] + latency[lutCal] + DPUClusterSizes[nr_dpu][DPUClusterSizes[nr_dpu].size() - 1] * (latency[distCal] + latency[topkSort]);
                if (queryCIds.find(nr_dpu) != queryCIds.end()) {
                    ++queryCIds[nr_dpu];
                } else {
                    queryCIds[nr_dpu] = 1;
                }
            }
        }
        for (auto &queryCId: queryCIds) {
            selClusterSizes[queryCId.first].push_back(queryCId.second);
            distribQueryIds[queryCId.first].push_back(qId);
            queryDPUIds[qId].push_back(std::move(std::make_pair(queryCId.first, distribQueryIds[queryCId.first].size())));  // <dpuId, clusterAmtPerDPU> -- For mergeTopk
        }
    }
}

// The input topk arrays can be out of order
void mergeTopk(const ADDRTYPE globalQueryId, const ADDRTYPE queryId, const std::vector<std::vector<std::pair<ADDRTYPE, ADDRTYPE>>> &queryDPUIds, std::vector<std::vector<pqueue_elem_t_mram>> &neighborsDPU, const uint32_t &neighborAmt, std::vector<pqueue_elem_t_mram> &neighbors) {
    pqueue_elem_t_mram *localNeighborStart = (pqueue_elem_t_mram *)(neighbors.data()) + globalQueryId * neighborAmt;
    if (queryDPUIds[queryId].size() < 2) {
        std::sort(neighborsDPU[queryDPUIds[queryId][0].first].begin() + (queryDPUIds[queryId][0].second - 1) * neighborAmt, neighborsDPU[queryDPUIds[queryId][0].first].begin() + queryDPUIds[queryId][0].second * neighborAmt, [](const pqueue_elem_t_mram &a, const pqueue_elem_t_mram &b) { return a.pri < b.pri; });
        memcpy(localNeighborStart, (const pqueue_elem_t_mram *)(neighborsDPU[queryDPUIds[queryId][0].first].data()) + (queryDPUIds[queryId][0].second - 1) * neighborAmt, sizeof(pqueue_elem_t_mram) * neighborAmt);
        return;
    }
    std::vector<pqueue_elem_t_mram> concatBuf(queryDPUIds[queryId].size() * neighborAmt);
    pqueue_elem_t_mram *concatBufPt = concatBuf.data();
    for (auto &queryDPUId: queryDPUIds[queryId]) {
        memcpy(concatBufPt, (const pqueue_elem_t_mram *)(neighborsDPU[queryDPUId.first].data()) + (queryDPUId.second - 1) * neighborAmt, sizeof(pqueue_elem_t_mram) * neighborAmt);
        concatBufPt += neighborAmt;
    }
    std::partial_sort(concatBuf.begin(), concatBuf.begin() + neighborAmt, concatBuf.end(), [](const pqueue_elem_t_mram &a, const pqueue_elem_t_mram &b) { return a.pri < b.pri; });
    memcpy(localNeighborStart, concatBuf.data(), sizeof(pqueue_elem_t_mram) * neighborAmt);
}
