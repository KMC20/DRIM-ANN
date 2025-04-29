/*
Author: Chen Mingkai
Date: 2024/2/29
Function: Auxiliary functions for general modules of the host side.
*/

#ifndef TOOLS_H
#define TOOLS_H
#include "request.h"
#include <string.h>
#include <vector>
#include <pqueue.h>
#include <cstdlib>
#include <memory>
#include <map>
#include <algorithm>
#include <cmath>
#include <assert.h>
#include "publicDefs.h"
#include <numeric>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <mutex>

/***************************************************************************************** This part is appended for load balance optimization ****************************************************************************************/
class ClusterLayout {
    public:
    ADDRTYPE dpuId;
    ADDRTYPE clusterId;
    ADDRTYPE startSliceId;
    ADDRTYPE endSliceId;

    ClusterLayout() {}
    ClusterLayout(const ADDRTYPE &dpuIdIn, const ADDRTYPE &clusterIdIn, const ADDRTYPE &startSliceIdIn, const ADDRTYPE &endSliceIdIn): dpuId(dpuIdIn), clusterId(clusterIdIn), startSliceId(startSliceIdIn), endSliceId(endSliceIdIn) {}
    ~ClusterLayout() {}
};
class ClusterIndex {
    public:
    ADDRTYPE dpuId;
    ADDRTYPE localClusterId;

    ClusterIndex(const ADDRTYPE &dpuIdIn, const ADDRTYPE &localClusterIdIn): dpuId(dpuIdIn), localClusterId(localClusterIdIn) {}
    ~ClusterIndex() {}
};
/***************************************************************************************** End of appended part for load balance optimization *****************************************************************************************/

uint64_t Log(const uint64_t k);
void mergeTopkQueueSort(const pqueue_elem_t_mram *topKQueues, const ADDRTYPE k, const ADDRTYPE nq, pqueue_elem_t_mram *topK);  // Each of topKQueues: ascent order!
void mergeTopkPairByPair(const pqueue_elem_t_mram *topKQueues, const ADDRTYPE k, const ADDRTYPE nq, pqueue_elem_t_mram *topK);  // Each of topKQueues: ascent order!
void IVFsearch(const float *const queriesF, const ADDRTYPE &queryAmt, size_t dimAmt, const faiss::Index *const quantizer, const ADDRTYPE &nprobe, const ADDRTYPE &pointAmtPerSlice, std::vector<std::vector<ADDRTYPE>> &selClusterIds, std::vector<std::vector<ADDRTYPE>> &selClusterSizes, std::vector<std::vector<ADDRTYPE>> &distribQueryIds, std::vector<std::vector<std::pair<ADDRTYPE, ADDRTYPE>>> &queryDPUIds, const std::vector<std::vector<ADDRTYPE>> &clusterAddrs, const std::vector<std::vector<CLUSTER_SIZES_TYPE>> &clusterSizes, const UINT64 *const latency, const std::vector<CLUSTER_SIZES_TYPE> &clusterSizesFlat, const std::vector<std::vector<ClusterLayout>>&clusterLayout, std::vector<std::vector<ClusterIndex>> &clusterDirectory, std::vector<std::vector<ADDRTYPE>> &DPUClusterAddrs, std::vector<std::vector<CLUSTER_SIZES_TYPE>> &DPUClusterSizes);  // Note: the order of elements of the input `clusterDirectory[clusterId]` may be changed in this function!
void mergeTopk(const ADDRTYPE globalQueryId, const ADDRTYPE queryId, const std::vector<std::vector<std::pair<ADDRTYPE, ADDRTYPE>>> &queryDPUIds, std::vector<std::vector<pqueue_elem_t_mram>> &neighborsDPU, const uint32_t &neighborAmt, std::vector<pqueue_elem_t_mram> &neighbors);

#endif