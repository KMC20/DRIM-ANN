/*
Author: KMC20
Date: 2024/8/20
Function: Generate the cluster layout on DRAM-PIMs offline for DRIM-ANN.
Usage:
 > g++ genClusterLayout.cpp ../DRIM-ANN/host/tools/src/*.cpp -std=c++11 -I ../DRIM-ANN/common/inc -I ../DRIM-ANN/host/tools/inc -fopenmp -march=native -o genClusterLayout
 > ./genClusterLayout -D 128 -K 10 -C 64 -Q 10000 -M 8 -U 2543 -S 2560 -p ../DRIM-ANN/offlineFiles/sift100m.clusters -q ../DRIM-ANN/datasets/SIFT100M/pured_bigann_query_sample.bvecs -c ../DRIM-ANN/offlineFiles/sift100m.centroids -b ../DRIM-ANN/offlineFiles/sift100m.codebook -a ../DRIM-ANN/offlineFiles/sift100m.clusterSizes -l ../DRIM-ANN/offlineFiles/sift100m.layout
 > rm genClusterLayout
*/

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include "request.h"
#include <map>
#include <memory>
#include <cmath>
#include "tools.h"
#include <numeric>
#include <utility>
#include <queue>
#include <functional>
#include <regex>

#define NR_JOB_PARALLEL 64
#define FLOAT_ERROR 1e-5

using UINT64 = unsigned long long;

class ClusterHeatInfo {
    public:
    double thres;
    ADDRTYPE dupTimes;  // Start from 0 (i.e. no copies)
    ADDRTYPE prefixNlist;  // = thres * nlist

    ClusterHeatInfo(const double &thresIn, const ADDRTYPE &dupTimesIn, const ADDRTYPE &prefixNlistIn): thres(thresIn), dupTimes(dupTimesIn), prefixNlist(prefixNlistIn) {}
    ~ClusterHeatInfo() {}
};


ADDRTYPE getPointsAmount(const char *const pointsFileName, const uint32_t pointSize) {
    FILE *fp = fopen(pointsFileName, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open the input point file: %s! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    ADDRTYPE pointsAmt = ftell(fp) / pointSize;
    fclose(fp);
    return pointsAmt;
}

template <typename T>
void load1DPointsFromFile(const char *const pointsFileName, std::vector<T> &points) {
    FILE *fp = fopen(pointsFileName, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open the input point file: %s! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    long long int pointsElemSize = ftell(fp) / sizeof(T);
    fseek(fp, 0, SEEK_SET);
    if (fread(points.data(), sizeof(T), pointsElemSize, fp) == 0 && pointsElemSize != 0) {
        fclose(fp);
        fprintf(stderr, "The input point file: %s is an empty file! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fclose(fp);
}

void saveDataToFile(const char *const dataFileName, const void *data, const size_t size, const size_t nmemb) {
    FILE *fp = fopen(dataFileName, "ab+");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open the output point file: %s! Exit now!\n", dataFileName);
        exit(-1);
    }
    fwrite(data, size, nmemb, fp);
    fclose(fp);
}

void clearFile(const char *const dataFileName) {
    FILE *fp = fopen(dataFileName, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open the data file: %s! Exit now!\n", dataFileName);
        exit(-1);
    }
    fclose(fp);
}

void parseDict(std::vector<ClusterHeatInfo> &clusterHeatThresInfos, const std::string &clusterHeatThreses, const ADDRTYPE &clusterAmt) {
    std::regex pat("([[:digit:]]+.[[:digit:]]+):([[:digit:]]+)");
    std::smatch curObj;
    std::string::const_iterator clusterHeatThresesStart = clusterHeatThreses.begin();
    std::string::const_iterator clusterHeatThresesEnd = clusterHeatThreses.end();
    ADDRTYPE prefixClusterAmt = 0;
    while (regex_search(clusterHeatThresesStart, clusterHeatThresesEnd, curObj, pat)) {
        double thres = std::stof(curObj[1]);
        prefixClusterAmt += std::floor(thres * clusterAmt);
        assert(prefixClusterAmt <= clusterAmt && "Error input cluster heat thresholds! Exit now!");
        clusterHeatThresInfos.push_back(ClusterHeatInfo(thres, std::stoi(curObj[2]), prefixClusterAmt));
        clusterHeatThresesStart = curObj[0].second;
    }
}

__attribute__((noreturn)) static void usage(FILE *f, int exit_code, const char *exec_name) {
    /* clang-format off */
    fprintf(f,
            "\nusage: %s [-q <queries_path>] [-c <centroids_path>] [-b <codebook_path>] [-a <cluster_size_path>] [-l <cluster_layout_result_path>] [-D <number_of_dimension>] [-K <number_of_neighbors>] [-C <number_of_searched_clusters>] [-Q <number_of_queries_per_batch>] [-M <number_of_subvectors>] [-S <size_of_cluster_slices>] [-H <thresholds_of_cluster_heat>] [-U <number_of_mrams>]\n"
            "\n"
            "\t-q \tthe path to the query location (default: query.bin)\n"
            "\t-c \tthe path to the centroid location (default: centroids.bin)\n"
            "\t-b \tthe path to the codebook location (default: codebook.bin)\n"
            "\t-a \tthe path to the cluster size location (default: clusterSizes.bin)\n"
            "\t-l \tthe path to the cluster layout location (default: clusterLayout.bin)\n"
            "\t-D \tthe number of dimensions of input points (default: 128)\n"
            "\t-K \tthe number of neighbors (default: 10)\n"
            "\t-C \tthe number of searched clusters for each query (default: 1)\n"
            "\t-Q \tthe number of queries in each batch (default: 1)\n"
            "\t-M \tthe number of subvectors generated by each point (default: 8)\n"
            "\t-S \tthe size of split cluster slices. Unit: Byte (default: 1)\n"
            "\t-H \tthe thresholds of cluster heat (format: {0.05:5,0.15:2,0.15:1} means the hottest 5\% clusters are duplicated 5 times, the 5\%~20\% hottest for twice, the 20\%~35\% hottest for once, and others have no copies) (default: {})\n"
            "\t-U \tthe number of mram to used (default: DPU_ALLOCATE_ALL)\n"
            "\t-h \tshow the usage message\n",
            exec_name);
    /* clang-format on */
    exit(exit_code);
}

static void verify_path_exists(const char *path) {
    if (access(path, R_OK)) {
        fprintf(stderr, "path '%s' does not exist or is not readable (errno: %i)\n", path, errno);
        exit(EXIT_FAILURE);
    }
}

static void parse_args(int argc, char **argv, uint32_t *dimAmt, uint32_t *neighborAmt, uint32_t *sliceAmt, uint32_t *queryBatchSize, ADDRTYPE *nprobe, uint32_t *clusterSliceSize, uint32_t *nb_mram, std::string &clusterHeatThreses, std::string &queriesFileName, std::string &centroidsFileName, std::string &codebookFileName, std::string &clusterSizesFileName, std::string &clusterLayoutFileName) {
    int opt;
    extern char *optarg;
    while ((opt = getopt(argc, argv, "hD:K:C:Q:M:S:H:U:q:c:b:a:l:")) != -1) {
        switch (opt) {
            case 'q':
                queriesFileName = optarg;
                break;
            case 'D':
                *dimAmt = (uint32_t)atoi(optarg);
                break;
            case 'K':
                *neighborAmt = (uint32_t)atoi(optarg);
                break;
            case 'C':
                *nprobe = (ADDRTYPE)atoi(optarg);
                break;
            case 'Q':
                *queryBatchSize = (uint32_t)atoi(optarg);
                break;
            case 'M':
                *sliceAmt = (uint32_t)atoi(optarg);
                break;
            case 'S':
                *clusterSliceSize = (uint32_t)atoi(optarg);
                break;
            case 'H':
                clusterHeatThreses = optarg;
                break;
            case 'c':
                centroidsFileName = optarg;
                break;
            case 'b':
                codebookFileName = optarg;
                break;
            case 'a':
                clusterSizesFileName = optarg;
                break;
            case 'l':
                clusterLayoutFileName = optarg;
                break;
            case 'U':
                *nb_mram = (uint32_t)atoi(optarg);
                break;
            case 'h':
                usage(stdout, EXIT_SUCCESS, argv[0]);
            default:
                usage(stderr, EXIT_FAILURE, argv[0]);
        }
    }
    verify_path_exists(queriesFileName.c_str());
    verify_path_exists(centroidsFileName.c_str());
    verify_path_exists(codebookFileName.c_str());
    verify_path_exists(clusterSizesFileName.c_str());
}

inline UINT64 getHeatByFP(const UINT64 &frequency_clusterSize, const UINT64 &frequency, const UINT64 *latency) {  // Get the heat of a cluster with its `accessed_frequency x cluster_size` and `accessed_frequency`
    return frequency_clusterSize * (latency[distCal] + latency[topkSort]) + frequency * (latency[resiCal] + latency[lutCal]);
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
void coarseScheduler(const std::vector<UINT64> &heatClusters, const std::vector<ADDRTYPE> &batchClusterIdxs, const std::map<ADDRTYPE, UINT64> &frequency, std::vector<std::vector<std::pair<UINT64, ADDRTYPE>>> &coarseClusterLayout, std::vector<UINT64> &heatDPUs, const UINT64 *latency, const uint32_t &nb_mram) {
    // Greedy assignment
    std::vector<coarseScheduler_pqType> pqInit(nb_mram);
    for (ADDRTYPE nr_dpu = 0; nr_dpu < nb_mram; ++nr_dpu)
        pqInit[nr_dpu].dpuIdx = nr_dpu;
    std::priority_queue<coarseScheduler_pqType> pq(pqInit.begin(), pqInit.end());
    std::vector<std::vector<ADDRTYPE>> clusterDirectory(heatClusters.size());
    for (auto &batchClusterIdx: batchClusterIdxs) {
        std::vector<coarseScheduler_pqType> residedDPUs;  // size: <= clusterDirectory[batchClusterIdx].size()
        coarseScheduler_pqType pqTop = pq.top();
        pq.pop();
        if (clusterDirectory[batchClusterIdx].size() > 0) {  // Search a DPU without the clusters[batchClusterIdx]
            bool isLocated = true;
            while (true) {
                isLocated = false;
                for (auto &dpuIdx: clusterDirectory[batchClusterIdx]) {
                    if (dpuIdx == pqTop.dpuIdx) {
                        isLocated = true;
                        break;
                    }
                }
                if (isLocated) {
                    residedDPUs.push_back(pqTop);
                    pqTop = pq.top();
                    pq.pop();
                } else {
                    break;
                }
            }
        }
        coarseClusterLayout[pqTop.dpuIdx].push_back(std::make_pair(heatClusters[batchClusterIdx], batchClusterIdx));
        pqTop.dpuHeat += getHeatByFP(heatClusters[batchClusterIdx], frequency.at(batchClusterIdx), latency);
        pq.push(pqTop);
        for (auto &residedDPU: residedDPUs) {
            pq.push(residedDPU);
        }
        clusterDirectory[batchClusterIdx].push_back(pqTop.dpuIdx);
    }
    while (!pq.empty()) {
        heatDPUs[pq.top().dpuIdx] = pq.top().dpuHeat;
        pq.pop();
    }
}

std::pair<UINT64, ADDRTYPE> splitLongTail(const std::vector<std::pair<UINT64, ADDRTYPE>> &clusterList, const UINT64 &thres, const std::vector<CLUSTER_SIZES_TYPE> &clusterSizes, const UINT64 *latency) {  // Linear search, since we need to compare the threshold with the prefix sum of heat instead of the exact values
                                                                                                                                                                                                           // Start from the beginning
    UINT64 accumulateHeat = 0;
    for (ADDRTYPE clusterId = 0, clusterAmt = clusterList.size(); clusterId < clusterAmt; ++clusterId) {
        accumulateHeat += getHeatByFP(clusterList[clusterId].first, clusterList[clusterId].first / clusterSizes[clusterList[clusterId].second], latency);
        if (accumulateHeat >= thres)
            return std::make_pair(accumulateHeat - thres, clusterId);
    }
    return std::make_pair(thres - accumulateHeat, clusterList.size());  // This line is expected never to be executed
}


int main(int argc, char **argv)
{
    const double DPUfrequency = 350000000;
    const double WRAMbandwidth = 1612.56 * 1024 * 1024;  // WRAM(ADD)
    const double MRAMbandwidth =  573.79 * 1024 * 1024;  // MRAM(ADD)
    uint32_t dimAmt = 128;
    uint32_t neighborAmt = 10;
    uint32_t sliceAmt = 8;
    uint32_t queryBatchSize = 1;
    ADDRTYPE nprobe = 1;
    uint32_t clusterSliceSize = 1;  // Unit: Byte. Note: it would be better to make this a multiple of `pointSize`
    std::string clusterHeatThreses = "{}";
    uint32_t nb_mram = 2530;
    std::string queriesFileName = "query.bin";
    std::string centroidsFileName = "centroids.bin";
    std::string codebookFileName = "codebook.bin";
    std::string clusterSizesFileName = "clusterSizes.bin";
    std::string clusterLayoutFileName = "clusterLayout.bin";
    parse_args(argc, argv, &dimAmt, &neighborAmt, &sliceAmt, &queryBatchSize, &nprobe, &clusterSliceSize, &nb_mram, clusterHeatThreses, queriesFileName, centroidsFileName, codebookFileName, clusterSizesFileName, clusterLayoutFileName);
    std::vector<ELEMTYPE> queries(getPointsAmount(queriesFileName.c_str(), sizeof(ELEMTYPE)));
    std::vector<ELEMTYPE> centroids(getPointsAmount(centroidsFileName.c_str(), sizeof(ELEMTYPE)));
    std::vector<CLUSTER_SIZES_TYPE> clusterSizes(getPointsAmount(clusterSizesFileName.c_str(), sizeof(CLUSTER_SIZES_TYPE)));
    std::vector<std::vector<ClusterLayout>> clusterLayout(nb_mram);
    load1DPointsFromFile(queriesFileName.c_str(), queries);
    load1DPointsFromFile(centroidsFileName.c_str(), centroids);
    load1DPointsFromFile(clusterSizesFileName.c_str(), clusterSizes);
    const ADDRTYPE queryAmt = getPointsAmount(queriesFileName.c_str(), dimAmt * sizeof(ELEMTYPE));
    const ADDRTYPE centroidAmt = getPointsAmount(centroidsFileName.c_str(), dimAmt * sizeof(ELEMTYPE));
    const ADDRTYPE codebookEntryAmt = getPointsAmount(codebookFileName.c_str(), dimAmt * sizeof(CB_TYPE));
    std::vector<ClusterHeatInfo> clusterHeatThresInfos;
    parseDict(clusterHeatThresInfos, clusterHeatThreses, centroidAmt);

    std::vector<UINT64> latencyVec = { 0, 
                                       static_cast<UINT64>(std::ceil((((sizeof(ELEMTYPE) + sizeof(CENTROIDS_TYPE)) / MRAMbandwidth + (sizeof(ELEMTYPE) + sizeof(CENTROIDS_TYPE)) / WRAMbandwidth) * DPUfrequency + 1) * dimAmt)),  // query/centroid: MRAM -> WRAM -> register + ADD
                                       static_cast<UINT64>(std::ceil((((sizeof(CB_TYPE) / MRAMbandwidth + (sizeof(CENTROIDS_TYPE) + sizeof(CB_TYPE)) / WRAMbandwidth + sizeof(ELEMTYPE) / WRAMbandwidth) * DPUfrequency + 1) * dimAmt + (dimAmt - sliceAmt)) * codebookEntryAmt)),  // residual: WRAM -> register; codebook: MRAM -> WRAM -> register + (SUB + MUL); ADD(accumulate: dim - sliceAmt). MUL = WRAM -> register
                                       static_cast<UINT64>(std::ceil((sizeof(POINTTYPE) / MRAMbandwidth + (sizeof(POINTTYPE) + sizeof(VECSUMTYPE)) / WRAMbandwidth) * sliceAmt + (sliceAmt - 1))),  // point: MRAM -> WRAM -> register; LUT[point]: WRAM -> register + ADD
                                       static_cast<UINT64>(std::ceil(((sizeof(VECSUMTYPE) + sizeof(CLUSTER_SIZES_TYPE)) / WRAMbandwidth + 1) * log2(neighborAmt) + sizeof(VECSUMTYPE) / WRAMbandwidth))  // dist: WRAM -> register; pQueue: WRAM -> register -> WRAM + CMP
                                       };  // Coarse-grained estimated by ideal model (only read since DMA is applied to write)
    const UINT64 *const latency = latencyVec.data();

    uint32_t pointSize = sliceAmt * sizeof(ELEMTYPE);
    std::vector<CLUSTER_SIZES_TYPE> clusterSliceAmts(clusterSizes.size());
    {
        ADDRTYPE clusterAmt = clusterSizes.size();
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE clusterId = 0; clusterId < clusterAmt; ++clusterId)
            clusterSliceAmts[clusterId] = std::ceil((float)(clusterSizes[clusterId]) * pointSize / clusterSliceSize);
    }
    
    std::vector<std::map<ADDRTYPE, ClusterLayout>> finalClusterLayout(nb_mram);  // For each DPU: <clusterId, { startSliceId, endSliceId }>
    std::vector<std::map<ADDRTYPE, ClusterLayout>> finalFineClusterLayout(nb_mram);  // For each DPU: <clusterId, { startSliceId, endSliceId }>
    std::vector<std::map<ADDRTYPE, UINT64>> frequency(queryAmt / queryBatchSize + 1);  // <clusterId, accessTimes> of the i th query batch; the map has been automatically sorted by its key in ascent order
    for (ADDRTYPE queryId = 0, qBatchId = 0; queryId < queryAmt; queryId += queryBatchSize, ++qBatchId) {
        ADDRTYPE processedQuerySize = queryId * dimAmt;
        uint32_t realQueryBatchSize = std::min(queryBatchSize, queryAmt - queryId);
        // 1. Get the access frequency of each cluster with faiss in each batch
        std::unique_ptr<ADDRTYPE[]> selClusterIdxs(new ADDRTYPE[realQueryBatchSize * nprobe]);
        {  // IVF search
            // metric_type == METRIC_L2
            // we see the distances and labels as heaps
            std::unique_ptr<pqueue_pri_t[]> selClusterDists(new pqueue_pri_t[realQueryBatchSize * nprobe]);
            knn_L2sqr(reinterpret_cast<ELEMTYPE *>(queries.data()) + processedQuerySize, reinterpret_cast<ELEMTYPE *>(centroids.data()), dimAmt, realQueryBatchSize, centroidAmt, nprobe, selClusterDists.get(), selClusterIdxs.get(), NULL, nullptr);
        }
        {
            auto &curFreq = frequency[qBatchId];
            for (uint32_t idxId = 0, idxAmt = realQueryBatchSize * nprobe; idxId < idxAmt; ++idxId) {
                ADDRTYPE key = selClusterIdxs[idxId];
                if (curFreq.find(key) != curFreq.end()) {
                    ++curFreq[key];
                } else {
                    curFreq[key] = 1;
                }
            }
        }
        // 2. Count and record the amount of accessed cluster slices in each batch. The size of slices is determined by the split strategy
        //    Count the sum of heat of all accessed cluster slices. The heat is calculated by `access_frequency x cluster_size`
        std::vector<UINT64> heatClusters(clusterSizes.size(), 0);
        UINT64 accessedSliceAmt = 0;
        {
            auto &curFreq = frequency[qBatchId];
            {
                ADDRTYPE clusterAmt = curFreq.size();
                #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
                for (ADDRTYPE clusterId = 0; clusterId < clusterAmt; ++clusterId) {
                    auto curClusterFreqIter = curFreq.begin();
                    std::advance(curClusterFreqIter, clusterId);
                    heatClusters[curClusterFreqIter->first] = curClusterFreqIter->second * clusterSizes[curClusterFreqIter->first];
                }
            }
            for (ADDRTYPE clusterId = 0, clusterAmt = curFreq.size(); clusterId < clusterAmt; ++clusterId) {
                auto curClusterFreqIter = curFreq.begin();
                std::advance(curClusterFreqIter, clusterId);
                accessedSliceAmt += clusterSliceAmts[curClusterFreqIter->first];
            }
        }
        // 3. Assign cluster slices to DPUs to balance the heat of each DPU in each batch. Remove redundant cluster slices on the same DPU. -- coarse-grained scheduling
        //    Assign clusters that are not accessed to DPUs based on the occupied capacity of MRAM on each DPU
        //    Order groups of <clusterId, sliceId> on each DPU
        std::vector<ADDRTYPE> batchClusterIdxs(frequency[qBatchId].size());
        {
            ADDRTYPE clusterAmt = frequency[qBatchId].size();
            #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
            for (ADDRTYPE clusterId = 0; clusterId < clusterAmt; ++clusterId) {
                auto curClusterFreqIter = frequency[qBatchId].begin();
                std::advance(curClusterFreqIter, clusterId);
                batchClusterIdxs[clusterId] = curClusterFreqIter->first;
            }
        }
        UINT64 sumHeatClusters = std::accumulate(heatClusters.begin(), heatClusters.end(), static_cast<UINT64>(0));
        if (clusterHeatThresInfos.size() > 0) {  // Cluster duplication
            std::vector<ADDRTYPE> dupBatchClusterIdxs;
            sort(batchClusterIdxs.begin(), batchClusterIdxs.end(), [&heatClusters] (const ADDRTYPE &a, const ADDRTYPE &b) { return heatClusters[a] > heatClusters[b]; });  // Order the indices of accessed clusters in descent order of heat
            auto &curFreq = frequency[qBatchId];
            if (clusterHeatThresInfos[0].prefixNlist > 0) {
                for (ADDRTYPE dupTimes = 0; dupTimes < clusterHeatThresInfos[0].dupTimes; ++dupTimes) {
                    dupBatchClusterIdxs.insert(dupBatchClusterIdxs.end(), batchClusterIdxs.begin(), batchClusterIdxs.begin() + clusterHeatThresInfos[0].prefixNlist);
                }
                UINT64 sumHeatClustersLocal = 0;
                for (auto batchClusterIdxsIter = batchClusterIdxs.begin(), batchClusterIdxsEnd = batchClusterIdxs.begin() + clusterHeatThresInfos[0].prefixNlist; batchClusterIdxsIter < batchClusterIdxsEnd; ++batchClusterIdxsIter) {
                    auto curClusterFreqIter = curFreq.begin();
                    std::advance(curClusterFreqIter, *batchClusterIdxsIter);
                    sumHeatClustersLocal += curClusterFreqIter->second * clusterSizes[curClusterFreqIter->first];
                }
                sumHeatClusters += sumHeatClustersLocal * clusterHeatThresInfos[0].dupTimes;
            }
            for (ADDRTYPE clusterHeatThresInfoIdx = 1, clusterHeatThresInfoEnd = clusterHeatThresInfos.size(); clusterHeatThresInfoIdx < clusterHeatThresInfoEnd; ++clusterHeatThresInfoIdx) {
                if (clusterHeatThresInfos[clusterHeatThresInfoIdx].prefixNlist > 0) {
                    ADDRTYPE copiedClusterAmt = clusterHeatThresInfos[clusterHeatThresInfoIdx - 1].prefixNlist, nextCopiedClusterAmt = clusterHeatThresInfos[clusterHeatThresInfoIdx].prefixNlist;
                    for (ADDRTYPE dupTimes = 0; dupTimes < clusterHeatThresInfos[clusterHeatThresInfoIdx].dupTimes; ++dupTimes) {
                        dupBatchClusterIdxs.insert(dupBatchClusterIdxs.end(), batchClusterIdxs.begin() + copiedClusterAmt, batchClusterIdxs.begin() + nextCopiedClusterAmt);
                    }
                    UINT64 sumHeatClustersLocal = 0;
                    for (auto batchClusterIdxsIter = batchClusterIdxs.begin() + copiedClusterAmt, batchClusterIdxsEnd = batchClusterIdxs.begin() + nextCopiedClusterAmt; batchClusterIdxsIter < batchClusterIdxsEnd; ++batchClusterIdxsIter) {
                        auto curClusterFreqIter = curFreq.begin();
                        std::advance(curClusterFreqIter, *batchClusterIdxsIter);
                        sumHeatClustersLocal += curClusterFreqIter->second * clusterSizes[curClusterFreqIter->first];
                    }
                    sumHeatClusters += sumHeatClustersLocal * clusterHeatThresInfos[clusterHeatThresInfoIdx].dupTimes;
                }
            }
            batchClusterIdxs.insert(batchClusterIdxs.end(), dupBatchClusterIdxs.begin(), dupBatchClusterIdxs.end());
        }
        sort(batchClusterIdxs.begin(), batchClusterIdxs.end(), [&clusterSizes] (const ADDRTYPE &a, const ADDRTYPE &b) { return clusterSizes[a] > clusterSizes[b]; });  // Order the indices of accessed clusters in descent order of size
        std::vector<std::vector<std::pair<UINT64, ADDRTYPE>>> coarseClusterLayout(nb_mram);  // <clusterHeat, clusterId>
        std::vector<UINT64> coarseHeatDPUs(nb_mram, 0);
        coarseScheduler(heatClusters, batchClusterIdxs, frequency[qBatchId], coarseClusterLayout, coarseHeatDPUs, latency, nb_mram);
        UINT64 dupAppendedFrequency = 0;
        if (clusterHeatThresInfos.size() > 0) {
            ADDRTYPE accessedClusterAmt = realQueryBatchSize * nprobe;
            dupAppendedFrequency = clusterHeatThresInfos[0].dupTimes * std::max(clusterHeatThresInfos[0].prefixNlist, accessedClusterAmt);
            for (UINT64 clusterHeatThresInfosBeg = 1, clusterHeatThresInfosEnd = clusterHeatThresInfos.size(); clusterHeatThresInfosBeg < clusterHeatThresInfosEnd; ++clusterHeatThresInfosBeg) {
                if (clusterHeatThresInfos[clusterHeatThresInfosBeg - 1].prefixNlist >= accessedClusterAmt) {
                    break;
                }
                dupAppendedFrequency += clusterHeatThresInfos[clusterHeatThresInfosBeg].dupTimes * (std::max(clusterHeatThresInfos[clusterHeatThresInfosBeg].prefixNlist, accessedClusterAmt) - clusterHeatThresInfos[clusterHeatThresInfosBeg - 1].prefixNlist);
            }
        }
        float avgLatency = (float)getHeatByFP(sumHeatClusters, realQueryBatchSize * nprobe + dupAppendedFrequency, latency) / nb_mram;
        avgLatency += latency[resiCal] + latency[lutCal];  // Average cost of cluster split
        // 4. Reorder clusters on each DPU in each batch by the estimated costs of RC and LC (viz. `access_frequency`) for balance -- fine-grained scheduling
        std::vector<ADDRTYPE> dpuIdxs(nb_mram);
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nb_mram; ++nr_dpu) {
            dpuIdxs[nr_dpu] = nr_dpu;
        }
        sort(dpuIdxs.begin(), dpuIdxs.end(), [&coarseHeatDPUs] (const ADDRTYPE &a, const ADDRTYPE &b) { return coarseHeatDPUs[a] > coarseHeatDPUs[b]; });
        ADDRTYPE longTailDPUAmt = 0;
        while (coarseHeatDPUs[dpuIdxs[longTailDPUAmt]] > avgLatency)
            ++longTailDPUAmt;
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < longTailDPUAmt; ++nr_dpu) {
            auto singleClusterLayoutIter = coarseClusterLayout.begin() + dpuIdxs[nr_dpu];
            sort(singleClusterLayoutIter->begin(), singleClusterLayoutIter->end());  // In ascent order of coarse-grained heat
        }
        std::vector<std::vector<std::pair<std::pair<UINT64, ADDRTYPE>, std::pair<ADDRTYPE, ADDRTYPE>>>> fineClusterLayout(nb_mram);  // <<clusterHeat, clusterId>, <startSliceId, endSliceId>>
        // std::vector<std::pair<UINT64, ADDRTYPE>> longTailClusterInfos(longTailDPUAmt + 1);  // <exceededHeatOfBorderCluster, borderLocalClusterId>
        std::vector<std::pair<UINT64, ADDRTYPE>> longTailClusterInfos(longTailDPUAmt);  // <exceededHeatOfBorderCluster, borderLocalClusterId>
        ADDRTYPE appendedClusterDPUId = nb_mram - 1;
        {  // Fine-grained scheduling. Get the `fineClusterLayout` (cluster slices for appending) and `longTailClusterInfos` (the border cluster and related slice for pruning) to adjust the coarse-grained layout
            std::vector<UINT64> fineHeatDPUs(nb_mram, 0);
            for (ADDRTYPE longTailDPUCnt = 0; longTailDPUCnt < longTailDPUAmt; ++longTailDPUCnt) {
                auto singleClusterLayoutIter = coarseClusterLayout.begin() + dpuIdxs[longTailDPUCnt];
                longTailClusterInfos[longTailDPUCnt] = splitLongTail(*singleClusterLayoutIter, coarseHeatDPUs[dpuIdxs[longTailDPUCnt]] - avgLatency, clusterSizes, latency);
                auto &longTailClusterInfo = longTailClusterInfos[longTailDPUCnt];
                for (ADDRTYPE clusterId = 0; clusterId < longTailClusterInfo.second; ++clusterId) {
                    {  // Avoid the case that the cluster to be transferred has already located on the cold DPU. If so, look for another candidate transferred cluster on the hot DPU. 
                       // To simplify the process, incomplete cluster on the cold DPU is seemed as a complete one. The process regarding the fine layout records of the cold cluster might be changed if more balanced layout is required
                        bool alreadyLocated = false;
                        ADDRTYPE hotClusterId = (*singleClusterLayoutIter)[clusterId].second;
                        auto &coldClusterCoarseList = coarseClusterLayout[dpuIdxs[appendedClusterDPUId]];
                        for (ADDRTYPE coldClusterId = 0, coldClusterAmt = coldClusterCoarseList.size(); coldClusterId < coldClusterAmt; ++coldClusterId) {
                            if (hotClusterId == coldClusterCoarseList[coldClusterId].second) {
                                alreadyLocated = true;
                                break;
                            }
                        }
                        if (!alreadyLocated) {  // Check the fine layout records of the cold cluster
                            auto &coldClusterFineList = fineClusterLayout[dpuIdxs[appendedClusterDPUId]];
                            for (ADDRTYPE coldClusterId = 0, coldClusterAmt = coldClusterFineList.size(); coldClusterId < coldClusterAmt; ++coldClusterId) {
                                if (hotClusterId == coldClusterFineList[coldClusterId].first.second) {
                                    alreadyLocated = true;
                                    break;
                                }
                            }
                        }
                        if (alreadyLocated) {
                            // 1. Adjust longTailClusterInfo (similar to the operations in `splitLongTail`)
                            UINT64 repeatedClusterHeat = getHeatByFP((*singleClusterLayoutIter)[clusterId].first, (*singleClusterLayoutIter)[clusterId].first / clusterSizes[(*singleClusterLayoutIter)[clusterId].second], latency);
                            UINT64 repeatedClusterHeatArchive = repeatedClusterHeat;
                            if (longTailClusterInfo.first >= repeatedClusterHeat) {
                                longTailClusterInfo.first -= repeatedClusterHeat;
                            } else {
                                hotClusterId = longTailClusterInfo.second + 1;
                                for (ADDRTYPE hotClusterAmt = singleClusterLayoutIter->size(); hotClusterId < hotClusterAmt; ++hotClusterId) {
                                    UINT64 clusterHeat = getHeatByFP((*singleClusterLayoutIter)[hotClusterId].first, (*singleClusterLayoutIter)[hotClusterId].first / clusterSizes[(*singleClusterLayoutIter)[hotClusterId].second], latency);
                                    if (clusterHeat >= repeatedClusterHeat) {
                                        longTailClusterInfo.first = clusterHeat - repeatedClusterHeat, longTailClusterInfo.second = hotClusterId;
                                        repeatedClusterHeat = 0;
                                    } else {
                                        repeatedClusterHeat -= clusterHeat;
                                    }
                                }
                                if (repeatedClusterHeat > FLOAT_ERROR) {
                                    longTailClusterInfo.first = repeatedClusterHeat, longTailClusterInfo.second = singleClusterLayoutIter->size();  // Might be executed due to coarse-grained filter for duplicated cluster slices
                                    repeatedClusterHeat = 0;
                                }
                            }
                            // 2. Push the cluster info into fineClusterLayout of the hot cluster
                            fineClusterLayout[singleClusterLayoutIter - coarseClusterLayout.begin()].push_back(std::make_pair(std::make_pair(repeatedClusterHeatArchive, (*singleClusterLayoutIter)[clusterId].second), std::make_pair(0, std::ceil((float)(clusterSizes[(*singleClusterLayoutIter)[clusterId].second]) * pointSize / clusterSliceSize))));
                            continue;
                        }
                    }
                    UINT64 clusterAccessedGFrequency = heatClusters[(*singleClusterLayoutIter)[clusterId].second] / clusterSizes[(*singleClusterLayoutIter)[clusterId].second];
                    UINT64 clusterHeat = getHeatByFP(heatClusters[(*singleClusterLayoutIter)[clusterId].second], clusterAccessedGFrequency, latency);
                    UINT64 clusterBaseHeat = clusterAccessedGFrequency * (latency[resiCal] + latency[lutCal]);
                    CLUSTER_SIZES_TYPE remainClusterSize = clusterSizes[(*singleClusterLayoutIter)[clusterId].second];
                    UINT64 partSliceStart = 0;
                    while (clusterHeat > FLOAT_ERROR) {
                        UINT64 newHeat = coarseHeatDPUs[dpuIdxs[appendedClusterDPUId]] + fineHeatDPUs[dpuIdxs[appendedClusterDPUId]] + clusterHeat;
                        if (newHeat >= avgLatency) {
                            UINT64 exceededHeat = newHeat > clusterBaseHeat + avgLatency ? newHeat - clusterBaseHeat - avgLatency : 0;
                            if (exceededHeat > FLOAT_ERROR) {  // Split
                                UINT64 partSliceAmt = std::ceil((remainClusterSize - (float)(exceededHeat) / clusterAccessedGFrequency / (latency[distCal] + latency[topkSort])) * pointSize / clusterSliceSize);
                                remainClusterSize -= partSliceAmt * std::ceil(clusterSliceSize / (float)(pointSize));
                                UINT64 appendedHeat = std::ceil(clusterAccessedGFrequency * partSliceAmt * clusterSliceSize / (float)(pointSize)) * (latency[distCal] + latency[topkSort]);
                                clusterHeat -= appendedHeat;
                                if (clusterHeat <= clusterBaseHeat) {  // Since the `ceil` result is adopted by `partSliceAmt`, the heat of current appended DPU with `appendedHeat` appended may exceed `avgLatency` even without `clusterBaseHeat` appended. In this case, the remained heat of the border cluster `clusterHeat` is no larger than `clusterBaseHeat`, so it should be cleared
                                    clusterHeat = 0;
                                }
                                appendedHeat += clusterBaseHeat;  // Add the extra costs of RC and LC
                                fineHeatDPUs[dpuIdxs[appendedClusterDPUId]] += appendedHeat;
                                fineClusterLayout[dpuIdxs[appendedClusterDPUId]].push_back(std::make_pair(std::make_pair(appendedHeat, (*singleClusterLayoutIter)[clusterId].second), std::make_pair(partSliceStart, partSliceStart + partSliceAmt)));
                                partSliceStart += partSliceAmt;
                            } else {  // Append
                                fineHeatDPUs[dpuIdxs[appendedClusterDPUId]] += clusterHeat;
                                fineClusterLayout[dpuIdxs[appendedClusterDPUId]].push_back(std::make_pair(std::make_pair(clusterHeat, (*singleClusterLayoutIter)[clusterId].second), std::make_pair(partSliceStart, std::ceil((float)(clusterSizes[(*singleClusterLayoutIter)[clusterId].second]) * pointSize / clusterSliceSize))));
                                clusterHeat = 0;
                            }
                            --appendedClusterDPUId;
                        } else {  // Append
                            fineHeatDPUs[dpuIdxs[appendedClusterDPUId]] += clusterHeat;
                            fineClusterLayout[dpuIdxs[appendedClusterDPUId]].push_back(std::make_pair(std::make_pair(clusterHeat, (*singleClusterLayoutIter)[clusterId].second), std::make_pair(partSliceStart, std::ceil((float)(clusterSizes[(*singleClusterLayoutIter)[clusterId].second]) * pointSize / clusterSliceSize))));
                            clusterHeat = 0;
                        }
                    }
                }
                UINT64 clusterAccessedGFrequency, clusterHeat;
                try {
                    clusterAccessedGFrequency = heatClusters[(*singleClusterLayoutIter)[longTailClusterInfo.second].second] / clusterSizes[(*singleClusterLayoutIter)[longTailClusterInfo.second].second];
                    clusterHeat = getHeatByFP(heatClusters[(*singleClusterLayoutIter)[longTailClusterInfo.second].second], clusterAccessedGFrequency, latency) - longTailClusterInfo.first;
                } catch (...) {
                    fprintf(stderr, "Warning: long-tailed DPU[%u] (DPU[%u]) may be overheated due to coarse-grained duplication filter for duplicated cluster slices! Check at runtime!\n", longTailDPUCnt, dpuIdxs[longTailDPUCnt]);
                    continue;  // `longTailClusterInfo.second` might be equal to `singleClusterLayoutIter->size()` due to coarse-grained filter for duplicated cluster slices. In this case, the following split is useless
                };
                if (clusterHeat > FLOAT_ERROR) {
                    UINT64 clusterBaseHeat = clusterAccessedGFrequency * (latency[resiCal] + latency[lutCal]);
                    UINT64 movedSliceAmt = clusterHeat <= clusterBaseHeat ? 0 : std::floor((float)(clusterSizes[(*singleClusterLayoutIter)[longTailClusterInfo.second].second]) * pointSize * (clusterHeat - clusterBaseHeat) / (clusterHeat + longTailClusterInfo.first - clusterBaseHeat) / clusterSliceSize);
                    UINT64 errorHeat = getHeatByFP(clusterAccessedGFrequency * movedSliceAmt * std::ceil(clusterSliceSize / (float)(pointSize)), clusterAccessedGFrequency, latency);
                    if (clusterHeat < errorHeat) {
                        errorHeat = 0;
                    } else {
                        errorHeat = clusterHeat - errorHeat;  // Since the `floor` result is adopted by `movedSliceAmt`, the maximal heat of the moved slices may be smaller than `clusterHeat`, leading to a redundant invalid record of the moved slice whose `startSliceId` is equal to its `endSliceId`. So just record the error in `errorHeat` and extract it from `clusterHeat` to the heat `longTailClusterInfo.first` on the original DPU
                    }
                    clusterHeat -= errorHeat;
                    if (movedSliceAmt > 0) {
                        CLUSTER_SIZES_TYPE remainClusterSize = movedSliceAmt * clusterSliceSize / (float)(pointSize);
                        UINT64 partSliceStart = 0;  // Move front slices since the retrieval of moved clusters is from the beginning of the cluster list
                        while (clusterHeat > FLOAT_ERROR) {
                            UINT64 newHeat = coarseHeatDPUs[dpuIdxs[appendedClusterDPUId]] + fineHeatDPUs[dpuIdxs[appendedClusterDPUId]] + clusterHeat;
                            if (newHeat >= avgLatency) {
                                UINT64 exceededHeat = newHeat > clusterBaseHeat + avgLatency ? newHeat - clusterBaseHeat - avgLatency : 0;
                                if (exceededHeat > FLOAT_ERROR) {  // Split
                                    UINT64 partSliceAmt = std::ceil((remainClusterSize - (float)(exceededHeat) / clusterAccessedGFrequency / (latency[distCal] + latency[topkSort])) * pointSize / clusterSliceSize);
                                    remainClusterSize -= partSliceAmt * std::ceil(clusterSliceSize / (float)(pointSize));
                                    UINT64 appendedHeat = std::ceil(clusterAccessedGFrequency * partSliceAmt * clusterSliceSize / (float)(pointSize)) * (latency[distCal] + latency[topkSort]);
                                    clusterHeat -= appendedHeat;
                                    if (clusterHeat <= clusterBaseHeat) {  // Since the `ceil` result is adopted by `partSliceAmt`, the heat of current appended DPU with `appendedHeat` appended may exceed `avgLatency` even without `clusterBaseHeat` appended. In this case, the remained heat of the border cluster `clusterHeat` is no larger than `clusterBaseHeat`, so it should be cleared
                                        clusterHeat = 0;
                                    }
                                    appendedHeat += clusterBaseHeat;  // Add the extra costs of RC and LC
                                    fineHeatDPUs[dpuIdxs[appendedClusterDPUId]] += appendedHeat;
                                    fineClusterLayout[dpuIdxs[appendedClusterDPUId]].push_back(std::make_pair(std::make_pair(appendedHeat, (*singleClusterLayoutIter)[longTailClusterInfo.second].second), std::make_pair(partSliceStart, partSliceStart + partSliceAmt)));
                                    partSliceStart += partSliceAmt;
                                } else {  // Append
                                    fineHeatDPUs[dpuIdxs[appendedClusterDPUId]] += clusterHeat;
                                    fineClusterLayout[dpuIdxs[appendedClusterDPUId]].push_back(std::make_pair(std::make_pair(clusterHeat, (*singleClusterLayoutIter)[longTailClusterInfo.second].second), std::make_pair(partSliceStart, movedSliceAmt)));
                                    clusterHeat = 0;
                                }
                                --appendedClusterDPUId;
                            } else {  // Append
                                fineHeatDPUs[dpuIdxs[appendedClusterDPUId]] += clusterHeat;
                                fineClusterLayout[dpuIdxs[appendedClusterDPUId]].push_back(std::make_pair(std::make_pair(clusterHeat, (*singleClusterLayoutIter)[longTailClusterInfo.second].second), std::make_pair(partSliceStart, movedSliceAmt)));
                                clusterHeat = 0;
                            }
                        }
                    }
                    fineClusterLayout[singleClusterLayoutIter - coarseClusterLayout.begin()].push_back(std::make_pair(std::make_pair(longTailClusterInfo.first + errorHeat, (*singleClusterLayoutIter)[longTailClusterInfo.second].second), std::make_pair(movedSliceAmt, std::ceil((float)(clusterSizes[(*singleClusterLayoutIter)[longTailClusterInfo.second].second]) * pointSize / clusterSliceSize))));
                } else {
                    fineClusterLayout[singleClusterLayoutIter - coarseClusterLayout.begin()].push_back(std::make_pair(std::make_pair(longTailClusterInfo.first, (*singleClusterLayoutIter)[longTailClusterInfo.second].second), std::make_pair(0, std::ceil((float)(clusterSizes[(*singleClusterLayoutIter)[longTailClusterInfo.second].second]) * pointSize / clusterSliceSize))));
                }
            }
        }
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nb_mram; ++nr_dpu) {  // A cluster may locates on different DPUs due to duplication by multiple queries
            // Record coarse-grained cluster slice layout
            auto &coarseClusterLayoutDPU = coarseClusterLayout[dpuIdxs[nr_dpu]];
            auto &finalClusterList = finalClusterLayout[dpuIdxs[nr_dpu]];
            for (auto clusterInfoIter = coarseClusterLayoutDPU.begin() + (nr_dpu < longTailDPUAmt ? longTailClusterInfos[nr_dpu].second + 1 : 0); clusterInfoIter < coarseClusterLayoutDPU.end(); ++clusterInfoIter) {
                auto &clusterId = clusterInfoIter->second;
                if (finalClusterList.find(clusterId) == finalClusterList.end()) {
                    finalClusterList[clusterId] = ClusterLayout(dpuIdxs[nr_dpu], clusterId, 0, std::ceil((float)(clusterSizes[clusterId]) * pointSize / clusterSliceSize));
                }
            }
        }
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nb_mram; ++nr_dpu) {
            // Record fine-grained cluster slice layout
            auto &fineClusterLayoutDPU = fineClusterLayout[nr_dpu];
            auto &finalFineClusterList = finalFineClusterLayout[nr_dpu];
            for (auto &clusterInfo: fineClusterLayoutDPU) {
                auto &clusterId = clusterInfo.first.second;
                if (finalFineClusterList.find(clusterId) == finalFineClusterList.end()) {
                    finalFineClusterList[clusterId] = ClusterLayout(nr_dpu, clusterId, clusterInfo.second.first, clusterInfo.second.second);
                } else {
                    auto &finalFineClusterListElem = finalFineClusterList[clusterId];
                    if (finalFineClusterListElem.startSliceId > clusterInfo.second.first)
                        finalFineClusterListElem.startSliceId = clusterInfo.second.first;
                    if (finalFineClusterListElem.endSliceId < clusterInfo.second.second)
                        finalFineClusterListElem.endSliceId = clusterInfo.second.second;
                }
            }
        }
    }
    // 5. Merge same clusters on each DPU of different batches and generate the final cluster slice layout
    #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
    for (ADDRTYPE nr_dpu = 0; nr_dpu < nb_mram; ++nr_dpu) {  // Merge fine-grained overall cluster slice layout into the overall cluster slice layout. The overall cluster slice layout only records the coarse-grained overall cluster slice layout in the loop above
        auto &finalClusterList = finalClusterLayout[nr_dpu];
        auto &finalFineClusterList = finalFineClusterLayout[nr_dpu];
        for (auto &clusterSliceInfo: finalFineClusterList) {
            auto &clusterId = clusterSliceInfo.first;
            if (finalClusterList.find(clusterId) == finalClusterList.end()) {
                finalClusterList[clusterId] = clusterSliceInfo.second;
            }
        }
    }

    // Runtime scheduler:
    // 1) Coarse-grained phase: keep all DPUs in a priority queue. Set the latency as the priority
    // 2) Split & duplication: If the required cluster is not on the top DPU, find it in a list that records all DPUs with the cluster slices. Sort them by the latency before searching.
    //    Find the optimal group that has the entire cluster
    // The runtime scheduler may not achieve the goal of the offline layout since the intrerrupts across query batches.
    // Thus, for large batches, the effects of the interrupts should be tolerant. The quantitative effects should be measured by experiments
    {  // Append missing clusters
        std::vector<bool> clusterIsAccessed(clusterSizes.size(), false);
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nb_mram; ++nr_dpu) {
            for (auto &clusterInfo: finalClusterLayout[nr_dpu]) {
                clusterIsAccessed[clusterInfo.second.clusterId] = true;  // Thread conflicts will not matter, but the results become incorrect with parallel operations of different DPUs due to unknown reasons
            }
        }
        std::vector<std::pair<UINT64, ADDRTYPE>> dpuClusterSliceAmts(nb_mram);  // <clusterSliceAmt, dpuId>
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nb_mram; ++nr_dpu) {
            dpuClusterSliceAmts[nr_dpu].first = 0, dpuClusterSliceAmts[nr_dpu].second = nr_dpu;
            for (auto &clusterInfo: finalClusterLayout[nr_dpu]) {
                dpuClusterSliceAmts[nr_dpu].first += clusterInfo.second.endSliceId - clusterInfo.second.startSliceId;
            }
        }
        std::priority_queue<std::pair<UINT64, ADDRTYPE>, std::vector<std::pair<UINT64, ADDRTYPE>>, std::greater<std::pair<UINT64, ADDRTYPE>>> clusterSliceAmtsPQ(dpuClusterSliceAmts.begin(), dpuClusterSliceAmts.end());  // Min heap. <clusterSliceAmt, dpuId>
        for (CLUSTER_SIZES_TYPE clusterId = 0, clusterAmt = clusterIsAccessed.size(); clusterId < clusterAmt; ++clusterId) {
            if (!clusterIsAccessed[clusterId]) {
                auto clusterSliceAmtsPQTop = clusterSliceAmtsPQ.top();
                clusterSliceAmtsPQ.pop();
                ADDRTYPE clusterSliceAmt = std::ceil((float)(clusterSizes[clusterId]) * pointSize / clusterSliceSize);
                finalClusterLayout[clusterSliceAmtsPQTop.second][clusterId] = ClusterLayout(clusterSliceAmtsPQTop.second, clusterId, 0, clusterSliceAmt);
                clusterSliceAmtsPQTop.first += clusterSliceAmt;
                clusterSliceAmtsPQ.push(clusterSliceAmtsPQTop);
            }
        }
    }
    std::vector<std::vector<ClusterIndex>> clusterDirectory(clusterSizes.size());  // Record the mapping from cluster to the located DPUs. E.g. if cluster n locates on DPU a, b and c, then clusterDirectory[n] = { a, b, c }

    // 6. Save results, including cluster slice information of each DPU, and the mapping from each cluster to the corresponding DPUs
    //    Datum format: finalClusterLayout: nb_mram * clusterSizeOnTheDPU + clusterSliceInfos; clusterDirectory: clusterSizes.size() * DPUAmtOfTheCluster + DPUIds
    clearFile(clusterLayoutFileName.c_str());
    {  // Save the vector size part of cluster slice information of each DPU
        std::vector<CLUSTER_SIZES_TYPE> clusterSizesOnTheDPU(nb_mram);
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nb_mram; ++nr_dpu) {
            clusterSizesOnTheDPU[nr_dpu] = finalClusterLayout[nr_dpu].size();
        }
        saveDataToFile(clusterLayoutFileName.c_str(), clusterSizesOnTheDPU.data(), sizeof(CLUSTER_SIZES_TYPE), clusterSizesOnTheDPU.size());  // Save the vector size part of cluster slice information of each DPU
    }
    {  // Save the vector data part of cluster slice information of each DPU
        std::vector<ClusterLayout> finalClusterLayout1D(clusterSizes.size());
        finalClusterLayout1D.clear();  // Clear empty elements of `finalClusterLayout1D` while keep the capacity to reduce the cost of push_back
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nb_mram; ++nr_dpu) {
            ADDRTYPE localClusterId = 0;
            for (auto &clusterInfo: finalClusterLayout[nr_dpu]) {
                finalClusterLayout1D.push_back(clusterInfo.second);
                clusterDirectory[clusterInfo.first].push_back(ClusterIndex(nr_dpu, localClusterId++));  // nr_dpu == clusterSliceInfo.dpuId
            }
        }
        saveDataToFile(clusterLayoutFileName.c_str(), finalClusterLayout1D.data(), sizeof(ClusterLayout), finalClusterLayout1D.size());  // Save the vector data part of cluster slice information of each DPU
    }
    {  // Save the vector size part of cluster-DPU mapping
        CLUSTER_SIZES_TYPE clusterAmt = clusterSizes.size();
        std::vector<CLUSTER_SIZES_TYPE> dpuSizesOfClusters(clusterAmt);
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (CLUSTER_SIZES_TYPE clusterId = 0; clusterId < clusterAmt; ++clusterId) {
            dpuSizesOfClusters[clusterId] = clusterDirectory[clusterId].size();
        }
        saveDataToFile(clusterLayoutFileName.c_str(), dpuSizesOfClusters.data(), sizeof(CLUSTER_SIZES_TYPE), dpuSizesOfClusters.size());
    }
    {  // Save the vector data part of cluster-DPU mapping
        std::vector<ClusterIndex> clusterDirectory1D;
        for (auto &clusterDPUMappingInfo: clusterDirectory) {
            clusterDirectory1D.insert(clusterDirectory1D.end(), clusterDPUMappingInfo.begin(), clusterDPUMappingInfo.end());
        }
        saveDataToFile(clusterLayoutFileName.c_str(), clusterDirectory1D.data(), sizeof(ClusterIndex), clusterDirectory1D.size());
    }
    return 0;
}
