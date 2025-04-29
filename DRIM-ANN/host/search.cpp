/*
Author: KMC20
Date: 2024/3/10
Function: Management including data transfer with DPU and for the query distribution to different MRAMs.
*/

#include <math.h>
#include <time.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <dpu>
#include <sys/time.h>
#include <dpu_error.h>
#include "request.h"
#ifdef ENERGY_EVAL
#include "measureEnergy.h"
#endif
#include "tools.h"
#include <numeric>
#include <vector>
#include <string>
#include <assert.h>
#define CPP_DPU_ASSERT assert

#ifdef PERF_EVAL
typedef uint64_t perfcounter_t;
#endif

#define XSTR(x) #x
#define STR(x) XSTR(x)

#define NR_JOB_PER_RANK 64
#define NR_JOB_PARALLEL NR_JOB_PER_RANK

#define dpu_binary_CBS "build/dpu_task_CBS"


template <typename T>
inline uint64_t align8bytes(const T &nBytes) {
    return nBytes + (((~nBytes & 7) + 1) & 7);
}

UINT64 getPointsAmount(const char *const pointsFileName, const uint32_t dimAmt) {
    FILE *fp = fopen(pointsFileName, "rb");
    if (fp == NULL) {
        printf("Failed to open the input point file: %s! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    UINT64 pointsAmt = ftell(fp) / (sizeof(ELEMTYPE) * dimAmt);
    fclose(fp);
    return pointsAmt;
}

UINT64 getElemsAmount(const char *const pointsFileName, const uint32_t elemSize) {
    FILE *fp = fopen(pointsFileName, "rb");
    if (fp == NULL) {
        printf("Failed to open the input file: %s! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    UINT64 pointsAmt = ftell(fp) / elemSize;
    fclose(fp);
    return pointsAmt;
}

template<typename T>
void load1DDataFromFile(const char *const dataFileName, std::vector<T> &data) {
    FILE *fp = fopen(dataFileName, "rb");
    if (fp == NULL) {
        printf("Failed to open the input data file: %s! Exit now!\n", dataFileName);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    long long int dataElemSize = ftell(fp) / sizeof(T);
    fseek(fp, 0, SEEK_SET);
    if (fread(data.data(), sizeof(T), dataElemSize, fp) == 0 && dataElemSize != 0) {
        fclose(fp);
        printf("The input data file: %s is an empty file! Exit now!\n", dataFileName);
        exit(-1);
    }
    fclose(fp);
}

template<typename T>
void load1DDataFromFile(const char *const dataFileName, std::vector<T> &data, const uint64_t &loadDataAmt, const uint64_t &fileOffset = 0) {
    FILE *fp = fopen(dataFileName, "rb");
    if (fp == NULL) {
        printf("Failed to open the input data file: %s! Exit now!\n", dataFileName);
        exit(-1);
    }
    fseek(fp, fileOffset, SEEK_SET);
    if (fread(data.data(), sizeof(T), loadDataAmt, fp) == 0 && loadDataAmt != 0) {
        fclose(fp);
        printf("Failed to read %lu Bytes from the input data file: %s from offset = %lu! Exit now!\n", loadDataAmt * sizeof(T), dataFileName, fileOffset);
        exit(-1);
    }
    fclose(fp);
}

void load1DUint16sFromFile(const char *const uint16sFileName, std::vector<uint16_t> &uint16s) {
    FILE *fp = fopen(uint16sFileName, "rb");
    if (fp == NULL) {
        printf("Failed to open the input uint16 file: %s! Exit now!\n", uint16sFileName);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    long long int uint16sElemSize = ftell(fp) / sizeof(uint16_t);
    fseek(fp, 0, SEEK_SET);
    if (fread(uint16s.data(), sizeof(uint16_t), uint16sElemSize, fp) == 0 && uint16sElemSize != 0) {
        fclose(fp);
        printf("The input uint16 file: %s is an empty file! Exit now!\n", uint16sFileName);
        exit(-1);
    }
    fclose(fp);
}

void load1DPointsFromFile(const char *const pointsFileName, std::vector<ELEMTYPE> &points) {
    FILE *fp = fopen(pointsFileName, "rb");
    if (fp == NULL) {
        printf("Failed to open the input point file: %s! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    long long int pointsElemSize = ftell(fp) / sizeof(ELEMTYPE);
    fseek(fp, 0, SEEK_SET);
    if (fread(points.data(), sizeof(ELEMTYPE), pointsElemSize, fp) == 0 && pointsElemSize != 0) {
        fclose(fp);
        printf("The input point file: %s is an empty file! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fclose(fp);
}

void load1DPointsFromFile(const char *const pointsFileName, std::vector<ELEMTYPE> &points, const std::vector<ADDRTYPE> &pointSizesPerVec, const ADDRTYPE &stride, const ADDRTYPE &startReadAddr) {
    FILE *fp = fopen(pointsFileName, "rb");
    if (fp == NULL) {
        printf("Failed to open the input point file: %s! Exit now!\n", pointsFileName);
        exit(-1);
    }
    fseek(fp, startReadAddr, SEEK_SET);
    {
        auto pointSizesPerVecIter = pointSizesPerVec.begin();
        for (ELEMTYPE *pointsIter = (ELEMTYPE *)(points.data()); pointSizesPerVecIter < pointSizesPerVec.end(); pointsIter += stride, ++pointSizesPerVecIter) {
            if (fread(pointsIter, sizeof(ELEMTYPE), *pointSizesPerVecIter, fp) == 0 && *pointSizesPerVecIter != 0) {
                fclose(fp);
                printf("The size of input point file: %s is wrong! Exit now!\n", pointsFileName);
                exit(-1);
            }
        }
    }
    fclose(fp);
}

void saveDataToFile(const char *const dataFileName, const void *data, const size_t size, const size_t nmemb) {
    FILE *fp = fopen(dataFileName, "wb");
    if (fp == NULL) {
        printf("Failed to open the output point file: %s! Exit now!\n", dataFileName);
        exit(-1);
    }
    fwrite(data, size, nmemb, fp);
    fclose(fp);
}

template<typename T>
void cvtDataToFloat(const std::vector<T> &data, std::vector<float> &dataF) {
    UINT64 dataSize = data.size();
    #pragma omp parallel for num_threads(NR_JOB_PER_RANK)
    for (UINT64 dataId = 0; dataId < dataSize; ++dataId) {
        dataF[dataId] = data[dataId];
    }
}

typedef struct {
    ADDRTYPE max_dpus;
    uint32_t nr_all_dpus;
    std::vector<std::vector<ADDRTYPE>> *queryAmtBufs;
    std::vector<std::vector<ADDRTYPE>> *selClusterSizes;
    std::vector<std::vector<ADDRTYPE>> *selClusterIds;
    std::vector<std::vector<ADDRTYPE>> *distribQueryIds;
    std::vector<std::vector<ADDRTYPE>> *DPUClusterAddrs;
    std::vector<std::vector<CLUSTER_SIZES_TYPE>> *DPUClusterSizes;
} loadBatchesIntoDPUsContext;
dpu_error_t loadBatchesIntoDPUs(void *args) {
    loadBatchesIntoDPUsContext *ctx = (loadBatchesIntoDPUsContext *)args;
    ADDRTYPE &max_dpus = ctx->max_dpus;
    uint32_t &nr_all_dpus = ctx->nr_all_dpus;
    std::vector<std::vector<ADDRTYPE>> *queryAmtBufs = ctx->queryAmtBufs;
    std::vector<std::vector<ADDRTYPE>> *selClusterSizes = ctx->selClusterSizes;
    std::vector<std::vector<ADDRTYPE>> *selClusterIds = ctx->selClusterIds;
    std::vector<std::vector<ADDRTYPE>> *distribQueryIds = ctx->distribQueryIds;
    std::vector<std::vector<ADDRTYPE>> *DPUClusterAddrs = ctx->DPUClusterAddrs;
    std::vector<std::vector<CLUSTER_SIZES_TYPE>> *DPUClusterSizes = ctx->DPUClusterSizes;

    ADDRTYPE maxSelClusterSizesAmt = 0;
    ADDRTYPE maxSelClusterIdsAmt = 0;
    ADDRTYPE maxDistribQueryIdsAmt = 0;
    ADDRTYPE maxDPUClusterAddrsAmt = 0;
    ADDRTYPE maxDPUClusterSizesAmt = 0;
    for (uint32_t nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
        if ((*selClusterSizes)[nr_dpu].size() > maxSelClusterSizesAmt)
            maxSelClusterSizesAmt = (*selClusterSizes)[nr_dpu].size();
        if ((*selClusterIds)[nr_dpu].size() > maxSelClusterIdsAmt)
            maxSelClusterIdsAmt = (*selClusterIds)[nr_dpu].size();
        if ((*distribQueryIds)[nr_dpu].size() > maxDistribQueryIdsAmt)
            maxDistribQueryIdsAmt = (*distribQueryIds)[nr_dpu].size();
        if ((*DPUClusterAddrs)[nr_dpu].size() > maxDPUClusterAddrsAmt)
            maxDPUClusterAddrsAmt = (*DPUClusterAddrs)[nr_dpu].size();
        if ((*DPUClusterSizes)[nr_dpu].size() > maxDPUClusterSizesAmt)
            maxDPUClusterSizesAmt = (*DPUClusterSizes)[nr_dpu].size();
    }
    if (sizeof(ADDRTYPE) < 8) {  // Make the size of parallel transferred data a multiple of 8
        ADDRTYPE MOD = 8 / sizeof(ADDRTYPE) - 1;
        ADDRTYPE remain = maxSelClusterSizesAmt & MOD;
        if (remain)
            maxSelClusterSizesAmt += (~remain & MOD) + 1;
        remain = maxSelClusterIdsAmt & MOD;
        if (remain)
            maxSelClusterIdsAmt += (~remain & MOD) + 1;
        remain = maxDistribQueryIdsAmt & MOD;
        if (remain)
            maxDistribQueryIdsAmt += (~remain & MOD) + 1;
        remain = maxDPUClusterAddrsAmt & MOD;
        if (remain)
            maxDPUClusterAddrsAmt += (~remain & MOD) + 1;
    }
    if (sizeof(CLUSTER_SIZES_TYPE) < 8) {  // Make the size of parallel transferred data a multiple of 8
        ADDRTYPE MOD = 8 / sizeof(CLUSTER_SIZES_TYPE) - 1;
        ADDRTYPE remain = maxDPUClusterSizesAmt & MOD;
        if (remain)
            maxDPUClusterSizesAmt += (~remain & MOD) + 1;
    }
    #pragma omp parallel for num_threads(NR_JOB_PER_RANK)
    for (uint32_t nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
        if (nr_dpu < max_dpus) {
            (*queryAmtBufs)[nr_dpu][0] = (*selClusterSizes)[nr_dpu].size();
            (*selClusterSizes)[nr_dpu].resize(maxSelClusterSizesAmt);
            (*selClusterIds)[nr_dpu].resize(maxSelClusterIdsAmt);
            (*distribQueryIds)[nr_dpu].resize(maxDistribQueryIdsAmt);
            (*DPUClusterAddrs)[nr_dpu].resize(maxDPUClusterAddrsAmt);
            (*DPUClusterSizes)[nr_dpu].resize(maxDPUClusterSizesAmt);
        } else {
            (*queryAmtBufs)[nr_dpu][0] = (*selClusterSizes)[max_dpus - 1].size();
            (*selClusterSizes)[nr_dpu].resize(maxSelClusterSizesAmt);
            (*selClusterIds)[nr_dpu].resize(maxSelClusterIdsAmt);
            (*distribQueryIds)[nr_dpu].resize(maxDistribQueryIdsAmt);
        }
    }

    return DPU_OK;
}

__attribute__((noreturn)) static void usage(FILE *f, int exit_code, const char *exec_name) {
    /* clang-format off */
    fprintf(f,
#ifdef PERF_EVAL
            "\nusage: %s [-p <points_path>] [-q <queries_path>] [-s <square_result_path>] [-c <centroids_path>] [-b <codebook_path>] [-a <cluster_size_path>] [-l <cluster_layout_result_path>] [-r <radius_result_path>] [-t <square_root_result_path>] [-k <knn_result_path>] [-D <number_of_dimension>] [-F <frequency_of_dpus>] [-K <number_of_neighbors>] [-C <number_of_searched_clusters>] [-Q <number_of_queries_per_batch>] [-M <number_of_subvectors>] [-G <size_of_DPU_groups>] [-S <size_of_cluster_slices>] [-U <number_of_mrams>]\n"
#else
            "\nusage: %s [-p <points_path>] [-q <queries_path>] [-s <square_result_path>] [-c <centroids_path>] [-b <codebook_path>] [-a <cluster_size_path>] [-l <cluster_layout_result_path>] [-r <radius_result_path>] [-t <square_root_result_path>] [-k <knn_result_path>] [-D <number_of_dimension>] [-K <number_of_neighbors>] [-C <number_of_searched_clusters>] [-Q <number_of_queries_per_batch>] [-M <number_of_subvectors>] [-G <size_of_DPU_groups>] [-S <size_of_cluster_slices>] [-U <number_of_mrams>]\n"
#endif
            "\n"
            "\t-p \tthe path to the cluster location (default: clusters.bin)\n"
            "\t-q \tthe path to the query location (default: query.bin)\n"
            "\t-s \tthe path to the square result location (default: squareRes.bin)\n"
            "\t-c \tthe path to the centroid location (default: centroids.bin)\n"
            "\t-b \tthe path to the codebook location (default: codebook.bin)\n"
            "\t-a \tthe path to the cluster size location (default: clusterSizes.bin)\n"
            "\t-l \tthe path to the cluster layout location (default: clusterLayout.bin)\n"
            "\t-r \tthe path to the radius location (default: radii.bin)\n"
            "\t-t \tthe path to the file to save the square result (default: squareRoots.bin)\n"
            "\t-k \tthe path to the k nearest neighbor, a.k.a, the result graph, location (default: knn.bin)\n"
            "\t-D \tthe number of dimensions of input points (default: 128)\n"
#ifdef CYCLE_PERF_EVAL
            "\t-F \tthe frequency of DPUs (default: 450000000)\n"
#endif
            "\t-K \tthe number of neighbors (default: 10)\n"
            "\t-C \tthe number of searched clusters for each query (default: 1)\n"
            "\t-Q \tthe number of queries in each batch (default: 1)\n"
            "\t-M \tthe number of subvectors generated by each point (default: 8)\n"
            "\t-G \tthe maximal size of each DPU group (default: 1)\n"
            "\t-S \tthe size of split cluster slices. Unit: Byte (default: 1)\n"
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

#ifdef CYCLE_PERF_EVAL
static void parse_args(int argc, char **argv, uint32_t *dimAmt, uint32_t *neighborAmt, uint32_t *sliceAmt, uint32_t *queryBatchSize, ADDRTYPE *nprobe, uint32_t *DPUGroupSize, uint32_t *clusterSliceSize, uint32_t *nb_mram, uint64_t *frequency, std::string &clustersFileName, std::string &queriesFileName, std::string &squareResFileName, std::string &centroidsFileName, std::string &codebookFileName, std::string &clusterSizesFileName, std::string &clusterLayoutFileName, std::string &radiiFileName, std::string &squareRootsFileName, std::string &knnFileName) {
#else
static void parse_args(int argc, char **argv, uint32_t *dimAmt, uint32_t *neighborAmt, uint32_t *sliceAmt, uint32_t *queryBatchSize, ADDRTYPE *nprobe, uint32_t *DPUGroupSize, uint32_t *clusterSliceSize, uint32_t *nb_mram, std::string &clustersFileName, std::string &queriesFileName, std::string &squareResFileName, std::string &centroidsFileName, std::string &codebookFileName, std::string &clusterSizesFileName, std::string &clusterLayoutFileName, std::string &radiiFileName, std::string &squareRootsFileName, std::string &knnFileName) {
#endif
    int opt;
    extern char *optarg;
#ifdef PERF_EVAL
    while ((opt = getopt(argc, argv, "hD:K:C:Q:M:G:S:F:U:p:q:s:c:b:a:l:r:t:k:")) != -1) {
#else
    while ((opt = getopt(argc, argv, "hD:K:C:Q:M:G:S:U:p:q:s:c:b:a:l:r:t:k:")) != -1) {
#endif
        switch (opt) {
            case 'p':
                clustersFileName = optarg;
                break;
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
            case 'G':
                *DPUGroupSize = (uint32_t)atoi(optarg);
                break;
            case 'S':
                *clusterSliceSize = (uint32_t)atoi(optarg);
                break;
#ifdef CYCLE_PERF_EVAL
            case 'F':
                *frequency = (uint64_t)atoi(optarg);
                break;
#endif
            case 's':
                squareResFileName = optarg;
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
            case 'r':
                radiiFileName = optarg;
                break;
            case 't':
                squareRootsFileName = optarg;
                break;
            case 'k':
                knnFileName = optarg;
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
    verify_path_exists(clustersFileName.c_str());
    verify_path_exists(queriesFileName.c_str());
    verify_path_exists(squareResFileName.c_str());
    verify_path_exists(centroidsFileName.c_str());
    verify_path_exists(codebookFileName.c_str());
    verify_path_exists(clusterSizesFileName.c_str());
    verify_path_exists(clusterLayoutFileName.c_str());
    verify_path_exists(radiiFileName.c_str());
    verify_path_exists(squareRootsFileName.c_str());
}

#ifdef CYCLE_PERF_EVAL
static void allocated_and_compute(dpu::DpuSet &dpu_set, const uint32_t dimAmt, const uint32_t neighborAmt, const uint32_t sliceAmt, const uint32_t queryBatchSize, const ADDRTYPE nprobe, const uint32_t DPUGroupSize, const uint32_t clusterSliceSize, const uint64_t &frequency, const UINT64 *const latency, const char *const clustersFileName, const char *const queriesFileName, const char *const squareResFileName, const char *const centroidsFileName, const char *const codebookFileName, const char *const clusterSizesFileName, const char *const clusterLayoutFileName, const char *const radiiFileName, const char *const squareRootsFileName, const char *const knnFileName) {
#else
static void allocated_and_compute(dpu::DpuSet &dpu_set, const uint32_t dimAmt, const uint32_t neighborAmt, const uint32_t sliceAmt, const uint32_t queryBatchSize, const ADDRTYPE nprobe, const uint32_t DPUGroupSize, const uint32_t clusterSliceSize, const UINT64 *const latency, const char *const clustersFileName, const char *const queriesFileName, const char *const squareResFileName, const char *const centroidsFileName, const char *const codebookFileName, const char *const clusterSizesFileName, const char *const clusterLayoutFileName, const char *const radiiFileName, const char *const squareRootsFileName, const char *const knnFileName) {
#endif
#ifdef ENERGY_EVAL
    double ESU = getEnergyUnit();
    uint32_t nr_sockets = getNRSockets();
    uint32_t nr_cpus = getNRPhyCPUs();
    uint32_t evalCPUIds[nr_sockets];
    for (uint32_t nr_cpu = 0, nr_socket = 0, coresPerSocket = nr_cpus / nr_sockets; nr_socket < nr_sockets; nr_cpu += coresPerSocket, ++nr_socket)
        evalCPUIds[nr_socket] = nr_cpu;
    uint64_t totalExecEnergy = 0;
    uint64_t startEnergy[nr_sockets], endEnergy[nr_sockets];
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
#ifdef PERF_EVAL
    uint64_t totalExecTime = 0;  // Unit: us
    uint64_t dpuExecTime = 0, hostExecTime = 0, dataTransferhost2DPUTime = 0, dataTransferDPU2hostTime = 0;  // Unit: us
    uint64_t staticDataTransferTime = 0, IVFExecTime = 0, PQExecTime = 0, prepareTime = 0;  // Unit: us
#endif
#if (defined PRINT_PERF_EACH_PHASE || defined PERF_EVAL)
    long start, end;
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif

    uint32_t nr_ranks = dpu_set.ranks().size();
    uint32_t nr_all_dpus = dpu_set.dpus().size();
    std::vector<uint16_t> squareRes(1 << (sizeof(ELEMTYPE) << 3));
    load1DUint16sFromFile(squareResFileName, squareRes);
    const ADDRTYPE queryAmt = getPointsAmount(queriesFileName, dimAmt);
    std::vector<ELEMTYPE> queries(queryAmt * dimAmt);
    load1DPointsFromFile(queriesFileName, queries);
    // Set dpu_offset
    uint32_t dpu_offset[nr_ranks + 1];
    dpu_offset[0] = 0;
    #pragma omp parallel for num_threads(NR_JOB_PER_RANK)
    for (ADDRTYPE i = 0; i < nr_ranks; ++i)
        dpu_offset[i + 1] = dpu_offset[i] + dpu_set.ranks()[i]->dpus().size();

    // 1. Load index
    printf("Loading indices:\n");
    const ADDRTYPE centroidAmt = getElemsAmount(centroidsFileName, dimAmt * sizeof(CENTROIDS_TYPE));
    std::vector<std::vector<CENTROIDS_TYPE>> centroids(nr_all_dpus);
    std::vector<CENTROIDS_TYPE> centroidsFlat(centroidAmt * dimAmt);
    uint32_t nr_groups = nr_all_dpus / DPUGroupSize + (nr_all_dpus % DPUGroupSize > 0 ? 1 : 0);
    --nr_groups, ++nr_groups;
    ADDRTYPE clusterAmt = centroidAmt;
    std::vector<std::vector<CLUSTER_SIZES_TYPE>> clusterSizes(nr_all_dpus, std::vector<CLUSTER_SIZES_TYPE>());
    std::vector<CLUSTER_SIZES_TYPE> clusterSizesFlat(clusterAmt);
    load1DDataFromFile(clusterSizesFileName, clusterSizesFlat);
    std::vector<SQUAREROOTDISTTYPE> radiiFlat(clusterAmt);
    load1DDataFromFile(radiiFileName, radiiFlat);
    std::vector<std::vector<ClusterLayout>> clusterLayout(nr_all_dpus);
    std::vector<std::vector<ClusterIndex>> clusterDirectory(clusterAmt);
    {  // Load cluster layout
        uint64_t clusterLayoutFileOffset = 0;
        // Load cluster slice information of each DPU
        std::vector<CLUSTER_SIZES_TYPE> clusterSizesOnTheDPU(nr_all_dpus);
        load1DDataFromFile(clusterLayoutFileName, clusterSizesOnTheDPU, clusterSizesOnTheDPU.size(), clusterLayoutFileOffset);
        clusterLayoutFileOffset += clusterSizesOnTheDPU.size() * sizeof(CLUSTER_SIZES_TYPE);
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            clusterLayout[nr_dpu].resize(clusterSizesOnTheDPU[nr_dpu]);
        }
        std::vector<CLUSTER_SIZES_TYPE> clusterSizesOnTheDPUPrefix(nr_all_dpus);
        clusterSizesOnTheDPUPrefix[0] = 0;
        for (ADDRTYPE nr_dpu = 1; nr_dpu < nr_all_dpus; ++nr_dpu) {
            clusterSizesOnTheDPUPrefix[nr_dpu] = clusterSizesOnTheDPUPrefix[nr_dpu - 1] + clusterSizesOnTheDPU[nr_dpu - 1];
        }
        {
            std::vector<ClusterLayout> clusterLayout1D(clusterSizesOnTheDPUPrefix[nr_all_dpus - 1] + clusterSizesOnTheDPU[nr_all_dpus - 1]);
            load1DDataFromFile(clusterLayoutFileName, clusterLayout1D, clusterLayout1D.size(), clusterLayoutFileOffset);
            clusterLayoutFileOffset += clusterLayout1D.size() * sizeof(ClusterLayout);
            #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
            for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
                memcpy(reinterpret_cast<uint8_t *>(clusterLayout[nr_dpu].data()), reinterpret_cast<uint8_t *>(clusterLayout1D.data()) + clusterSizesOnTheDPUPrefix[nr_dpu] * sizeof(ClusterLayout), clusterSizesOnTheDPU[nr_dpu] * sizeof(ClusterLayout));
            }
        }
        // Load cluster-DPU mapping
        std::vector<CLUSTER_SIZES_TYPE> dpuSizesOfClusters(clusterAmt);
        load1DDataFromFile(clusterLayoutFileName, dpuSizesOfClusters, dpuSizesOfClusters.size(), clusterLayoutFileOffset);
        clusterLayoutFileOffset += dpuSizesOfClusters.size() * sizeof(CLUSTER_SIZES_TYPE);
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE clusterId = 0; clusterId < clusterAmt; ++clusterId) {
            clusterDirectory[clusterId].resize(dpuSizesOfClusters[clusterId], ClusterIndex(-1, -1));
        }
        std::vector<CLUSTER_SIZES_TYPE> dpuSizesOfClustersPrefix(clusterAmt);
        dpuSizesOfClustersPrefix[0] = 0;
        for (ADDRTYPE clusterId = 1; clusterId < clusterAmt; ++clusterId) {
            dpuSizesOfClustersPrefix[clusterId] = dpuSizesOfClustersPrefix[clusterId - 1] + dpuSizesOfClusters[clusterId - 1];
        }
        {
            std::vector<ClusterIndex> clusterDirectory1D(dpuSizesOfClustersPrefix[clusterAmt - 1] + dpuSizesOfClusters[clusterAmt - 1], ClusterIndex(-1, -1));
            load1DDataFromFile(clusterLayoutFileName, clusterDirectory1D, clusterDirectory1D.size(), clusterLayoutFileOffset);
            clusterLayoutFileOffset += clusterDirectory1D.size() * sizeof(ClusterIndex);
            #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
            for (ADDRTYPE clusterId = 0; clusterId < clusterAmt; ++clusterId) {
                memcpy(reinterpret_cast<uint8_t *>(clusterDirectory[clusterId].data()), reinterpret_cast<uint8_t *>(clusterDirectory1D.data()) + dpuSizesOfClustersPrefix[clusterId] * sizeof(ClusterIndex), dpuSizesOfClusters[clusterId] * sizeof(ClusterIndex));
            }
        }
    }
    ADDRTYPE pointAmtPerSlice = 1;
    {
        ADDRTYPE pointSize = sizeof(POINTTYPE) * sliceAmt;
        pointAmtPerSlice = std::ceil(static_cast<float>(clusterSliceSize) / pointSize);
    }
    const ADDRTYPE codebookEntryAmt = getElemsAmount(codebookFileName, dimAmt * sizeof(CB_TYPE));
    std::vector<CB_TYPE> codebook(dimAmt * codebookEntryAmt);
    load1DDataFromFile(codebookFileName, codebook);
    std::vector<std::vector<ADDRTYPE>> localClusterAddrs(nr_all_dpus);
    {  // Set clusterAddrs, load clusterSizes
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            ADDRTYPE localClusterAddr = 0;
            for (auto &clusterLayoutDPU: clusterLayout[nr_dpu]) {
                localClusterAddrs[nr_dpu].push_back(localClusterAddr);
                localClusterAddr += (clusterLayoutDPU.endSliceId - clusterLayoutDPU.startSliceId) * pointAmtPerSlice;
            }
        }
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            clusterSizes[nr_dpu].resize(localClusterAddrs[nr_dpu].size());
            if (localClusterAddrs[nr_dpu].size() < 1)
                continue;
            ADDRTYPE localClusterAmtIdx = localClusterAddrs[nr_dpu].size() - 1;
            for (ADDRTYPE localClusterId = 0; localClusterId < localClusterAmtIdx; ++localClusterId) {
                clusterSizes[nr_dpu][localClusterId] = localClusterAddrs[nr_dpu][localClusterId + 1] - localClusterAddrs[nr_dpu][localClusterId];
                if (clusterSizesFlat[clusterLayout[nr_dpu][localClusterId].clusterId] <= clusterLayout[nr_dpu][localClusterId].endSliceId * pointAmtPerSlice) {  // The last slice of a cluster may be not full
                    ADDRTYPE pointAmtInLastSlice = clusterSizesFlat[clusterLayout[nr_dpu][localClusterId].clusterId] % pointAmtPerSlice;
                    if (pointAmtInLastSlice)
                        clusterSizes[nr_dpu][localClusterId] -= pointAmtPerSlice - pointAmtInLastSlice;
                }
            }
            clusterSizes[nr_dpu][localClusterAmtIdx] = (clusterLayout[nr_dpu][localClusterAmtIdx].endSliceId - clusterLayout[nr_dpu][localClusterAmtIdx].startSliceId) * pointAmtPerSlice;
            if (clusterSizesFlat[clusterLayout[nr_dpu][localClusterAmtIdx].clusterId] <= clusterLayout[nr_dpu][localClusterAmtIdx].endSliceId * pointAmtPerSlice) {  // The last slice of a cluster may be not full
                ADDRTYPE pointAmtInLastSlice = clusterSizesFlat[clusterLayout[nr_dpu][localClusterAmtIdx].clusterId] % pointAmtPerSlice;
                if (pointAmtInLastSlice)
                    clusterSizes[nr_dpu][localClusterAmtIdx] -= pointAmtPerSlice - pointAmtInLastSlice;
            }
        }
    }
    load1DDataFromFile(centroidsFileName, centroidsFlat);
    std::vector<std::vector<SQUAREROOTDISTTYPE>> radii(nr_all_dpus);
    #pragma omp parallel for num_threads(NR_JOB_PER_RANK)
    for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
        for (auto &clusterLayoutDPU: clusterLayout[nr_dpu]) {
            // Set centroids
            auto centroidIter = centroidsFlat.begin() + clusterLayoutDPU.clusterId * dimAmt;
            centroids[nr_dpu].insert(centroids[nr_dpu].end(), centroidIter, centroidIter + dimAmt);
            // Set radii
            radii[nr_dpu].push_back(radiiFlat[clusterLayoutDPU.clusterId]);
        }
    }
    std::vector<std::vector<POINTTYPE>> clusters(nr_all_dpus);
    std::vector<std::vector<UINT64>> globalPointIds(nr_all_dpus);
    {  // Load clusters. Load them from disk to memory at first, then distribute them to the corresponding sub-vector list of each DPU from the host memory
        std::vector<POINTTYPE> clustersFlat(getElemsAmount(clustersFileName, sizeof(POINTTYPE)));
        load1DDataFromFile(clustersFileName, clustersFlat);
        std::vector<ADDRTYPE> globalClusterAddrs(clusterAmt);
        globalClusterAddrs[0] = 0;
        for (ADDRTYPE clusterId = 1; clusterId < clusterAmt; ++clusterId) {
            globalClusterAddrs[clusterId] = globalClusterAddrs[clusterId - 1] + clusterSizesFlat[clusterId - 1];
        }
        std::vector<UINT64> basePointIds(clustersFlat.size() / sliceAmt);
        #pragma omp parallel for num_threads(NR_JOB_PER_RANK)
        for (UINT64 pId = 0; pId < basePointIds.size(); ++pId) {
            basePointIds[pId] = pId;
        }
        #pragma omp parallel for num_threads(NR_JOB_PER_RANK)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            // Load and distribute clusters upon each DPU
            auto &clusterSizesDPU = clusterSizes[nr_dpu];
            for (ADDRTYPE localClusterId = 0, localClusterAmt = clusterLayout[nr_dpu].size(); localClusterId < localClusterAmt; ++localClusterId) {
                auto &clusterLayoutDPU = clusterLayout[nr_dpu][localClusterId];
                auto clusterPointsIter = clustersFlat.begin() + (globalClusterAddrs[clusterLayoutDPU.clusterId] + clusterLayoutDPU.startSliceId * pointAmtPerSlice) * sliceAmt;
                size_t curClusterSize = clusters[nr_dpu].size();
                clusters[nr_dpu].insert(clusters[nr_dpu].end(), clusterPointsIter, clusterPointsIter + clusterSizesDPU[localClusterId] * sliceAmt);
                clusters[nr_dpu].resize(curClusterSize + (clusterLayoutDPU.endSliceId - clusterLayoutDPU.startSliceId) * pointAmtPerSlice * sliceAmt);  // Take cluster slice as the unit
                UINT64 preLocalPointAmt = globalPointIds[nr_dpu].size();
                globalPointIds[nr_dpu].resize(preLocalPointAmt + (clusterLayoutDPU.endSliceId - clusterLayoutDPU.startSliceId) * pointAmtPerSlice);
                memcpy(reinterpret_cast<UINT64 *>(globalPointIds[nr_dpu].data()) + preLocalPointAmt, reinterpret_cast<UINT64 *>(basePointIds.data()) + globalClusterAddrs[clusterLayoutDPU.clusterId] + clusterLayoutDPU.startSliceId * pointAmtPerSlice, clusterSizesDPU[localClusterId] * sizeof(UINT64));
            }
        }
    }
    {  // Padding for parallel transfer to DPUs
        ADDRTYPE maxLocalClusterAddrsSize = 0;
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            if (localClusterAddrs[nr_dpu].size() > maxLocalClusterAddrsSize)
                maxLocalClusterAddrsSize = localClusterAddrs[nr_dpu].size();
        }
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            localClusterAddrs[nr_dpu].resize(maxLocalClusterAddrsSize);
            clusterSizes[nr_dpu].resize(maxLocalClusterAddrsSize);
            radii[nr_dpu].resize(maxLocalClusterAddrsSize);
        }
        ADDRTYPE maxLocalCentroidsSize = maxLocalClusterAddrsSize * dimAmt;
        #pragma omp parallel for num_threads(NR_JOB_PER_RANK)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            centroids[nr_dpu].resize(maxLocalCentroidsSize);
        }
        ADDRTYPE maxLocalClusterPointsSize = 0;
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            if (clusters[nr_dpu].size() > maxLocalClusterPointsSize)
                maxLocalClusterPointsSize = clusters[nr_dpu].size();
        }
        #pragma omp parallel for num_threads(NR_JOB_PER_RANK)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            clusters[nr_dpu].resize(maxLocalClusterPointsSize);
        }
    }
    std::vector<VECSUMTYPE> largeSquareRes(getElemsAmount(squareRootsFileName, sizeof(VECSUMTYPE)));
    load1DDataFromFile(squareRootsFileName, largeSquareRes);
    // Convert data to float to use faiss for IVF coarse search
    std::vector<float> queriesF(queries.size());
    std::vector<float> centroidsFlatF(centroidsFlat.size());
    cvtDataToFloat(queries, queriesF);
    cvtDataToFloat(centroidsFlat, centroidsFlatF);
    faiss::IndexFlatL2 interQuantizer(dimAmt);
    interQuantizer.add(centroidAmt, const_cast<float *>(centroidsFlatF.data()));
    faiss::Index *quantizer = &interQuantizer;
    dpu_set.load(dpu_binary_CBS);
    std::vector<pqueue_elem_t_mram> neighbors(queryAmt * neighborAmt);
    std::vector<std::vector<ADDRTYPE>> queryAmtBufs(nr_all_dpus, std::vector<ADDRTYPE>(1));
    std::vector<std::vector<pqueue_elem_t_mram>> neighborsDPU(nr_all_dpus, std::vector<pqueue_elem_t_mram>(queryBatchSize * neighborAmt));
#ifdef PERF_EVAL
    gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    totalExecTime += end - start;
    prepareTime += end - start;
#ifdef ENERGY_EVAL
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
#if (defined MODULE_PERF_EVAL || defined CYCLE_PERF_EVAL)
    std::vector<std::vector<perfcounter_t>> exec_times_trans(nr_all_dpus, std::vector<perfcounter_t>(MODULE_TYPES_END * NR_TASKLETS));
    std::vector<std::vector<unsigned long long int>> exec_times(nr_all_dpus, std::vector<unsigned long long int>(MODULE_TYPES_END * NR_TASKLETS, 0));
    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
    {
        dpu_set.copy(STR(centroids), 0, centroids, align8bytes(centroids[0].size() * sizeof(CENTROIDS_TYPE)));
        dpu_set.copy("points", 0, clusters, align8bytes(clusters[0].size() * sizeof(POINTTYPE)));
        dpu_set.copy(STR(radii), 0, radii, align8bytes(radii[0].size() * sizeof(SQUAREROOTDISTTYPE)));
        // Broadcast
        dpu_set.copy(STR(squareRes), 0, squareRes, align8bytes((1 << (sizeof(ELEMTYPE) << 3)) * sizeof(uint16_t)));
        dpu_set.copy(STR(codebook), 0, codebook, align8bytes(codebook.size() * sizeof(CB_TYPE)));
        dpu_set.copy(STR(codebookEntryAmt), 0, std::vector<ADDRTYPE>({codebookEntryAmt}), sizeof(ADDRTYPE));
        dpu_set.copy(STR(neighborAmt), 0, std::vector<uint32_t>({neighborAmt}), sizeof(uint32_t));
        dpu_set.copy(STR(dimAmt), 0, std::vector<uint32_t>({dimAmt}), sizeof(uint32_t));
        dpu_set.copy("svDimAmt", 0, std::vector<uint32_t>({static_cast<uint32_t>(std::ceil(static_cast<float>(dimAmt) / sliceAmt))}), sizeof(uint32_t));
        dpu_set.copy(STR(largeSquareRes), 0, largeSquareRes, align8bytes(largeSquareRes.size() * sizeof(VECSUMTYPE)));
    }
    printf("End of preparing...\n");
    fflush(stdout);
#ifdef PERF_EVAL
    gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    totalExecTime += end - start;
    prepareTime += end - start;
    staticDataTransferTime += end - start;
#ifdef ENERGY_EVAL
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
    for (ADDRTYPE queryId = 0; queryId < queryAmt; queryId += queryBatchSize) {
        std::vector<std::vector<ADDRTYPE>> selClusterIds(nr_all_dpus);
        std::vector<std::vector<ADDRTYPE>> selClusterSizes(nr_all_dpus);
        std::vector<std::vector<ADDRTYPE>> distribQueryIds(nr_all_dpus);
        std::vector<std::vector<ADDRTYPE>> DPUClusterAddrs(nr_all_dpus);
        std::vector<std::vector<CLUSTER_SIZES_TYPE>> DPUClusterSizes(nr_all_dpus);
        ADDRTYPE processedQuerySize = queryId * dimAmt;
        auto queryBatch = queries.begin() + processedQuerySize;
        uint32_t realQueryBatchSize = std::min(queryBatchSize, queryAmt - queryId);
        uint32_t realQueryBatchSizeSpace = realQueryBatchSize * dimAmt;
        std::vector<std::vector<std::pair<ADDRTYPE, ADDRTYPE>>> queryDPUIds(realQueryBatchSize);

        // 2. IVF coarse search -- CPU(HyperThreading + AVX512)
        printf("IVF searching phase:\n");
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        hostExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        // Note: the order of elements of the input `clusterDirectory[clusterId]` may be changed in this function!
        IVFsearch(reinterpret_cast<float *>(queriesF.data()) + processedQuerySize, realQueryBatchSize, dimAmt, quantizer, nprobe, pointAmtPerSlice, selClusterIds, selClusterSizes, distribQueryIds, queryDPUIds, localClusterAddrs, clusterSizes, latency, clusterSizesFlat, clusterLayout, clusterDirectory, DPUClusterAddrs, DPUClusterSizes);
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        hostExecTime += end - start;
        IVFExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif

        // 3. PQ cluster scan -- DPU
        printf("Cluster searching phase:\n");
        // Send data to DPUs
        dpu_set.copy(STR(sliceAmt), 0, std::vector<uint32_t>({sliceAmt}), sizeof(uint32_t));
        dpu_set.copy(STR(queries), 0, std::vector<ELEMTYPE>(queryBatch, queryBatch + realQueryBatchSizeSpace), align8bytes(sizeof(ELEMTYPE) * realQueryBatchSizeSpace));
        loadBatchesIntoDPUsContext loadBatchesIntoDPUsContext_ctx = { .max_dpus = nr_all_dpus, .nr_all_dpus = nr_all_dpus, .queryAmtBufs = &queryAmtBufs, .selClusterSizes = &selClusterSizes, .selClusterIds = &selClusterIds, .distribQueryIds = &distribQueryIds, .DPUClusterAddrs = &DPUClusterAddrs, .DPUClusterSizes = &DPUClusterSizes };
        CPP_DPU_ASSERT(loadBatchesIntoDPUs(&loadBatchesIntoDPUsContext_ctx) == DPU_OK);
        dpu_set.copy(STR(queryAmt), 0, queryAmtBufs, sizeof(ADDRTYPE));
        dpu_set.copy("localClusterIDIdxs", 0, selClusterSizes, align8bytes(selClusterSizes[0].size() * sizeof(ADDRTYPE)));
        dpu_set.copy("localClusterIDs", 0, selClusterIds, align8bytes(selClusterIds[0].size() * sizeof(ADDRTYPE)));
        dpu_set.copy("localQueryIDs", 0, distribQueryIds, align8bytes(distribQueryIds[0].size() * sizeof(ADDRTYPE)));
        dpu_set.copy("localClusterAddrs", 0, DPUClusterAddrs, align8bytes(DPUClusterAddrs[0].size() * sizeof(ADDRTYPE)));
        dpu_set.copy("localClusterSizes", 0, DPUClusterSizes, align8bytes(DPUClusterSizes[0].size() * sizeof(CLUSTER_SIZES_TYPE)));
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dataTransferhost2DPUTime += end - start;
        PQExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        // Execute on DPUs
        dpu_set.exec();

        // 4. Transfer results from DPU
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dpuExecTime += end - start;
        PQExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
#if (defined MODULE_PERF_EVAL || defined CYCLE_PERF_EVAL)
        dpu_set.copy(exec_times_trans, MODULE_TYPES_END * NR_TASKLETS * sizeof(perfcounter_t), STR(exec_times), 0);
        {
            auto exec_times_transIter = exec_times_trans.begin();
            for (auto exec_timesIter = exec_times.begin(); exec_timesIter < exec_times.end(); ++exec_timesIter, ++exec_times_transIter) {
                auto exec_times_transElemIter = exec_times_transIter->begin();
                for (auto exec_timesElemIter = exec_timesIter->begin(); exec_timesElemIter < exec_timesIter->end(); ++exec_timesElemIter, ++exec_times_transElemIter) {
                    *exec_timesElemIter += *exec_times_transElemIter;
                }
            }
        }
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        // Get responses and merge top-k results of the same query
        dpu_set.copy(neighborsDPU, neighborAmt * distribQueryIds[0].size() * sizeof(pqueue_elem_t_mram), STR(neighbors), 0);
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        dataTransferDPU2hostTime += end - start;
        PQExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        // Convert local point ids to global ones
        #pragma omp parallel for num_threads(NR_JOB_PARALLEL)
        for (ADDRTYPE nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            auto &neighborDPU = neighborsDPU[nr_dpu];
            for (ADDRTYPE neighborId = 0, neighborIdEnd = queryAmtBufs[nr_dpu][0] * neighborAmt; neighborId < neighborIdEnd; ++neighborId) {
                neighborDPU[neighborId].val = globalPointIds[nr_dpu][neighborDPU[neighborId].val >> 8];  // Cover the `pos` field
            }
        }
        ADDRTYPE batchQueryEnd = queryId + realQueryBatchSize;
        for (ADDRTYPE batchQueryId = queryId, localQueryId = 0; batchQueryId < batchQueryEnd; ++batchQueryId, ++localQueryId) {
            mergeTopk(batchQueryId, localQueryId, queryDPUIds, neighborsDPU, neighborAmt, neighbors);
        }
#ifdef PERF_EVAL
        gettimeofday(&timecheck, NULL);
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
#endif
        end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
        totalExecTime += end - start;
        hostExecTime += end - start;
        PQExecTime += end - start;
#ifdef ENERGY_EVAL
        for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
            startEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
#endif
        gettimeofday(&timecheck, NULL);
        start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
        DPUClusterSizes.clear();
        DPUClusterAddrs.clear();
        selClusterIds.clear();
        selClusterSizes.clear();
        distribQueryIds.clear();
        queryDPUIds.clear();
    }
    dpu_set.async().sync();
#ifdef PRINT_PERF_EACH_PHASE
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    printf("[Host]  Total time until cluster searching phase completed: %.3lfs\n", (end - start) / 1e6);
#endif
#ifdef PERF_EVAL
    gettimeofday(&timecheck, NULL);
#endif
#ifdef ENERGY_EVAL
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        endEnergy[nr_socket] = getEnergy(evalCPUIds[nr_socket]);
    for (uint32_t nr_socket = 0; nr_socket < nr_sockets; ++nr_socket)
        totalExecEnergy += (endEnergy[nr_socket] - startEnergy[nr_socket]) & MSR_ENERGY_MASK;
    printf("[Host]  Total energy for IVFPQ searching: %.6lfJ\n", totalExecEnergy * ESU);
#endif
#ifdef PERF_EVAL
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    totalExecTime += end - start;
#endif
#ifdef CYCLE_PERF_EVAL
    {
        std::vector<unsigned long long int> max_exec_times(1, 0);
        std::vector<unsigned long long int> min_exec_times(1, ((unsigned long long int)(1) << ((sizeof(unsigned long long int) << 3) - 1)) - 1 + ((unsigned long long int)(1) << ((sizeof(unsigned long long int) << 3) - 1)));
        std::vector<double> avg_exec_times(1, 0);
        for (auto &exec_time: exec_times) {
            unsigned long long int exec_time_idx = 0, exec_time_size = exec_time.size();
            for (; exec_time_idx < exec_time_size; exec_time_idx += MODULE_TYPES_END) {
                // Max
                if (exec_time[exec_time_idx + AFFILIATE_OPS] > max_exec_times[0])
                    max_exec_times[0] = exec_time[exec_time_idx + AFFILIATE_OPS];
            }
        }
        dpuExecTime = max_exec_times[0] / frequency * 1e6;
    }
    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif
#ifdef PERF_EVAL
    printf("[Host]  Total time for query searching: %.6lfs\n", std::max(dpuExecTime, hostExecTime + dataTransferhost2DPUTime + dataTransferDPU2hostTime) / 1e6);
    // printf("[Host]  Time for DPU execution: %.6lfs, host execution: %.6lfs, data transfer host2DPU: %.6lfs, data transfer DPU2host: %.6lfs\n", dpuExecTime / 1e6, hostExecTime / 1e6, dataTransferhost2DPUTime / 1e6, dataTransferDPU2hostTime / 1e6);
    // printf("[Host]  Time for data preparing Phase: %.6lfs\n", prepareTime / 1e6);
#endif
#ifdef MODULE_PERF_EVAL
    printf("DPU Summary:\n");
    {
        std::vector<unsigned long long int> max_exec_times(MODULE_TYPES_END, 0);
        std::vector<unsigned long long int> min_exec_times(MODULE_TYPES_END, ((unsigned long long int)(1) << ((sizeof(unsigned long long int) << 3) - 1)) - 1 + ((unsigned long long int)(1) << ((sizeof(unsigned long long int) << 3) - 1)));
        std::vector<double> avg_exec_times(MODULE_TYPES_END, 0);
        for (auto &exec_time: exec_times) {
            unsigned long long int exec_time_idx = 0, exec_time_size = exec_time.size();
            for (; exec_time_idx < exec_time_size; exec_time_idx += MODULE_TYPES_END) {
                // Max
                if (exec_time[exec_time_idx + PRE_DEFINING] > max_exec_times[PRE_DEFINING])
                    max_exec_times[PRE_DEFINING] = exec_time[exec_time_idx + PRE_DEFINING];
                if (exec_time[exec_time_idx + AFFILIATE_OPS] > max_exec_times[AFFILIATE_OPS])
                    max_exec_times[AFFILIATE_OPS] = exec_time[exec_time_idx + AFFILIATE_OPS];
                if (exec_time[exec_time_idx + CAL_RESIDUAL] > max_exec_times[CAL_RESIDUAL])
                    max_exec_times[CAL_RESIDUAL] = exec_time[exec_time_idx + CAL_RESIDUAL];
                if (exec_time[exec_time_idx + CONSTR_LUT] > max_exec_times[CONSTR_LUT])
                    max_exec_times[CONSTR_LUT] = exec_time[exec_time_idx + CONSTR_LUT];
                if (exec_time[exec_time_idx + CLUSTER_LOADING] > max_exec_times[CLUSTER_LOADING])
                    max_exec_times[CLUSTER_LOADING] = exec_time[exec_time_idx + CLUSTER_LOADING];
                if (exec_time[exec_time_idx + CAL_DISTANCE] > max_exec_times[CAL_DISTANCE])
                    max_exec_times[CAL_DISTANCE] = exec_time[exec_time_idx + CAL_DISTANCE];
                if (exec_time[exec_time_idx + TOPK_SORT] > max_exec_times[TOPK_SORT])
                    max_exec_times[TOPK_SORT] = exec_time[exec_time_idx + TOPK_SORT];
                if (exec_time[exec_time_idx + TOPK_SAVING] > max_exec_times[TOPK_SAVING])
                    max_exec_times[TOPK_SAVING] = exec_time[exec_time_idx + TOPK_SAVING];
                // Min
                if (exec_time[exec_time_idx + PRE_DEFINING] < min_exec_times[PRE_DEFINING])
                    min_exec_times[PRE_DEFINING] = exec_time[exec_time_idx + PRE_DEFINING];
                if (exec_time[exec_time_idx + AFFILIATE_OPS] < min_exec_times[AFFILIATE_OPS])
                    min_exec_times[AFFILIATE_OPS] = exec_time[exec_time_idx + AFFILIATE_OPS];
                if (exec_time[exec_time_idx + CAL_RESIDUAL] < min_exec_times[CAL_RESIDUAL])
                    min_exec_times[CAL_RESIDUAL] = exec_time[exec_time_idx + CAL_RESIDUAL];
                if (exec_time[exec_time_idx + CONSTR_LUT] < min_exec_times[CONSTR_LUT])
                    min_exec_times[CONSTR_LUT] = exec_time[exec_time_idx + CONSTR_LUT];
                if (exec_time[exec_time_idx + CLUSTER_LOADING] < min_exec_times[CLUSTER_LOADING])
                    min_exec_times[CLUSTER_LOADING] = exec_time[exec_time_idx + CLUSTER_LOADING];
                if (exec_time[exec_time_idx + CAL_DISTANCE] < min_exec_times[CAL_DISTANCE])
                    min_exec_times[CAL_DISTANCE] = exec_time[exec_time_idx + CAL_DISTANCE];
                if (exec_time[exec_time_idx + TOPK_SORT] < min_exec_times[TOPK_SORT])
                    min_exec_times[TOPK_SORT] = exec_time[exec_time_idx + TOPK_SORT];
                if (exec_time[exec_time_idx + TOPK_SAVING] < min_exec_times[TOPK_SAVING])
                    min_exec_times[TOPK_SAVING] = exec_time[exec_time_idx + TOPK_SAVING];
                // Avg
                avg_exec_times[PRE_DEFINING] += exec_time[exec_time_idx + PRE_DEFINING];
                avg_exec_times[AFFILIATE_OPS] += exec_time[exec_time_idx + AFFILIATE_OPS];
                avg_exec_times[CAL_RESIDUAL] += exec_time[exec_time_idx + CAL_RESIDUAL];
                avg_exec_times[CONSTR_LUT] += exec_time[exec_time_idx + CONSTR_LUT];
                avg_exec_times[CLUSTER_LOADING] += exec_time[exec_time_idx + CLUSTER_LOADING];
                avg_exec_times[CAL_DISTANCE] += exec_time[exec_time_idx + CAL_DISTANCE];
                avg_exec_times[TOPK_SORT] += exec_time[exec_time_idx + TOPK_SORT];
                avg_exec_times[TOPK_SAVING] += exec_time[exec_time_idx + TOPK_SAVING];
            }
        }
        printf("Max: PRE_DEFINING: %llu, AFFILIATE_OPS: %llu, CAL_RESIDUAL: %llu, CONSTR_LUT: %llu, CLUSTER_LOADING: %llu, CAL_DISTANCE: %llu, TOPK_SORT: %llu, TOPK_SAVING: %llu\n",  max_exec_times[PRE_DEFINING], max_exec_times[AFFILIATE_OPS], max_exec_times[CAL_RESIDUAL], max_exec_times[CONSTR_LUT], max_exec_times[CLUSTER_LOADING], max_exec_times[CAL_DISTANCE], max_exec_times[TOPK_SORT], max_exec_times[TOPK_SAVING]);
        printf("Min: PRE_DEFINING: %llu, AFFILIATE_OPS: %llu, CAL_RESIDUAL: %llu, CONSTR_LUT: %llu, CLUSTER_LOADING: %llu, CAL_DISTANCE: %llu, TOPK_SORT: %llu, TOPK_SAVING: %llu\n",  min_exec_times[PRE_DEFINING], min_exec_times[AFFILIATE_OPS], min_exec_times[CAL_RESIDUAL], min_exec_times[CONSTR_LUT], min_exec_times[CLUSTER_LOADING], min_exec_times[CAL_DISTANCE], min_exec_times[TOPK_SORT], min_exec_times[TOPK_SAVING]);
        printf("Avg: PRE_DEFINING: %lf, AFFILIATE_OPS: %lf, CAL_RESIDUAL: %lf, CONSTR_LUT: %lf, CLUSTER_LOADING: %lf, CAL_DISTANCE: %lf, TOPK_SORT: %lf, TOPK_SAVING: %lf\n",  avg_exec_times[PRE_DEFINING] / NR_TASKLETS / nr_all_dpus, avg_exec_times[AFFILIATE_OPS] / NR_TASKLETS / nr_all_dpus, avg_exec_times[CAL_RESIDUAL] / NR_TASKLETS / nr_all_dpus, avg_exec_times[CONSTR_LUT] / NR_TASKLETS / nr_all_dpus, avg_exec_times[CLUSTER_LOADING] / NR_TASKLETS / nr_all_dpus, avg_exec_times[CAL_DISTANCE] / NR_TASKLETS / nr_all_dpus, avg_exec_times[TOPK_SORT] / NR_TASKLETS / nr_all_dpus, avg_exec_times[TOPK_SAVING] / NR_TASKLETS / nr_all_dpus);
    }
    {
        unsigned long long int max_dpu_exec_time = 0;
        uint32_t max_dpu_id = 0;
        for (uint32_t nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
            auto &exec_time = exec_times[nr_dpu];
            for (unsigned long long int exec_time_idx = 0, exec_time_size = exec_time.size(); exec_time_idx < exec_time_size; ) {
                unsigned long long int dpu_exec_time = 0;
                unsigned long long int exec_time_end = exec_time_idx + MODULE_TYPES_END;
                for (; exec_time_idx < exec_time_end; ++exec_time_idx) {
                    dpu_exec_time += exec_time[exec_time_idx];
                }
                if (dpu_exec_time > max_dpu_exec_time) {
                    max_dpu_exec_time = dpu_exec_time, max_dpu_id = nr_dpu;
                }
            }
        }
        printf("Max DPU: id = %u, cycles = %llu, DPU[%u]:\n", max_dpu_id, max_dpu_exec_time, max_dpu_id);
        for (uint32_t nr_tasklet = 0; nr_tasklet < NR_TASKLETS; ++nr_tasklet) {
            printf("Tasklet[%u], sum = %llu: ", nr_tasklet, exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + PRE_DEFINING] + exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + AFFILIATE_OPS] + exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + CAL_RESIDUAL] + exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + CONSTR_LUT] + exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + CLUSTER_LOADING] + exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + CAL_DISTANCE] + exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + TOPK_SORT] + exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + TOPK_SAVING]);
            printf("PRE_DEFINING: %llu, AFFILIATE_OPS: %llu, CAL_RESIDUAL: %llu, CONSTR_LUT: %llu, CLUSTER_LOADING: %llu, CAL_DISTANCE: %llu, TOPK_SORT: %llu, TOPK_SAVING: %llu\n", exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + PRE_DEFINING], exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + AFFILIATE_OPS], exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + CAL_RESIDUAL], exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + CONSTR_LUT], exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + CLUSTER_LOADING], exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + CAL_DISTANCE], exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + TOPK_SORT], exec_times[max_dpu_id][nr_tasklet * MODULE_TYPES_END + TOPK_SAVING]);
        }
    }
    printf("DPU[0]:\n");
    for (uint32_t nr_tasklet = 0; nr_tasklet < NR_TASKLETS; ++nr_tasklet) {
        printf("Tasklet[%u]: PRE_DEFINING: %llu, AFFILIATE_OPS: %llu, CAL_RESIDUAL: %llu, CONSTR_LUT: %llu, CLUSTER_LOADING: %llu, CAL_DISTANCE: %llu, TOPK_SORT: %llu, TOPK_SAVING: %llu\n", nr_tasklet, exec_times[0][nr_tasklet * MODULE_TYPES_END + PRE_DEFINING], exec_times[0][nr_tasklet * MODULE_TYPES_END + AFFILIATE_OPS], exec_times[0][nr_tasklet * MODULE_TYPES_END + CAL_RESIDUAL], exec_times[0][nr_tasklet * MODULE_TYPES_END + CONSTR_LUT], exec_times[0][nr_tasklet * MODULE_TYPES_END + CLUSTER_LOADING], exec_times[0][nr_tasklet * MODULE_TYPES_END + CAL_DISTANCE], exec_times[0][nr_tasklet * MODULE_TYPES_END + TOPK_SORT], exec_times[0][nr_tasklet * MODULE_TYPES_END + TOPK_SAVING]);
    }
    printf("DPU[%u]:\n", nr_all_dpus - 1);
    for (uint32_t nr_tasklet = 0; nr_tasklet < NR_TASKLETS; ++nr_tasklet) {
        printf("Tasklet[%u]: PRE_DEFINING: %llu, AFFILIATE_OPS: %llu, CAL_RESIDUAL: %llu, CONSTR_LUT: %llu, CLUSTER_LOADING: %llu, CAL_DISTANCE: %llu, TOPK_SORT: %llu, TOPK_SAVING: %llu\n", nr_tasklet, exec_times[nr_all_dpus - 1][nr_tasklet * MODULE_TYPES_END + PRE_DEFINING], exec_times[nr_all_dpus - 1][nr_tasklet * MODULE_TYPES_END + AFFILIATE_OPS], exec_times[nr_all_dpus - 1][nr_tasklet * MODULE_TYPES_END + CAL_RESIDUAL], exec_times[nr_all_dpus - 1][nr_tasklet * MODULE_TYPES_END + CONSTR_LUT], exec_times[nr_all_dpus - 1][nr_tasklet * MODULE_TYPES_END + CLUSTER_LOADING], exec_times[nr_all_dpus - 1][nr_tasklet * MODULE_TYPES_END + CAL_DISTANCE], exec_times[nr_all_dpus - 1][nr_tasklet * MODULE_TYPES_END + TOPK_SORT], exec_times[nr_all_dpus - 1][nr_tasklet * MODULE_TYPES_END + TOPK_SAVING]);
    }
    printf("Tasklet[0]:\n");
    for (uint32_t nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
        printf("DPU[%u]: PRE_DEFINING: %llu, AFFILIATE_OPS: %llu, CAL_RESIDUAL: %llu, CONSTR_LUT: %llu, CLUSTER_LOADING: %llu, CAL_DISTANCE: %llu, TOPK_SORT: %llu, TOPK_SAVING: %llu\n", nr_dpu, exec_times[nr_dpu][PRE_DEFINING], exec_times[nr_dpu][AFFILIATE_OPS], exec_times[nr_dpu][CAL_RESIDUAL], exec_times[nr_dpu][CONSTR_LUT], exec_times[nr_dpu][CLUSTER_LOADING], exec_times[nr_dpu][CAL_DISTANCE], exec_times[nr_dpu][TOPK_SORT], exec_times[nr_dpu][TOPK_SAVING]);
    }
    printf("Tasklet[%u]:\n", NR_TASKLETS - 1);
    for (uint32_t nr_dpu = 0; nr_dpu < nr_all_dpus; ++nr_dpu) {
        printf("DPU[%u]: PRE_DEFINING: %llu, AFFILIATE_OPS: %llu, CAL_RESIDUAL: %llu, CONSTR_LUT: %llu, CLUSTER_LOADING: %llu, CAL_DISTANCE: %llu, TOPK_SORT: %llu, TOPK_SAVING: %llu\n", nr_dpu, exec_times[nr_dpu][MODULE_TYPES_END * (NR_TASKLETS - 1) + PRE_DEFINING], exec_times[nr_dpu][MODULE_TYPES_END * (NR_TASKLETS - 1) + AFFILIATE_OPS], exec_times[nr_dpu][MODULE_TYPES_END * (NR_TASKLETS - 1) + CAL_RESIDUAL], exec_times[nr_dpu][MODULE_TYPES_END * (NR_TASKLETS - 1) + CONSTR_LUT], exec_times[nr_dpu][MODULE_TYPES_END * (NR_TASKLETS - 1) + CLUSTER_LOADING], exec_times[nr_dpu][MODULE_TYPES_END * (NR_TASKLETS - 1) + CAL_DISTANCE], exec_times[nr_dpu][MODULE_TYPES_END * (NR_TASKLETS - 1) + TOPK_SORT], exec_times[nr_dpu][MODULE_TYPES_END * (NR_TASKLETS - 1) + TOPK_SAVING]);
    }
    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
#endif

    // 5. Save results
    printf("Result saving:\n");
    saveDataToFile(knnFileName, neighbors.data(), sizeof(pqueue_elem_t_mram), queryAmt * neighborAmt);
    printf("End of result saving:\n");
#ifdef PRINT_PERF_EACH_PHASE
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1e6 + (long)timecheck.tv_usec;
    printf("[Host]  Total time for IVFPQ searching: %.3lfs\n", (end - start) / 1e6);
#endif
}


int main(int argc, char **argv) {
    uint32_t dimAmt = 128;
    uint32_t neighborAmt = 10;
    uint32_t sliceAmt = 8;
    uint32_t queryBatchSize = 1;
    ADDRTYPE nprobe = 1;
    uint32_t DPUGroupSize = 1;
    uint32_t clusterSliceSize = 1;  // Unit: Byte. Note: it would be better to make this a multiple of `pointSize`
    uint32_t nb_mram = DPU_ALLOCATE_ALL;
    std::string clustersFileName = "clusters.bin";
    std::string queriesFileName = "query.bin";
    std::string squareResFileName = "squareRes.bin";
    std::string centroidsFileName = "centroids.bin";
    std::string codebookFileName = "codebook.bin";
    std::string clusterSizesFileName = "clusterSizes.bin";
    std::string clusterLayoutFileName = "clusterLayout.bin";
    std::string radiiFileName = "radii.bin";
    std::string squareRootsFileName = "squareRoots.bin";
    std::string knnFileName = "knn.bin";
#ifdef CYCLE_PERF_EVAL
    uint64_t frequency = 450 << 20;
    parse_args(argc, argv, &dimAmt, &neighborAmt, &sliceAmt, &queryBatchSize, &nprobe, &DPUGroupSize, &clusterSliceSize, &nb_mram, &frequency, clustersFileName, queriesFileName, squareResFileName, centroidsFileName, codebookFileName, clusterSizesFileName, clusterLayoutFileName, radiiFileName, squareRootsFileName, knnFileName);
#else
    parse_args(argc, argv, &dimAmt, &neighborAmt, &sliceAmt, &queryBatchSize, &nprobe, &DPUGroupSize, &clusterSliceSize, &nb_mram, clustersFileName, queriesFileName, squareResFileName, centroidsFileName, codebookFileName, clusterSizesFileName, clusterLayoutFileName, radiiFileName, squareRootsFileName, knnFileName);
#endif

    // Check parameters
    if (clusterSliceSize < sizeof(POINTTYPE) * sliceAmt) {
        fprintf(stderr, "Error: the parameter `clusterSliceSize` = %u, is not allowed to be smaller than the vector size = %lu! Exit now!\n", clusterSliceSize, sizeof(POINTTYPE) * sliceAmt);
        return 0;
    }

    printf("Allocating DPUs\n");
    dpu::DpuSet dpu_set = dpu::DpuSet::allocate(nb_mram, ("nrJobPerRank=" + std::to_string(NR_JOB_PER_RANK) + ",dispatchOnAllRanks=true,cycleAccurate=true"));
    printf("DPUs allocated\n");
    printf("Using %u MRAMs already loaded\n", nb_mram);
#ifdef CYCLE_PERF_EVAL
    printf("Config: dimAmt = %u, neighborAmt = %u, sliceAmt = %u, queryBatchSize = %u, nprobe = %u, DPUGroupSize = %u, clusterSliceSize = %u, pruneSliceAmt = %u, nb_mram = %u, nr_tasklets = %u, frequency = %lu, clustersFileName = %s, queriesFileName = %s, squareResFileName = %s, centroidsFileName = %s, codebookFileName = %s, clusterSizesFileName = %s, clusterLayoutFileName = %s, radiiFileName = %s, squareRootsFileName = %s, knnFileName = %s\n", dimAmt, neighborAmt, sliceAmt, queryBatchSize, nprobe, DPUGroupSize, clusterSliceSize, PRUNE_SLICE_AMT, nb_mram, NR_TASKLETS, frequency, clustersFileName.c_str(), queriesFileName.c_str(), squareResFileName.c_str(), centroidsFileName.c_str(), codebookFileName.c_str(), clusterSizesFileName.c_str(), clusterLayoutFileName.c_str(), radiiFileName.c_str(), squareRootsFileName.c_str(), knnFileName.c_str());
#else
    printf("Config: dimAmt = %u, neighborAmt = %u, sliceAmt = %u, queryBatchSize = %u, nprobe = %u, DPUGroupSize = %u, clusterSliceSize = %u, pruneSliceAmt = %u, nb_mram = %u, nr_tasklets = %u, clustersFileName = %s, queriesFileName = %s, squareResFileName = %s, centroidsFileName = %s, codebookFileName = %s, clusterSizesFileName = %s, clusterLayoutFileName = %s, radiiFileName = %s, squareRootsFileName = %s, knnFileName = %s\n", dimAmt, neighborAmt, sliceAmt, queryBatchSize, nprobe, DPUGroupSize, clusterSliceSize, PRUNE_SLICE_AMT, nb_mram, NR_TASKLETS, clustersFileName.c_str(), queriesFileName.c_str(), squareResFileName.c_str(), centroidsFileName.c_str(), codebookFileName.c_str(), clusterSizesFileName.c_str(), clusterLayoutFileName.c_str(), radiiFileName.c_str(), squareRootsFileName.c_str(), knnFileName.c_str());
#endif

    /************************************************************************************* Same as the definition in the layout generation program *************************************************************************************/
    const double DPUfrequency = 350000000;
    const double WRAMbandwidth = 1612.56 * 1024 * 1024;  // WRAM(ADD)
    const double MRAMbandwidth =  573.79 * 1024 * 1024;  // MRAM(ADD)
    const ADDRTYPE codebookEntryAmt = getElemsAmount(codebookFileName.c_str(), dimAmt * sizeof(CB_TYPE));
    std::vector<UINT64> latencyVec = { 0, 
                                       static_cast<UINT64>(std::ceil((((sizeof(ELEMTYPE) + sizeof(CENTROIDS_TYPE)) / MRAMbandwidth + (sizeof(ELEMTYPE) + sizeof(CENTROIDS_TYPE)) / WRAMbandwidth) * DPUfrequency + 1) * dimAmt)),  // query/centroid: MRAM -> WRAM -> register + ADD
                                       static_cast<UINT64>(std::ceil((((sizeof(CB_TYPE) / MRAMbandwidth + (sizeof(CENTROIDS_TYPE) + sizeof(CB_TYPE)) / WRAMbandwidth + sizeof(ELEMTYPE) / WRAMbandwidth) * DPUfrequency + 1) * dimAmt + (dimAmt - sliceAmt)) * codebookEntryAmt)),  // residual: WRAM -> register; codebook: MRAM -> WRAM -> register + (SUB + MUL); ADD(accumulate: dim - sliceAmt). MUL = WRAM -> register
                                       static_cast<UINT64>(std::ceil((sizeof(POINTTYPE) / MRAMbandwidth + (sizeof(POINTTYPE) + sizeof(VECSUMTYPE)) / WRAMbandwidth) * sliceAmt + (sliceAmt - 1))),  // point: MRAM -> WRAM -> register; LUT[point]: WRAM -> register + ADD
                                       static_cast<UINT64>(std::ceil(((sizeof(VECSUMTYPE) + sizeof(CLUSTER_SIZES_TYPE)) / WRAMbandwidth + 1) * log2(neighborAmt) + sizeof(VECSUMTYPE) / WRAMbandwidth))  // dist: WRAM -> register; pQueue: WRAM -> register -> WRAM + CMP
                                       };
    const UINT64 *latency = latencyVec.data();
    /**************************************************************************************************** End of latenct definition ****************************************************************************************************/

#ifdef CYCLE_PERF_EVAL
    allocated_and_compute(dpu_set, dimAmt, neighborAmt, sliceAmt, queryBatchSize, nprobe, DPUGroupSize, clusterSliceSize, frequency, latency, clustersFileName.c_str(), queriesFileName.c_str(), squareResFileName.c_str(), centroidsFileName.c_str(), codebookFileName.c_str(), clusterSizesFileName.c_str(), clusterLayoutFileName.c_str(), radiiFileName.c_str(), squareRootsFileName.c_str(), knnFileName.c_str());
#else
    allocated_and_compute(dpu_set, dimAmt, neighborAmt, sliceAmt, queryBatchSize, nprobe, DPUGroupSize, clusterSliceSize, latency, clustersFileName.c_str(), queriesFileName.c_str(), squareResFileName.c_str(), centroidsFileName.c_str(), codebookFileName.c_str(), clusterSizesFileName.c_str(), clusterLayoutFileName.c_str(), radiiFileName.c_str(), squareRootsFileName.c_str(), knnFileName.c_str());
#endif

return 0;
}
