BUILDDIR ?= build

HOST_BINARY=${BUILDDIR}/host_app
HOST_SOURCES=$(wildcard host/*.cpp host/*.c host/tools/src/*.cpp)
HOST_HEADERS=$(wildcard host/*.h host/tools/inc/*.h)

DPU_SOURCES=$(wildcard dpu/src/*.c dpu/libpqueue/src/pqueue.c)
DPU_HEADERS=$(wildcard dpu/inc/*.h dpu/libpqueue/src/pqueue.h)
DPU_MAIN_CBS=dpu/CBS.c
DPU_BINARY_CBS=${BUILDDIR}/dpu_task_CBS

COMMONS_HEADERS=$(wildcard common/inc/*.h)

OUTPUT_FILE=${BUILDDIR}/output.txt
PLOTDATA_FILE=${BUILDDIR}/plotdata.csv

CHECK_FORMAT_FILES=${HOST_SOURCES} ${HOST_HEADERS} ${DPU_SOURCES} ${DPU_HEADERS} ${COMMONS_HEADERS} ${DPU_MAIN_CBS}
CHECK_FORMAT_DEPENDENCIES=$(addsuffix -check-format,${CHECK_FORMAT_FILES})

NR_TASKLETS ?= 12

PRUNE_SLICE_AMT ?= 0  # This param should fall in [0, M]

__dirs := $(shell mkdir -p ${BUILDDIR})

.PHONY: all clean run plotdata check check-format

all: ${HOST_BINARY} ${DPU_BINARY_CBS}
clean:
	rm -rf ${BUILDDIR}

###
### HOST APPLICATION
###
CXXFLAGS=-g -Wall -Werror -Wextra -O3 -std=c++11 `dpu-pkg-config --cflags dpu` -Ihost/tools/inc -Ihost/inc -Icommon/inc -mavx512bw -mavx512vl -mavx512f -mavx512cd -mavx512dq -mavx2 -mavx -msse2 -msse -DNR_TASKLETS=${NR_TASKLETS}
LDFLAGS=`dpu-pkg-config --libs dpu` -fopenmp -lfaiss

${HOST_BINARY}: ${HOST_SOURCES} ${HOST_HEADERS} ${COMMONS_HEADERS} ${DPU_BINARY_CBS}
	$(CXX) -o $@ ${HOST_SOURCES} $(LDFLAGS) $(CXXFLAGS) -DDPU_BINARY_CBS=\"$(realpath ${DPU_BINARY_CBS})\" -DPERF_EVAL -DCYCLE_PERF_EVAL -DPRUNE_SLICE_AMT=${PRUNE_SLICE_AMT}  # Performance evaluation
# 	$(CXX) -o $@ ${HOST_SOURCES} $(LDFLAGS) $(CXXFLAGS) -DDPU_BINARY_CBS=\"$(realpath ${DPU_BINARY_CBS})\" -DPERF_EVAL -DMODULE_PERF_EVAL -DPRUNE_SLICE_AMT=${PRUNE_SLICE_AMT}  # Fine-grained performance evaluation
# 	$(CXX) -o $@ ${HOST_SOURCES} $(LDFLAGS) $(CXXFLAGS) -DDPU_BINARY_CBS=\"$(realpath ${DPU_BINARY_CBS})\" -DPERF_EVAL -DENERGY_EVAL -DPRUNE_SLICE_AMT=${PRUNE_SLICE_AMT}  # Energy evaluation

###
### DPU BINARY
###
DPU_FLAGS=-g -O2 -Wall -Wextra -flto=thin -Idpu/inc -Icommon/inc -Idpu/libpqueue/src -DNR_TASKLETS=${NR_TASKLETS} -DSTACK_SIZE_DEFAULT=512 -DPERF_EVAL -DCYCLE_PERF_EVAL -DPRUNE_SLICE_AMT=${PRUNE_SLICE_AMT}  # Performance/energy evaluation
# DPU_FLAGS=-g -O2 -Wall -Wextra -flto=thin -Idpu/inc -Icommon/inc -Idpu/libpqueue/src -DNR_TASKLETS=${NR_TASKLETS} -DSTACK_SIZE_DEFAULT=512 -DPERF_EVAL -DMODULE_PERF_EVAL -DPRUNE_SLICE_AMT=${PRUNE_SLICE_AMT}  # Fine-grained performance evaluation

${DPU_BINARY_CBS}: ${DPU_MAIN_CBS} ${DPU_SOURCES} ${DPU_HEADERS} ${COMMONS_HEADERS}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} ${DPU_SOURCES} ${DPU_MAIN_CBS} -o $@

###
### EXECUTION & TEST
###
run: all
	${HOST_BINARY} -p offlineFiles/sift100mC8192M16.clusters -q datasets/SIFT100M/pured_bigann_query.bvecs -s offlineFiles/squareResUint8 -c offlineFiles/sift100mC8192M16.centroids -b offlineFiles/sift100mC8192M16.codebook -a offlineFiles/sift100mC8192M16U32.clusterSizes -r offlineFiles/sift100mC8192M16.radii -l offlineFiles/sift100mC8192P96M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/SIFT100M/sift100mC8192M16P96.neighbors -D 128 -K 10 -C 96 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/sift100mC16384M16.clusters -q datasets/SIFT100M/pured_bigann_query.bvecs -s offlineFiles/squareResUint8 -c offlineFiles/sift100mC16384M16.centroids -b offlineFiles/sift100mC16384M16.codebook -a offlineFiles/sift100mC16384M16U32.clusterSizes -r offlineFiles/sift100mC16384M16.radii -l offlineFiles/sift100mC16384P32M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/SIFT100M/sift100mC16384M16P32.neighbors -D 128 -K 10 -C 32 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/sift100mC16384M16.clusters -q datasets/SIFT100M/pured_bigann_query.bvecs -s offlineFiles/squareResUint8 -c offlineFiles/sift100mC16384M16.centroids -b offlineFiles/sift100mC16384M16.codebook -a offlineFiles/sift100mC16384M16U32.clusterSizes -r offlineFiles/sift100mC16384M16.radii -l offlineFiles/sift100mC16384P64M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/SIFT100M/sift100mC16384M16P64.neighbors -D 128 -K 10 -C 64 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/sift100mC16384M16.clusters -q datasets/SIFT100M/pured_bigann_query.bvecs -s offlineFiles/squareResUint8 -c offlineFiles/sift100mC16384M16.centroids -b offlineFiles/sift100mC16384M16.codebook -a offlineFiles/sift100mC16384M16U32.clusterSizes -r offlineFiles/sift100mC16384M16.radii -l offlineFiles/sift100mC16384P96M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/SIFT100M/sift100mC16384M16P96.neighbors -D 128 -K 10 -C 96 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/sift100mC16384M16.clusters -q datasets/SIFT100M/pured_bigann_query.bvecs -s offlineFiles/squareResUint8 -c offlineFiles/sift100mC16384M16.centroids -b offlineFiles/sift100mC16384M16.codebook -a offlineFiles/sift100mC16384M16U32.clusterSizes -r offlineFiles/sift100mC16384M16.radii -l offlineFiles/sift100mC16384P128M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/SIFT100M/sift100mC16384M16P128.neighbors -D 128 -K 10 -C 128 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/sift100mC32768M16.clusters -q datasets/SIFT100M/pured_bigann_query.bvecs -s offlineFiles/squareResUint8 -c offlineFiles/sift100mC32768M16.centroids -b offlineFiles/sift100mC32768M16.codebook -a offlineFiles/sift100mC32768M16U32.clusterSizes -r offlineFiles/sift100mC32768M16.radii -l offlineFiles/sift100mC32768P96M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/SIFT100M/sift100mC32768M16P96.neighbors -D 128 -K 10 -C 96 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/sift100mC65536M16.clusters -q datasets/SIFT100M/pured_bigann_query.bvecs -s offlineFiles/squareResUint8 -c offlineFiles/sift100mC65536M16.centroids -b offlineFiles/sift100mC65536M16.codebook -a offlineFiles/sift100mC65536M16U32.clusterSizes -r offlineFiles/sift100mC65536M16.radii -l offlineFiles/sift100mC65536P96M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/SIFT100M/sift100mC65536M16P96.neighbors -D 128 -K 10 -C 96 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/deep100mC8192M16.clusters -q datasets/DEEP100M/quantiUFB_query_b -s offlineFiles/squareResUint8 -c offlineFiles/deep100mC8192M16.centroids -b offlineFiles/deep100mC8192M16.codebook -a offlineFiles/deep100mC8192M16U32.clusterSizes -r offlineFiles/deep100mC8192M16.radii -l offlineFiles/deep100mC8192P96M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/DEEP100M/deep100mC8192M16P96.neighbors -D 96 -K 10 -C 96 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/deep100mC16384M16.clusters -q datasets/DEEP100M/quantiUFB_query_b -s offlineFiles/squareResUint8 -c offlineFiles/deep100mC16384M16.centroids -b offlineFiles/deep100mC16384M16.codebook -a offlineFiles/deep100mC16384M16U32.clusterSizes -r offlineFiles/deep100mC16384M16.radii -l offlineFiles/deep100mC16384P32M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/DEEP100M/deep100mC16384M16P32.neighbors -D 96 -K 10 -C 32 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/deep100mC16384M16.clusters -q datasets/DEEP100M/quantiUFB_query_b -s offlineFiles/squareResUint8 -c offlineFiles/deep100mC16384M16.centroids -b offlineFiles/deep100mC16384M16.codebook -a offlineFiles/deep100mC16384M16U32.clusterSizes -r offlineFiles/deep100mC16384M16.radii -l offlineFiles/deep100mC16384P64M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/DEEP100M/deep100mC16384M16P64.neighbors -D 96 -K 10 -C 64 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/deep100mC16384M16.clusters -q datasets/DEEP100M/quantiUFB_query_b -s offlineFiles/squareResUint8 -c offlineFiles/deep100mC16384M16.centroids -b offlineFiles/deep100mC16384M16.codebook -a offlineFiles/deep100mC16384M16U32.clusterSizes -r offlineFiles/deep100mC16384M16.radii -l offlineFiles/deep100mC16384P96M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/DEEP100M/deep100mC16384M16P96.neighbors -D 96 -K 10 -C 96 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/deep100mC16384M16.clusters -q datasets/DEEP100M/quantiUFB_query_b -s offlineFiles/squareResUint8 -c offlineFiles/deep100mC16384M16.centroids -b offlineFiles/deep100mC16384M16.codebook -a offlineFiles/deep100mC16384M16U32.clusterSizes -r offlineFiles/deep100mC16384M16.radii -l offlineFiles/deep100mC16384P128M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/DEEP100M/deep100mC16384M16P128.neighbors -D 96 -K 10 -C 128 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/deep100mC32768M16.clusters -q datasets/DEEP100M/quantiUFB_query_b -s offlineFiles/squareResUint8 -c offlineFiles/deep100mC32768M16.centroids -b offlineFiles/deep100mC32768M16.codebook -a offlineFiles/deep100mC32768M16U32.clusterSizes -r offlineFiles/deep100mC32768M16.radii -l offlineFiles/deep100mC32768P96M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/DEEP100M/deep100mC32768M16P96.neighbors -D 96 -K 10 -C 96 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1
	${HOST_BINARY} -p offlineFiles/deep100mC65536M16.clusters -q datasets/DEEP100M/quantiUFB_query_b -s offlineFiles/squareResUint8 -c offlineFiles/deep100mC65536M16.centroids -b offlineFiles/deep100mC65536M16.codebook -a offlineFiles/deep100mC65536M16U32.clusterSizes -r offlineFiles/deep100mC65536M16.radii -l offlineFiles/deep100mC65536P96M16S2560.clusterLayout -t offlineFiles/squareRootsUint8 -k ckpts/DEEP100M/deep100mC65536M16P96.neighbors -D 96 -K 10 -C 96 -Q 10000 -M 16 -S 2560 -U 2543 -G 1 >> ${OUTPUT_FILE} 2>&1

check:
	cat ${OUTPUT_FILE} | grep "matchs found" | diff output.txt -  # Useless

plotdata:
	echo "Mcc" > ${PLOTDATA_FILE}  # Useless
	cat ${OUTPUT_FILE} | grep "average execution time" | sed 's/\[DPU\]  average execution time.*= .* ms (\(.*\) Mcc)/\1/' >> ${PLOTDATA_FILE}  # Useless

%-check-format: %
	clang-format $< | diff -y --suppress-common-lines $< -  # Useless

check-format: ${CHECK_FORMAT_DEPENDENCIES}  # Useless
