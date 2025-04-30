# DRIM-ANN
## Dependency
* The faiss library: integrated to the cluster locating phase on CPU. Install by conda at: https://github.com/facebookresearch/faiss
  
To avoid dependency on faiss, this repo provides a version without faiss as well. You can try this by replacing the folder `host` by `hostNoFaiss`, and replacing `Makefile` by `Makefile.noFaiss`. This version supports hyperthreading and AVX2 by default, and you can change the use of `knn_L2sqr` into `knn_L2sqrAVX512` in `host/tools/src/tools.cpp` and the `CXXFLAGS` in `Makefile.noFaiss` into the annotated one if your machine supports AVX512. However, the implementation has not been deeply optimized, so it is 2~4 times slower than the faiss implementation, especially in cases that `nlist` is large.
## Index & query data
1. Browse the cloud storage link: https://pan.baidu.com/s/17qbkuQC80EE0dE_SDdljrA
2. Input the extracting code: cv43
3. Download the whole data folder and move it to the local folder.
## How to test
Make sure that index & query data have been downloaded and placed in this folder. Then execute the following commands:
```
mv data/* ./
rmdir data
mkdir build
mkdir -p ckpts/SIFT100M
mkdir ckpts/DEEP100M
make run
```
## How to check the results
Check the log and top-k neighbor files in the `build` and `ckpts` folders respectively. Please refer to the Appendix of Artifact Description/Artifact Evaluation of the SC2025 article for more details.
## Thanks
 * Library of priority queue. The implementations of priority queue in `dpu/libpqueue` in UPMEM_d and UPMEM_h are modified to suit the demands of UPMEM from the implementation here: https://github.com/vy/libpqueue
 * MSR fetching. The implementations of `rdmsr` and related functions in `host/measureEnergy.c` in UPMEM_d and UPMEM_h for energy measurement are modified from the implementation here: https://github.com/lixiaobai09/intel_power_consumption_get/blob/master/powerget.c
 * The basic directory structure of the repo is learned from the upmemGCiM project: https://github.com/KMC20/upmemGCiM
 * The faiss library: https://github.com/facebookresearch/faiss
 * SIFT dataset: http://corpus-texmex.irisa.fr
 * DEEP dataset: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
## Reference
If you feel this repo is useful for you, don't hesitate to star this!😀 And it is really kind of you to cite this repo in your paper or project.

If you feel DRIM-ANN is interesting, please cite:

```
@misc{chen2024drimannapproximatenearestneighbor,
      title={DRIM-ANN: An Approximate Nearest Neighbor Search Engine based on Commercial DRAM-PIMs}, 
      author={Mingkai Chen and Tianhua Han and Cheng Liu and Shengwen Liang and Kuai Yu and Lei Dai and Ziming Yuan and Ying Wang and Lei Zhang and Huawei Li and Xiaowei Li},
      year={2024},
      eprint={2410.15621},
      archivePrefix={arXiv},
      primaryClass={cs.PF},
      url={https://arxiv.org/abs/2410.15621}, 
}
```
