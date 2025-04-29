# DRIM-ANN
## How to test
```
unzip -q data.zip
rm data.zip
mkdir build
mkdir -p ckpts/SIFT100M
mkdir -p ckpts/DEEP100M
make run
```
## How to check the results
Please refer to the Appendix of Artifact Description/Artifact Evaluation of the SC2025 article.
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
