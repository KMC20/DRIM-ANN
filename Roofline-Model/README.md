# Roofline model
A roofline model of ANNS on various platforms with several datasets.
## How to test
```
python roofline_model.py --datasets-config-file-name config/datasets.json --platforms-config-file-name config/platforms.json --result-excel-file-name result/result.xls --result-plot-file-name result/result.png
```
## How to check the results
Check the generated png and excel files in the `result` folder. Please refer to Section2.1 of the SC2025 article for more details.
## Reference
If you feel this repo is useful for you, don't hesitate to star this!😀 And it is really kind of you to cite this repo in your paper or project.

If you feel DRIM-ANN is interesting, please cite:

```
@inproceedings{10.1145/3712285.3759801,
author = {Chen, Mingkai and Han, Tianhua and Liu, Cheng and Liang, Shengwen and Yu, Kuai and Dai, Lei and Yuan, Ziming and Wang, Ying and Zhang, Lei and Li, Huawei and Li, Xiaowei},
title = {DRIM-ANN: An Approximate Nearest Neighbor Search Engine based on Commercial DRAM-PIMs},
year = {2025},
isbn = {9798400714665},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3712285.3759801},
doi = {10.1145/3712285.3759801},
abstract = {Approximate nearest neighbor search (ANNS) is essential for applications like recommendation systems and retrieval-augmented generation (RAG) but is highly I/O-intensive and memory-demanding. CPUs face I/O bottlenecks, while GPUs are constrained by limited memory. DRAM-based Processing-in-Memory (DRAM-PIM) offers a promising alternative by providing high bandwidth, large memory capacity, and near-data computation. This work introduces DRIM-ANN, the first optimized ANNS engine leveraging UPMEM’s DRAM-PIM. While UPMEM scales memory bandwidth and capacity, it suffers from low computing power because of the limited processor embedded in each DRAM bank. To address this, we systematically optimize ANNS approximation configurations and replace expensive squaring operations with lookup tables to align the computing requirements with UPMEM’s architecture. Additionally, we propose load-balancing and I/O optimization strategies to maximize parallel processing efficiency. Experimental results show that DRIM-ANN achieves a 2.46\texttimes{} speedup over a 32-thread CPU and up to 2.67\texttimes{} over a GPU when deployed on computationally enhanced PIM platforms.},
booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
pages = {820–836},
numpages = {17},
keywords = {Processing in memory (PIM), DRAM PIM, ANNS, approximate computing},
location = {
},
series = {SC '25}
}
```
or
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
