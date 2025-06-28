# Roofline model
A roofline model of ANNS on various platforms with several datasets. (The files are going to be public later after arrangement.)
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
