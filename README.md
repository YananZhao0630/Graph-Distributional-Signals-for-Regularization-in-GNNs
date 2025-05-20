# Graph Distributional Signals for Regularization in Graph Neural Networks

This repository contains the code for our IEEE TSPIN accepted paper.

## Table of Contents
- [Requirements](#requirements)
- [Reproducing Results](#reproducing-results)
- [Reference](#reference)
- [Citation](#citation)

## Requirements
To install the required dependencies, refer to the `environment.yml` file.

## Reproducing Results
We use GCN as the baseline model, implemented in the GCN file. To reproduce the results in Table 3, run the following command:

```bash
python train_GCN_RLoss.py 
python train_GCN_Preg.py 
python train_GCN_Lreg.py 


```
These scripts correspond to different regularization methods, referred to as R-GCN, P-GCN, and L-GCN, respectively. Among them, R-GCN is our proposed method.

For applying our regularization methods to other baseline models, please refer to the respective folders: CGNN/, GAT/, GraphCON/, PN-SI/, and WGNN/. Each folder contains the corresponding training scripts named xxx_RLoss.py, xxx_PLoss.py, and xxx_LLoss.py, where xxx denotes the model name.

## Reference

Our proposed regularization method is implemented based on different models (e.g., CGNN, GraphCON, PN-SI, and WGNN) , developed based on the following repo: 

GraphCON: https://github.com/tk-rusch/GraphCON

PN-SI: https://github.com/LingxiaoShawn/PairNorm

WGNN: https://github.com/amblee0306/label-non-uniformity-gnn

CGNN: https://github.com/GeoX-Lab/CGNN


## Citation
If you find our helpful, consider to cite us:

```bibtex
@ARTICLE{JiZha2025,
	author = {Feng Ji, Yanan Zhao, See Hian Lee, Kai Zhao, Wee Peng Tay and Jielong Yang},
	title={Graph Distributional Signals for Regularization in Graph Neural Networks},
	journal= {{IEEE} Transactions on Signal and Information Processing over Networks}, 
	year={2025}
}
```
